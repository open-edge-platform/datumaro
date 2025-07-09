# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""Automatic mask generation using Segment Anything Model"""

import os.path as osp
from typing import List, Optional

import numpy as np

import datumaro.plugins.sam_transforms.interpreters.sam_decoder_for_amg as sam_decoder_for_amg
import datumaro.plugins.sam_transforms.interpreters.sam_encoder as sam_encoder_interp
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetItem, IDataset
from datumaro.components.transformer import ModelTransform
from datumaro.plugins.inference_server_plugin import OVMSLauncher, TritonLauncher
from datumaro.plugins.inference_server_plugin.base import (
    InferenceServerType,
    ProtocolType,
    TLSConfig,
)
from datumaro.plugins.sam_transforms.interpreters.sam_decoder_for_amg import AMGMasks, AMGPoints
from datumaro.util.deprecation import deprecated

__all__ = ["SAMAutomaticMaskGeneration"]


@deprecated(deprecated_version="1.11", removed_version="1.12")
class SAMAutomaticMaskGeneration(ModelTransform, CliPlugin):
    """Produce instance segmentation masks automatically using Segment Anything Model (SAM).

    This transform can produce instance segmentation mask annotations for each given image.
    It samples single-point input prompts on a uniform 2D grid over the image.
    For each prompt, SAM can predict multiple masks. After obtaining the mask candidates,
    it post-processes them using the given parameters to improve quality and remove duplicates.

    It uses the Segment Anything Model deployed in the OpenVINO™ Model Server
    or NVIDIA Triton™ Inference Server instance. To launch the server instance,
    please see the guide in this link:
    https://github.com/open-edge-platform/datumaro/tree/develop/docker/segment-anything/README.md

    Parameters:
        extractor: Dataset to transform
        inference_server_type: Inference server type:
            `InferenceServerType.ovms` or `InferenceServerType.triton`
        host: Host address of the server instance
        port: Port number of the server instance
        timeout: Timeout limit during communication between the client and the server instance
        tls_config: Configuration required if the server instance is in the secure mode
        protocol_type: Communication protocol type with the server instance
        num_workers: The number of worker threads to use for parallel inference.
            Set to 0 for single-process mode. Default is 0.
        points_per_side (int): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2 on a uniform 2d grid.
        points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
        pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
        stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
        stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
        box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
        min_mask_region_area (int): If >0, postprocessing will be applied
            to remove the binary mask which has the number of 1s less than min_mask_region_area.
    """

    def __init__(
        self,
        extractor: IDataset,
        inference_server_type: InferenceServerType = InferenceServerType.ovms,
        host: str = "localhost",
        port: int = 9000,
        timeout: float = 10.0,
        tls_config: Optional[TLSConfig] = None,
        protocol_type: ProtocolType = ProtocolType.grpc,
        num_workers: int = 0,
        points_per_side: int = 32,
        points_per_batch: int = 128,
        mask_threshold: float = 0.0,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 0,
    ):
        if inference_server_type == InferenceServerType.ovms:
            launcher_cls = OVMSLauncher
        elif inference_server_type == InferenceServerType.triton:
            launcher_cls = TritonLauncher
        else:
            raise ValueError(inference_server_type)

        self._sam_encoder_launcher = launcher_cls(
            model_name="sam_encoder",
            model_interpreter_path=osp.abspath(sam_encoder_interp.__file__),
            model_version=1,
            host=host,
            port=port,
            timeout=timeout,
            tls_config=tls_config,
            protocol_type=protocol_type,
        )
        self._sam_decoder_launcher = launcher_cls(
            model_name="sam_decoder",
            model_interpreter_path=osp.abspath(sam_decoder_for_amg.__file__),
            model_version=1,
            host=host,
            port=port,
            timeout=timeout,
            tls_config=tls_config,
            protocol_type=protocol_type,
        )

        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.mask_threshold = mask_threshold
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_offset = stability_score_offset
        self.stability_score_thresh = stability_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.min_mask_region_area = min_mask_region_area

        super().__init__(
            extractor,
            launcher=self._sam_encoder_launcher,
            batch_size=1,
            append_annotation=False,
            num_workers=num_workers,
        )

    @property
    def points_per_side(self) -> int:
        return self._points_per_side

    @points_per_side.setter
    def points_per_side(self, points_per_side: int) -> None:
        points_y = (np.arange(points_per_side) + 0.5) / points_per_side
        points_x = (np.arange(points_per_side) + 0.5) / points_per_side

        points_x = np.tile(points_x[None, :], (points_per_side, 1))
        points_y = np.tile(points_y[:, None], (1, points_per_side))
        self._points_grid = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
        self._points_per_side = points_per_side

    def _process_batch(
        self,
        batch: List[DatasetItem],
    ) -> List[DatasetItem]:
        img_embeds = self._sam_encoder_launcher.launch(
            batch=[item for item in batch if self._sam_encoder_launcher.type_check(item)]
        )

        items = []
        for item, img_embed in zip(batch, img_embeds):
            amg_masks: List[AMGMasks] = []

            for i in range(0, len(self._points_grid), self.points_per_batch):
                amg_points = [AMGPoints(points=self._points_grid[i : i + self.points_per_batch])]
                item_to_decode = item.wrap(annotations=amg_points + img_embed)

                # Nested list of mask [[mask_0, ...]]
                nested_masks: List[List[AMGMasks]] = self._sam_decoder_launcher.launch(
                    [item_to_decode],
                    stack=False,
                )
                amg_masks += nested_masks[0]

            mask_anns = AMGMasks.cat(amg_masks).postprocess(
                mask_threshold=self.mask_threshold,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_offset=self.stability_score_offset,
                stability_score_thresh=self.stability_score_thresh,
                box_nms_thresh=self.box_nms_thresh,
                min_mask_region_area=self.min_mask_region_area,
            )

            items.append(item.wrap(annotations=mask_anns))

        return items
