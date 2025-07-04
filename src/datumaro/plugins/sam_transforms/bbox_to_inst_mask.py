# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""Bbox-to-instance mask transform using Segment Anything Model"""

import os.path as osp
from typing import List, Optional

import datumaro.plugins.sam_transforms.interpreters.sam_decoder_for_bbox as sam_decoder_for_bbox_interp
import datumaro.plugins.sam_transforms.interpreters.sam_encoder as sam_encoder_interp
from datumaro.components.annotation import Bbox, Mask, Polygon
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetItem, IDataset
from datumaro.components.transformer import ModelTransform
from datumaro.plugins.inference_server_plugin import OVMSLauncher, TritonLauncher
from datumaro.plugins.inference_server_plugin.base import (
    InferenceServerType,
    ProtocolType,
    TLSConfig,
)
from datumaro.util.deprecation import deprecated
from datumaro.util.mask_tools import extract_contours

__all__ = ["SAMBboxToInstanceMask"]


@deprecated(deprecated_version="1.11", removed_version="1.12")
class SAMBboxToInstanceMask(ModelTransform, CliPlugin):
    """Convert bounding boxes to instance mask using Segment Anything Model.

    This transform convert all the `Bbox` annotations in the dataset item to
    `Mask` or `Polygon` annotations (`Mask` is default).
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
        to_polygon: If true, the output `Mask` annotations will be converted to `Polygon` annotations.
        num_workers: The number of worker threads to use for parallel inference.
            Set to 0 for single-process mode. Default is 0.
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
        to_polygon: bool = False,
        num_workers: int = 0,
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
            model_interpreter_path=osp.abspath(sam_decoder_for_bbox_interp.__file__),
            model_version=1,
            host=host,
            port=port,
            timeout=timeout,
            tls_config=tls_config,
            protocol_type=protocol_type,
        )

        super().__init__(
            extractor,
            launcher=self._sam_encoder_launcher,
            batch_size=1,
            append_annotation=False,
            num_workers=num_workers,
        )
        self._to_polygon = to_polygon

    def _process_batch(
        self,
        batch: List[DatasetItem],
    ) -> List[DatasetItem]:
        img_embeds = self._sam_encoder_launcher.launch(
            batch=[item for item in batch if self._sam_encoder_launcher.type_check(item)]
        )

        items = []
        for item, img_embed in zip(batch, img_embeds):
            item_to_decode = item.wrap(annotations=item.annotations + img_embed)

            if not any(isinstance(ann, Bbox) for ann in item_to_decode.annotations):
                item_to_decode.annotations.pop()  # Pop the added image embedding
                items.append(item_to_decode)
                continue

            # Nested list of mask [[mask_0, ...]]
            nested_masks: List[List[Mask]] = self._sam_decoder_launcher.launch(
                [item_to_decode],
                stack=False,
            )

            # Pop the added image embedding
            item_to_decode.annotations.pop()
            # Leave non-bbox annotations only
            item_to_decode.annotations = [
                ann for ann in item_to_decode.annotations if not isinstance(ann, Bbox)
            ]

            item_to_decode.annotations += (
                self._convert_to_polygon(nested_masks[0]) if self._to_polygon else nested_masks[0]
            )

            items.append(item_to_decode)

        return items

    @staticmethod
    def _convert_to_polygon(masks: List[Mask]):
        return [
            Polygon(
                points=contour,
                id=mask.id,
                attributes=mask.attributes,
                group=mask.group,
                object_id=mask.object_id,
                label=mask.label,
                z_order=mask.z_order,
            )
            for mask in masks
            for contour in extract_contours(mask.image)
        ]
