# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=unused-variable

import cv2
import numpy as np

from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.util import take_by
from datumaro.util.deprecation import deprecated

__all__ = ["RISE"]


@deprecated(deprecated_version="1.11", removed_version="1.12")
class RISE:
    """
    Implements RISE: Randomized Input Sampling for
    Explanation of Black-box Models algorithm.
    See explanations at: https://arxiv.org/pdf/1806.07421.pdf
    """

    def __init__(
        self,
        model,
        num_masks: int = 100,
        mask_size: int = 7,
        prob: float = 0.5,
        batch_size: int = 1,
    ):
        assert prob >= 0 and prob <= 1
        self.model = model
        self.num_masks = num_masks
        self.mask_size = mask_size
        self.prob = prob
        self.batch_size = batch_size

    def normalize_saliency(self, saliency):
        normalized_saliency = np.empty_like(saliency)
        for idx, sal in enumerate(saliency):
            normalized_saliency[idx, ...] = (sal - np.min(sal)) / (np.max(sal) - np.min(sal))
        return normalized_saliency

    def generate_masks(self, image_size):
        cell_size = np.ceil(np.array(image_size) / self.mask_size).astype(np.int8)
        up_size = tuple([(self.mask_size + 1) * int(cs) for cs in cell_size])

        grid = np.random.rand(self.num_masks, self.mask_size, self.mask_size) < self.prob
        grid = grid.astype("float32")

        masks = np.empty((self.num_masks, *image_size))
        for i in range(self.num_masks):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])

            # Linear upsampling and cropping
            masks[i, ...] = cv2.resize(grid[i], up_size, interpolation=cv2.INTER_LINEAR)[
                x : x + image_size[0], y : y + image_size[1]
            ]

        return masks

    def generate_masked_dataset(self, image, image_size, masks):
        input_image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)

        items = []
        for id, mask in enumerate(masks):
            masked_image = np.expand_dims(mask, axis=-1) * input_image
            items.append(
                DatasetItem(
                    id=id,
                    media=Image.from_numpy(masked_image),
                )
            )
        return Dataset.from_iterable(items)

    def apply(self, image, progressive=False):
        assert len(image.shape) in [2, 3], "Expected an input image in (H, W, C) format"
        if len(image.shape) == 3:
            assert image.shape[2] in [3, 4], "Expected BGR or BGRA input"
        image = image[:, :, :3].astype(np.float32)

        model = self.model

        image_size = model.inputs[0].shape
        logit_size = model.outputs[0].shape

        batch_size = image_size[0]
        if image_size[1] in [1, 3]:  # for CxHxW
            image_size = (image_size[2], image_size[3])
        elif image_size[3] in [1, 3]:  # for HxWxC
            image_size = (image_size[1], image_size[2])

        masks = self.generate_masks(image_size=image_size)
        masked_dataset = self.generate_masked_dataset(image, image_size, masks)

        saliency = np.zeros((logit_size[1], *image_size), dtype=np.float32)
        for batch_id, batch in enumerate(take_by(masked_dataset, batch_size)):
            outputs = model.launch(batch)

            for sample_id in range(len(batch)):
                mask = masks[batch_size * batch_id + sample_id]
                for class_idx in range(logit_size[1]):
                    score = outputs[sample_id][class_idx].attributes["score"]
                    saliency[class_idx, ...] += score * mask

                # [TODO] wonjuleee: support DRISE for detection model explainability
                # if isinstance(self.target, Label):
                #     logits = outputs[sample_id][0].vector
                #     max_score = logits[self.target.label]
                # elif isinstance(self.target, Bbox):
                #     preds = outputs[sample_id][0]
                #     max_score = 0
                #     for box in preds:
                #         if box[0] == self.target.label:
                #             confidence, box = box[1], box[2]
                #             score = iou(self.target.get_bbox, box) * confidence
                #             if score > max_score:
                #                 max_score = score
                # saliency += max_score * mask

                if progressive:
                    yield self.normalize_saliency(saliency)

        yield self.normalize_saliency(saliency)
