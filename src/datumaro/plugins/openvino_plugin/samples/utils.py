# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from datumaro.components.annotation import Bbox


def create_bboxes_with_rescaling(
    bboxes: np.ndarray, labels: np.ndarray, r_scale: float
) -> List[Bbox]:
    idx = 0
    anns = []
    for bbox, label in zip(bboxes, labels):
        points = r_scale * bbox[:4]
        x1, y1, x2, y2 = points
        conf = bbox[4]
        anns.append(
            Bbox(
                x=x1,
                y=y1,
                w=x2 - x1,
                h=y2 - y1,
                id=idx,
                label=label,
                attributes={"score": conf},
            )
        )
        idx += 1
    return anns


@dataclass
class RescaledImage:
    """Dataclass for a rescaled image.

    This dataclass represents a rescaled image along with the scaling information.

    Attributes:
        image: The rescaled image as a NumPy array.
        scale: The scale factor by which the image was resized to fit the model input size.
               The scale factor is the same for both height and width.

    Note:
        The `image` attribute stores the rescaled image as a NumPy array.
        The `scale` attribute represents the scale factor used to resize the image.
        The scale factor indicates how much the image was scaled to fit the model's input size.
    """

    image: np.ndarray
    scale: float


def rescale_img_keeping_aspect_ratio(
    img: np.ndarray, h_model: int, w_model: int, padding: bool = True
) -> RescaledImage:
    """
    Rescale image while maintaining its aspect ratio.

    This function rescales the input image to fit the requirements of the model input.
    It also attempts to preserve the original aspect ratio of the input image.
    If the aspect ratio of the input image does not match the aspect ratio required by the model,
    the function applies zero padding to the image boundaries to maintain the aspect ratio if `padding` option is true.

    Parameters:
        img: The image to be rescaled.
        h_model: The desired height of the image required by the model.
        w_model: The desired width of the image required by the model.
        padding: If true, pad the output image boundaries to make the output image size `(h_model, w_model).
            Otherwise, there is no pad, so that the output image size can be different with `(h_model, w_model)`.
    """
    assert len(img.shape) == 3

    h_img, w_img = img.shape[:2]

    scale = min(h_model / h_img, w_model / w_img)

    h_resize = min(int(scale * h_img), h_model)
    w_resize = min(int(scale * w_img), w_model)

    num_channel = img.shape[-1]

    if padding:
        resized_inputs = np.zeros((h_model, w_model, num_channel), dtype=np.uint8)

        resized_inputs[:h_resize, :w_resize, :] = cv2.resize(
            img,
            (w_resize, h_resize),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        resized_inputs = cv2.resize(
            img,
            (w_resize, h_resize),
            interpolation=cv2.INTER_LINEAR,
        )

    return RescaledImage(image=resized_inputs, scale=scale)
