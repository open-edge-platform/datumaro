# Copyright (C) 2022-2026 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import numpy as np
import polars as pl

from datumaro.experimental import Sample
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.fields import (
    ImageInfo,
    Subset,
    bbox_field,
    image_info_field,
    image_path_field,
    label_field,
    subset_field,
)


class YoloSample(Sample):
    """
    Sample class for YOLO format datasets.

    YOLO format stores object detection annotations with bounding boxes
    in normalized xywh format (center_x, center_y, width, height) relative
    to image dimensions, with each box having a class label.
    """

    # Basic image information
    image: str = image_path_field()
    image_info: ImageInfo = image_info_field()

    # Object detection annotations
    bboxes: np.ndarray | None = bbox_field(dtype=pl.Float32(), format="xywh")
    labels: np.ndarray | None = label_field(dtype=pl.UInt8(), is_list=True)

    # Dataset organization
    subset: Subset = subset_field()


class YoloCategories(LabelCategories):
    """
    Categories class for YOLO format datasets.

    YOLO stores class names in either:
    - obj.names file (one class per line)
    - data.yaml file (for Ultralytics format)
    """

    def __init__(self, labels: tuple[str, ...] | list[str] | None = None):
        """
        Initialize YoloCategories with a list of labels.

        Args:
            labels: Tuple or list of label names. If None, creates empty categories.
        """
        provided = tuple(labels) if labels is not None else ()
        super().__init__(labels=provided)
