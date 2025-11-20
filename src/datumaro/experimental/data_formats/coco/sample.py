# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE
from functools import cached_property

import numpy as np
import polars as pl

from datumaro.experimental import (
    ImageInfo,
    Sample,
    Semantic,
    bbox_field,
    caption_field,
    image_id_field,
    image_info_field,
    image_path_field,
    label_field,
    polygon_field,
)
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.data_formats.coco.constants import COCO_LABEL_TO_SUPER
from datumaro.experimental.fields import Subset, bool_field, keypoints_field, numeric_field, subset_field


class CocoSample(Sample):
    # Basic image information
    image: str = image_path_field()
    image_info: ImageInfo = image_info_field()

    # Instance annotations (from instances_train/val)
    bboxes: np.ndarray | None = bbox_field(dtype=pl.Float32(), format="xywh")
    polygons: np.ndarray | None = polygon_field(dtype=pl.Float32())
    labels: np.ndarray | None = label_field(dtype=pl.Int32(), is_list=True)
    areas: np.ndarray | None = numeric_field(dtype=pl.Float32(), is_list=True)
    iscrowd: np.ndarray | None = bool_field(is_list=True)

    # Keypoint annotations (from person_keypoints_train/val)
    keypoints: np.ndarray | None = keypoints_field(dtype=pl.Float32())

    # Caption annotations (from captions_train/val)
    captions: np.ndarray | None = caption_field(is_list=True, semantic=Semantic.Caption)
    caption_group_ids: np.ndarray | None = label_field(dtype=pl.Int32(), is_list=True, semantic=Semantic.Caption)

    # Dataset organization
    subset: Subset = subset_field()
    image_id: int | None = image_id_field()

    @cached_property
    def area(self) -> float:
        """Compute area of the sample based on its polygons."""
        area = 0.0
        for polygon in self.polygons or []:
            if len(polygon) < 6:
                continue  # Not enough points to form a polygon
            x = polygon[0::2]
            y = polygon[1::2]
            area += 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area


class CocoCategories(LabelCategories):
    # Unified mapping of label -> super-category (preserves COCO order)
    label_to_super: dict[str, str] = COCO_LABEL_TO_SUPER
    # Convenience: ordered tuple of super-categories aligned with labels order
    super_categories: tuple[str, ...] = tuple(label_to_super.values())

    """Initializes CocoCategories object with all the default coco labels"""

    def __init__(self):
        # Initialize using the order defined in the mapping
        super().__init__(labels=tuple(self.label_to_super.keys()))

    def get_labels_by_super_category(self, super_category: str) -> list[str]:
        """
        Get all labels that belong to a specific super category.

        Args:
            super_category: The name of the super category

        Returns:
            List of label names that belong to the specified super category
        """
        return [label for label, sc in self.label_to_super.items() if sc == super_category]
