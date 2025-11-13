# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Base Field class implementations for various data types.

This module provides concrete field implementations that handle serialization
to/from Polars DataFrames for different data types commonly used in machine
learning and computer vision applications.
"""

from datumaro.v2.fields.base import Field, Semantic, convert_numpy_object_array_to_series
from datumaro.v2.fields.annotations import (
    BBoxField,
    RotatedBBoxField,
    LabelField,
    ScoreField,
    PolygonField,
    KeypointsField,
    EllipseField,
    bbox_field,
    rotated_bbox_field,
    label_field,
    score_field,
    polygon_field,
    keypoints_field,
)
from datumaro.v2.fields.datasets import (
    Subset,
    TileInfo,
    TileField,
    SubsetField,
    tile_field,
    subset_field,
)
from datumaro.v2.fields.images import (
    TensorField,
    ImageField,
    ImageBytesField,
    ImageInfoField,
    ImagePathField,
    ImageCallableField,
    tensor_field,
    image_field,
    image_bytes_field,
    image_info_field,
    image_path_field,
    image_callable_field,
)
from datumaro.v2.fields.masks import (
    MaskField,
    InstanceMaskField,
    InstanceMaskCallableField,
    MaskCallableField,
    mask_field,
    instance_mask_field,
    instance_mask_callable_field,
    mask_callable_field,
)

__all__ = [
    "Field",
    "Semantic",
    "convert_numpy_object_array_to_series",
    "BBoxField",
    "RotatedBBoxField",
    "LabelField",
    "ScoreField",
    "PolygonField",
    "KeypointsField",
    "EllipseField",
    "bbox_field",
    "rotated_bbox_field",
    "label_field",
    "score_field",
    "polygon_field",
    "keypoints_field",
    "Subset",
    "TileInfo",
    "TileField",
    "SubsetField",
    "tile_field",
    "subset_field",
    "TensorField",
    "ImageField",
    "ImageBytesField",
    "ImageInfoField",
    "ImagePathField",
    "ImageCallableField",
    "tensor_field",
    "image_field",
    "image_bytes_field",
    "image_info_field",
    "image_path_field",
    "image_callable_field",
    "MaskField",
    "InstanceMaskField",
    "InstanceMaskCallableField",
    "MaskCallableField",
    "mask_field",
    "instance_mask_field",
    "instance_mask_callable_field",
    "mask_callable_field",
]