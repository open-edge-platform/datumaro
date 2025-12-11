# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Base Field class implementations for various data types.

This module provides concrete field implementations that handle serialization
to/from Polars DataFrames for different data types commonly used in machine
learning and computer vision applications.
"""

from datumaro.experimental.fields.annotations import (
    BBoxField,
    EllipseField,
    KeypointsField,
    LabelField,
    PolygonField,
    RotatedBBoxField,
    bbox_field,
    caption_field,
    keypoints_field,
    label_field,
    polygon_field,
    rotated_bbox_field,
)
from datumaro.experimental.fields.base import Field, convert_numpy_object_array_to_series
from datumaro.experimental.fields.datasets import Subset, SubsetField, TileField, TileInfo, subset_field, tile_field
from datumaro.experimental.fields.images import (
    ImageBytesField,
    ImageCallableField,
    ImageField,
    ImageInfo,
    ImageInfoField,
    ImagePathField,
    ImagePathLike,
    LazyImage,
    TensorField,
    image_bytes_field,
    image_callable_field,
    image_field,
    image_info_field,
    image_path_field,
    tensor_field,
)
from datumaro.experimental.fields.masks import (
    InstanceMaskCallableField,
    InstanceMaskField,
    MaskCallableField,
    MaskField,
    instance_mask_callable_field,
    instance_mask_field,
    mask_callable_field,
    mask_field,
)
from datumaro.experimental.fields.types import NumericField, bool_field, numeric_field

__all__ = [
    "BBoxField",
    "EllipseField",
    "Field",
    "ImageBytesField",
    "ImageCallableField",
    "ImageField",
    "ImageInfo",
    "ImageInfoField",
    "ImagePathField",
    "ImagePathLike",
    "InstanceMaskCallableField",
    "InstanceMaskField",
    "KeypointsField",
    "LabelField",
    "LazyImage",
    "MaskCallableField",
    "MaskField",
    "NumericField",
    "PolygonField",
    "RotatedBBoxField",
    "Subset",
    "SubsetField",
    "TensorField",
    "TileField",
    "TileInfo",
    "bbox_field",
    "bool_field",
    "caption_field",
    "convert_numpy_object_array_to_series",
    "image_bytes_field",
    "image_callable_field",
    "image_field",
    "image_info_field",
    "image_path_field",
    "instance_mask_callable_field",
    "instance_mask_field",
    "keypoints_field",
    "label_field",
    "mask_callable_field",
    "mask_field",
    "numeric_field",
    "polygon_field",
    "rotated_bbox_field",
    "subset_field",
    "tensor_field",
    "tile_field",
]
