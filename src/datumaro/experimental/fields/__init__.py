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
    CaptionField,
    EllipseField,
    KeypointsField,
    LabelField,
    PolygonField,
    RotatedBBoxField,
    bbox_field,
    caption_field,
    ellipse_field,
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
from datumaro.experimental.fields.types import (
    BoolField,
    NumericField,
    StringField,
    bool_field,
    numeric_field,
    string_field,
)
from datumaro.experimental.fields.videos import (
    MediaPathField,
    VideoFrameCallableField,
    VideoFramePathField,
    VideoInfoField,
    media_path_field,
    video_frame_callable_field,
    video_frame_path_field,
    video_info_field,
)

__all__ = [
    "BBoxField",
    "BoolField",
    "CaptionField",
    "EllipseField",
    "Field",
    "ImageBytesField",
    "ImageCallableField",
    "ImageField",
    "ImageInfo",
    "ImageInfoField",
    "ImagePathField",
    "InstanceMaskCallableField",
    "InstanceMaskField",
    "KeypointsField",
    "LabelField",
    "MaskCallableField",
    "MaskField",
    "MediaPathField",
    "NumericField",
    "PolygonField",
    "RotatedBBoxField",
    "StringField",
    "Subset",
    "SubsetField",
    "TensorField",
    "TileField",
    "TileInfo",
    "VideoFrameCallableField",
    "VideoFramePathField",
    "VideoInfoField",
    "bbox_field",
    "bool_field",
    "caption_field",
    "convert_numpy_object_array_to_series",
    "ellipse_field",
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
    "media_path_field",
    "numeric_field",
    "polygon_field",
    "rotated_bbox_field",
    "string_field",
    "subset_field",
    "tensor_field",
    "tile_field",
    "video_frame_callable_field",
    "video_frame_path_field",
    "video_info_field",
]
