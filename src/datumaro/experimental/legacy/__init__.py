# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
This module provides functionality to convert legacy Datumaro datasets to the new
dataset format with automatic schema inference and type conversion.
"""

from .annotation_converters import (
    BackwardAnnotationConverter,
    BackwardBboxAnnotationConverter,
    BackwardPolygonAnnotationConverter,
    BackwardRotatedBboxAnnotationConverter,
    ForwardAnnotationConverter,
    ForwardBboxAnnotationConverter,
    ForwardEllipseAnnotationConverter,
    ForwardKeypointAnnotationConverter,
    ForwardLabelAnnotationConverter,
    ForwardMaskAnnotationConverter,
    ForwardPolygonAnnotationConverter,
    ForwardRotatedBboxAnnotationConverter,
)
from .dataset_converters import (
    AnalysisResult,
    BackwardAnalysisResult,
    analyze_experimental_dataset,
    analyze_legacy_dataset,
    convert_from_legacy,
    convert_to_legacy,
)
from .media_converters import (
    BackwardImageMediaConverter,
    BackwardMediaConverter,
    BackwardMixedMediaConverter,
    BackwardVideoMediaConverter,
    ForwardImageMediaConverter,
    ForwardMediaConverter,
    ForwardMixedMediaConverter,
    ForwardVideoMediaConverter,
)
from .register_legacy_converters import (
    register_backward_annotation_converter,
    register_backward_media_converter,
    register_builtin_backward_converters,
    register_builtin_forward_converters,
    register_forward_annotation_converter,
    register_forward_media_converter,
)

__all__ = [
    "AnalysisResult",
    "BackwardAnalysisResult",
    "BackwardAnnotationConverter",
    "BackwardBboxAnnotationConverter",
    "BackwardImageMediaConverter",
    "BackwardMediaConverter",
    "BackwardMixedMediaConverter",
    "BackwardPolygonAnnotationConverter",
    "BackwardRotatedBboxAnnotationConverter",
    "BackwardVideoMediaConverter",
    "ForwardAnnotationConverter",
    "ForwardBboxAnnotationConverter",
    "ForwardEllipseAnnotationConverter",
    "ForwardImageMediaConverter",
    "ForwardKeypointAnnotationConverter",
    "ForwardLabelAnnotationConverter",
    "ForwardMaskAnnotationConverter",
    "ForwardMediaConverter",
    "ForwardMixedMediaConverter",
    "ForwardPolygonAnnotationConverter",
    "ForwardRotatedBboxAnnotationConverter",
    "ForwardVideoMediaConverter",
    "analyze_experimental_dataset",
    "analyze_legacy_dataset",
    "convert_from_legacy",
    "convert_to_legacy",
    "register_backward_annotation_converter",
    "register_backward_media_converter",
    "register_builtin_backward_converters",
    "register_builtin_forward_converters",
    "register_forward_annotation_converter",
    "register_forward_media_converter",
]
