# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Legacy dataset conversion functionality.

This module provides functionality to convert legacy Datumaro datasets to the new
v2 dataset format with automatic schema inference and type conversion.
"""
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
This module provides functionality to convert legacy Datumaro datasets to the new
v2 dataset format with automatic schema inference and type conversion.
"""

from .annotation_converters import (
    ForwardAnnotationConverter,
    ForwardBboxAnnotationConverter,
    ForwardRotatedBboxAnnotationConverter,
    ForwardPolygonAnnotationConverter,
    ForwardLabelAnnotationConverter,
    ForwardKeypointAnnotationConverter,
    ForwardEllipseAnnotationConverter,
    ForwardMaskAnnotationConverter,
    BackwardAnnotationConverter,
    BackwardBboxAnnotationConverter,
    BackwardRotatedBboxAnnotationConverter,
    BackwardPolygonAnnotationConverter,
)
from .dataset_converters import (
    AnalysisResult,
    BackwardAnalysisResult,
    analyze_legacy_dataset,
    analyze_experimental_dataset,
    convert_from_legacy,
    convert_to_legacy,
)
from .media_converters import (
    ForwardMediaConverter,
    ForwardImageMediaConverter,
    BackwardMediaConverter,
    BackwardImageMediaConverter,
)
from .register_legacy_converters import (
    register_forward_media_converter,
    register_forward_annotation_converter,
    register_builtin_forward_converters,
    register_backward_media_converter,
    register_backward_annotation_converter,
    register_builtin_backward_converters,
)

__all__ = [
    "ForwardAnnotationConverter",
    "ForwardBboxAnnotationConverter",
    "ForwardRotatedBboxAnnotationConverter",
    "ForwardPolygonAnnotationConverter",
    "ForwardLabelAnnotationConverter",
    "ForwardKeypointAnnotationConverter",
    "ForwardEllipseAnnotationConverter",
    "ForwardMaskAnnotationConverter",
    "BackwardAnnotationConverter",
    "BackwardBboxAnnotationConverter",
    "BackwardRotatedBboxAnnotationConverter",
    "BackwardPolygonAnnotationConverter",
    "AnalysisResult",
    "BackwardAnalysisResult",
    "analyze_legacy_dataset",
    "analyze_experimental_dataset",
    "convert_from_legacy",
    "convert_to_legacy",
    "ForwardMediaConverter",
    "ForwardImageMediaConverter",
    "BackwardMediaConverter",
    "BackwardImageMediaConverter",
    "register_forward_media_converter",
    "register_forward_annotation_converter",
    "register_builtin_forward_converters",
    "register_backward_media_converter",
    "register_backward_annotation_converter",
    "register_builtin_backward_converters",
]
