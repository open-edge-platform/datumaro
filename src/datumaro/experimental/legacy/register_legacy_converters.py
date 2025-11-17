# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Legacy dataset conversion functionality.

This module provides functionality to convert legacy Datumaro datasets to the new
dataset format with automatic schema inference and type conversion.
"""

from __future__ import annotations

from typing import Any, cast

from datumaro import Dataset as LegacyDataset
from datumaro.components.media import MediaElement
from datumaro.experimental import Semantic
from datumaro.experimental.legacy.annotation_converters import (
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
    _annotation_converters,
)
from datumaro.experimental.legacy.media_converters import (
    BackwardImageMediaConverter,
    BackwardMediaConverter,
    ForwardImageMediaConverter,
    ForwardMediaConverter,
)

# Global registries
_media_converter_classes: dict[type[MediaElement[Any]], type[ForwardMediaConverter]] = {}


def register_forward_media_converter(converter_class: type[ForwardMediaConverter]) -> None:
    """Register a forward converter class for media types it supports."""
    for media_type in converter_class.get_supported_media_types():
        _media_converter_classes[media_type] = converter_class


def register_forward_annotation_converter(
    converter_class: type[ForwardAnnotationConverter],
) -> None:
    """Register a forward converter class for annotation types it supports."""
    for annotation_type in converter_class.get_supported_annotation_types():
        _annotation_converters[annotation_type] = converter_class


def register_builtin_forward_converters() -> None:
    """Register built-in forward converters for common types."""

    # Media converters
    register_forward_media_converter(ForwardImageMediaConverter)

    # Annotation converters
    register_forward_annotation_converter(ForwardMaskAnnotationConverter)
    register_forward_annotation_converter(ForwardBboxAnnotationConverter)
    register_forward_annotation_converter(ForwardLabelAnnotationConverter)
    register_forward_annotation_converter(ForwardKeypointAnnotationConverter)
    register_forward_annotation_converter(ForwardPolygonAnnotationConverter)
    register_forward_annotation_converter(ForwardRotatedBboxAnnotationConverter)
    register_forward_annotation_converter(ForwardEllipseAnnotationConverter)


# Global registries for backward converters
_backward_media_converter_classes: list[type[BackwardMediaConverter]] = []
_backward_annotation_converter_classes: list[type[BackwardAnnotationConverter]] = []


def register_backward_media_converter(converter_class: type[BackwardMediaConverter]) -> None:
    """Register a backward converter class for a media type."""
    _backward_media_converter_classes.append(converter_class)


def register_backward_annotation_converter(
    converter_class: type[BackwardAnnotationConverter],
) -> None:
    """Register a backward converter class for an annotation type."""
    _backward_annotation_converter_classes.append(converter_class)


def register_builtin_backward_converters() -> None:
    """Register built-in backward converters."""

    # Register backward media converters
    register_backward_media_converter(BackwardImageMediaConverter)

    # Register backward annotation converters
    register_backward_annotation_converter(BackwardBboxAnnotationConverter)
    register_backward_annotation_converter(BackwardPolygonAnnotationConverter)
    register_backward_annotation_converter(BackwardRotatedBboxAnnotationConverter)


# Auto-register built-in converters when module is imported
register_builtin_forward_converters()
register_builtin_backward_converters()


def get_forward_media_converter(
    dataset: LegacyDataset, semantic: Semantic = Semantic.Default, name_prefix: str = ""
) -> ForwardMediaConverter | None:
    """Get forward converter for a dataset by trying registered converters.

    Args:
        dataset: Legacy dataset to create converter from
        semantic: The semantic type for the converted fields
        name_prefix: Prefix to prepend to all field names
    """
    # Get the dataset's media type
    media_type = cast("type[MediaElement[Any]]", dataset.media_type())

    # Try converter registered for this specific media type
    if media_type in _media_converter_classes:
        converter_class = _media_converter_classes[media_type]
        return converter_class.create(dataset, semantic, name_prefix)

    return None
