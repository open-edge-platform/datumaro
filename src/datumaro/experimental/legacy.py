# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Legacy dataset conversion functionality.

This module provides functionality to convert legacy Datumaro datasets to the new
experimental dataset format with automatic schema inference and type conversion.
"""

from abc import ABC, abstractmethod
from typing import Any, Type, cast

import numpy as np
import polars as pl

from datumaro.components.annotation import Annotation, AnnotationType, Bbox
from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import CategoriesInfo, DatasetItem
from datumaro.components.media import FromFileMixin, Image, MediaElement

from .dataset import Dataset, Sample
from .fields import bbox_field, image_path_field, tensor_field
from .schema import AttributeInfo, Schema


class MediaConverter(ABC):
    """Base class for media type converters."""

    @abstractmethod
    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        """Return schema attributes for this media type."""
        pass

    @abstractmethod
    def convert_item_media(self, item: DatasetItem) -> dict[str, Any]:
        """Convert media from a DatasetItem to experimental format."""
        pass


class AnnotationConverter(ABC):
    """Base class for annotation type converters."""

    @abstractmethod
    def get_schema_attributes(self, categories: CategoriesInfo) -> dict[str, AttributeInfo]:
        """Return schema attributes for this annotation type."""
        pass

    @abstractmethod
    def convert_annotations(
        self, annotations: list[Annotation], item: DatasetItem
    ) -> dict[str, Any]:
        """Convert annotations of this type to experimental format."""
        pass


# Global registries
_media_converters: dict[Type[MediaElement[Any]], MediaConverter] = {}
_annotation_converters: dict[AnnotationType, AnnotationConverter] = {}


def register_media_converter(
    media_type: Type[MediaElement[Any]], converter: MediaConverter
) -> None:
    """Register a converter for a media type."""
    _media_converters[media_type] = converter


def register_annotation_converter(
    annotation_type: AnnotationType, converter: AnnotationConverter
) -> None:
    """Register a converter for an annotation type."""
    _annotation_converters[annotation_type] = converter


def get_media_converter(media_type: Type[MediaElement[Any]]) -> MediaConverter:
    """Get converter for a media type."""
    for registered_type, converter in _media_converters.items():
        if issubclass(media_type, registered_type):
            return converter
    raise ValueError(f"No converter registered for media type: {media_type}")


def get_annotation_converter(annotation_type: AnnotationType) -> AnnotationConverter:
    """Get converter for an annotation type."""
    if annotation_type not in _annotation_converters:
        raise ValueError(f"No converter registered for annotation type: {annotation_type}")
    return _annotation_converters[annotation_type]


class ImageMediaConverter(MediaConverter):
    """Converter for Image media type."""

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        return {"image_path": AttributeInfo(type=str, annotation=image_path_field())}

    def convert_item_media(self, item: DatasetItem) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if isinstance(item.media, Image):  # pyright: ignore[reportUnknownMemberType]
            if isinstance(item.media, FromFileMixin):
                # Handle image path
                result["image_path"] = item.media.path
            else:
                raise ValueError(f"Unsupported media type for Image: {type(item.media)}")

        return result


class BboxAnnotationConverter(AnnotationConverter):
    """Converter for Bbox annotations."""

    def get_schema_attributes(self, categories: CategoriesInfo) -> dict[str, AttributeInfo]:
        return {
            "bboxes": AttributeInfo(
                type=np.ndarray, annotation=bbox_field(dtype=pl.Float32, format="x1y1x2y2")
            ),
            "bbox_labels": AttributeInfo(type=np.ndarray, annotation=tensor_field(dtype=pl.Int32)),
        }

    def convert_annotations(
        self, annotations: list[Annotation], item: DatasetItem
    ) -> dict[str, Any]:
        bboxes: list[list[float]] = []
        labels: list[int | None] = []

        for ann in annotations:
            if isinstance(ann, Bbox):
                # Convert from x,y,w,h to x1,y1,x2,y2 format
                bboxes.append([ann.x, ann.y, ann.x + ann.w, ann.y + ann.h])
                labels.append(ann.label)

        # Ensure proper shapes for empty arrays
        bboxes_array = np.array(bboxes, dtype=np.float32)
        if bboxes_array.shape == (0,):
            bboxes_array = bboxes_array.reshape(0, 4)

        return {"bboxes": bboxes_array, "bbox_labels": np.array(labels, dtype=np.int32)}


def register_builtin_converters():
    """Register built-in converters for common types."""

    # Media converters
    register_media_converter(Image, ImageMediaConverter())

    # Annotation converters
    register_annotation_converter(AnnotationType.bbox, BboxAnnotationConverter())


from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Result of legacy dataset analysis."""

    schema: Schema
    media_converter: MediaConverter | None
    ann_converters: dict[AnnotationType, AnnotationConverter]


def analyze_legacy_dataset(legacy_dataset: LegacyDataset) -> AnalysisResult:
    """Analyze legacy dataset and generate schema using registered converters.

    Args:
        legacy_dataset: The legacy Datumaro dataset to analyze

    Returns:
        AnalysisResult containing the inferred schema and converters
    """
    categories = legacy_dataset.categories()
    media_type = cast(
        Type[MediaElement[Any]], legacy_dataset.media_type()
    )  # pyright: ignore[reportUnknownMemberType]
    ann_types = legacy_dataset.ann_types()

    attributes: dict[str, AttributeInfo] = {}
    media_converter: MediaConverter | None = None
    ann_converters: dict[AnnotationType, AnnotationConverter] = {}

    # Get media attributes from converter
    try:
        media_converter = get_media_converter(media_type)
        attributes.update(media_converter.get_schema_attributes())
    except ValueError:
        # No converter for this media type - skip
        media_converter = None

    # Get annotation attributes from converters
    for ann_type in ann_types:
        try:
            ann_converter = get_annotation_converter(ann_type)
            ann_converters[ann_type] = ann_converter
            attributes.update(ann_converter.get_schema_attributes(categories))
        except ValueError:
            # No converter for this annotation type - skip
            continue

    schema = Schema(attributes=attributes)
    return AnalysisResult(
        schema=schema, media_converter=media_converter, ann_converters=ann_converters
    )


def _convert_legacy_item(item: DatasetItem, analysis_result: AnalysisResult) -> dict[str, Any]:
    """Convert item using converters from analysis result."""

    attributes: dict[str, Any] = {}

    # Convert media using the analyzed converter
    if analysis_result.media_converter:
        attributes.update(analysis_result.media_converter.convert_item_media(item))

    # Group annotations by type
    annotations_by_type: dict[AnnotationType, list[Annotation]] = {}
    for ann in item.annotations:
        ann_type = ann.type
        if ann_type not in annotations_by_type:
            annotations_by_type[ann_type] = []
        annotations_by_type[ann_type].append(ann)

    # Convert each annotation type using the analyzed converters
    for ann_type, anns in annotations_by_type.items():
        if ann_type in analysis_result.ann_converters:
            ann_converter = analysis_result.ann_converters[ann_type]
            attributes.update(ann_converter.convert_annotations(anns, item))

    return attributes


def convert_from_legacy(legacy_dataset: LegacyDataset) -> Dataset[Sample]:
    """Convert legacy dataset to experimental format with automatic schema inference.

    Args:
        legacy_dataset: The legacy Datumaro dataset to convert

    Returns:
        A new experimental Dataset with inferred schema and converted data

    Example:
        >>> legacy_ds = Dataset.import_from("path/to/coco", "coco")
        >>> experimental_ds = convert_from_legacy(legacy_ds)
        >>> sample = experimental_ds[0]
        >>> print(sample.image_path)
        >>> print(sample.bboxes.shape)
    """

    # Step 1: Analyze dataset to infer schema
    analysis_result = analyze_legacy_dataset(legacy_dataset)

    # Step 2: Create experimental dataset with inferred schema
    experimental_dataset = Dataset(analysis_result.schema)

    # Step 3: Convert all items
    for legacy_item in legacy_dataset:
        # Convert legacy item to experimental sample
        sample_data = _convert_legacy_item(legacy_item, analysis_result)

        # Create sample and add to dataset
        sample = Sample(**sample_data)
        experimental_dataset.append(sample)

    return experimental_dataset


# Auto-register built-in converters when module is imported
register_builtin_converters()
