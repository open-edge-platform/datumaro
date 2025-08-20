# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Legacy dataset conversion functionality.

This module provides functionality to convert legacy Datumaro datasets to the new
experimental dataset format with automatic schema inference and type conversion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type, cast

import numpy as np
import polars as pl

from datumaro.components.annotation import Annotation, AnnotationType, Bbox
from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import CategoriesInfo, DatasetItem
from datumaro.components.media import FromFileMixin, Image, MediaElement

from .dataset import Dataset, Sample
from .fields import (
    BBoxField,
    ImagePathField,
    TensorField,
    bbox_field,
    image_path_field,
    tensor_field,
)
from .schema import AttributeInfo, Schema


class ForwardMediaConverter(ABC):
    """Base class for forward media type converters."""

    @abstractmethod
    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        """Return schema attributes for this media type."""
        pass

    @abstractmethod
    def convert_item_media(self, item: DatasetItem) -> dict[str, Any]:
        """Convert media from a DatasetItem to experimental format."""
        pass


class ForwardAnnotationConverter(ABC):
    """Base class for forward annotation type converters."""

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
_media_converters: dict[Type[MediaElement[Any]], ForwardMediaConverter] = {}
_annotation_converters: dict[AnnotationType, ForwardAnnotationConverter] = {}


def register_forward_media_converter(
    media_type: Type[MediaElement[Any]], converter: ForwardMediaConverter
) -> None:
    """Register a forward converter for a media type."""
    _media_converters[media_type] = converter


def register_forward_annotation_converter(
    annotation_type: AnnotationType, converter: ForwardAnnotationConverter
) -> None:
    """Register a forward converter for an annotation type."""
    _annotation_converters[annotation_type] = converter


def get_forward_media_converter(media_type: Type[MediaElement[Any]]) -> ForwardMediaConverter:
    """Get forward converter for a media type."""
    for registered_type, converter in _media_converters.items():
        if issubclass(media_type, registered_type):
            return converter
    raise ValueError(f"No converter registered for media type: {media_type}")


def get_forward_annotation_converter(annotation_type: AnnotationType) -> ForwardAnnotationConverter:
    """Get forward converter for an annotation type."""
    if annotation_type not in _annotation_converters:
        raise ValueError(f"No converter registered for annotation type: {annotation_type}")
    return _annotation_converters[annotation_type]


class ForwardImageMediaConverter(ForwardMediaConverter):
    """Forward converter for Image media type."""

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


class ForwardBboxAnnotationConverter(ForwardAnnotationConverter):
    """Forward converter for Bbox annotations."""

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


def register_builtin_forward_converters():
    """Register built-in forward converters for common types."""

    # Media converters
    register_forward_media_converter(Image, ForwardImageMediaConverter())

    # Annotation converters
    register_forward_annotation_converter(AnnotationType.bbox, ForwardBboxAnnotationConverter())


from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Result of legacy dataset analysis."""

    schema: Schema
    media_converter: ForwardMediaConverter | None
    ann_converters: dict[AnnotationType, ForwardAnnotationConverter]


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
    media_converter: ForwardMediaConverter | None
    ann_converters: dict[AnnotationType, ForwardAnnotationConverter] = {}

    # Get media attributes from converter
    try:
        media_converter = get_forward_media_converter(media_type)
        attributes.update(media_converter.get_schema_attributes())
    except ValueError:
        # No converter for this media type - skip
        media_converter = None

    # Get annotation attributes from converters
    for ann_type in ann_types:
        try:
            ann_converter = get_forward_annotation_converter(ann_type)
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


class BackwardMediaConverter(ABC):
    """Base class for backward media type converters."""

    @classmethod
    @abstractmethod
    def create_from_schema(cls, schema: Schema) -> "BackwardMediaConverter | None":
        """Create converter instance if schema is supported, None otherwise."""
        pass

    @abstractmethod
    def get_media_type(self) -> Type[MediaElement[Any]]:
        """Get the legacy media type this converter produces."""
        pass

    @abstractmethod
    def convert_to_legacy_media(self, sample: Sample) -> MediaElement[Any]:
        """Convert experimental sample media to legacy MediaElement."""
        pass


class BackwardAnnotationConverter(ABC):
    """Base class for backward annotation type converters."""

    @classmethod
    @abstractmethod
    def create_from_schema(cls, schema: Schema) -> "BackwardAnnotationConverter | None":
        """Create converter instance if schema is supported, None otherwise."""
        pass

    @abstractmethod
    def get_annotation_type(self) -> AnnotationType:
        """Get the legacy annotation type this converter produces."""
        pass

    @abstractmethod
    def infer_categories(self, experimental_dataset: Dataset[Sample]) -> CategoriesInfo:
        """Infer legacy categories from experimental dataset."""
        pass

    @abstractmethod
    def convert_to_legacy_annotations(
        self, sample: Sample, categories: CategoriesInfo
    ) -> list[Annotation]:
        """Convert experimental sample annotations to legacy format."""
        pass


# Global registries for backward converters
_backward_media_converter_classes: list[Type[BackwardMediaConverter]] = []
_backward_annotation_converter_classes: list[Type[BackwardAnnotationConverter]] = []


def register_backward_media_converter(converter_class: Type[BackwardMediaConverter]) -> None:
    """Register a backward converter class for a media type."""
    _backward_media_converter_classes.append(converter_class)


def register_backward_annotation_converter(
    converter_class: Type[BackwardAnnotationConverter],
) -> None:
    """Register a backward converter class for an annotation type."""
    _backward_annotation_converter_classes.append(converter_class)


class BackwardImageMediaConverter(BackwardMediaConverter):
    """Backward converter for Image media type."""

    def __init__(self, image_path_attr: str):
        """Initialize with the name of the image path attribute."""
        self.image_path_attr = image_path_attr

    @classmethod
    def create_from_schema(cls, schema: Schema) -> "BackwardImageMediaConverter | None":
        """Create converter instance if schema contains image_path field."""
        for attr_name, attr_info in schema.attributes.items():
            if isinstance(attr_info.annotation, ImagePathField):
                return cls(image_path_attr=attr_name)
        return None

    def get_media_type(self) -> Type[MediaElement[Any]]:
        return Image

    def convert_to_legacy_media(self, sample: Sample) -> MediaElement[Any]:
        """Convert image_path back to Image MediaElement."""
        image_path = getattr(sample, self.image_path_attr)
        return Image.from_file(path=image_path)  # pyright: ignore[reportUnknownMemberType]


class BackwardBboxAnnotationConverter(BackwardAnnotationConverter):
    """Backward converter for Bbox annotations."""

    def __init__(self, bboxes_attr: str, bbox_labels_attr: str):
        """Initialize with the names of the bbox-related attributes."""
        self.bboxes_attr = bboxes_attr
        self.bbox_labels_attr = bbox_labels_attr

    @classmethod
    def create_from_schema(cls, schema: Schema) -> "BackwardBboxAnnotationConverter | None":
        """Create converter instance if schema contains bbox-related fields."""
        bboxes_attr = None
        bbox_labels_attr = None

        # Find bbox field
        for attr_name, attr_info in schema.attributes.items():
            if isinstance(attr_info.annotation, BBoxField):
                bboxes_attr = attr_name
                break

        # Find bbox_labels field (look for tensor field with 'bbox_labels' in name or similar pattern)
        for attr_name, attr_info in schema.attributes.items():
            if isinstance(attr_info.annotation, TensorField) and (
                "bbox_label" in attr_name.lower() or "label" in attr_name.lower()
            ):
                bbox_labels_attr = attr_name
                break

        if bboxes_attr and bbox_labels_attr:
            return cls(bboxes_attr=bboxes_attr, bbox_labels_attr=bbox_labels_attr)
        return None

    def get_annotation_type(self) -> AnnotationType:
        return AnnotationType.bbox

    def convert_to_legacy_annotations(
        self, sample: Sample, categories: CategoriesInfo
    ) -> list[Annotation]:
        """Convert bboxes and bbox_labels back to legacy Bbox annotations."""
        bboxes = getattr(sample, self.bboxes_attr, None)
        bbox_labels = getattr(sample, self.bbox_labels_attr, None)

        if bboxes is None or bbox_labels is None:
            return []

        annotations: list[Annotation] = []
        for i in range(len(bboxes)):
            # Convert from x1,y1,x2,y2 back to x,y,w,h format
            x1, y1, x2, y2 = bboxes[i]
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            label_id = int(bbox_labels[i]) if bbox_labels[i] is not None else None

            bbox = Bbox(x=x, y=y, w=w, h=h, label=label_id)
            annotations.append(bbox)

        return annotations

    def infer_categories(self, experimental_dataset: Dataset[Sample]) -> CategoriesInfo:
        """Infer label categories from bbox_labels."""
        from datumaro.components.annotation import LabelCategories

        # Collect all unique label IDs
        label_ids: set[int] = set()
        for sample in experimental_dataset:
            bbox_labels = getattr(sample, self.bbox_labels_attr, None)
            if bbox_labels is not None:
                for label_id in bbox_labels:
                    if label_id is not None:
                        label_ids.add(int(label_id))

        # Create label categories
        label_categories = LabelCategories()
        for label_id in sorted(label_ids):
            label_categories.add(f"class_{label_id}")

        return {AnnotationType.label: label_categories}


@dataclass
class BackwardAnalysisResult:
    """Result of experimental dataset analysis for backward conversion."""

    media_type: Type[MediaElement[Any]] | None
    ann_types: set[AnnotationType]
    categories: CategoriesInfo
    media_converter: BackwardMediaConverter | None
    ann_converters: dict[AnnotationType, BackwardAnnotationConverter]


def analyze_experimental_dataset(experimental_dataset: Dataset[Sample]) -> BackwardAnalysisResult:
    """Analyze experimental dataset schema to determine legacy format.

    Args:
        experimental_dataset: The experimental dataset to analyze

    Returns:
        BackwardAnalysisResult containing legacy format information
    """
    schema = experimental_dataset.schema

    # Find compatible media converter
    media_converter: BackwardMediaConverter | None = None
    media_type: Type[MediaElement[Any]] | None = None

    for converter_class in _backward_media_converter_classes:
        converter_instance = converter_class.create_from_schema(schema)
        if converter_instance is not None:
            media_converter = converter_instance
            media_type = converter_instance.get_media_type()
            break

    # Find compatible annotation converters
    ann_converters: dict[AnnotationType, BackwardAnnotationConverter] = {}
    ann_types: set[AnnotationType] = set()
    categories: CategoriesInfo = {}

    for converter_class in _backward_annotation_converter_classes:
        converter_instance = converter_class.create_from_schema(schema)
        if converter_instance is not None:
            ann_type = converter_instance.get_annotation_type()
            ann_converters[ann_type] = converter_instance
            ann_types.add(ann_type)

            # Merge categories from this converter
            converter_categories = converter_instance.infer_categories(experimental_dataset)
            categories.update(converter_categories)

    return BackwardAnalysisResult(
        media_type=media_type,
        ann_types=ann_types,
        categories=categories,
        media_converter=media_converter,
        ann_converters=ann_converters,
    )


def _convert_experimental_item(
    index: int, sample: Sample, backward_analysis: BackwardAnalysisResult
) -> DatasetItem:
    """Convert experimental sample to legacy DatasetItem."""

    # Convert media
    media: MediaElement[Any] | None = None
    if backward_analysis.media_converter:
        media = backward_analysis.media_converter.convert_to_legacy_media(sample)

    # Convert annotations
    annotations: list[Annotation] = []
    for converter in backward_analysis.ann_converters.values():
        ann_list = converter.convert_to_legacy_annotations(sample, backward_analysis.categories)
        annotations.extend(ann_list)

    item_id = str(index)

    return DatasetItem(
        id=item_id,
        media=media,
        annotations=annotations,
        attributes={},  # Could be extended to convert attributes
    )


def convert_to_legacy(experimental_dataset: Dataset[Sample]) -> LegacyDataset:
    """Convert experimental dataset to legacy format.

    Args:
        experimental_dataset: The experimental Dataset to convert

    Returns:
        A new legacy Datumaro Dataset with converted data

    Example:
        >>> experimental_ds = Dataset(MySchema)
        >>> # ... add samples to experimental_ds
        >>> legacy_ds = convert_to_legacy(experimental_ds)
        >>> legacy_ds.export("output", "coco")
    """

    # Step 1: Analyze experimental dataset
    backward_analysis = analyze_experimental_dataset(experimental_dataset)

    # Step 2: Create legacy dataset items
    legacy_items: list[DatasetItem] = []
    for i, sample in enumerate(experimental_dataset):
        legacy_item = _convert_experimental_item(i, sample, backward_analysis)
        legacy_items.append(legacy_item)

    # Step 3: Create legacy dataset
    legacy_dataset = LegacyDataset.from_iterable(  # pyright: ignore[reportUnknownMemberType]
        legacy_items,
        categories=backward_analysis.categories,
        media_type=backward_analysis.media_type or MediaElement,
    )

    return legacy_dataset


def register_builtin_backward_converters():
    """Register built-in backward converters."""

    # Register backward media converters
    register_backward_media_converter(BackwardImageMediaConverter)

    # Register backward annotation converters
    register_backward_annotation_converter(BackwardBboxAnnotationConverter)


# Auto-register built-in converters when module is imported
register_builtin_forward_converters()
register_builtin_backward_converters()
