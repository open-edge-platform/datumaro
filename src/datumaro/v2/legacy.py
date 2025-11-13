# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Legacy dataset conversion functionality.

This module provides functionality to convert legacy Datumaro datasets to the new
v2 dataset format with automatic schema inference and type conversion.
"""

from __future__ import annotations

import io
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial as _partial
from typing import Any, cast

import numpy as np
import polars as pl
from PIL import Image as PILImage

from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    Ellipse,
    ExtractedMask,
    Label,
    Points,
    Polygon,
    RotatedBbox,
)
from datumaro.components.annotation import LabelCategories as LegacyLabelCategories
from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import CategoriesInfo, DatasetItem
from datumaro.components.media import FromDataMixin, FromFileMixin, Image, MediaElement

from .categories import (
    GroupType,
    HierarchicalLabelCategories,
    HierarchicalLabelCategory,
    LabelCategories,
    LabelGroup,
    LabelSemantic,
    MaskCategories,
    RgbColor,
)
from .converters import generate_colormap
from .dataset import Dataset, Sample
from .fields import (
    BBoxField,
    EllipseField,
    ImageInfo,
    ImagePathField,
    LabelField,
    PolygonField,
    RotatedBBoxField,
    Subset,
    bbox_field,
    image_bytes_field,
    image_callable_field,
    image_info_field,
    image_path_field,
    instance_mask_callable_field,
    keypoints_field,
    label_field,
    mask_callable_field,
    polygon_field,
    rotated_bbox_field,
    subset_field,
)
from .schema import AttributeInfo, Schema, Semantic


class ForwardMediaConverter(ABC):
    """Base class for forward media type converters."""

    @classmethod
    @abstractmethod
    def get_supported_media_types(cls) -> list[type[MediaElement[Any]]]:
        """Return list of media types this converter can handle."""

    @classmethod
    @abstractmethod
    def create(
        cls, dataset: LegacyDataset, semantic: Semantic = Semantic.Default, name_prefix: str = ""
    ) -> ForwardMediaConverter | None:
        """Create converter instance if dataset is supported, None otherwise.

        Args:
            dataset: Legacy dataset to create converter from
            semantic: The semantic type for the converted fields
            name_prefix: Prefix to prepend to all field names
        """

    @abstractmethod
    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        """Return schema attributes for this media type."""

    @abstractmethod
    def convert_item_media(self, item: DatasetItem) -> dict[str, Any]:
        """Convert media from a DatasetItem to v2 format."""


class ForwardAnnotationConverter(ABC):
    """Base class for forward annotation type converters."""

    @classmethod
    @abstractmethod
    def get_supported_annotation_types(cls) -> list[AnnotationType]:
        """Return list of annotation types this converter can handle."""

    @classmethod
    @abstractmethod
    def create(
        cls, dataset: LegacyDataset, semantic: Semantic = Semantic.Default, name_prefix: str = ""
    ) -> ForwardAnnotationConverter | None:
        """Create converter instance if dataset supports this annotation type.

        Args:
            dataset: Legacy dataset to create converter from
            semantic: The semantic type for the converted fields
            name_prefix: Prefix to prepend to all field names
        """

    @abstractmethod
    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        """Return schema attributes for this annotation type."""

    @abstractmethod
    def convert_annotations(self, annotations: list[Annotation], item: DatasetItem) -> dict[str, Any]:
        """Convert annotations of this type to v2 format."""


# Global registries
_media_converter_classes: dict[type[MediaElement[Any]], type[ForwardMediaConverter]] = {}
_annotation_converters: dict[AnnotationType, type[ForwardAnnotationConverter]] = {}


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


def get_forward_annotation_converter(
    annotation_type: AnnotationType,
    dataset: LegacyDataset,
    semantic: Semantic = Semantic.Default,
    name_prefix: str = "",
) -> ForwardAnnotationConverter | None:
    """Get forward converter for an annotation type from the dataset.

    Args:
        annotation_type: The type of annotation to get a converter for
        dataset: The legacy dataset to create a converter from
        semantic: The semantic type for the converted fields
        name_prefix: Prefix to prepend to all field names

    Returns:
        A forward converter instance if one can handle the annotation type, None otherwise
    """
    if annotation_type not in _annotation_converters:
        return None
    converter_class = _annotation_converters[annotation_type]
    return converter_class.create(dataset, semantic, name_prefix)


def _image_callable_impl(bytes_source: Any, is_callable: bool = False):
    """Convert image bytes (or bytes provider) to a numpy array.

    Implemented at module scope so that partials of this function are pickleable
    and thus safe to use with multi-processing data loaders.
    """
    # Get the bytes data (either directly or from callable)
    bytes_data = bytes_source() if is_callable else bytes_source
    if not isinstance(bytes_data, bytes):
        raise TypeError(f"Expected bytes data, got {type(bytes_data)}")
    # Convert bytes to image array using PIL
    with PILImage.open(io.BytesIO(bytes_data)) as pil_image:
        processed_image = pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
        return np.array(processed_image, dtype=np.uint8)


class ForwardImageMediaConverter(ForwardMediaConverter):
    """Forward converter for Image media type supporting both file paths and byte data."""

    def __init__(
        self,
        media_mixin: type,
        has_image_info: bool,
        semantic: Semantic = Semantic.Default,
        name_prefix: str = "",
        has_callable_data: bool = False,
    ):
        """Initialize converter with format preference and image info availability."""
        self.media_mixin = media_mixin
        self.has_image_info = has_image_info
        self.has_callable_data = has_callable_data
        self.semantic = semantic
        self.name_prefix = name_prefix

    @classmethod
    def get_supported_media_types(cls) -> list[type[MediaElement[Any]]]:
        """Return list of media types this converter can handle."""
        return [Image]

    @classmethod
    def create(
        cls, dataset: LegacyDataset, semantic: Semantic = Semantic.Default, name_prefix: str = ""
    ) -> ForwardImageMediaConverter | None:
        """Create converter instance, detecting whether to use paths or bytes.

        Args:
            dataset: Legacy dataset to create converter from
            semantic: The semantic type for the converted fields
        """
        found_media_type: type | None = None
        has_image_info = True  # Assume all images have size until proven otherwise
        has_callable_data = False  # Track if any FromDataMixin has callable _data

        for item in dataset:
            if isinstance(item.media, Image):
                media_type = type(item.media)
                if found_media_type is not None and media_type != found_media_type:
                    raise ValueError(
                        f"The dataset has a mix of different image media types: "
                        f"{found_media_type} and {media_type}. This is not supported by the converter."
                    )

                found_media_type = media_type

                # Check if this image has size info
                if not item.media.has_size:
                    has_image_info = False

                # Check if this is FromDataMixin with callable _data
                if isinstance(item.media, FromDataMixin) and callable(item.media._data):
                    has_callable_data = True

        if found_media_type is None:
            return None

        if issubclass(found_media_type, FromDataMixin):
            media_mixin = FromDataMixin
        elif issubclass(found_media_type, FromFileMixin):
            media_mixin = FromFileMixin
        else:
            raise ValueError(f"Unknown media mixin for {found_media_type}.")

        return cls(
            media_mixin=media_mixin,
            has_image_info=has_image_info,
            semantic=semantic,
            name_prefix=name_prefix,
            has_callable_data=has_callable_data,
        )

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        attributes: dict[str, AttributeInfo] = {}

        if self.media_mixin == FromDataMixin:
            if self.has_callable_data:
                attributes[self.name_prefix + "image_callable"] = AttributeInfo(
                    type=callable, field=image_callable_field(semantic=self.semantic)
                )
            else:
                attributes[self.name_prefix + "image_bytes"] = AttributeInfo(
                    type=bytes, field=image_bytes_field(semantic=self.semantic)
                )
        elif self.media_mixin == FromFileMixin:
            attributes[self.name_prefix + "image_path"] = AttributeInfo(
                type=str, field=image_path_field(semantic=self.semantic)
            )
        else:
            raise RuntimeError(f"Media mixin not implemented: {self.media_mixin}")

        # Add image info field if all images have size
        if self.has_image_info:
            attributes[self.name_prefix + "image_info"] = AttributeInfo(
                type=ImageInfo, field=image_info_field(semantic=self.semantic)
            )

        return attributes

    def convert_item_media(self, item: DatasetItem) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if isinstance(item.media, Image):  # pyright: ignore[reportUnknownMemberType]
            if self.media_mixin == FromDataMixin:
                if self.has_callable_data:
                    # Use a top-level callable to ensure picklability across workers
                    is_callable = callable(item.media._data)
                    bytes_source = item.media._data
                    result[self.name_prefix + "image_callable"] = _partial(
                        _image_callable_impl, bytes_source, is_callable
                    )
                else:
                    result[self.name_prefix + "image_bytes"] = item.media._data
            elif self.media_mixin == FromFileMixin:
                result[self.name_prefix + "image_path"] = item.media.path
            else:
                raise RuntimeError(f"Media mixin not implemented: {self.media_mixin}")

            # Add image info if available
            if self.has_image_info and item.media.has_size:
                height, width = item.media.size  # size returns (H, W)
                result[self.name_prefix + "image_info"] = ImageInfo(width=width, height=height)

        return result


class ForwardBboxAnnotationConverter(ForwardAnnotationConverter):
    """Forward converter for Bbox annotations."""

    def __init__(
        self,
        bbox_attribute: AttributeInfo,
        bbox_labels_attribute: AttributeInfo | None,
        name_prefix: str,
    ):
        """Initialize with bbox attributes and label attribute name."""
        super().__init__()
        self.bbox_attribute = bbox_attribute
        self.bbox_labels_attribute = bbox_labels_attribute
        self.name_prefix = name_prefix

    @classmethod
    def get_supported_annotation_types(cls) -> list[AnnotationType]:
        """Return list of annotation types this converter can handle."""
        return [AnnotationType.bbox]

    @classmethod
    def create(
        cls, dataset: LegacyDataset, semantic: Semantic = Semantic.Default, name_prefix: str = ""
    ) -> ForwardBboxAnnotationConverter | None:
        """Create converter instance for bbox annotations."""
        categories = dataset.categories()
        # Extract label categories if available
        legacy_label_categories = categories.get(AnnotationType.label, None)

        bbox_attribute = AttributeInfo(type=np.ndarray, field=bbox_field(dtype=pl.Float32, semantic=semantic))

        bbox_labels_attribute = None
        # Only add bbox_labels if we have label categories
        if legacy_label_categories is not None:
            # Convert legacy label categories to new format
            labels = tuple(label_item.name for label_item in legacy_label_categories.items)
            new_label_categories = LabelCategories(labels=labels)

            bbox_labels_attribute = AttributeInfo(
                type=np.ndarray,
                field=label_field(is_list=True, semantic=semantic),
                categories=new_label_categories,
            )

        return cls(
            bbox_attribute=bbox_attribute,
            bbox_labels_attribute=bbox_labels_attribute,
            name_prefix=name_prefix,
        )

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        attributes = {self.name_prefix + "bboxes": self.bbox_attribute}
        if self.bbox_labels_attribute is not None:
            attributes[self.name_prefix + "labels"] = self.bbox_labels_attribute
        return attributes

    def convert_annotations(self, annotations: list[Annotation], item: DatasetItem) -> dict[str, Any]:  # noqa: ARG002
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

        result = {self.name_prefix + "bboxes": bboxes_array}

        # Only add bbox_labels if we have label categories
        if self.bbox_labels_attribute is not None:
            result[self.name_prefix + "labels"] = np.array(labels, dtype=np.int32)

        return result


class ForwardRotatedBboxAnnotationConverter(ForwardAnnotationConverter):
    """Forward converter for RotatedBbox annotations."""

    def __init__(
        self,
        rotated_bbox_attribute: AttributeInfo,
        rotated_bbox_labels_attribute: AttributeInfo | None = None,
        name_prefix: str = "",
    ):
        """Initialize converter with rotated bbox attributes."""
        self.rotated_bbox_attribute = rotated_bbox_attribute
        self.rotated_bbox_labels_attribute = rotated_bbox_labels_attribute
        self.name_prefix = name_prefix

    @classmethod
    def get_supported_annotation_types(cls) -> list[AnnotationType]:
        """Return list of annotation types this converter can handle."""
        return [AnnotationType.rotated_bbox]

    @classmethod
    def create(
        cls, dataset: LegacyDataset, semantic: Semantic = Semantic.Default, name_prefix: str = ""
    ) -> ForwardRotatedBboxAnnotationConverter | None:
        """Create converter instance from dataset."""
        categories = dataset.categories()
        # Create attribute for rotated bboxes (cx, cy, w, h, r)
        rotated_bbox_attribute = AttributeInfo(
            type=np.ndarray,
            field=rotated_bbox_field(dtype=pl.Float32, semantic=semantic),
        )

        # Create attribute for labels if we have label categories
        rotated_bbox_labels_attribute = None
        # Extract label categories if available
        legacy_label_categories = categories.get(AnnotationType.label, None)

        if legacy_label_categories is not None and len(legacy_label_categories.items) > 0:
            # Convert legacy label categories to new format
            labels = tuple(label_item.name for label_item in legacy_label_categories.items)
            new_label_categories = LabelCategories(labels=labels)

            rotated_bbox_labels_attribute = AttributeInfo(
                type=np.ndarray,
                field=label_field(is_list=True, semantic=semantic),
                categories=new_label_categories,
            )

        return cls(
            rotated_bbox_attribute=rotated_bbox_attribute,
            rotated_bbox_labels_attribute=rotated_bbox_labels_attribute,
            name_prefix=name_prefix,
        )

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        attributes = {self.name_prefix + "rotated_bboxes": self.rotated_bbox_attribute}
        if self.rotated_bbox_labels_attribute is not None:
            attributes[self.name_prefix + "labels"] = self.rotated_bbox_labels_attribute
        return attributes

    def convert_annotations(self, annotations: list[Annotation], item: DatasetItem) -> dict[str, Any]:  # noqa: ARG002
        rotated_bboxes: list[list[float]] = []
        labels: list[int | None] = []

        for ann in annotations:
            if isinstance(ann, RotatedBbox):
                # Convert from degrees to radians for rotation angle
                r_radians = math.radians(ann.r)
                rotated_bboxes.append([ann.cx, ann.cy, ann.w, ann.h, r_radians])
                labels.append(ann.label)

        # Ensure proper shapes for empty arrays
        rotated_bboxes_array = np.array(rotated_bboxes, dtype=np.float32)
        if rotated_bboxes_array.shape == (0,):
            rotated_bboxes_array = rotated_bboxes_array.reshape(0, 5)

        result = {self.name_prefix + "rotated_bboxes": rotated_bboxes_array}

        # Only add rotated_bbox_labels if we have label categories
        if self.rotated_bbox_labels_attribute is not None:
            result[self.name_prefix + "labels"] = np.array(labels, dtype=np.int32)

        return result


class ForwardPolygonAnnotationConverter(ForwardAnnotationConverter):
    """Forward converter for Polygon annotations."""

    def __init__(
        self,
        polygon_attribute: AttributeInfo,
        polygon_labels_attribute: AttributeInfo | None,
        name_prefix: str,
    ):
        """Initialize with polygon attributes and label attribute."""
        self.polygon_attribute = polygon_attribute
        self.polygon_labels_attribute = polygon_labels_attribute
        self.name_prefix = name_prefix

    @classmethod
    def get_supported_annotation_types(cls) -> list[AnnotationType]:
        """Return list of annotation types this converter can handle."""
        return [AnnotationType.polygon]

    @classmethod
    def create(
        cls, dataset: LegacyDataset, semantic: Semantic = Semantic.Default, name_prefix: str = ""
    ) -> ForwardPolygonAnnotationConverter | None:
        """Create converter instance for polygon annotations."""
        categories = dataset.categories()
        # Extract label categories if available
        legacy_label_categories = categories.get(AnnotationType.label, None)

        polygon_attribute = AttributeInfo(
            type=np.ndarray,
            field=polygon_field(dtype=pl.Float32, format="xy", semantic=semantic),
        )

        polygon_labels_attribute = None
        # Only add polygon_labels if we have label categories
        if legacy_label_categories is not None:
            # Convert legacy label categories to new format
            labels = tuple(label_item.name for label_item in legacy_label_categories.items)
            new_label_categories = LabelCategories(labels=labels)

            polygon_labels_attribute = AttributeInfo(
                type=np.ndarray,
                field=label_field(is_list=True, semantic=semantic),
                categories=new_label_categories,
            )

        return cls(
            polygon_attribute=polygon_attribute,
            polygon_labels_attribute=polygon_labels_attribute,
            name_prefix=name_prefix,
        )

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        attributes = {self.name_prefix + "polygons": self.polygon_attribute}
        if self.polygon_labels_attribute is not None:
            attributes[self.name_prefix + "labels"] = self.polygon_labels_attribute
        return attributes

    def convert_annotations(self, annotations: list[Annotation], item: DatasetItem) -> dict[str, Any]:  # noqa: ARG002
        polygons: list[list[float]] = []
        labels: list[int | None] = []

        for ann in annotations:
            if isinstance(ann, Polygon):
                # Points are stored as flat coordinates in Polygon
                # ann.points in the format [x1,y1,x2,y2,...] format
                polygons.append(np.array(ann.points).reshape(-1, 2))
                labels.append(ann.label)

        # Convert to numpy array - polygons is a list of variable-length coordinate lists
        # We'll store it as a ragged array (object dtype to handle different lengths)

        # When using np.array, there is a corner case for the case where len(polygons) == 1 where
        # Numpy creates a 2D array of objects instead of a 1D array of objects.
        # We may be able to solve this in the upcoming version of Numpy with the argument ndmax.
        # In the meantime, create an empty array, then assign to avoid the corner case
        polygons_array = np.empty((len(polygons),), dtype=object)
        polygons_array[:] = polygons
        result = {self.name_prefix + "polygons": polygons_array}

        # Only add polygon_labels if we have label categories
        if self.polygon_labels_attribute is not None:
            result[self.name_prefix + "labels"] = np.array(labels, dtype=np.int32)

        return result


class ForwardLabelAnnotationConverter(ForwardAnnotationConverter):
    """Forward converter for Label (single label classification) annotations."""

    def __init__(
        self,
        label_attribute: AttributeInfo,
        semantic: Semantic = Semantic.Default,
        name_prefix: str = "",
    ):
        """Initialize with label attribute."""
        super().__init__()
        self.label_attribute = label_attribute
        self.semantic = semantic
        self.name_prefix = name_prefix

    @classmethod
    def create(
        cls, dataset: LegacyDataset, semantic: Semantic = Semantic.Default, name_prefix: str = ""
    ) -> ForwardLabelAnnotationConverter | None:
        """Create converter instance for label annotations."""
        categories = dataset.categories()
        legacy_label_categories = categories.get(AnnotationType.label, None)

        if legacy_label_categories is None:
            return None

        labels = tuple(label_item.name for label_item in legacy_label_categories.items)
        new_label_categories = LabelCategories(labels=labels)

        label_attribute = AttributeInfo(
            type=int,
            field=label_field(semantic=semantic),
            categories=new_label_categories,
        )

        return cls(label_attribute=label_attribute, semantic=semantic, name_prefix=name_prefix)

    @classmethod
    def get_supported_annotation_types(cls) -> list[AnnotationType]:
        return [AnnotationType.label]

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        return {self.name_prefix + "label": self.label_attribute}

    def convert_annotations(self, annotations: list[Annotation], item: DatasetItem) -> dict[str, Any]:  # noqa: ARG002
        labels = [ann for ann in annotations if isinstance(ann, Label)]
        result = {}
        if len(labels) > 0:
            if "multi_label_ids" in labels[0].attributes:
                result[self.name_prefix + "label"] = labels[0].attributes["multi_label_ids"]
            elif self.label_attribute.field.multi_label:
                result[self.name_prefix + "label"] = [labels[0].label]
            else:
                result[self.name_prefix + "label"] = labels[0].label
        else:
            result[self.name_prefix + "label"] = None
        return result


class ForwardKeypointAnnotationConverter(ForwardAnnotationConverter):
    """Forward converter for Points (keypoints) annotations."""

    def __init__(
        self,
        keypoints_attribute: AttributeInfo,
        keypoints_labels_attribute: AttributeInfo | None,
        name_prefix: str,
    ):
        """Initialize with keypoints attributes and label attribute name."""
        super().__init__()
        self.keypoints_attribute = keypoints_attribute
        self.keypoints_labels_attribute = keypoints_labels_attribute
        self.name_prefix = name_prefix

    @classmethod
    def create(
        cls, dataset: LegacyDataset, semantic: Semantic = Semantic.Default, name_prefix: str = ""
    ) -> ForwardKeypointAnnotationConverter | None:
        """Create converter instance for keypoints annotations."""
        categories = dataset.categories()
        # Extract label categories if available
        legacy_label_categories = categories.get(AnnotationType.label, None)

        keypoints_attribute = AttributeInfo(type=np.ndarray, field=keypoints_field(dtype=pl.Float32, semantic=semantic))

        keypoints_labels_attribute = None
        # Only add keypoints_labels if we have label categories
        if legacy_label_categories is not None:
            # Convert legacy label categories to new format
            labels = tuple(label_item.name for label_item in legacy_label_categories.items)
            new_label_categories = LabelCategories(labels=labels)

            keypoints_labels_attribute = AttributeInfo(
                type=np.ndarray,
                field=label_field(is_list=True, semantic=semantic),
                categories=new_label_categories,
            )

        return cls(
            keypoints_attribute=keypoints_attribute,
            keypoints_labels_attribute=keypoints_labels_attribute,
            name_prefix=name_prefix,
        )

    @classmethod
    def get_supported_annotation_types(cls) -> list[AnnotationType]:
        return [AnnotationType.points]

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        attributes = {self.name_prefix + "keypoints": self.keypoints_attribute}
        if self.keypoints_labels_attribute is not None:
            attributes[self.name_prefix + "labels"] = self.keypoints_labels_attribute
        return attributes

    def convert_annotations(self, annotations: list[Annotation], item: DatasetItem) -> dict[str, Any]:  # noqa: ARG002
        keypoints = [ann for ann in annotations if isinstance(ann, Points)]
        # KeypointsField expects individual Points objects, not arrays
        # Only supports single keypoint case
        result = {self.name_prefix + "labels": None, self.name_prefix + "keypoints": None}

        if len(keypoints) > 0:
            result["keypoints"] = keypoints[0]  # Pass the Points object directly
            if self.keypoints_labels_attribute is not None and "keypoint_label_ids" in keypoints[0].attributes:
                result[self.name_prefix + "labels"] = keypoints[0].attributes["keypoint_label_ids"]
            else:
                result[self.name_prefix + "labels"] = None
        else:
            result[self.name_prefix + "keypoints"] = None

        return result


class ForwardEllipseAnnotationConverter(ForwardAnnotationConverter):
    """Forward converter for Ellipse annotations."""

    def __init__(
        self,
        ellipse_attribute: AttributeInfo,
        ellipse_labels_attribute: AttributeInfo | None,
        name_prefix: str,
    ):
        """Initialize with ellipse attributes and ellipse label attribute name."""
        super().__init__()
        self.ellipse_attribute = ellipse_attribute
        self.ellipse_labels_attribute = ellipse_labels_attribute
        self.name_prefix = name_prefix

    @classmethod
    def get_supported_annotation_types(cls) -> list[AnnotationType]:
        """Return list of annotation types this converter can handle."""
        return [AnnotationType.ellipse]

    @classmethod
    def create(
        cls, dataset: LegacyDataset, semantic: Semantic = Semantic.Default, name_prefix: str = ""
    ) -> ForwardEllipseAnnotationConverter | None:
        """Create converter instance for ellipse annotations."""
        categories = dataset.categories()
        # Extract label categories if available
        legacy_label_categories = categories.get(AnnotationType.label, None)

        ellipse_attribute = AttributeInfo(type=np.ndarray, field=EllipseField(dtype=pl.Float32, semantic=semantic))

        ellipse_labels_attribute = None
        # Only add ellipse_labels if we have label categories
        if legacy_label_categories is not None:
            # Convert legacy label categories to new format
            labels = tuple(label_item.name for label_item in legacy_label_categories.items)
            new_label_categories = LabelCategories(labels=labels)

            ellipse_labels_attribute = AttributeInfo(
                type=np.ndarray,
                field=label_field(is_list=True, semantic=semantic),
                categories=new_label_categories,
            )

        return cls(
            ellipse_attribute=ellipse_attribute,
            ellipse_labels_attribute=ellipse_labels_attribute,
            name_prefix=name_prefix,
        )

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        attributes = {self.name_prefix + "ellipses": self.ellipse_attribute}
        if self.ellipse_labels_attribute is not None:
            attributes[self.name_prefix + "labels"] = self.ellipse_labels_attribute
        return attributes

    def convert_annotations(self, annotations: list[Annotation], item: DatasetItem) -> dict[str, Any]:  # noqa: ARG002
        ellipses: list[list[float]] = []
        labels: list[int | None] = []

        for ann in annotations:
            if isinstance(ann, Ellipse):
                ellipses.append([ann.x1, ann.y1, ann.x2, ann.y2])
                labels.append(ann.label)

        # Ensure proper shapes for empty arrays
        ellipses_array = np.array(ellipses, dtype=np.float32)
        if ellipses_array.shape == (0,):
            ellipses_array = ellipses_array.reshape(0, 4)

        result = {self.name_prefix + "ellipses": ellipses_array}

        # Only add ellipse_labels if we have label categories
        if self.ellipse_labels_attribute is not None:
            result[self.name_prefix + "labels"] = np.array(labels, dtype=np.int32)

        return result


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


@dataclass
class AnalysisResult:
    """Result of legacy dataset analysis."""

    schema: Schema
    media_converter: ForwardMediaConverter | None
    ann_converters: dict[AnnotationType, ForwardAnnotationConverter]
    subset_converter: SubsetConverter
    is_anomaly: bool
    is_hierarchical: bool


def analyze_legacy_dataset(legacy_dataset: LegacyDataset, semantic: Semantic = Semantic.Default) -> AnalysisResult:
    """Analyze legacy dataset and generate schema using registered converters.

    Args:
        legacy_dataset: The legacy Datumaro dataset to analyze
        semantic: The semantic type for the converted fields

    Returns:
        AnalysisResult containing the inferred schema and converters
    """
    ann_types = legacy_dataset.ann_types()

    attributes: dict[str, AttributeInfo] = {}
    ann_converters: dict[AnnotationType, ForwardAnnotationConverter] = {}

    if AnnotationType.label in legacy_dataset.categories():
        label_groups = legacy_dataset.categories()[AnnotationType.label].label_groups
        # Convert to new label group class
        label_groups = [
            LabelGroup(name=group.name, labels=group.labels, group_type=GroupType[group.group_type.name])
            for group in label_groups
        ]
        label_group_names = [group.name for group in label_groups] if label_groups else []

        # Check if project has a hierarchical structure
        label_names = [item.name for item in legacy_dataset.categories()[AnnotationType.label].items]
        is_hierarchical = _has_derived_labels(label_group_names) or _has_derived_labels(label_names)

        # Look for multi label classification groups
        multi_label_group_names = [name for name in label_group_names if name.startswith("Classification labels__")]
        is_multi_label = len(multi_label_group_names) > 1 and not is_hierarchical
    else:
        is_hierarchical = False
        is_multi_label = False

    media_converter = get_forward_media_converter(legacy_dataset, semantic)
    if media_converter is not None:
        attributes.update(media_converter.get_schema_attributes())

    # Add SubsetConverter since it's always needed and not tied to specific annotation types
    subset_converter = SubsetConverter(semantic=semantic)
    attributes.update(subset_converter.get_schema_attributes())

    # If we have a label converter plus other converters, assume that this is an anomaly task.
    # To avoid conflicts between the label attribute and the other ones, use semantic to distinguish them.
    is_anomaly = AnnotationType.label in ann_types and len(ann_types) > 1

    for ann_type in ann_types:
        ann_semantic = Semantic.Anomaly if is_anomaly and ann_type != AnnotationType.label else semantic
        name_prefix = "anomaly_" if is_anomaly and ann_type != AnnotationType.label else ""
        converter = get_forward_annotation_converter(ann_type, legacy_dataset, ann_semantic, name_prefix)
        if converter is not None:
            ann_converters[ann_type] = converter
            ann_attributes = converter.get_schema_attributes()
            if is_multi_label:
                ann_attributes[AnnotationType.label.name].field = label_field(multi_label=True)
            if is_hierarchical:
                ann_attributes[AnnotationType.label.name].field = label_field(is_list=True)
                categories = legacy_dataset.categories()[AnnotationType.label].items
                label_categories = tuple(
                    HierarchicalLabelCategory(
                        name=category.name,
                        parent=category.parent,
                        label_semantics=_attributes_to_dict(category.attributes),
                    )
                    for category in categories
                )
                hierarchical_categories = HierarchicalLabelCategories(
                    items=label_categories, label_groups=tuple(label_groups)
                )
                ann_attributes[AnnotationType.label.name].categories = hierarchical_categories
            attributes.update(ann_attributes)
    schema = Schema(attributes=attributes)
    return AnalysisResult(
        schema=schema,
        media_converter=media_converter,
        ann_converters=ann_converters,
        subset_converter=subset_converter,
        is_anomaly=is_anomaly,
        is_hierarchical=is_hierarchical,
    )


def _attributes_to_dict(attributes: list[str]) -> dict[str, str]:
    """Convert a list of Attribute objects to a dictionary. Used for hierarchical label legacy conversion"""
    attr_dict = {}
    for attr in attributes:
        if "__" in attr:
            try:
                attr_values = attr.split("__")
                attr_dict[attr_values[1]] = attr_values[2]
            except IndexError:
                pass
    return attr_dict


def _has_derived_labels(labels: list[str]) -> bool:
    """
    Check if any item in the list is any other label + "__" + anything. This indicates that the
    labels are from a hierarchical structure

    Args:
        labels: List of label strings

    Returns:
        bool: True if any derived labels exist, False otherwise
    """
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and labels[i].startswith(labels[j] + "__"):
                return True
    return False


def _convert_legacy_item(item: DatasetItem, analysis_result: AnalysisResult) -> dict[str, Any]:
    """Convert item using converters from analysis result."""

    attributes: dict[str, Any] = {}

    # Convert media using the analyzed converter
    if analysis_result.media_converter:
        attributes.update(analysis_result.media_converter.convert_item_media(item))

    # Convert subset using the subset converter
    attributes.update(analysis_result.subset_converter.convert_annotations(item.annotations, item))

    # Convert each annotation type using the analyzed converters
    for ann_converter in analysis_result.ann_converters.values():
        attributes.update(ann_converter.convert_annotations(item.annotations, item))

    return attributes


def convert_from_legacy(legacy_dataset: LegacyDataset) -> Dataset[Sample]:
    """Convert legacy dataset to v2 format with automatic schema inference.

    Args:
        legacy_dataset: The legacy Datumaro dataset to convert
    Returns:
        A new v2 Dataset with inferred schema and converted data

    Example:
        >>> legacy_ds = Dataset.import_from("path/to/coco", "coco")
        >>> experimental_ds = convert_from_legacy(legacy_ds)
        >>> sample = experimental_ds[0]
        >>> print(sample.image_path)
        >>> print(sample.bboxes.shape)
    """

    # Step 1: Analyze dataset to infer schema
    analysis_result = analyze_legacy_dataset(legacy_dataset)

    # Step 2: Create v2 dataset with inferred schema
    experimental_dataset = Dataset(analysis_result.schema)

    # Step 3: Convert all items
    for legacy_item in legacy_dataset:
        # Convert legacy item to v2 sample
        sample_data = _convert_legacy_item(legacy_item, analysis_result)
        if analysis_result.is_hierarchical and isinstance(sample_data["label"], int):
            # Convert single labels in hierarchical project to be a list
            sample_data["label"] = [sample_data["label"]]
        # Create sample and add to dataset
        sample = Sample(**sample_data)
        experimental_dataset.append(sample)

    if analysis_result.is_anomaly:
        categories = experimental_dataset.schema.attributes["label"].categories
        if not isinstance(categories, LabelCategories):
            raise ValueError("Expected label categories for anomaly detection.")

        good_categories = [index for index, label in enumerate(categories.labels) if label == "good"]

        if len(good_categories) != 1:
            raise ValueError("Expected exactly one 'good' label for anomaly detection.")

        good_category_index = good_categories[0]

        new_categories = LabelCategories(
            labels=("normal", "anomalous"),
            label_semantics={LabelSemantic.NORMAL: "normal", LabelSemantic.ANOMALOUS: "anomalous"},
        )
        experimental_dataset.schema.attributes["label"].categories = new_categories
        experimental_dataset.df = experimental_dataset.df.with_columns(
            pl.col("label").replace([good_category_index], [0], default=1)
        )

    return experimental_dataset


class BackwardMediaConverter(ABC):
    """Base class for backward media type converters."""

    @classmethod
    @abstractmethod
    def create_from_schema(cls, schema: Schema) -> BackwardMediaConverter | None:
        """Create converter instance if schema is supported, None otherwise."""

    @abstractmethod
    def get_media_type(self) -> type[MediaElement[Any]]:
        """Get the legacy media type this converter produces."""

    @abstractmethod
    def convert_to_legacy_media(self, sample: Sample) -> MediaElement[Any]:
        """Convert v2 sample media to legacy MediaElement."""


class BackwardAnnotationConverter(ABC):
    """Base class for backward annotation type converters."""

    @classmethod
    @abstractmethod
    def create_from_schema(cls, schema: Schema) -> BackwardAnnotationConverter | None:
        """Create converter instance if schema is supported, None otherwise."""

    @abstractmethod
    def get_annotation_type(self) -> AnnotationType:
        """Get the legacy annotation type this converter produces."""

    @abstractmethod
    def infer_categories(self, experimental_dataset: Dataset[Sample]) -> CategoriesInfo:
        """Infer legacy categories from v2 dataset."""

    @abstractmethod
    def convert_to_legacy_annotations(self, sample: Sample, categories: CategoriesInfo) -> list[Annotation]:
        """Convert v2 sample annotations to legacy format."""


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


class ForwardMaskAnnotationConverter(ForwardAnnotationConverter):
    """Forward converter for mask annotations handling both semantic and instance segmentation.

    For semantic segmentation:
    - Creates a single uint8 mask where pixel values = class labels

    For instance segmentation:
    - Creates N binary masks (N = number of instances)
    - Each mask represents a single instance
    - Labels array stores class label for each instance
    """

    def __init__(
        self,
        mask_attribute: AttributeInfo,
        instance_mask_attribute: AttributeInfo,
        mask_labels_attribute: AttributeInfo | None,
        is_semantic: bool,
        name_prefix: str,
    ):
        """Initialize with mask attributes and optional label attribute."""
        self.mask_attribute = mask_attribute
        self.instance_mask_attribute = instance_mask_attribute
        self.mask_labels_attribute = mask_labels_attribute
        self.is_semantic = is_semantic
        self.name_prefix = name_prefix

    @classmethod
    def create(
        cls, dataset: LegacyDataset, semantic: Semantic = Semantic.Default, name_prefix: str = ""
    ) -> ForwardMaskAnnotationConverter | None:
        """Create converter instance for mask annotations.

        Determines if the dataset uses semantic or instance segmentation by checking
        if mask indices match their labels across all mask annotations.
        """
        categories = dataset.categories()
        # Extract label categories if available
        legacy_label_categories = categories.get(AnnotationType.label, None)

        # Check all masks in the dataset to determine if semantic or instance segmentation
        is_semantic = all(
            ann.label is not None and ann.index == ann.label
            for item in dataset
            for ann in item.annotations
            if isinstance(ann, ExtractedMask)
        )

        # Create categories based on legacy label categories
        new_label_categories = None
        labels = []
        if legacy_label_categories is not None:
            labels = tuple(label_item.name for label_item in legacy_label_categories.items)

        mask_labels_attribute = None

        if is_semantic:
            # For semantic segmentation, use MaskCategories for mask_attribute
            colormap = {}

            # Have at least one label for the background
            if len(labels) == 0:
                labels = ("background",)

            # Generate colors for all labels plus background
            num_classes = len(labels)
            colormap_dict = generate_colormap(num_classes, include_background=True)

            # Convert colors to RgbColor
            for index, color in colormap_dict.items():
                if isinstance(color, tuple):
                    colormap[index] = RgbColor(*color)
                else:
                    colormap[index] = color

            mask_categories = MaskCategories(labels=labels, colormap=colormap)
            mask_attribute = AttributeInfo(
                type=callable,
                field=mask_callable_field(dtype=pl.UInt8, semantic=semantic),
                categories=mask_categories,
            )
            instance_mask_attribute = None
        else:
            # For instance segmentation, use no categories
            mask_attribute = None

            # Configure instance mask attribute with Boolean dtype for binary instance masks
            instance_mask_attribute = AttributeInfo(
                type=callable,
                field=instance_mask_callable_field(dtype=pl.Boolean, semantic=semantic),
            )

            # Only add mask_labels if we have label categories
            if len(labels) > 0:
                new_label_categories = LabelCategories(labels=labels)

                mask_labels_attribute = AttributeInfo(
                    type=np.ndarray,
                    field=label_field(is_list=True, semantic=semantic),  # Labels for each instance
                    categories=new_label_categories,
                )

        return cls(
            mask_attribute=mask_attribute,
            instance_mask_attribute=instance_mask_attribute,
            mask_labels_attribute=mask_labels_attribute,
            is_semantic=is_semantic,
            name_prefix=name_prefix,
        )

    @classmethod
    def get_supported_annotation_types(cls) -> list[AnnotationType]:
        """Return list of annotation types this converter can handle."""
        return [AnnotationType.mask]

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        """Return schema attributes."""
        attributes = {}

        if self.mask_attribute is not None:
            attributes[self.name_prefix + "mask_callable"] = self.mask_attribute
        if self.instance_mask_attribute is not None:
            attributes[self.name_prefix + "instance_mask_callable"] = self.instance_mask_attribute
        if self.mask_labels_attribute is not None:
            attributes[self.name_prefix + "labels"] = self.mask_labels_attribute
        return attributes

    def convert_annotations(self, annotations: list[Annotation], item: DatasetItem) -> dict[str, Any]:  # noqa: ARG002
        """Convert legacy mask annotations to either semantic or instance segmentation format."""
        # Extract mask annotations
        extracted_masks = [ann for ann in annotations if isinstance(ann, ExtractedMask)]

        results = {}

        if self.is_semantic:
            # Convert to semantic segmentation mask
            def get_semantic_mask() -> np.ndarray:
                if len(extracted_masks) == 0:
                    return None

                # Initialize empty mask
                # Get first mask to determine shape
                first_mask = extracted_masks[0].image
                output_mask = np.zeros(first_mask.shape, dtype=np.uint8)

                # Combine all masks into a single semantic mask
                for mask in extracted_masks:
                    if mask.label is not None:
                        mask_data = mask.image
                        output_mask[mask_data] = mask.label

                return output_mask

            results[self.name_prefix + "mask_callable"] = get_semantic_mask

        else:
            # Convert to instance segmentation masks
            def get_instance_masks() -> np.ndarray:
                if len(extracted_masks) == 0:
                    return np.zeros((0, 0, 0), dtype=bool)

                # Get first mask to determine shape
                first_mask = extracted_masks[0].image
                shape = (len(extracted_masks), *first_mask.shape)

                # Create array of binary instance masks
                instance_masks = np.empty(shape, dtype=bool)

                # Fill instance masks
                for i, mask in enumerate(extracted_masks):
                    instance_masks[i] = mask.image

                return instance_masks

            results[self.name_prefix + "instance_mask_callable"] = get_instance_masks

            # Add instance labels
            labels = [mask.label if mask.label is not None else 0 for mask in extracted_masks]

            # Add labels only for instance segmentation
            if self.mask_labels_attribute is not None:
                results[self.name_prefix + "labels"] = np.array(labels, dtype=np.int32)

        return results


class BackwardImageMediaConverter(BackwardMediaConverter):
    """Backward converter for Image media type."""

    def __init__(self, image_path_attr: str):
        """Initialize with the name of the image path attribute."""
        self.image_path_attr = image_path_attr

    @classmethod
    def create_from_schema(cls, schema: Schema) -> BackwardImageMediaConverter | None:
        """Create converter instance if schema contains image_path field."""
        for attr_name, attr_info in schema.attributes.items():
            if isinstance(attr_info.field, ImagePathField):
                return cls(image_path_attr=attr_name)
        return None

    def get_media_type(self) -> type[MediaElement[Any]]:
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
    def create_from_schema(cls, schema: Schema) -> BackwardBboxAnnotationConverter | None:
        """Create converter instance if schema contains bbox-related fields."""
        bboxes_attr = None
        bbox_labels_attr = None

        # Find bbox field
        for attr_name, attr_info in schema.attributes.items():
            if isinstance(attr_info.field, BBoxField):
                bboxes_attr = attr_name
                break

        # Find bbox_labels field (look for label field with 'bbox_labels' in name or similar pattern)
        for attr_name, attr_info in schema.attributes.items():
            if isinstance(attr_info.field, LabelField):
                bbox_labels_attr = attr_name
                break

        if bboxes_attr and bbox_labels_attr:
            return cls(bboxes_attr=bboxes_attr, bbox_labels_attr=bbox_labels_attr)
        return None

    def get_annotation_type(self) -> AnnotationType:
        return AnnotationType.bbox

    def convert_to_legacy_annotations(self, sample: Sample, categories: CategoriesInfo) -> list[Annotation]:  # noqa: ARG002
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

        # Collect all unique label IDs
        label_ids: set[int] = set()
        for sample in experimental_dataset:
            bbox_labels = getattr(sample, self.bbox_labels_attr, None)
            if bbox_labels is not None:
                for label_id in bbox_labels:
                    if label_id is not None:
                        label_ids.add(int(label_id))

        # Create label categories
        label_categories = LegacyLabelCategories()
        for label_id in sorted(label_ids):
            label_categories.add(f"class_{label_id}")

        return {AnnotationType.label: label_categories}


class BackwardRotatedBboxAnnotationConverter(BackwardAnnotationConverter):
    """Backward converter for RotatedBbox annotations."""

    def __init__(self, rotated_bboxes_attr: str, rotated_bbox_labels_attr: str | None):
        """Initialize with the names of the rotated bbox-related attributes."""
        self.rotated_bboxes_attr = rotated_bboxes_attr
        self.rotated_bbox_labels_attr = rotated_bbox_labels_attr

    @classmethod
    def create_from_schema(cls, schema: Schema) -> BackwardRotatedBboxAnnotationConverter | None:
        """Create converter if schema contains rotated bbox fields."""
        rotated_bboxes_attr: str | None = None
        rotated_bbox_labels_attr: str | None = None

        for attr_name, attr_info in schema.attributes.items():
            if isinstance(attr_info.field, RotatedBBoxField):
                rotated_bboxes_attr = attr_name
            elif isinstance(attr_info.field, LabelField):
                rotated_bbox_labels_attr = attr_name

        if rotated_bboxes_attr is None:
            return None

        return cls(rotated_bboxes_attr, rotated_bbox_labels_attr)

    def get_annotation_type(self) -> AnnotationType:
        return AnnotationType.rotated_bbox

    def convert_to_legacy_annotations(self, sample: Sample, categories: CategoriesInfo) -> list[Annotation]:  # noqa: ARG002
        """Convert v2 rotated bbox data to legacy RotatedBbox annotations."""
        rotated_bboxes = getattr(sample, self.rotated_bboxes_attr, None)
        if rotated_bboxes is None or len(rotated_bboxes) == 0:
            return []

        rotated_bbox_labels = None
        if self.rotated_bbox_labels_attr is not None:
            rotated_bbox_labels = getattr(sample, self.rotated_bbox_labels_attr, None)

        annotations: list[Annotation] = []
        for i, bbox in enumerate(rotated_bboxes):
            cx, cy, w, h, r_radians = bbox
            # Convert from radians to degrees
            r_degrees = math.degrees(r_radians)

            label = None
            if rotated_bbox_labels is not None and i < len(rotated_bbox_labels):
                label = int(rotated_bbox_labels[i])

            annotation = RotatedBbox(
                cx=float(cx),
                cy=float(cy),
                w=float(w),
                h=float(h),
                r=float(r_degrees),
                label=label,
            )
            annotations.append(annotation)

        return annotations

    def infer_categories(self, experimental_dataset: Dataset[Sample]) -> CategoriesInfo:
        """Infer label categories from rotated_bbox_labels."""

        # Collect all unique label IDs
        label_ids: set[int] = set()
        for sample in experimental_dataset:
            if self.rotated_bbox_labels_attr is not None:
                rotated_bbox_labels = getattr(sample, self.rotated_bbox_labels_attr, None)
                if rotated_bbox_labels is not None:
                    for label_id in rotated_bbox_labels:
                        if label_id is not None:
                            label_ids.add(int(label_id))

        # Create label categories
        label_categories = LegacyLabelCategories()
        for label_id in sorted(label_ids):
            label_categories.add(f"class_{label_id}")

        return {AnnotationType.label: label_categories}


class BackwardPolygonAnnotationConverter(BackwardAnnotationConverter):
    """Backward converter for Polygon annotations."""

    def __init__(self, polygons_attr: str, polygon_labels_attr: str | None):
        """Initialize with the names of the polygon-related attributes."""
        self.polygons_attr = polygons_attr
        self.polygon_labels_attr = polygon_labels_attr

    @classmethod
    def create_from_schema(cls, schema: Schema) -> BackwardPolygonAnnotationConverter | None:
        """Create converter instance if schema contains polygon-related fields."""
        polygons_attr = None
        polygon_labels_attr = None

        # Find polygon field
        for attr_name, attr_info in schema.attributes.items():
            if isinstance(attr_info.field, PolygonField):
                polygons_attr = attr_name
                break

        # Find polygon_labels field
        for attr_name, attr_info in schema.attributes.items():
            if isinstance(attr_info.field, LabelField):
                polygon_labels_attr = attr_name
                break

        if polygons_attr:
            return cls(polygons_attr=polygons_attr, polygon_labels_attr=polygon_labels_attr)
        return None

    def get_annotation_type(self) -> AnnotationType:
        return AnnotationType.polygon

    def convert_to_legacy_annotations(self, sample: Sample, categories: CategoriesInfo) -> list[Annotation]:  # noqa: ARG002
        """Convert polygons and polygon_labels back to legacy Polygon annotations."""
        polygons = getattr(sample, self.polygons_attr)
        polygon_labels = getattr(sample, self.polygon_labels_attr) if self.polygon_labels_attr else None

        annotations: list[Annotation] = []
        for i in range(len(polygons)):
            flat_coords = polygons[i].reshape(-1)
            label = int(polygon_labels[i]) if polygon_labels is not None else None

            polygon = Polygon(points=flat_coords, label=label)
            annotations.append(polygon)

        return annotations

    def infer_categories(self, experimental_dataset: Dataset[Sample]) -> CategoriesInfo:
        """Infer label categories from polygon_labels."""

        if self.polygon_labels_attr is None:
            return {}

        # Collect all unique label IDs
        label_ids: set[int] = set()
        for sample in experimental_dataset:
            polygon_labels = getattr(sample, self.polygon_labels_attr, None)
            if polygon_labels is not None:
                for label_id in polygon_labels:
                    if label_id is not None:
                        label_ids.add(int(label_id))

        # Create label categories
        label_categories = LegacyLabelCategories()
        for label_id in sorted(label_ids):
            label_categories.add(f"class_{label_id}")

        return {AnnotationType.label: label_categories}


@dataclass
class BackwardAnalysisResult:
    """Result of v2 dataset analysis for backward conversion."""

    media_type: type[MediaElement[Any]] | None
    ann_types: set[AnnotationType]
    categories: CategoriesInfo
    media_converter: BackwardMediaConverter | None
    ann_converters: dict[AnnotationType, BackwardAnnotationConverter]


def analyze_experimental_dataset(experimental_dataset: Dataset[Sample]) -> BackwardAnalysisResult:
    """Analyze v2 dataset schema to determine legacy format.

    Args:
        experimental_dataset: The v2 dataset to analyze

    Returns:
        BackwardAnalysisResult containing legacy format information
    """
    schema = experimental_dataset.schema

    # Find compatible media converter
    media_converter: BackwardMediaConverter | None = None
    media_type: type[MediaElement[Any]] | None = None

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


def _convert_experimental_item(index: int, sample: Sample, backward_analysis: BackwardAnalysisResult) -> DatasetItem:
    """Convert v2 sample to legacy DatasetItem."""

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
    """Convert v2 dataset to legacy format.

    Args:
        experimental_dataset: The v2 Dataset to convert

    Returns:
        A new legacy Datumaro Dataset with converted data

    Example:
        >>> experimental_ds = Dataset(MySchema)
        >>> # ... add samples to experimental_ds
        >>> legacy_ds = convert_to_legacy(experimental_ds)
        >>> legacy_ds.export("output", "coco")
    """

    # Step 1: Analyze v2 dataset
    backward_analysis = analyze_experimental_dataset(experimental_dataset)

    # Step 2: Create legacy dataset items
    legacy_items: list[DatasetItem] = []
    for i, sample in enumerate(experimental_dataset):
        legacy_item = _convert_experimental_item(i, sample, backward_analysis)
        legacy_items.append(legacy_item)

    # Step 3: Create legacy dataset
    return LegacyDataset.from_iterable(  # pyright: ignore[reportUnknownMemberType]
        legacy_items,
        categories=backward_analysis.categories,
        media_type=backward_analysis.media_type or MediaElement,
    )


class SubsetConverter(ForwardAnnotationConverter):
    """Converts legacy subset strings to Subset enum values.

    This converter handles mapping of legacy subset strings to their standardized
    Subset enum values while preserving unrecognized values as strings.

    The following mappings are supported:
    - TRAINING: "train", "training" -> Subset.TRAINING
    - VALIDATION: "val", "validation" -> Subset.VALIDATION
    - TESTING: "test", "testing" -> Subset.TESTING
    """

    # Case-insensitive + synonym lookup table mapping strings to enum values
    _BASE_SUBSET_SYNONYMS = {
        "train": Subset.TRAINING,
        "training": Subset.TRAINING,
        "val": Subset.VALIDATION,
        "validation": Subset.VALIDATION,
        "test": Subset.TESTING,
        "testing": Subset.TESTING,
    }

    _SUBSET_MAP: dict[str, Subset] = {}
    for key, value in _BASE_SUBSET_SYNONYMS.items():
        for variant in {key, key.upper(), key.capitalize()}:
            _SUBSET_MAP[variant] = value

    def __init__(self, semantic: Semantic = Semantic.Default, name_prefix: str = ""):
        """Initialize the converter.

        Args:
            semantic: The semantic type for the converted fields
            name_prefix: Prefix to prepend to all field names
        """
        self._semantic = semantic
        self._name_prefix = name_prefix

    @classmethod
    def get_supported_annotation_types(cls) -> list[AnnotationType]:
        """Return list of annotation types this converter can handle."""
        return []  # Subset conversion does not handle any specific annotation type

    @classmethod
    def create(
        cls,
        dataset: LegacyDataset,  # noqa: ARG003
        semantic: Semantic = Semantic.Default,
        name_prefix: str = "",
    ) -> ForwardAnnotationConverter | None:
        """Create converter instance if dataset supports this annotation type.

        Args:
            dataset: Legacy dataset to create converter from
            semantic: The semantic type for the converted fields
            name_prefix: Prefix to prepend to all field names
        """
        return cls(semantic=semantic, name_prefix=name_prefix)

    def get_schema_attributes(self) -> dict[str, AttributeInfo]:
        """Return schema attributes for this annotation type.

        Returns:
            A dictionary with a single entry for the subset field, using SubsetField
            with the configured semantic type.
        """
        field_name = f"{self._name_prefix}subset" if self._name_prefix else "subset"
        return {
            field_name: AttributeInfo(
                type=str,
                field=subset_field(semantic=self._semantic),
            )
        }

    def convert_annotations(self, annotations: list[Annotation], item: DatasetItem) -> dict[str, Any]:  # noqa: ARG002
        """Convert dataset item subset to standardized format.

        Args:
            annotations: List of annotations (not used by this converter)
            item: Legacy dataset item containing the subset information

        Returns:
            Dict containing the converted subset field
        """
        subset = item.subset
        if subset is None:
            return {}

        # Convert legacy subset name to standardized format
        converted_subset = self._SUBSET_MAP.get(subset, subset)
        field_name = f"{self._name_prefix}subset" if self._name_prefix else "subset"
        return {field_name: converted_subset}


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
