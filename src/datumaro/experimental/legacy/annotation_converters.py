from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import polars as pl

from datumaro import Annotation, AnnotationType, Bbox, CategoriesInfo, DatasetItem, Ellipse, Label, Points, Polygon
from datumaro import Dataset as LegacyDataset
from datumaro import LabelCategories as LegacyLabelCategories
from datumaro.components.annotation import ExtractedMask, RotatedBbox
from datumaro.experimental import (
    AttributeInfo,
    BBoxField,
    Dataset,
    LabelField,
    PolygonField,
    RotatedBBoxField,
    Sample,
    Schema,
    Semantic,
    bbox_field,
    label_field,
    polygon_field,
    rotated_bbox_field,
)
from datumaro.experimental.categories import LabelCategories, MaskCategories, RgbColor
from datumaro.experimental.fields import (
    EllipseField,
    instance_mask_callable_field,
    keypoints_field,
    mask_callable_field,
)
from datumaro.util.mask_tools import generate_colormap


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


_annotation_converters: dict[AnnotationType, type[ForwardAnnotationConverter]] = {}


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
