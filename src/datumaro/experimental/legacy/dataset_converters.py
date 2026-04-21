from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from datumaro import Annotation, AnnotationType, CategoriesInfo, DatasetItem, MediaElement
from datumaro import Dataset as LegacyDataset
from datumaro.experimental.categories import (
    GroupType,
    HierarchicalLabelCategories,
    HierarchicalLabelCategory,
    LabelCategories,
    LabelGroup,
    LabelSemantic,
)
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields.annotations import label_field
from datumaro.experimental.fields.datasets import Subset, subset_field
from datumaro.experimental.legacy.annotation_converters import (
    BackwardAnnotationConverter,
    ForwardAnnotationConverter,
    get_forward_annotation_converter,
)
from datumaro.experimental.legacy.media_converters import BackwardMediaConverter, ForwardMediaConverter
from datumaro.experimental.legacy.register_legacy_converters import (
    _backward_annotation_converter_classes,
    _backward_media_converter_classes,
    get_forward_media_converter,
)
from datumaro.experimental.schema import AttributeInfo, Schema


@dataclass
class AnalysisResult:
    """Result of legacy dataset analysis."""

    schema: Schema
    media_converter: ForwardMediaConverter | None
    ann_converters: dict[AnnotationType, ForwardAnnotationConverter]
    subset_converter: SubsetConverter
    is_anomaly: bool
    is_hierarchical: bool


def analyze_legacy_dataset(
    legacy_dataset: LegacyDataset,
    semantic: str = "default",
    hierarchical: bool = False,
    multi_label: bool = False,
    anomaly: bool = False,
) -> AnalysisResult:
    """Analyze legacy dataset and generate schema using registered converters.

    Args:
        legacy_dataset: The legacy Datumaro dataset to analyze
        semantic: The semantic type for the converted fields
        hierarchical: Boolean indicating if the dataset should be treated as hierarchical
        multi_label: Boolean indicating if the dataset should be treated as multi-label
        anomaly: Boolean indicating if the dataset should be treated as anomaly

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
        is_hierarchical = (_has_derived_labels(label_group_names) or _has_derived_labels(label_names)) or hierarchical

        # Look for multi label classification groups
        multi_label_group_names = [name for name in label_group_names if name.startswith("Classification labels__")]
        is_multi_label = (len(multi_label_group_names) > 1 and not is_hierarchical) or multi_label
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
    is_anomaly = (AnnotationType.label in ann_types and len(ann_types) > 1) or anomaly

    # Skip bbox converter when polygon converter exists, since polygon converter
    # now generates bboxes from polygon bounds for proper instance segmentation alignment
    skip_bbox_converter = AnnotationType.polygon in ann_types and AnnotationType.bbox in ann_types

    for ann_type in ann_types:
        # Skip bbox converter if polygon converter will handle bboxes
        if skip_bbox_converter and ann_type == AnnotationType.bbox:
            continue

        ann_semantic = "anomaly" if is_anomaly and ann_type != AnnotationType.label else semantic
        name_prefix = "anomaly_" if is_anomaly and ann_type != AnnotationType.label else ""
        converter = get_forward_annotation_converter(ann_type, legacy_dataset, ann_semantic, name_prefix)
        if converter is not None:
            ann_converters[ann_type] = converter
            ann_attributes = converter.get_schema_attributes()
            if (is_multi_label or is_hierarchical) and AnnotationType.label.name in ann_attributes:
                ann_attributes[AnnotationType.label.name].field = label_field(multi_label=True)
                ann_attributes[AnnotationType.label.name].type = np.ndarray
            if is_hierarchical and AnnotationType.label.name in ann_attributes:
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


def convert_from_legacy(
    legacy_dataset: LegacyDataset, hierarchical: bool = False, multi_label: bool = False, anomaly: bool = False
) -> Dataset[Sample]:
    """Convert legacy dataset to new dataset format with automatic schema inference.

    Args:
        legacy_dataset: The legacy Datumaro dataset to convert
        hierarchical: If True, forces hierarchical classification; otherwise, uses automatic detection.
        multi_label: If True, forces multi-label classification; otherwise, uses automatic detection.
        anomaly: If True, forces anomaly detection; otherwise, uses automatic detection.
    Returns:
        A new Dataset with inferred schema and converted data

    Example:
        >>> legacy_ds = Dataset.import_from("path/to/coco", "coco")
        >>> experimental_ds = convert_from_legacy(legacy_ds)
        >>> sample = experimental_ds[0]
        >>> print(sample.image_path)
        >>> print(sample.bboxes.shape)
    """

    # Step 1: Analyze dataset to infer schema
    analysis_result = analyze_legacy_dataset(
        legacy_dataset, hierarchical=hierarchical, multi_label=multi_label, anomaly=anomaly
    )

    # Step 2: Build samples from the legacy items.
    samples: list[Sample] = []
    for legacy_item in legacy_dataset:
        if analysis_result.media_converter is not None and analysis_result.media_converter.should_skip_item(
            legacy_item
        ):
            continue

        sample_data = _convert_legacy_item(legacy_item, analysis_result)

        if analysis_result.is_hierarchical and isinstance(sample_data.get("label"), int):
            sample_data["label"] = [sample_data["label"]]

        samples.append(Sample(**sample_data))

    # Step 3: Create the dataset and bulk-append the samples.
    experimental_dataset: Dataset[Sample] = Dataset(analysis_result.schema)
    experimental_dataset.append_batch(samples)

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


@dataclass
class BackwardAnalysisResult:
    """Result of dataset analysis for backward conversion to legacy dataset."""

    media_type: type[MediaElement[Any]] | None
    ann_types: set[AnnotationType]
    categories: CategoriesInfo
    media_converter: BackwardMediaConverter | None
    ann_converters: dict[AnnotationType, BackwardAnnotationConverter]


def analyze_experimental_dataset(experimental_dataset: Dataset[Sample]) -> BackwardAnalysisResult:
    """Analyze dataset schema to determine legacy format.

    Args:
        experimental_dataset: The dataset to analyze

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
    """Convert dataset sample to legacy DatasetItem."""

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
    """Convert dataset to legacy dataset format.

    Args:
        experimental_dataset: The Dataset to convert

    Returns:
        A new legacy Datumaro Dataset with converted data

    Example:
        >>> experimental_ds = Dataset(MySchema)
        >>> # ... add samples to experimental_ds
        >>> legacy_ds = convert_to_legacy(experimental_ds)
        >>> legacy_ds.export("output", "coco")
    """

    # Step 1: Analyze the dataset
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

    def __init__(self, semantic: str = "default", name_prefix: str = ""):
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
        semantic: str = "default",
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
