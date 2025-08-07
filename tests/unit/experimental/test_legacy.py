"""Unit tests for legacy dataset conversion functionality."""

import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from typing_extensions import Annotated

from datumaro.components.annotation import AnnotationType, Bbox
from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import CategoriesInfo, DatasetItem
from datumaro.components.media import Image, ImageFromFile
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import bbox_field, image_path_field, tensor_field
from datumaro.experimental.legacy import (
    BackwardBboxAnnotationConverter,
    BackwardImageMediaConverter,
    ForwardBboxAnnotationConverter,
    ForwardImageMediaConverter,
    analyze_experimental_dataset,
    analyze_legacy_dataset,
    convert_from_legacy,
    convert_to_legacy,
    get_forward_annotation_converter,
    get_forward_media_converter,
    register_forward_annotation_converter,
    register_forward_media_converter,
)


def test_image_media_converter_get_schema_attributes():
    """Test schema attribute generation for images."""
    converter = ForwardImageMediaConverter()
    attributes = converter.get_schema_attributes()

    assert "image_path" in attributes
    assert attributes["image_path"].type == str


def test_image_media_converter_convert_item_media_with_path_and_size():
    """Test media conversion with path and size."""
    converter = ForwardImageMediaConverter()

    image_media = Image.from_file(
        "/path/to/image.jpg", size=(480, 640)
    )  # pyright: ignore[reportUnknownMemberType]

    item = DatasetItem(id="test", media=image_media)
    result = converter.convert_item_media(item)

    assert result["image_path"] == "/path/to/image.jpg"


def test_image_media_converter_convert_item_media_with_path_only():
    """Test media conversion with only path."""
    converter = ForwardImageMediaConverter()

    image_media = Image.from_file("/path/to/image.jpg")  # pyright: ignore[reportUnknownMemberType]

    item = DatasetItem(id="test", media=image_media)
    result = converter.convert_item_media(item)

    assert result["image_path"] == "/path/to/image.jpg"


def test_image_media_converter_convert_item_media_no_media():
    """Test media conversion with no media."""
    converter = ForwardImageMediaConverter()
    item = DatasetItem(id="test", media=None)
    result = converter.convert_item_media(item)

    assert result == {}


def test_bbox_annotation_converter_get_schema_attributes():
    """Test schema attribute generation for bboxes."""
    converter = ForwardBboxAnnotationConverter()
    categories: CategoriesInfo = {}  # Empty categories

    attributes = converter.get_schema_attributes(categories)

    assert "bboxes" in attributes
    assert "bbox_labels" in attributes
    assert attributes["bboxes"].type == np.ndarray
    assert attributes["bbox_labels"].type == np.ndarray


def test_bbox_annotation_converter_convert_annotations_single_bbox():
    """Test conversion of single bbox annotation."""
    converter = ForwardBboxAnnotationConverter()

    bbox = Bbox(10, 20, 30, 40, label=1)  # x=10, y=20, w=30, h=40
    item = DatasetItem(id="test")

    result = converter.convert_annotations([bbox], item)

    expected_bbox = np.array([[10, 20, 40, 60]], dtype=np.float32)  # x1,y1,x2,y2 format
    expected_labels = np.array([1], dtype=np.int32)

    np.testing.assert_array_equal(result["bboxes"], expected_bbox)
    np.testing.assert_array_equal(result["bbox_labels"], expected_labels)


def test_bbox_annotation_converter_convert_annotations_multiple_bboxes():
    """Test conversion of multiple bbox annotations."""
    converter = ForwardBboxAnnotationConverter()

    bbox1 = Bbox(10, 20, 30, 40, label=1)
    bbox2 = Bbox(50, 60, 70, 80, label=2)
    item = DatasetItem(id="test")

    result = converter.convert_annotations([bbox1, bbox2], item)

    expected_bboxes = np.array(
        [
            [10, 20, 40, 60],  # x1,y1,x2,y2 for first bbox
            [50, 60, 120, 140],  # x1,y1,x2,y2 for second bbox
        ],
        dtype=np.float32,
    )
    expected_labels = np.array([1, 2], dtype=np.int32)

    np.testing.assert_array_equal(result["bboxes"], expected_bboxes)
    np.testing.assert_array_equal(result["bbox_labels"], expected_labels)


def test_bbox_annotation_converter_convert_annotations_empty_list():
    """Test conversion of empty annotation list."""
    converter = ForwardBboxAnnotationConverter()
    item = DatasetItem(id="test")

    result = converter.convert_annotations([], item)

    # Empty arrays with proper shapes
    assert result["bboxes"].shape == (0, 4)
    assert result["bboxes"].dtype == np.float32
    assert result["bbox_labels"].shape == (0,)
    assert result["bbox_labels"].dtype == np.int32


# Converter registry tests


def test_register_and_get_media_converter():
    """Test media converter registration and retrieval."""
    converter = ForwardImageMediaConverter()
    register_forward_media_converter(Image, converter)

    retrieved = get_forward_media_converter(Image)
    assert retrieved is converter


def test_register_and_get_annotation_converter():
    """Test annotation converter registration and retrieval."""
    converter = ForwardBboxAnnotationConverter()
    register_forward_annotation_converter(AnnotationType.bbox, converter)

    retrieved = get_forward_annotation_converter(AnnotationType.bbox)
    assert retrieved is converter


def test_get_nonexistent_media_converter():
    """Test getting converter for unregistered media type."""
    from datumaro.components.media import Video

    with pytest.raises(ValueError, match="No converter registered for media type"):
        get_forward_media_converter(Video)


def test_get_nonexistent_annotation_converter():
    """Test getting converter for unregistered annotation type."""
    with pytest.raises(ValueError, match="No converter registered for annotation type"):
        get_forward_annotation_converter(AnnotationType.polygon)


def test_analyze_image_and_bbox_dataset():
    """Test analysis of dataset with images and bboxes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        image_path = str(temp_path / "image1.jpg")
        image_media = Image.from_file(
            image_path, size=(480, 640)
        )  # pyright: ignore[reportUnknownMemberType]
        bbox = Bbox(10, 20, 30, 40, label=1)
        item = DatasetItem(id="item1", media=image_media, annotations=[bbox])

        dataset = LegacyDataset.from_iterable(
            [item], ann_types={AnnotationType.bbox}
        )  # pyright: ignore[reportUnknownMemberType]

        analysis_result = analyze_legacy_dataset(dataset)

        # Should have image and bbox attributes
        assert "image_path" in analysis_result.schema.attributes
        assert "bboxes" in analysis_result.schema.attributes
        assert "bbox_labels" in analysis_result.schema.attributes


def test_analyze_image_only_dataset():
    """Test analysis of dataset with only images."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        image_path = str(temp_path / "image1.jpg")
        image_media = Image.from_file(
            image_path, size=(480, 640)
        )  # pyright: ignore[reportUnknownMemberType]
        item = DatasetItem(id="item1", media=image_media, annotations=[])

        dataset = LegacyDataset.from_iterable([item])  # pyright: ignore[reportUnknownMemberType]

        analysis_result = analyze_legacy_dataset(dataset)

        # Should have only image attributes
        assert "image_path" in analysis_result.schema.attributes
        assert "bboxes" not in analysis_result.schema.attributes
        assert "bbox_labels" not in analysis_result.schema.attributes


def test_analyze_unknown_annotation_type():
    """Test analysis with unknown annotation type."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Polygon annotation (not registered)
        from datumaro.components.annotation import Polygon

        image_path = str(temp_path / "image1.jpg")
        image_media = Image.from_file(
            image_path, size=(480, 640)
        )  # pyright: ignore[reportUnknownMemberType]
        polygon = Polygon([10, 20, 30, 40, 50, 60], label=1)
        item = DatasetItem(id="item1", media=image_media, annotations=[polygon])

        dataset = LegacyDataset.from_iterable(
            [item], ann_types={AnnotationType.polygon}
        )  # pyright: ignore[reportUnknownMemberType]

        analysis_result = analyze_legacy_dataset(dataset)

        # Should skip unknown annotation type
        assert "image_path" in analysis_result.schema.attributes
        assert len(analysis_result.schema.attributes) == 1  # Only image_path field


def test_convert_simple_bbox_dataset():
    """Test conversion of a simple dataset with bboxes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        image_path = str(temp_path / "image1.jpg")
        image_media = Image.from_file(
            image_path, size=(480, 640)
        )  # pyright: ignore[reportUnknownMemberType]

        bbox1 = Bbox(10, 20, 30, 40, label=1)
        bbox2 = Bbox(50, 60, 70, 80, label=2)

        item1 = DatasetItem(id="item1", media=image_media, annotations=[bbox1, bbox2])

        dataset = LegacyDataset.from_iterable(
            [item1], ann_types={AnnotationType.bbox}
        )  # pyright: ignore[reportUnknownMemberType]

        # Convert dataset
        experimental_ds = convert_from_legacy(dataset)

        assert len(experimental_ds.df) == 1

        sample = experimental_ds[0]

        # Check attributes
        assert hasattr(sample, "image_path")
        assert Path(getattr(sample, "image_path")) == Path(image_path)
        assert hasattr(sample, "bboxes")
        assert hasattr(sample, "bbox_labels")

        expected_bboxes = np.array(
            [
                [10, 20, 40, 60],  # First bbox in x1,y1,x2,y2 format
                [50, 60, 120, 140],  # Second bbox in x1,y1,x2,y2 format
            ],
            dtype=np.float32,
        )
        expected_labels = np.array([1, 2], dtype=np.int32)

        np.testing.assert_array_equal(getattr(sample, "bboxes"), expected_bboxes)
        np.testing.assert_array_equal(getattr(sample, "bbox_labels"), expected_labels)


def test_convert_empty_dataset():
    """Test conversion of empty dataset."""
    dataset = LegacyDataset.from_iterable([])  # pyright: ignore[reportUnknownMemberType]

    experimental_ds = convert_from_legacy(dataset)
    assert len(experimental_ds.df) == 0


def test_convert_image_only_dataset():
    """Test conversion of dataset with only images, no annotations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        image_path = str(temp_path / "image1.jpg")
        image_media = Image.from_file(
            image_path, size=(480, 640)
        )  # pyright: ignore[reportUnknownMemberType]
        item = DatasetItem(id="item1", media=image_media, annotations=[])

        dataset = LegacyDataset.from_iterable([item])  # pyright: ignore[reportUnknownMemberType]

        experimental_ds = convert_from_legacy(dataset)

        assert len(experimental_ds.df) == 1
        sample = experimental_ds[0]

        # Should have image attributes only
        assert hasattr(sample, "image_path")
        assert Path(getattr(sample, "image_path")) == Path(image_path)
        assert not hasattr(sample, "bboxes")
        assert not hasattr(sample, "bbox_labels")


def test_builtin_converters_registration():
    """Test that built-in converters are registered on import."""
    image_converter = get_forward_media_converter(Image)
    assert isinstance(image_converter, ForwardImageMediaConverter)

    bbox_converter = get_forward_annotation_converter(AnnotationType.bbox)
    assert isinstance(bbox_converter, ForwardBboxAnnotationConverter)


# Define a sample schema for testing convert_to_legacy
class DetectionSample(Sample):
    image_path: Annotated[str, image_path_field()]
    bboxes: Annotated[
        np.ndarray[Any, np.dtype[np.float32]], bbox_field(dtype=pl.Float32, format="x1y1x2y2")
    ]
    bbox_labels: Annotated[np.ndarray[Any, np.dtype[np.int32]], tensor_field(dtype=pl.Int32)]


def test_convert_to_legacy_simple():
    """Test basic convert_to_legacy functionality."""
    # Create experimental dataset with sample data
    experimental_dataset = Dataset(DetectionSample)

    # Add sample data
    sample1 = DetectionSample(
        image_path="/path/to/image1.jpg",
        bboxes=np.array([[10, 20, 50, 60], [100, 120, 150, 160]], dtype=np.float32),
        bbox_labels=np.array([0, 1], dtype=np.int32),
    )

    sample2 = DetectionSample(
        image_path="/path/to/image2.jpg",
        bboxes=np.array([[5, 15, 45, 55]], dtype=np.float32),
        bbox_labels=np.array([1], dtype=np.int32),
    )

    experimental_dataset.append(sample1)
    experimental_dataset.append(sample2)

    # Convert to legacy format
    legacy_dataset = convert_to_legacy(experimental_dataset)  # type: ignore

    # Verify conversion
    assert len(legacy_dataset) == 2

    # Check first item
    items = list(legacy_dataset)
    first_item = items[0]

    # Verify media
    assert isinstance(first_item.media, Image)  # type: ignore[reportUnknownMemberType]
    assert getattr(first_item.media, "path", None) == "/path/to/image1.jpg"

    # Verify annotations
    assert len(first_item.annotations) == 2

    # Check first bbox: [10, 20, 50, 60] -> Bbox(x=10, y=20, w=40, h=40)
    bbox1 = first_item.annotations[0]
    assert isinstance(bbox1, Bbox)
    assert bbox1.x == 10
    assert bbox1.y == 20
    assert bbox1.w == 40  # 50 - 10
    assert bbox1.h == 40  # 60 - 20
    assert bbox1.label == 0

    # Check second bbox: [100, 120, 150, 160] -> Bbox(x=100, y=120, w=50, h=40)
    bbox2 = first_item.annotations[1]
    assert isinstance(bbox2, Bbox)
    assert bbox2.x == 100
    assert bbox2.y == 120
    assert bbox2.w == 50  # 150 - 100
    assert bbox2.h == 40  # 160 - 120
    assert bbox2.label == 1

    # Check second item
    second_item = items[1]
    second_item_media = cast(Any, second_item.media)  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(second_item_media, ImageFromFile)
    assert second_item_media.path == "/path/to/image2.jpg"
    assert len(second_item.annotations) == 1

    # Check the bbox in second item: [5, 15, 45, 55] -> Bbox(x=5, y=15, w=40, h=40)
    bbox3 = second_item.annotations[0]
    assert isinstance(bbox3, Bbox)
    assert bbox3.x == 5
    assert bbox3.y == 15
    assert bbox3.w == 40  # 45 - 5
    assert bbox3.h == 40  # 55 - 15
    assert bbox3.label == 1


def test_convert_to_legacy_empty_dataset():
    """Test convert_to_legacy with empty experimental dataset."""
    experimental_dataset = Dataset(DetectionSample)

    legacy_dataset = convert_to_legacy(experimental_dataset)  # type: ignore

    assert len(legacy_dataset) == 0


def test_backward_image_media_converter_create_from_schema():
    """Test BackwardImageMediaConverter.create_from_schema method."""
    # Create experimental dataset to get schema
    experimental_dataset = Dataset(DetectionSample)
    schema = experimental_dataset.schema

    # Test that converter can be created from schema with image_path field
    converter = BackwardImageMediaConverter.create_from_schema(schema)
    assert converter is not None
    assert isinstance(converter, BackwardImageMediaConverter)
    assert converter.image_path_attr == "image_path"

    # Test get_media_type
    assert converter.get_media_type() == Image


def test_backward_image_media_converter_create_from_schema_no_image_field():
    """Test BackwardImageMediaConverter with schema that has no image field."""
    from datumaro.experimental.fields import tensor_field
    from datumaro.experimental.schema import AttributeInfo, Schema

    # Create schema without image_path field
    schema = Schema(
        attributes={
            "some_tensor": AttributeInfo(type=np.ndarray, annotation=tensor_field(dtype=pl.Float32))
        }
    )

    converter = BackwardImageMediaConverter.create_from_schema(schema)
    assert converter is None


def test_backward_image_media_converter_convert_to_legacy_media():
    """Test media conversion from experimental to legacy."""
    experimental_dataset = Dataset(DetectionSample)
    schema = experimental_dataset.schema

    converter = BackwardImageMediaConverter.create_from_schema(schema)
    assert converter is not None

    # Create sample with image path
    sample = DetectionSample(
        image_path="/test/image.jpg",
        bboxes=np.array([[1, 2, 3, 4]], dtype=np.float32),
        bbox_labels=np.array([0], dtype=np.int32),
    )

    # Convert to legacy media
    legacy_media = converter.convert_to_legacy_media(sample)

    assert isinstance(legacy_media, Image)
    assert getattr(legacy_media, "path", None) == "/test/image.jpg"


def test_backward_bbox_annotation_converter_create_from_schema():
    """Test BackwardBboxAnnotationConverter.create_from_schema method."""
    # Create experimental dataset to get schema
    experimental_dataset = Dataset(DetectionSample)
    schema = experimental_dataset.schema

    # Test that converter can be created from schema with bbox fields
    converter = BackwardBboxAnnotationConverter.create_from_schema(schema)
    assert converter is not None
    assert isinstance(converter, BackwardBboxAnnotationConverter)
    assert converter.bboxes_attr == "bboxes"
    assert converter.bbox_labels_attr == "bbox_labels"

    # Test get_annotation_type
    assert converter.get_annotation_type() == AnnotationType.bbox


def test_backward_bbox_annotation_converter_create_from_schema_missing_fields():
    """Test BackwardBboxAnnotationConverter with incomplete schema."""
    from datumaro.experimental.fields import image_path_field
    from datumaro.experimental.schema import AttributeInfo, Schema

    # Create schema without bbox fields
    schema = Schema(
        attributes={"image_path": AttributeInfo(type=str, annotation=image_path_field())}
    )

    converter = BackwardBboxAnnotationConverter.create_from_schema(schema)
    assert converter is None


def test_backward_bbox_annotation_converter_convert_to_legacy_annotations():
    """Test annotation conversion from experimental to legacy."""
    experimental_dataset = Dataset(DetectionSample)
    schema = experimental_dataset.schema

    converter = BackwardBboxAnnotationConverter.create_from_schema(schema)
    assert converter is not None

    # Create sample with bboxes
    sample = DetectionSample(
        image_path="/test/image.jpg",
        bboxes=np.array([[10, 20, 50, 60], [100, 150, 200, 250]], dtype=np.float32),
        bbox_labels=np.array([1, 2], dtype=np.int32),
    )

    # Convert to legacy annotations
    categories: CategoriesInfo = {}  # Empty categories for this test
    legacy_annotations = converter.convert_to_legacy_annotations(sample, categories)

    assert len(legacy_annotations) == 2

    # Check first bbox: [10, 20, 50, 60] -> Bbox(x=10, y=20, w=40, h=40)
    bbox1 = legacy_annotations[0]
    assert isinstance(bbox1, Bbox)
    assert bbox1.x == 10
    assert bbox1.y == 20
    assert bbox1.w == 40  # 50 - 10
    assert bbox1.h == 40  # 60 - 20
    assert bbox1.label == 1

    # Check second bbox: [100, 150, 200, 250] -> Bbox(x=100, y=150, w=100, h=100)
    bbox2 = legacy_annotations[1]
    assert isinstance(bbox2, Bbox)
    assert bbox2.x == 100
    assert bbox2.y == 150
    assert bbox2.w == 100  # 200 - 100
    assert bbox2.h == 100  # 250 - 150
    assert bbox2.label == 2


def test_backward_bbox_annotation_converter_convert_empty_annotations():
    """Test bbox converter with empty arrays."""
    experimental_dataset = Dataset(DetectionSample)
    schema = experimental_dataset.schema

    converter = BackwardBboxAnnotationConverter.create_from_schema(schema)
    assert converter is not None

    # Create sample with empty bboxes
    sample = DetectionSample(
        image_path="/test/image.jpg",
        bboxes=np.array([], dtype=np.float32).reshape(0, 4),
        bbox_labels=np.array([], dtype=np.int32),
    )

    # Convert to legacy annotations
    categories: CategoriesInfo = {}
    legacy_annotations = converter.convert_to_legacy_annotations(sample, categories)

    assert len(legacy_annotations) == 0


def test_backward_bbox_annotation_converter_infer_categories():
    """Test category inference from experimental dataset."""
    experimental_dataset = Dataset(DetectionSample)

    # Add samples with different labels
    sample1 = DetectionSample(
        image_path="/test/image1.jpg",
        bboxes=np.array([[10, 20, 50, 60]], dtype=np.float32),
        bbox_labels=np.array([0], dtype=np.int32),
    )

    sample2 = DetectionSample(
        image_path="/test/image2.jpg",
        bboxes=np.array([[100, 150, 200, 250], [50, 75, 100, 125]], dtype=np.float32),
        bbox_labels=np.array([2, 1], dtype=np.int32),
    )

    experimental_dataset.append(sample1)
    experimental_dataset.append(sample2)

    schema = experimental_dataset.schema
    converter = BackwardBboxAnnotationConverter.create_from_schema(schema)
    assert converter is not None

    # Infer categories
    categories = converter.infer_categories(experimental_dataset)  # type: ignore

    # Should have label categories
    assert AnnotationType.label in categories
    # Basic check that categories were created - detailed verification is complex due to legacy types
    assert categories[AnnotationType.label] is not None


def test_analyze_experimental_dataset():
    """Test analysis of experimental dataset for backward conversion."""
    experimental_dataset = Dataset(DetectionSample)

    # Add sample data
    sample = DetectionSample(
        image_path="/test/image.jpg",
        bboxes=np.array([[10, 20, 50, 60]], dtype=np.float32),
        bbox_labels=np.array([1], dtype=np.int32),
    )
    experimental_dataset.append(sample)

    # Analyze dataset
    analysis_result = analyze_experimental_dataset(experimental_dataset)  # type: ignore

    # Check media type and converter
    assert analysis_result.media_type == Image
    assert analysis_result.media_converter is not None
    assert isinstance(analysis_result.media_converter, BackwardImageMediaConverter)

    # Check annotation types and converters
    assert AnnotationType.bbox in analysis_result.ann_types
    assert AnnotationType.bbox in analysis_result.ann_converters
    assert isinstance(
        analysis_result.ann_converters[AnnotationType.bbox], BackwardBboxAnnotationConverter
    )

    # Check categories
    assert AnnotationType.label in analysis_result.categories


def test_analyze_experimental_dataset_no_compatible_converters():
    """Test analysis with dataset that has no compatible backward converters."""
    from datumaro.experimental.fields import tensor_field

    # Create a custom sample type without image_path or bbox fields
    class CustomSample(Sample):
        some_data: Annotated[np.ndarray[Any, np.dtype[np.float32]], tensor_field(dtype=pl.Float32)]

    experimental_dataset = Dataset(CustomSample)

    # Add sample
    sample = CustomSample(some_data=np.array([1, 2, 3], dtype=np.float32))
    experimental_dataset.append(sample)

    # Analyze dataset
    analysis_result = analyze_experimental_dataset(experimental_dataset)  # type: ignore

    # Should have no compatible converters
    assert analysis_result.media_type is None
    assert analysis_result.media_converter is None
    assert len(analysis_result.ann_types) == 0
    assert len(analysis_result.ann_converters) == 0
    assert len(analysis_result.categories) == 0


def test_convert_to_legacy_with_only_images():
    """Test convert_to_legacy with dataset containing only images."""
    from datumaro.experimental.fields import image_path_field

    # Define schema with only image_path
    class ImageOnlySample(Sample):
        image_path: Annotated[str, image_path_field()]

    experimental_dataset = Dataset(ImageOnlySample)

    # Add samples
    sample1 = ImageOnlySample(image_path="/path/to/image1.jpg")
    sample2 = ImageOnlySample(image_path="/path/to/image2.jpg")

    experimental_dataset.append(sample1)
    experimental_dataset.append(sample2)

    # Convert to legacy
    legacy_dataset = convert_to_legacy(experimental_dataset)  # type: ignore

    assert len(legacy_dataset) == 2

    items = list(legacy_dataset)

    # Check first item
    first_item = items[0]
    first_item_media = cast(Any, first_item.media)  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(first_item_media, ImageFromFile)
    assert first_item_media.path == "/path/to/image1.jpg"
    assert len(first_item.annotations) == 0  # No annotations

    # Check second item
    second_item = items[1]
    second_item_media = cast(Any, second_item.media)  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(second_item_media, ImageFromFile)
    assert second_item_media.path == "/path/to/image2.jpg"
    assert len(second_item.annotations) == 0
