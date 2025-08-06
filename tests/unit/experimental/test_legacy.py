"""Unit tests for legacy dataset conversion functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from datumaro.components.annotation import AnnotationType, Bbox
from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import CategoriesInfo, DatasetItem
from datumaro.components.media import Image
from datumaro.experimental.legacy import (
    BboxAnnotationConverter,
    ImageMediaConverter,
    analyze_legacy_dataset,
    convert_from_legacy,
    get_annotation_converter,
    get_media_converter,
    register_annotation_converter,
    register_media_converter,
)


def test_image_media_converter_get_schema_attributes():
    """Test schema attribute generation for images."""
    converter = ImageMediaConverter()
    attributes = converter.get_schema_attributes()

    assert "image_path" in attributes
    assert attributes["image_path"].type == str


def test_image_media_converter_convert_item_media_with_path_and_size():
    """Test media conversion with path and size."""
    converter = ImageMediaConverter()

    image_media = Image.from_file(
        "/path/to/image.jpg", size=(480, 640)
    )  # pyright: ignore[reportUnknownMemberType]

    item = DatasetItem(id="test", media=image_media)
    result = converter.convert_item_media(item)

    assert result["image_path"] == "/path/to/image.jpg"


def test_image_media_converter_convert_item_media_with_path_only():
    """Test media conversion with only path."""
    converter = ImageMediaConverter()

    image_media = Image.from_file("/path/to/image.jpg")  # pyright: ignore[reportUnknownMemberType]

    item = DatasetItem(id="test", media=image_media)
    result = converter.convert_item_media(item)

    assert result["image_path"] == "/path/to/image.jpg"


def test_image_media_converter_convert_item_media_no_media():
    """Test media conversion with no media."""
    converter = ImageMediaConverter()
    item = DatasetItem(id="test", media=None)
    result = converter.convert_item_media(item)

    assert result == {}


def test_bbox_annotation_converter_get_schema_attributes():
    """Test schema attribute generation for bboxes."""
    converter = BboxAnnotationConverter()
    categories: CategoriesInfo = {}  # Empty categories

    attributes = converter.get_schema_attributes(categories)

    assert "bboxes" in attributes
    assert "bbox_labels" in attributes
    assert attributes["bboxes"].type == np.ndarray
    assert attributes["bbox_labels"].type == np.ndarray


def test_bbox_annotation_converter_convert_annotations_single_bbox():
    """Test conversion of single bbox annotation."""
    converter = BboxAnnotationConverter()

    bbox = Bbox(10, 20, 30, 40, label=1)  # x=10, y=20, w=30, h=40
    item = DatasetItem(id="test")

    result = converter.convert_annotations([bbox], item)

    expected_bbox = np.array([[10, 20, 40, 60]], dtype=np.float32)  # x1,y1,x2,y2 format
    expected_labels = np.array([1], dtype=np.int32)

    np.testing.assert_array_equal(result["bboxes"], expected_bbox)
    np.testing.assert_array_equal(result["bbox_labels"], expected_labels)


def test_bbox_annotation_converter_convert_annotations_multiple_bboxes():
    """Test conversion of multiple bbox annotations."""
    converter = BboxAnnotationConverter()

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
    converter = BboxAnnotationConverter()
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
    converter = ImageMediaConverter()
    register_media_converter(Image, converter)

    retrieved = get_media_converter(Image)
    assert retrieved is converter


def test_register_and_get_annotation_converter():
    """Test annotation converter registration and retrieval."""
    converter = BboxAnnotationConverter()
    register_annotation_converter(AnnotationType.bbox, converter)

    retrieved = get_annotation_converter(AnnotationType.bbox)
    assert retrieved is converter


def test_get_nonexistent_media_converter():
    """Test getting converter for unregistered media type."""
    from datumaro.components.media import Video

    with pytest.raises(ValueError, match="No converter registered for media type"):
        get_media_converter(Video)


def test_get_nonexistent_annotation_converter():
    """Test getting converter for unregistered annotation type."""
    with pytest.raises(ValueError, match="No converter registered for annotation type"):
        get_annotation_converter(AnnotationType.polygon)


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
        assert getattr(sample, "image_path") == image_path
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
        assert getattr(sample, "image_path") == image_path
        assert not hasattr(sample, "bboxes")
        assert not hasattr(sample, "bbox_labels")


def test_builtin_converters_registration():
    """Test that built-in converters are registered on import."""
    image_converter = get_media_converter(Image)
    assert isinstance(image_converter, ImageMediaConverter)

    bbox_converter = get_annotation_converter(AnnotationType.bbox)
    assert isinstance(bbox_converter, BboxAnnotationConverter)
