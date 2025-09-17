"""Unit tests for legacy dataset conversion functionality."""

import io
import math
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from PIL import Image as PILImage
from typing_extensions import Annotated

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    LabelCategories,
    Polygon,
    RotatedBbox,
)
from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import CategoriesInfo, DatasetItem
from datumaro.components.media import Image, ImageFromData, ImageFromFile, MediaElement, Video
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import (
    ImageInfo,
    bbox_field,
    image_bytes_field,
    image_info_field,
    image_path_field,
    label_field,
    polygon_field,
    rotated_bbox_field,
    tensor_field,
)
from datumaro.experimental.legacy import (
    BackwardBboxAnnotationConverter,
    BackwardImageMediaConverter,
    BackwardPolygonAnnotationConverter,
    BackwardRotatedBboxAnnotationConverter,
    ForwardBboxAnnotationConverter,
    ForwardImageMediaConverter,
    ForwardPolygonAnnotationConverter,
    ForwardRotatedBboxAnnotationConverter,
    analyze_experimental_dataset,
    analyze_legacy_dataset,
    convert_from_legacy,
    convert_to_legacy,
    get_forward_annotation_converter,
    get_forward_media_converter,
    register_forward_annotation_converter,
    register_forward_media_converter,
)
from datumaro.experimental.schema import AttributeInfo, Schema
from datumaro.util.image import encode_image


def test_image_media_converter_get_schema_attributes():
    """Test schema attribute generation for images."""
    # Create a simple dataset with file-based images to get the converter
    items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None

    attributes = converter.get_schema_attributes()

    assert "image_path" in attributes
    assert attributes["image_path"].type == str


def test_image_media_converter_convert_item_media_with_path_and_size():
    """Test media conversion with path and size."""
    # Create a dataset to get the converter
    items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None

    image_media = Image.from_file(
        "/path/to/image.jpg", size=(480, 640)
    )  # pyright: ignore[reportUnknownMemberType]

    item = DatasetItem(id="test", media=image_media)
    result = converter.convert_item_media(item)

    assert result["image_path"] == "/path/to/image.jpg"


def test_image_media_converter_convert_item_media_with_path_only():
    """Test media conversion with only path."""
    # Create a dataset to get the converter
    items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None

    image_media = Image.from_file("/path/to/image.jpg")  # pyright: ignore[reportUnknownMemberType]

    item = DatasetItem(id="test", media=image_media)
    result = converter.convert_item_media(item)

    assert result["image_path"] == "/path/to/image.jpg"


def test_image_media_converter_convert_item_media_no_media():
    """Test media conversion with no media."""
    # Create a dataset to get the converter
    items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None

    item = DatasetItem(id="test", media=None)
    result = converter.convert_item_media(item)

    assert result == {}


def test_image_bytes_media_converter_get_schema_attributes():
    """Test schema attribute generation for image bytes."""
    # Create test image data
    test_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    image_bytes = encode_image(test_image, ".png")

    # Create a dataset with ImageFromBytes media
    items = [DatasetItem(id="test", media=Image.from_bytes(image_bytes))]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None

    attributes = converter.get_schema_attributes()

    assert "image_bytes" in attributes
    assert attributes["image_bytes"].type == bytes


def test_image_bytes_media_converter_convert_item_media():
    """Test media conversion with image bytes."""
    # Create test image data
    test_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    image_bytes = encode_image(test_image, ".png")

    # Create a dataset to get the converter
    items = [DatasetItem(id="test", media=Image.from_bytes(image_bytes))]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None

    image_media = Image.from_bytes(image_bytes)
    item = DatasetItem(id="test", media=image_media)
    result = converter.convert_item_media(item)

    assert "image_bytes" in result
    assert isinstance(result["image_bytes"], (bytes, np.ndarray))


def test_image_bytes_media_converter_convert_item_media_with_numpy():
    """Test media conversion with numpy-based image."""
    # Create test image data
    test_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

    # Create a dataset with ImageFromNumpy media
    items = [DatasetItem(id="test", media=Image.from_numpy(test_image))]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None

    image_media = Image.from_numpy(test_image)
    item = DatasetItem(id="test", media=image_media)
    result = converter.convert_item_media(item)

    assert "image_bytes" in result
    assert isinstance(result["image_bytes"], np.ndarray)
    np.testing.assert_array_equal(result["image_bytes"], test_image)


def test_unified_image_converter_handles_file_based_images():
    """Test that the unified ForwardImageMediaConverter handles file-based images."""
    # Create a dataset with file-based images
    items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None

    # Should generate image_path attributes for file-based images
    attributes = converter.get_schema_attributes()
    assert "image_path" in attributes


def test_unified_image_converter_handles_data_based_images():
    """Test that the unified ForwardImageMediaConverter handles data-based images."""
    # Create test image data
    test_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    image_bytes = encode_image(test_image, ".png")

    # Create a dataset with data-based images
    items = [DatasetItem(id="test", media=Image.from_bytes(image_bytes))]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None

    # Should generate image_bytes attributes for data-based images
    attributes = converter.get_schema_attributes()
    assert "image_bytes" in attributes


def test_image_converter_with_size_info_includes_image_info_field():
    """Test that ForwardImageMediaConverter includes ImageInfoField when all images have size."""
    # Create images with size information
    items = [
        DatasetItem(id="test1", media=Image.from_file("/path/to/image1.jpg", size=(480, 640))),
        DatasetItem(id="test2", media=Image.from_file("/path/to/image2.jpg", size=(720, 1280))),
    ]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None
    assert converter.has_image_info is True

    # Should include both image_path and image_info attributes
    attributes = converter.get_schema_attributes()
    assert "image_path" in attributes
    assert "image_info" in attributes
    assert attributes["image_info"].type == ImageInfo


def test_image_converter_without_size_info_excludes_image_info_field():
    """Test that ForwardImageMediaConverter excludes ImageInfoField when images don't have size."""
    # Create images without size information
    items = [
        DatasetItem(id="test1", media=Image.from_file("/path/to/image1.jpg")),
        DatasetItem(id="test2", media=Image.from_file("/path/to/image2.jpg")),
    ]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None
    assert converter.has_image_info is False

    # Should only include image_path attribute, not image_info
    attributes = converter.get_schema_attributes()
    assert "image_path" in attributes
    assert "image_info" not in attributes


def test_image_converter_mixed_size_info_excludes_image_info_field():
    """Test that ForwardImageMediaConverter excludes ImageInfoField when some images lack size."""
    # Create mixed dataset - some images with size, some without
    items = [
        DatasetItem(id="test1", media=Image.from_file("/path/to/image1.jpg", size=(480, 640))),
        DatasetItem(id="test2", media=Image.from_file("/path/to/image2.jpg")),  # No size
    ]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None
    assert converter.has_image_info is False

    # Should only include image_path attribute, not image_info
    attributes = converter.get_schema_attributes()
    assert "image_path" in attributes
    assert "image_info" not in attributes


def test_image_converter_converts_size_info():
    """Test that ForwardImageMediaConverter properly converts size information."""
    # Create images with size information
    items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg", size=(480, 640)))]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None
    assert converter.has_image_info is True

    # Test conversion
    image_media = Image.from_file("/path/to/image.jpg", size=(480, 640))
    item = DatasetItem(id="test", media=image_media)
    result = converter.convert_item_media(item)

    assert "image_path" in result
    assert "image_info" in result
    assert result["image_path"] == "/path/to/image.jpg"
    assert isinstance(result["image_info"], ImageInfo)
    assert result["image_info"].width == 640
    assert result["image_info"].height == 480


def test_image_converter_bytes_with_size_info():
    """Test ForwardImageMediaConverter with image bytes that have size information."""
    # Create test image data with size
    test_image = np.random.randint(0, 256, (64, 128, 3), dtype=np.uint8)
    image_bytes = encode_image(test_image, ".png")

    # Create images from bytes - they should have size info when loaded
    items = [DatasetItem(id="test", media=Image.from_bytes(image_bytes))]
    dataset = LegacyDataset.from_iterable(items)

    converter = ForwardImageMediaConverter.create(dataset)
    assert converter is not None

    # Test conversion - the exact behavior depends on whether from_bytes has size info
    image_media = Image.from_bytes(image_bytes)
    item = DatasetItem(id="test", media=image_media)
    result = converter.convert_item_media(item)

    assert "image_bytes" in result

    # If the image has size info, it should be included
    if converter.has_image_info and image_media.has_size:
        assert "image_info" in result
        assert isinstance(result["image_info"], ImageInfo)
        # Note: Image size is (H, W) but ImageInfo expects (width, height)
        height, width = image_media.size
        assert result["image_info"].width == width
        assert result["image_info"].height == height


def test_convert_from_legacy_with_image_bytes():
    """Test full conversion pipeline with ImageFromData images."""
    # Create test images
    test_image1 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    test_image2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    # Encode images as bytes
    image_bytes1 = encode_image(test_image1, ".png")
    image_bytes2 = encode_image(test_image2, ".jpg")

    # Create dataset items with ImageFromBytes media
    items = [
        DatasetItem(id="item1", media=Image.from_bytes(image_bytes1), annotations=[]),
        DatasetItem(id="item2", media=Image.from_bytes(image_bytes2), annotations=[]),
    ]

    # Create legacy dataset
    legacy_dataset = LegacyDataset.from_iterable(items)

    # Convert to experimental format
    experimental_dataset = convert_from_legacy(legacy_dataset)

    # Verify conversion
    assert "image_bytes" in experimental_dataset.schema.attributes
    assert "image_path" not in experimental_dataset.schema.attributes

    # Verify samples
    assert len(experimental_dataset) == 2

    sample1 = experimental_dataset[0]
    sample2 = experimental_dataset[1]

    # Check that image_bytes are present
    assert hasattr(sample1, "image_bytes")
    assert hasattr(sample2, "image_bytes")


def test_convert_from_legacy_with_image_paths():
    """Test that file-based images still use ImagePathField correctly."""
    # Create dataset items with ImageFromFile media
    items = [
        DatasetItem(id="item1", media=Image.from_file("test1.jpg"), annotations=[]),
        DatasetItem(id="item2", media=Image.from_file("test2.png"), annotations=[]),
    ]

    # Create legacy dataset
    legacy_dataset = LegacyDataset.from_iterable(items)

    # Convert to experimental format
    experimental_dataset = convert_from_legacy(legacy_dataset)

    # Verify conversion
    assert "image_path" in experimental_dataset.schema.attributes
    assert "image_bytes" not in experimental_dataset.schema.attributes

    # Verify samples
    assert len(experimental_dataset) == 2

    sample1 = experimental_dataset[0]
    sample2 = experimental_dataset[1]

    # Check that image_path are present
    assert hasattr(sample1, "image_path")
    assert hasattr(sample2, "image_path")


def test_convert_from_legacy_with_callable_image_data():
    """Test conversion with ImageFromData containing callable _data."""

    # Create test image bytes
    test_image1 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    test_image2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    # Create bytes using PIL
    def create_image_bytes(img_array):
        pil_image = PILImage.fromarray(img_array)
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    image_bytes1 = create_image_bytes(test_image1)
    image_bytes2 = create_image_bytes(test_image2)

    # Create callables that return these bytes
    def get_image_bytes1():
        return image_bytes1

    def get_image_bytes2():
        return image_bytes2

    # Create dataset items with ImageFromData containing callable _data
    items = [
        DatasetItem(id="item1", media=ImageFromData(data=get_image_bytes1), annotations=[]),
        DatasetItem(id="item2", media=ImageFromData(data=get_image_bytes2), annotations=[]),
    ]

    # Create legacy dataset
    legacy_dataset = LegacyDataset.from_iterable(items)

    # Convert to experimental format
    experimental_dataset = convert_from_legacy(legacy_dataset)

    # Verify conversion uses callable field instead of bytes field
    assert "image_callable" in experimental_dataset.schema.attributes
    assert "image_bytes" not in experimental_dataset.schema.attributes
    assert "image_path" not in experimental_dataset.schema.attributes

    # Verify samples
    assert len(experimental_dataset) == 2

    sample1 = experimental_dataset[0]
    sample2 = experimental_dataset[1]

    # Check that image_callable are present and work
    assert hasattr(sample1, "image_callable")
    assert hasattr(sample2, "image_callable")
    assert callable(sample1.image_callable)
    assert callable(sample2.image_callable)

    # Test that callables return proper image arrays
    result1 = sample1.image_callable()
    result2 = sample2.image_callable()

    assert isinstance(result1, np.ndarray)
    assert isinstance(result2, np.ndarray)
    assert result1.dtype == np.uint8
    assert result2.dtype == np.uint8
    assert result1.shape == (32, 32, 3)
    assert result2.shape == (64, 64, 3)


def test_image_converter_with_mixed_callable_and_bytes_data():
    """Test that mixed callable and non-callable data uses callable field."""

    # Create test image bytes
    test_image = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    pil_image = PILImage.fromarray(test_image)
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")
    image_bytes = img_bytes.getvalue()

    def get_image_bytes():
        return image_bytes

    # Create dataset with mixed callable and non-callable data
    items = [
        DatasetItem(
            id="item1", media=ImageFromData(data=get_image_bytes), annotations=[]
        ),  # callable
        DatasetItem(
            id="item2", media=ImageFromData(data=image_bytes), annotations=[]
        ),  # non-callable
    ]

    legacy_dataset = LegacyDataset.from_iterable(items)
    experimental_dataset = convert_from_legacy(legacy_dataset)

    # Should use callable field (callable takes precedence)
    assert "image_callable" in experimental_dataset.schema.attributes
    assert "image_bytes" not in experimental_dataset.schema.attributes

    # Both samples should have working callables
    sample1 = experimental_dataset[0]
    sample2 = experimental_dataset[1]

    assert hasattr(sample1, "image_callable")
    assert hasattr(sample2, "image_callable")

    result1 = sample1.image_callable()
    result2 = sample2.image_callable()

    assert isinstance(result1, np.ndarray)
    assert isinstance(result2, np.ndarray)
    assert result1.shape == (16, 16, 3)
    assert result2.shape == (16, 16, 3)


def test_bbox_annotation_converter_get_schema_attributes():
    """Test schema attribute generation for bboxes."""
    categories: CategoriesInfo = {}  # Empty categories
    converter = ForwardBboxAnnotationConverter.create_from_categories(categories)
    assert converter is not None

    attributes = converter.get_schema_attributes()

    assert "bboxes" in attributes
    # bbox_labels should not be present when there are no label categories
    assert "bbox_labels" not in attributes
    assert attributes["bboxes"].type == np.ndarray


def test_bbox_annotation_converter_convert_annotations_single_bbox():
    """Test conversion of single bbox annotation."""

    # Create categories with labels to test with labels
    label_categories = LabelCategories()
    label_categories.add("class_1")
    categories: CategoriesInfo = {AnnotationType.label: label_categories}

    converter = ForwardBboxAnnotationConverter.create_from_categories(categories)
    assert converter is not None

    bbox = Bbox(10, 20, 30, 40, label=1)  # x=10, y=20, w=30, h=40
    item = DatasetItem(id="test")

    result = converter.convert_annotations([bbox], item)

    expected_bbox = np.array([[10, 20, 40, 60]], dtype=np.float32)  # x1,y1,x2,y2 format
    expected_labels = np.array([1], dtype=np.int32)

    np.testing.assert_array_equal(result["bboxes"], expected_bbox)
    np.testing.assert_array_equal(result["bbox_labels"], expected_labels)


def test_bbox_annotation_converter_convert_annotations_multiple_bboxes():
    """Test conversion of multiple bbox annotations."""

    # Create categories with labels
    label_categories = LabelCategories()
    label_categories.add("class_1")
    label_categories.add("class_2")
    categories: CategoriesInfo = {AnnotationType.label: label_categories}

    converter = ForwardBboxAnnotationConverter.create_from_categories(categories)
    assert converter is not None

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
    categories: CategoriesInfo = {}  # Empty categories
    converter = ForwardBboxAnnotationConverter.create_from_categories(categories)
    assert converter is not None
    item = DatasetItem(id="test")

    result = converter.convert_annotations([], item)

    # Empty arrays with proper shapes
    assert result["bboxes"].shape == (0, 4)
    assert result["bboxes"].dtype == np.float32
    # No bbox_labels should be present when there are no categories
    assert "bbox_labels" not in result


# Converter registry tests


def test_register_and_get_media_converter():
    """Test media converter registration and retrieval."""
    register_forward_media_converter(ForwardImageMediaConverter)

    # Create a dataset with file-based images
    items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
    dataset = LegacyDataset.from_iterable(items)

    retrieved = get_forward_media_converter(dataset)
    assert retrieved is not None
    assert isinstance(retrieved, ForwardImageMediaConverter)


def test_register_and_get_annotation_converter():
    """Test annotation converter registration and retrieval."""
    register_forward_annotation_converter(ForwardBboxAnnotationConverter)

    categories: CategoriesInfo = {}
    retrieved = get_forward_annotation_converter(AnnotationType.bbox, categories)
    assert retrieved is not None
    assert isinstance(retrieved, ForwardBboxAnnotationConverter)


def test_get_nonexistent_media_converter():
    """Test getting converter for unsupported dataset."""
    # Create a dataset with Video media (not supported by current converters)
    items = [DatasetItem(id="test", media=Video("/path/to/video.mp4"))]
    dataset = LegacyDataset.from_iterable(items, media_type=Video)

    retrieved = get_forward_media_converter(dataset)
    assert retrieved is None


def test_get_nonexistent_annotation_converter():
    """Test getting converter for unregistered annotation type."""
    categories: CategoriesInfo = {}
    converter = get_forward_annotation_converter(AnnotationType.unknown, categories)
    assert converter is None


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

        # Create dataset with label categories
        label_categories = LabelCategories()
        label_categories.add("background")
        label_categories.add("class_1")

        dataset = LegacyDataset.from_iterable(
            [item],
            ann_types={AnnotationType.bbox},
            categories={AnnotationType.label: label_categories},
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

        image_path = str(temp_path / "image1.jpg")
        image_media = Image.from_file(
            image_path, size=(480, 640)
        )  # pyright: ignore[reportUnknownMemberType]

        class UnknownAnnotation(Bbox):
            _type = AnnotationType.unknown

        bbox = UnknownAnnotation(10, 20, 30, 40, label=1)
        item = DatasetItem(id="item1", media=image_media, annotations=[bbox])

        dataset = LegacyDataset.from_iterable([item])  # pyright: ignore[reportUnknownMemberType]

        analysis_result = analyze_legacy_dataset(dataset)

        # Should skip unknown annotation type
        assert "image_path" in analysis_result.schema.attributes
        assert "image_info" in analysis_result.schema.attributes
        assert len(analysis_result.schema.attributes) == 2  # Only image_path and image_info field


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

        # Create dataset with label categories so bbox_labels will be present
        label_categories = LabelCategories()
        label_categories.add("background")
        label_categories.add("class_1")
        label_categories.add("class_2")

        dataset = LegacyDataset.from_iterable(
            [item1],
            ann_types={AnnotationType.bbox},
            categories={AnnotationType.label: label_categories},
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
    # Create a dataset with file-based images
    items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
    dataset = LegacyDataset.from_iterable(items)

    image_converter = get_forward_media_converter(dataset)
    assert image_converter is not None
    assert isinstance(image_converter, ForwardImageMediaConverter)

    categories: CategoriesInfo = {}
    bbox_converter = get_forward_annotation_converter(AnnotationType.bbox, categories)
    assert bbox_converter is not None
    assert isinstance(bbox_converter, ForwardBboxAnnotationConverter)


# Define a sample schema for testing convert_to_legacy
class DetectionSample(Sample):
    image_path: Annotated[str, image_path_field()]
    bboxes: Annotated[
        np.ndarray[Any, np.dtype[np.float32]], bbox_field(dtype=pl.Float32, format="x1y1x2y2")
    ]
    bbox_labels: Annotated[
        np.ndarray[Any, np.dtype[np.int32]], label_field(dtype=pl.Int32, multi_label=True)
    ]


class RotatedDetectionSample(Sample):
    image_path: Annotated[str, image_path_field()]
    rotated_bboxes: Annotated[
        np.ndarray[Any, np.dtype[np.float32]], rotated_bbox_field(dtype=pl.Float32)
    ]
    rotated_bbox_labels: Annotated[
        np.ndarray[Any, np.dtype[np.int32]], label_field(dtype=pl.Int32, multi_label=True)
    ]


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


def test_forward_polygon_annotation_converter_get_schema_attributes():
    """Test schema attribute generation for polygons."""
    categories: CategoriesInfo = {}  # Empty categories
    converter = ForwardPolygonAnnotationConverter.create_from_categories(categories)
    assert converter is not None
    attributes = converter.get_schema_attributes()

    assert "polygons" in attributes
    # polygon_labels should not be present when there are no label categories
    assert "polygon_labels" not in attributes
    assert attributes["polygons"].type == np.ndarray


def test_forward_polygon_annotation_converter_convert_annotations():
    """Test polygon annotation conversion."""

    # Create categories with labels
    label_categories = LabelCategories()
    label_categories.add("class_1")
    label_categories.add("class_2")
    categories: CategoriesInfo = {AnnotationType.label: label_categories}

    converter = ForwardPolygonAnnotationConverter.create_from_categories(categories)
    assert converter is not None

    # Create polygon annotations with flat coordinates
    triangle = Polygon(points=[0, 0, 10, 0, 5, 10], label=1)  # Triangle as flat list
    rectangle = Polygon(points=[20, 20, 30, 20, 30, 30, 20, 30], label=2)  # Rectangle as flat list

    annotations = [triangle, rectangle]
    item = DatasetItem(id="test")

    result = converter.convert_annotations(annotations, item)

    assert "polygons" in result
    assert "polygon_labels" in result

    # Check polygon data
    polygons = result["polygons"]
    assert len(polygons) == 2

    # First polygon (triangle): [0,0,10,0,5,10] -> [0,0,10,0,5,10]
    assert np.all(polygons[0] == [[0, 0], [10, 0], [5, 10]])

    # Second polygon (rectangle): [20,20,30,20,30,30,20,30] -> [20,20,30,20,30,30,20,30]
    assert np.all(polygons[1] == [[20, 20], [30, 20], [30, 30], [20, 30]])

    # Check labels
    labels = result["polygon_labels"]
    assert len(labels) == 2
    assert labels[0] == 1
    assert labels[1] == 2


def test_backward_polygon_annotation_converter_create_from_schema():
    """Test BackwardPolygonAnnotationConverter schema detection."""

    # Create schema with polygon fields
    schema = Schema(
        attributes={
            "polygons": AttributeInfo(type=list, annotation=polygon_field(dtype=pl.Float32)),
            "polygon_labels": AttributeInfo(
                type=np.ndarray, annotation=label_field(dtype=pl.Int32, multi_label=True)
            ),
        }
    )

    converter = BackwardPolygonAnnotationConverter.create_from_schema(schema)
    assert converter is not None
    assert converter.polygons_attr == "polygons"
    assert converter.polygon_labels_attr == "polygon_labels"


def test_backward_polygon_annotation_converter_convert_to_legacy():
    """Test conversion from experimental to legacy polygon annotations."""
    converter = BackwardPolygonAnnotationConverter("polygons", "polygon_labels")

    # Create sample with polygon data
    class TestSample(Sample):
        pass

    sample = TestSample()
    # Simulate polygon data: triangle and rectangle
    sample.polygons = np.array(
        [
            np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float32),
            np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float32),
        ],
        dtype=object,
    )
    sample.polygon_labels = np.array([1, 2], dtype=np.int32)

    categories: CategoriesInfo = {}
    result = converter.convert_to_legacy_annotations(sample, categories)

    assert len(result) == 2

    # Check first polygon (triangle)
    poly1 = result[0]
    assert isinstance(poly1, Polygon)
    assert poly1.points == [0.0, 0.0, 10.0, 0.0, 5.0, 10.0]  # Flat coordinate format
    assert poly1.label == 1

    # Check second polygon (rectangle)
    poly2 = result[1]
    assert isinstance(poly2, Polygon)
    assert poly2.points == [
        20.0,
        20.0,
        30.0,
        20.0,
        30.0,
        30.0,
        20.0,
        30.0,
    ]  # Flat coordinate format
    assert poly2.label == 2


def test_polygon_conversion_with_labels():
    """Test polygon conversion between legacy and experimental formats with label categories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create an image file for the test
        image_path = str(temp_path / "image1.jpg")
        image_media = Image.from_file(
            image_path, size=(480, 640)
        )  # pyright: ignore[reportUnknownMemberType]

        # Create polygon annotations with different shapes
        triangle = Polygon(points=[10, 20, 30, 25, 20, 40], label=1)  # Triangle
        rectangle = Polygon(points=[50, 60, 80, 60, 80, 90, 50, 90], label=2)  # Rectangle

        item = DatasetItem(id="polygon_test", media=image_media, annotations=[triangle, rectangle])

        # Create label categories

        label_categories = LabelCategories()
        label_categories.add("background")
        label_categories.add("triangle_class")
        label_categories.add("rectangle_class")

        # Create legacy dataset with categories
        legacy_dataset = LegacyDataset.from_iterable(
            [item],
            ann_types={AnnotationType.polygon},
            categories={AnnotationType.label: label_categories},
        )  # pyright: ignore[reportUnknownMemberType]

        # Convert to experimental format
        experimental_dataset = convert_from_legacy(legacy_dataset)

        # Verify experimental dataset structure
        assert len(experimental_dataset) == 1
        exp_sample = experimental_dataset[0]

        # Check that polygons and polygon_labels are present
        assert hasattr(exp_sample, "polygons")
        assert hasattr(exp_sample, "polygon_labels")

        # Check polygon data
        assert len(exp_sample.polygons) == 2
        assert np.all(exp_sample.polygons[0] == [[10, 20], [30, 25], [20, 40]])  # Triangle
        assert np.all(
            exp_sample.polygons[1] == [[50, 60], [80, 60], [80, 90], [50, 90]]
        )  # Rectangle

        # Check labels
        np.testing.assert_array_equal(exp_sample.polygon_labels, [1, 2])

        # Convert back to legacy format
        restored_legacy_dataset = convert_to_legacy(experimental_dataset)

        # Verify restored dataset
        restored_items = list(restored_legacy_dataset)
        assert len(restored_items) == 1

        restored_item = restored_items[0]
        assert len(restored_item.annotations) == 2

        # Check restored polygons
        polygon_anns = [ann for ann in restored_item.annotations if isinstance(ann, Polygon)]
        assert len(polygon_anns) == 2

        # Sort by label for consistent comparison
        polygon_anns.sort(key=lambda x: x.label)

        # Verify triangle
        assert polygon_anns[0].points == [10, 20, 30, 25, 20, 40]
        assert polygon_anns[0].label == 1

        # Verify rectangle
        assert polygon_anns[1].points == [50, 60, 80, 60, 80, 90, 50, 90]
        assert polygon_anns[1].label == 2


def test_polygon_conversion_without_labels():
    """Test polygon conversion when no label categories are present."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create an image file for the test
        image_path = str(temp_path / "image1.jpg")
        image_media = Image.from_file(
            image_path, size=(480, 640)
        )  # pyright: ignore[reportUnknownMemberType]

        # Create polygon annotation without label categories
        triangle = Polygon(points=[10, 20, 30, 25, 20, 40])  # No label

        item = DatasetItem(id="polygon_no_labels", media=image_media, annotations=[triangle])

        # Create legacy dataset without label categories
        legacy_dataset = LegacyDataset.from_iterable(
            [item], ann_types={AnnotationType.polygon}
        )  # pyright: ignore[reportUnknownMemberType]

        # Convert to experimental format
        experimental_dataset = convert_from_legacy(legacy_dataset)

        # Verify experimental dataset structure
        assert len(experimental_dataset) == 1
        exp_sample = experimental_dataset[0]

        # Check that polygons is present but polygon_labels is not
        assert hasattr(exp_sample, "polygons")
        assert not hasattr(exp_sample, "polygon_labels")

        # Check polygon data
        assert len(exp_sample.polygons) == 1
        assert np.all(exp_sample.polygons[0] == [[10, 20], [30, 25], [20, 40]])

        # Convert back to legacy format
        restored_legacy_dataset = convert_to_legacy(experimental_dataset)

        # Verify restored dataset
        restored_items = list(restored_legacy_dataset)
        assert len(restored_items) == 1

        restored_item = restored_items[0]
        assert len(restored_item.annotations) == 1

        # Check restored polygon
        polygon_anns = [ann for ann in restored_item.annotations if isinstance(ann, Polygon)]
        assert len(polygon_anns) == 1

        restored_polygon = polygon_anns[0]
        assert restored_polygon.points == [10, 20, 30, 25, 20, 40]
        assert restored_polygon.label is None


def test_forward_rotated_bbox_annotation_converter_get_schema_attributes():
    """Test schema attribute generation for rotated bboxes."""
    categories: CategoriesInfo = {}  # Empty categories
    converter = ForwardRotatedBboxAnnotationConverter.create_from_categories(categories)
    assert converter is not None
    attributes = converter.get_schema_attributes()

    assert "rotated_bboxes" in attributes
    # rotated_bbox_labels should not be present when there are no label categories
    assert "rotated_bbox_labels" not in attributes
    assert attributes["rotated_bboxes"].type == np.ndarray


def test_forward_rotated_bbox_annotation_converter_convert_annotations():
    """Test rotated bbox annotation conversion."""

    # Create categories with labels
    label_categories = LabelCategories()
    label_categories.add("class_1")
    label_categories.add("class_2")
    categories: CategoriesInfo = {AnnotationType.label: label_categories}

    converter = ForwardRotatedBboxAnnotationConverter.create_from_categories(categories)
    assert converter is not None

    # Create rotated bbox annotations: RotatedBbox(cx, cy, w, h, r_degrees, ...)
    rotated_bbox1 = RotatedBbox(50, 60, 30, 20, 45.0, label=0)  # 45 degrees
    rotated_bbox2 = RotatedBbox(100, 120, 40, 25, -30.0, label=1)  # -30 degrees

    annotations = [rotated_bbox1, rotated_bbox2]
    item = DatasetItem(id="test")

    result = converter.convert_annotations(annotations, item)

    assert "rotated_bboxes" in result
    assert "rotated_bbox_labels" in result

    # Check rotated bbox data (should be converted to radians)
    rotated_bboxes = result["rotated_bboxes"]
    assert len(rotated_bboxes) == 2

    # First rotated bbox: (50, 60, 30, 20, 45°) -> (50, 60, 30, 20, π/4 radians)
    assert abs(rotated_bboxes[0, 0] - 50.0) < 1e-6
    assert abs(rotated_bboxes[0, 1] - 60.0) < 1e-6
    assert abs(rotated_bboxes[0, 2] - 30.0) < 1e-6
    assert abs(rotated_bboxes[0, 3] - 20.0) < 1e-6
    assert abs(rotated_bboxes[0, 4] - math.radians(45.0)) < 1e-6

    # Second rotated bbox: (100, 120, 40, 25, -30°) -> (100, 120, 40, 25, -π/6 radians)
    assert abs(rotated_bboxes[1, 0] - 100.0) < 1e-6
    assert abs(rotated_bboxes[1, 1] - 120.0) < 1e-6
    assert abs(rotated_bboxes[1, 2] - 40.0) < 1e-6
    assert abs(rotated_bboxes[1, 3] - 25.0) < 1e-6
    assert abs(rotated_bboxes[1, 4] - math.radians(-30.0)) < 1e-6

    # Check labels
    labels = result["rotated_bbox_labels"]
    assert len(labels) == 2
    assert labels[0] == 0
    assert labels[1] == 1


def test_backward_rotated_bbox_annotation_converter_create_from_schema():
    """Test backward rotated bbox converter creation from schema."""

    # Create schema with rotated bbox attributes
    attributes = {
        "rotated_bboxes": AttributeInfo(
            type=np.ndarray,
            annotation=rotated_bbox_field(dtype=pl.Float32),
        ),
        "rotated_bbox_labels": AttributeInfo(
            type=np.ndarray,
            annotation=label_field(is_list=True),
        ),
    }
    schema = Schema(attributes=attributes)

    converter = BackwardRotatedBboxAnnotationConverter.create_from_schema(schema)
    assert converter is not None
    assert converter.rotated_bboxes_attr == "rotated_bboxes"
    assert converter.rotated_bbox_labels_attr == "rotated_bbox_labels"


def test_backward_rotated_bbox_annotation_converter_convert_to_legacy():
    """Test backward conversion from experimental to legacy rotated bbox format."""

    # Create experimental dataset with rotated bbox data
    rotated_bbox_data = np.array(
        [
            [50.0, 60.0, 30.0, 20.0, math.radians(45.0)],  # 45 degrees in radians
            [100.0, 120.0, 40.0, 25.0, math.radians(-30.0)],  # -30 degrees in radians
        ],
        dtype=np.float32,
    )

    label_data = np.array([0, 1], dtype=np.int32)

    # Create experimental dataset and add sample with rotated bbox data
    experimental_dataset = Dataset(RotatedDetectionSample)

    sample = RotatedDetectionSample(
        image_path="/path/to/test.jpg",
        rotated_bboxes=rotated_bbox_data,
        rotated_bbox_labels=label_data,
    )
    experimental_dataset.append(sample)

    # Create categories for testing
    categories: CategoriesInfo = {
        AnnotationType.label: LabelCategories(),
    }

    # Create converter
    attributes = {
        "rotated_bboxes": AttributeInfo(
            type=np.ndarray,
            annotation=rotated_bbox_field(dtype=pl.Float32),
        ),
        "rotated_bbox_labels": AttributeInfo(
            type=np.ndarray,
            annotation=label_field(is_list=True),
        ),
    }
    schema = Schema(attributes=attributes)

    converter = BackwardRotatedBboxAnnotationConverter.create_from_schema(schema)
    assert converter is not None

    # Convert to legacy annotations
    annotations = converter.convert_to_legacy_annotations(sample, categories)

    # Check results
    assert len(annotations) == 2

    # Check first rotated bbox (should be converted back to degrees)
    rotated_bbox1 = annotations[0]
    assert isinstance(rotated_bbox1, RotatedBbox)
    assert abs(rotated_bbox1.cx - 50.0) < 1e-5
    assert abs(rotated_bbox1.cy - 60.0) < 1e-5
    assert abs(rotated_bbox1.w - 30.0) < 1e-5
    assert abs(rotated_bbox1.h - 20.0) < 1e-5
    assert abs(rotated_bbox1.r - 45.0) < 1e-5  # Back to degrees
    assert rotated_bbox1.label == 0

    # Check second rotated bbox
    rotated_bbox2 = annotations[1]
    assert isinstance(rotated_bbox2, RotatedBbox)
    assert abs(rotated_bbox2.cx - 100.0) < 1e-5
    assert abs(rotated_bbox2.cy - 120.0) < 1e-5
    assert abs(rotated_bbox2.w - 40.0) < 1e-5
    assert abs(rotated_bbox2.h - 25.0) < 1e-5
    assert abs(rotated_bbox2.r - (-30.0)) < 1e-5  # Back to degrees
    assert rotated_bbox2.label == 1


def test_rotated_bbox_conversion_with_labels():
    """Test end-to-end rotated bbox conversion with labels."""

    # Create legacy dataset with rotated bbox and label categories
    label_categories = LabelCategories()
    label_categories.add("person")
    label_categories.add("car")

    items = [
        DatasetItem(
            id="test_item",
            media=Image.from_file("test.jpg", size=(200, 150)),
            annotations=[
                RotatedBbox(75, 50, 40, 30, 30.0, label=0),  # person
                RotatedBbox(125, 100, 60, 40, -45.0, label=1),  # car
            ],
        )
    ]

    legacy_dataset = LegacyDataset.from_iterable(
        items,
        ann_types={AnnotationType.rotated_bbox},
        categories={AnnotationType.label: label_categories},
    )

    # Convert to experimental format
    experimental_dataset = convert_from_legacy(legacy_dataset)

    # Check experimental dataset
    assert len(experimental_dataset) == 1
    exp_sample = experimental_dataset[0]

    # Check rotated bbox data
    assert hasattr(exp_sample, "rotated_bboxes")
    assert hasattr(exp_sample, "rotated_bbox_labels")

    rotated_bboxes = exp_sample.rotated_bboxes
    labels = exp_sample.rotated_bbox_labels

    assert len(rotated_bboxes) == 2
    assert len(labels) == 2

    # Verify first rotated bbox (converted to radians)
    assert abs(rotated_bboxes[0, 0] - 75.0) < 1e-6  # cx
    assert abs(rotated_bboxes[0, 1] - 50.0) < 1e-6  # cy
    assert abs(rotated_bboxes[0, 2] - 40.0) < 1e-6  # w
    assert abs(rotated_bboxes[0, 3] - 30.0) < 1e-6  # h
    assert abs(rotated_bboxes[0, 4] - math.radians(30.0)) < 1e-6  # r in radians
    assert labels[0] == 0

    # Verify second rotated bbox
    assert abs(rotated_bboxes[1, 0] - 125.0) < 1e-6  # cx
    assert abs(rotated_bboxes[1, 1] - 100.0) < 1e-6  # cy
    assert abs(rotated_bboxes[1, 2] - 60.0) < 1e-6  # w
    assert abs(rotated_bboxes[1, 3] - 40.0) < 1e-6  # h
    assert abs(rotated_bboxes[1, 4] - math.radians(-45.0)) < 1e-6  # r in radians
    assert labels[1] == 1

    # Convert back to legacy format
    restored_legacy_dataset = convert_to_legacy(experimental_dataset)

    # Verify restored dataset
    restored_items = list(restored_legacy_dataset)
    assert len(restored_items) == 1

    restored_item = restored_items[0]
    rotated_bbox_anns = [ann for ann in restored_item.annotations if isinstance(ann, RotatedBbox)]
    assert len(rotated_bbox_anns) == 2

    # Check restored first rotated bbox (should be back to degrees)
    restored_bbox1 = rotated_bbox_anns[0]
    assert abs(restored_bbox1.cx - 75.0) < 1e-5
    assert abs(restored_bbox1.cy - 50.0) < 1e-5
    assert abs(restored_bbox1.w - 40.0) < 1e-5
    assert abs(restored_bbox1.h - 30.0) < 1e-5
    assert abs(restored_bbox1.r - 30.0) < 1e-5  # Back to degrees
    assert restored_bbox1.label == 0

    # Check restored second rotated bbox
    restored_bbox2 = rotated_bbox_anns[1]
    assert abs(restored_bbox2.cx - 125.0) < 1e-5
    assert abs(restored_bbox2.cy - 100.0) < 1e-5
    assert abs(restored_bbox2.w - 60.0) < 1e-5
    assert abs(restored_bbox2.h - 40.0) < 1e-5
    assert abs(restored_bbox2.r - (-45.0)) < 1e-5  # Back to degrees
    assert restored_bbox2.label == 1


def test_rotated_bbox_conversion_without_labels():
    """Test rotated bbox conversion without label categories."""

    # Create legacy dataset with rotated bbox but no label categories
    items = [
        DatasetItem(
            id="test_item",
            media=Image.from_file("test.jpg", size=(100, 100)),
            annotations=[
                RotatedBbox(50, 50, 20, 30, 0.0),  # No label, no rotation
                RotatedBbox(25, 75, 10, 15, 90.0),  # No label, 90 degrees
            ],
        )
    ]

    legacy_dataset = LegacyDataset.from_iterable(items, ann_types={AnnotationType.rotated_bbox})

    # Convert to experimental format
    experimental_dataset = convert_from_legacy(legacy_dataset)

    # Check experimental dataset
    assert len(experimental_dataset) == 1
    exp_sample = experimental_dataset[0]

    # Should have rotated_bboxes but not rotated_bbox_labels
    assert hasattr(exp_sample, "rotated_bboxes")
    assert not hasattr(exp_sample, "rotated_bbox_labels")

    rotated_bboxes = exp_sample.rotated_bboxes
    assert len(rotated_bboxes) == 2

    # Verify rotated bbox data
    assert abs(rotated_bboxes[0, 0] - 50.0) < 1e-6  # cx
    assert abs(rotated_bboxes[0, 1] - 50.0) < 1e-6  # cy
    assert abs(rotated_bboxes[1, 4] - math.radians(90.0)) < 1e-6  # 90 degrees in radians

    # Convert back to legacy format
    restored_legacy_dataset = convert_to_legacy(experimental_dataset)

    # Verify restored dataset
    restored_items = list(restored_legacy_dataset)
    assert len(restored_items) == 1

    restored_item = restored_items[0]
    rotated_bbox_anns = [ann for ann in restored_item.annotations if isinstance(ann, RotatedBbox)]
    assert len(rotated_bbox_anns) == 2

    # Check that labels are None (since we didn't have label categories)
    for bbox in rotated_bbox_anns:
        assert bbox.label is None
