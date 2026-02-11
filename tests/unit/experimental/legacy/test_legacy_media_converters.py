"""Unit tests for legacy dataset conversion functionality."""

from typing import Annotated, Any

import numpy as np
import polars as pl

from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import ImageInfo, bbox_field, image_path_field, label_field, tensor_field
from datumaro.experimental.legacy import BackwardImageMediaConverter, ForwardImageMediaConverter
from datumaro.experimental.schema import AttributeInfo, Schema
from datumaro.util.image import encode_image


# Define a sample schema for testing convert_to_legacy
class DetectionSample(Sample):
    image_path: Annotated[str, image_path_field()]
    bboxes: Annotated[np.ndarray[Any, np.dtype[np.float32]], bbox_field(dtype=pl.Float32(), format="x1y1x2y2")]
    bbox_labels: Annotated[np.ndarray[Any, np.dtype[np.uint32]], label_field(dtype=pl.UInt32(), is_list=True)]


class ForwardImageMediaConverterTest:
    """Tests for ForwardImageMediaConverter."""

    def test_image_media_converter_get_schema_attributes(self):
        """Test schema attribute generation for images."""
        # Create a simple dataset with file-based images to get the converter
        items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
        dataset = LegacyDataset.from_iterable(items)

        converter = ForwardImageMediaConverter.create(dataset)
        assert converter is not None

        attributes = converter.get_schema_attributes()

        assert "image_path" in attributes
        assert attributes["image_path"].type is str

    def test_image_media_converter_convert_item_media_with_path(self):
        """Test media conversion with path (with and without size)."""
        # Create a dataset to get the converter
        items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
        dataset = LegacyDataset.from_iterable(items)

        converter = ForwardImageMediaConverter.create(dataset)
        assert converter is not None

        # Test with size
        image_media_with_size = Image.from_file("/path/to/image.jpg", size=(480, 640))
        item_with_size = DatasetItem(id="test", media=image_media_with_size)
        result_with_size = converter.convert_item_media(item_with_size)
        assert result_with_size["image_path"] == "/path/to/image.jpg"

        # Test without size
        image_media_no_size = Image.from_file("/path/to/image.jpg")
        item_no_size = DatasetItem(id="test", media=image_media_no_size)
        result_no_size = converter.convert_item_media(item_no_size)
        assert result_no_size["image_path"] == "/path/to/image.jpg"

    def test_image_media_converter_convert_item_media_no_media(self):
        """Test media conversion with no media."""
        # Create a dataset to get the converter
        items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
        dataset = LegacyDataset.from_iterable(items)

        converter = ForwardImageMediaConverter.create(dataset)
        assert converter is not None

        item = DatasetItem(id="test", media=None)
        result = converter.convert_item_media(item)

        assert result == {}

    def test_image_bytes_media_converter_get_schema_attributes(self):
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
        assert attributes["image_bytes"].type is bytes

    def test_image_bytes_media_converter_convert_item_media(self):
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

    def test_image_bytes_media_converter_convert_item_media_with_numpy(self):
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

    def test_image_converter_image_info_field_inclusion(self):
        """Test ForwardImageMediaConverter includes/excludes ImageInfoField based on size availability."""
        # Test 1: All images have size - should include ImageInfo
        items_with_size = [
            DatasetItem(id="test1", media=Image.from_file("/path/to/image1.jpg", size=(480, 640))),
            DatasetItem(id="test2", media=Image.from_file("/path/to/image2.jpg", size=(720, 1280))),
        ]
        dataset_with_size = LegacyDataset.from_iterable(items_with_size)
        converter_with_size = ForwardImageMediaConverter.create(dataset_with_size)
        assert converter_with_size is not None
        assert converter_with_size.has_image_info is True
        attributes_with_size = converter_with_size.get_schema_attributes()
        assert "image_path" in attributes_with_size
        assert "image_info" in attributes_with_size
        assert attributes_with_size["image_info"].type == ImageInfo

        # Test 2: No images have size - should exclude ImageInfo
        items_no_size = [
            DatasetItem(id="test1", media=Image.from_file("/path/to/image1.jpg")),
            DatasetItem(id="test2", media=Image.from_file("/path/to/image2.jpg")),
        ]
        dataset_no_size = LegacyDataset.from_iterable(items_no_size)
        converter_no_size = ForwardImageMediaConverter.create(dataset_no_size)
        assert converter_no_size is not None
        assert converter_no_size.has_image_info is False
        attributes_no_size = converter_no_size.get_schema_attributes()
        assert "image_path" in attributes_no_size
        assert "image_info" not in attributes_no_size

        # Test 3: Mixed - some images with size, some without - should exclude ImageInfo
        items_mixed = [
            DatasetItem(id="test1", media=Image.from_file("/path/to/image1.jpg", size=(480, 640))),
            DatasetItem(id="test2", media=Image.from_file("/path/to/image2.jpg")),  # No size
        ]
        dataset_mixed = LegacyDataset.from_iterable(items_mixed)
        converter_mixed = ForwardImageMediaConverter.create(dataset_mixed)
        assert converter_mixed is not None
        assert converter_mixed.has_image_info is False
        attributes_mixed = converter_mixed.get_schema_attributes()
        assert "image_path" in attributes_mixed
        assert "image_info" not in attributes_mixed

    def test_image_converter_converts_size_info(self):
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

    def test_image_converter_bytes_with_size_info(self):
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


class BackwardImageMediaConverterTest:
    """Tests for BackwardImageMediaConverter."""

    def test_backward_image_media_converter_create_from_schema(self):
        """Test BackwardImageMediaConverter.create_from_schema method."""
        # Create v2 dataset to get schema
        experimental_dataset = Dataset(DetectionSample)
        schema = experimental_dataset.schema

        # Test that converter can be created from schema with image_path field
        converter = BackwardImageMediaConverter.create_from_schema(schema)
        assert converter is not None
        assert isinstance(converter, BackwardImageMediaConverter)
        assert converter.image_path_attr == "image_path"

        # Test get_media_type
        assert converter.get_media_type() == Image

    def test_backward_image_media_converter_create_from_schema_no_image_field(self):
        """Test BackwardImageMediaConverter with schema that has no image field."""

        # Create schema without image_path field
        schema = Schema(
            attributes={"some_tensor": AttributeInfo(type=np.ndarray, field=tensor_field(dtype=pl.Float32()))}
        )

        converter = BackwardImageMediaConverter.create_from_schema(schema)
        assert converter is None

    def test_backward_image_media_converter_convert_to_legacy_media(self):
        """Test media conversion from v2 to legacy."""
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
