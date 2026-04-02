"""Unit tests for legacy dataset conversion functionality."""

from typing import Annotated, Any

import numpy as np
import polars as pl

from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, MediaElement, Video, VideoFrame
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import (
    ImageInfo,
    MediaPathField,
    VideoFramePathField,
    bbox_field,
    image_path_field,
    label_field,
    media_path_field,
    tensor_field,
    video_frame_path_field,
)
from datumaro.experimental.legacy import (
    BackwardImageMediaConverter,
    BackwardVideoMediaConverter,
    ForwardImageMediaConverter,
    ForwardVideoMediaConverter,
)
from datumaro.experimental.legacy.media_converters import BackwardMixedMediaConverter, ForwardMixedMediaConverter
from datumaro.experimental.media import LazyImage, LazyVideoFrame
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


# Define a sample schema for testing backward video conversion
class VideoFrameSample(Sample):
    video_frame: Annotated[LazyVideoFrame, video_frame_path_field()]


class ForwardVideoMediaConverterTest:
    """Tests for ForwardVideoMediaConverter."""

    def test_video_media_converter_get_supported_media_types(self):
        """Test that ForwardVideoMediaConverter supports VideoFrame."""
        supported = ForwardVideoMediaConverter.get_supported_media_types()
        assert VideoFrame in supported

    def test_video_media_converter_create_with_video_frames(self):
        """Test creating converter from a dataset with VideoFrame media."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
            DatasetItem(id="frame_1", media=VideoFrame(video=video, index=1)),
        ]
        dataset = LegacyDataset.from_iterable(items, media_type=VideoFrame)

        converter = ForwardVideoMediaConverter.create(dataset)
        assert converter is not None
        assert isinstance(converter, ForwardVideoMediaConverter)

    def test_video_media_converter_create_with_no_video_frames(self):
        """Test that create returns None when no VideoFrame media exists."""
        items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
        dataset = LegacyDataset.from_iterable(items)

        converter = ForwardVideoMediaConverter.create(dataset)
        assert converter is None

    def test_video_media_converter_get_schema_attributes(self):
        """Test schema attribute generation for video frames."""
        video = Video(path="/path/to/video.mp4")
        items = [DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0))]
        dataset = LegacyDataset.from_iterable(items, media_type=VideoFrame)

        converter = ForwardVideoMediaConverter.create(dataset)
        assert converter is not None

        attributes = converter.get_schema_attributes()
        assert "video_frame" in attributes
        assert attributes["video_frame"].type is LazyVideoFrame
        assert isinstance(attributes["video_frame"].field, VideoFramePathField)

    def test_video_media_converter_get_schema_attributes_with_prefix(self):
        """Test schema attribute generation with name prefix."""
        video = Video(path="/path/to/video.mp4")
        items = [DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0))]
        dataset = LegacyDataset.from_iterable(items, media_type=VideoFrame)

        converter = ForwardVideoMediaConverter.create(dataset, name_prefix="anomaly_")
        assert converter is not None

        attributes = converter.get_schema_attributes()
        assert "anomaly_video_frame" in attributes

    def test_video_media_converter_convert_item_media(self):
        """Test converting VideoFrame media from a DatasetItem."""
        video = Video(path="/path/to/video.mp4")
        item = DatasetItem(id="frame_42", media=VideoFrame(video=video, index=42))

        converter = ForwardVideoMediaConverter(semantic="default")
        result = converter.convert_item_media(item)

        assert "video_frame" in result
        lazy_frame = result["video_frame"]
        assert isinstance(lazy_frame, LazyVideoFrame)
        assert str(lazy_frame.video_path) == "/path/to/video.mp4"
        assert lazy_frame.frame_index == 42

    def test_video_media_converter_convert_item_media_no_media(self):
        """Test converting an item with no media returns empty dict."""
        item = DatasetItem(id="test", media=None)

        converter = ForwardVideoMediaConverter(semantic="default")
        result = converter.convert_item_media(item)

        assert result == {}

    def test_video_media_converter_multiple_frames_same_video(self):
        """Test converting multiple frames from the same video."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
            DatasetItem(id="frame_10", media=VideoFrame(video=video, index=10)),
            DatasetItem(id="frame_20", media=VideoFrame(video=video, index=20)),
        ]

        converter = ForwardVideoMediaConverter(semantic="default")

        for item, expected_index in zip(items, [0, 10, 20]):
            result = converter.convert_item_media(item)
            assert result["video_frame"].frame_index == expected_index
            assert str(result["video_frame"].video_path) == "/path/to/video.mp4"


class BackwardVideoMediaConverterTest:
    """Tests for BackwardVideoMediaConverter."""

    def test_backward_video_media_converter_create_from_schema(self):
        """Test BackwardVideoMediaConverter.create_from_schema with video frame field."""
        experimental_dataset = Dataset(VideoFrameSample)
        schema = experimental_dataset.schema

        converter = BackwardVideoMediaConverter.create_from_schema(schema)
        assert converter is not None
        assert isinstance(converter, BackwardVideoMediaConverter)
        assert converter.video_frame_attr == "video_frame"

    def test_backward_video_media_converter_create_from_schema_no_video_field(self):
        """Test BackwardVideoMediaConverter with schema that has no video frame field."""
        schema = Schema(
            attributes={"some_tensor": AttributeInfo(type=np.ndarray, field=tensor_field(dtype=pl.Float32()))}
        )

        converter = BackwardVideoMediaConverter.create_from_schema(schema)
        assert converter is None

    def test_backward_video_media_converter_get_media_type(self):
        """Test that backward converter returns VideoFrame as media type."""
        experimental_dataset = Dataset(VideoFrameSample)
        schema = experimental_dataset.schema

        converter = BackwardVideoMediaConverter.create_from_schema(schema)
        assert converter is not None
        assert converter.get_media_type() == VideoFrame

    def test_backward_video_media_converter_convert_to_legacy_media(self):
        """Test media conversion from new format to legacy VideoFrame."""
        experimental_dataset = Dataset(VideoFrameSample)
        schema = experimental_dataset.schema

        converter = BackwardVideoMediaConverter.create_from_schema(schema)
        assert converter is not None

        sample = VideoFrameSample(
            video_frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=42),
        )

        legacy_media = converter.convert_to_legacy_media(sample)
        assert isinstance(legacy_media, VideoFrame)
        assert legacy_media.video.path == "/path/to/video.mp4"
        assert legacy_media.index == 42

    def test_backward_video_media_converter_caches_video_objects(self):
        """Test that backward converter caches Video objects for same video path."""
        experimental_dataset = Dataset(VideoFrameSample)
        schema = experimental_dataset.schema

        converter = BackwardVideoMediaConverter.create_from_schema(schema)
        assert converter is not None

        sample1 = VideoFrameSample(
            video_frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=0),
        )
        sample2 = VideoFrameSample(
            video_frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=10),
        )

        media1 = converter.convert_to_legacy_media(sample1)
        media2 = converter.convert_to_legacy_media(sample2)

        assert isinstance(media1, VideoFrame)
        assert isinstance(media2, VideoFrame)
        # Both frames should share the same Video object
        assert media1.video is media2.video
        assert media1.index == 0
        assert media2.index == 10


# Define a sample schema for testing backward mixed media conversion
class MixedMediaSample(Sample):
    media: Annotated[LazyImage | LazyVideoFrame, media_path_field()]


class ForwardMixedMediaConverterTest:
    """Tests for ForwardMixedMediaConverter."""

    def test_mixed_media_converter_create_with_mixed_items(self):
        """Test creating converter from a dataset with both Image and VideoFrame media."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="img_0", media=Image.from_file("/path/to/image.jpg")),
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
        ]
        dataset = LegacyDataset.from_iterable(items)

        converter = ForwardMixedMediaConverter.create(dataset)
        assert converter is not None
        assert isinstance(converter, ForwardMixedMediaConverter)

    def test_mixed_media_converter_create_with_only_images(self):
        """Test that create returns None when dataset has only images."""
        items = [
            DatasetItem(id="img_0", media=Image.from_file("/path/to/image1.jpg")),
            DatasetItem(id="img_1", media=Image.from_file("/path/to/image2.jpg")),
        ]
        dataset = LegacyDataset.from_iterable(items)

        converter = ForwardMixedMediaConverter.create(dataset)
        assert converter is None

    def test_mixed_media_converter_create_with_only_video_frames(self):
        """Test that create returns a converter when dataset has only video frames."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
            DatasetItem(id="frame_1", media=VideoFrame(video=video, index=1)),
        ]
        dataset = LegacyDataset.from_iterable(items, media_type=VideoFrame)

        converter = ForwardMixedMediaConverter.create(dataset)
        assert isinstance(converter, ForwardMixedMediaConverter)

    def test_mixed_media_converter_get_supported_media_types_is_empty(self):
        """Test that the mixed converter returns empty supported types (not registry-based)."""
        assert ForwardMixedMediaConverter.get_supported_media_types() == []

    def test_mixed_media_converter_get_schema_attributes(self):
        """Test schema attribute generation uses media_path_field."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="img_0", media=Image.from_file("/path/to/image.jpg")),
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
        ]
        dataset = LegacyDataset.from_iterable(items)

        converter = ForwardMixedMediaConverter.create(dataset)
        assert converter is not None

        attributes = converter.get_schema_attributes()
        assert "media" in attributes
        assert isinstance(attributes["media"].field, MediaPathField)

    def test_mixed_media_converter_get_schema_attributes_with_prefix(self):
        """Test schema attribute generation with name prefix."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="img_0", media=Image.from_file("/path/to/image.jpg")),
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
        ]
        dataset = LegacyDataset.from_iterable(items)

        converter = ForwardMixedMediaConverter.create(dataset, name_prefix="anomaly_")
        assert converter is not None

        attributes = converter.get_schema_attributes()
        assert "anomaly_media" in attributes

    def test_mixed_media_converter_convert_image_item(self):
        """Test converting an Image item to LazyImage."""
        converter = ForwardMixedMediaConverter(semantic="default")
        item = DatasetItem(id="img_0", media=Image.from_file("/path/to/image.jpg"))

        result = converter.convert_item_media(item)
        assert "media" in result
        assert isinstance(result["media"], LazyImage)
        assert str(result["media"].path) == "/path/to/image.jpg"

    def test_mixed_media_converter_convert_video_frame_item(self):
        """Test converting a VideoFrame item to LazyVideoFrame."""
        video = Video(path="/path/to/video.mp4")
        converter = ForwardMixedMediaConverter(semantic="default")
        item = DatasetItem(id="frame_0", media=VideoFrame(video=video, index=42))

        result = converter.convert_item_media(item)
        assert "media" in result
        assert isinstance(result["media"], LazyVideoFrame)
        assert str(result["media"].video_path) == "/path/to/video.mp4"
        assert result["media"].frame_index == 42

    def test_mixed_media_converter_convert_item_no_media(self):
        """Test converting an item with no media returns empty dict."""
        converter = ForwardMixedMediaConverter(semantic="default")
        item = DatasetItem(id="test", media=None)

        result = converter.convert_item_media(item)
        assert result == {}

    def test_mixed_media_converter_create_with_whole_video(self):
        """Test that create activates when dataset has a whole Video item."""
        items = [DatasetItem(id="vid", media=Video(path="/path/to/video.mp4"))]
        dataset = LegacyDataset.from_iterable(items, media_type=Video)

        converter = ForwardMixedMediaConverter.create(dataset)
        assert converter is not None
        assert isinstance(converter, ForwardMixedMediaConverter)

    def test_mixed_media_converter_create_with_video_and_video_frames(self):
        """Test that create activates with a mix of whole Video and VideoFrame items."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="vid", media=Video(path="/path/to/other_video.mp4")),
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
        ]
        dataset = LegacyDataset.from_iterable(items, media_type=MediaElement)

        converter = ForwardMixedMediaConverter.create(dataset)
        assert converter is not None
        assert isinstance(converter, ForwardMixedMediaConverter)

    def test_mixed_media_converter_create_tracks_annotated_video_paths(self):
        """Test that create tracks video paths that have VideoFrame items."""
        video = Video(path="/path/to/annotated_video.mp4")
        items = [
            DatasetItem(id="vid", media=Video(path="/path/to/unannotated_video.mp4")),
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
            DatasetItem(id="frame_10", media=VideoFrame(video=video, index=10)),
        ]
        dataset = LegacyDataset.from_iterable(items, media_type=MediaElement)

        converter = ForwardMixedMediaConverter.create(dataset)
        assert converter is not None
        assert converter._annotated_video_paths == {"/path/to/annotated_video.mp4"}

    def test_mixed_media_converter_convert_unannotated_whole_video_item(self):
        """Test converting a whole Video item (unannotated) stores it as frame 0."""
        converter = ForwardMixedMediaConverter(semantic="default")
        item = DatasetItem(id="vid", media=Video(path="/path/to/video.mp4"))

        result = converter.convert_item_media(item)
        assert "media" in result
        assert isinstance(result["media"], LazyVideoFrame)
        assert str(result["media"].video_path) == "/path/to/video.mp4"
        assert result["media"].frame_index == 0

    def test_mixed_media_converter_skips_annotated_whole_video_item(self):
        """Test that a whole Video item is skipped when its path has annotated VideoFrame items."""
        converter = ForwardMixedMediaConverter(
            semantic="default",
            annotated_video_paths={"/path/to/video.mp4"},
        )
        item = DatasetItem(id="vid", media=Video(path="/path/to/video.mp4"))

        result = converter.convert_item_media(item)
        assert result == {}

    def test_mixed_media_converter_does_not_skip_different_video_path(self):
        """Test that a whole Video item is NOT skipped when its path differs from annotated videos."""
        converter = ForwardMixedMediaConverter(
            semantic="default",
            annotated_video_paths={"/path/to/annotated.mp4"},
        )
        item = DatasetItem(id="vid", media=Video(path="/path/to/unannotated.mp4"))

        result = converter.convert_item_media(item)
        assert "media" in result
        assert isinstance(result["media"], LazyVideoFrame)
        assert str(result["media"].video_path) == "/path/to/unannotated.mp4"
        assert result["media"].frame_index == 0

    def test_mixed_media_converter_convert_mixed_video_and_frames(self):
        """Test converting items from a dataset with whole Video, VideoFrame, and Image.

        The whole Video here uses a path not in annotated_video_paths,
        so it is treated as unannotated and stored as frame 0.
        """
        converter = ForwardMixedMediaConverter(semantic="default")

        # Whole Video (unannotated, no annotated_video_paths set) → frame 0
        video_item = DatasetItem(id="vid", media=Video(path="/path/to/video1.mp4"))
        result_video = converter.convert_item_media(video_item)
        assert isinstance(result_video["media"], LazyVideoFrame)
        assert str(result_video["media"].video_path) == "/path/to/video1.mp4"
        assert result_video["media"].frame_index == 0

        # VideoFrame → frame N
        video2 = Video(path="/path/to/video2.mp4")
        frame_item = DatasetItem(id="frame_42", media=VideoFrame(video=video2, index=42))
        result_frame = converter.convert_item_media(frame_item)
        assert isinstance(result_frame["media"], LazyVideoFrame)
        assert str(result_frame["media"].video_path) == "/path/to/video2.mp4"
        assert result_frame["media"].frame_index == 42

        # Image → LazyImage
        img_item = DatasetItem(id="img", media=Image.from_file("/path/to/image.jpg"))
        result_img = converter.convert_item_media(img_item)
        assert isinstance(result_img["media"], LazyImage)
        assert str(result_img["media"].path) == "/path/to/image.jpg"

    def test_mixed_media_converter_same_video_as_whole_and_frames(self):
        """Test that when the same video path has both a Video item and VideoFrame items,
        the Video item is skipped and only the annotated frames are preserved."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="vid", media=Video(path="/path/to/video.mp4")),
            DatasetItem(id="frame_5", media=VideoFrame(video=video, index=5)),
            DatasetItem(id="frame_10", media=VideoFrame(video=video, index=10)),
        ]
        dataset = LegacyDataset.from_iterable(items, media_type=MediaElement)

        converter = ForwardMixedMediaConverter.create(dataset)
        assert converter is not None

        # The whole Video item should be skipped because its path has annotated frames
        vid_item = items[0]
        result_vid = converter.convert_item_media(vid_item)
        assert result_vid == {}

        # The annotated frames should still be converted
        result_frame5 = converter.convert_item_media(items[1])
        assert isinstance(result_frame5["media"], LazyVideoFrame)
        assert result_frame5["media"].frame_index == 5

        result_frame10 = converter.convert_item_media(items[2])
        assert isinstance(result_frame10["media"], LazyVideoFrame)
        assert result_frame10["media"].frame_index == 10


class BackwardMixedMediaConverterTest:
    """Tests for BackwardMixedMediaConverter."""

    def test_backward_mixed_media_converter_create_from_schema(self):
        """Test BackwardMixedMediaConverter.create_from_schema with MediaPathField."""
        experimental_dataset = Dataset(MixedMediaSample)
        schema = experimental_dataset.schema

        converter = BackwardMixedMediaConverter.create_from_schema(schema)
        assert converter is not None
        assert isinstance(converter, BackwardMixedMediaConverter)
        assert converter.media_attr == "media"

    def test_backward_mixed_media_converter_create_from_schema_no_media_path_field(self):
        """Test BackwardMixedMediaConverter with schema that has no MediaPathField."""
        schema = Schema(
            attributes={"some_tensor": AttributeInfo(type=np.ndarray, field=tensor_field(dtype=pl.Float32()))}
        )

        converter = BackwardMixedMediaConverter.create_from_schema(schema)
        assert converter is None

    def test_backward_mixed_media_converter_get_media_type(self):
        """Test that backward mixed converter returns MediaElement (common base)."""
        from datumaro.components.media import MediaElement

        experimental_dataset = Dataset(MixedMediaSample)
        schema = experimental_dataset.schema

        converter = BackwardMixedMediaConverter.create_from_schema(schema)
        assert converter is not None
        assert converter.get_media_type() == MediaElement

    def test_backward_mixed_media_converter_convert_lazy_image(self):
        """Test converting a LazyImage back to legacy Image."""
        experimental_dataset = Dataset(MixedMediaSample)
        schema = experimental_dataset.schema

        converter = BackwardMixedMediaConverter.create_from_schema(schema)
        assert converter is not None

        sample = MixedMediaSample(media=LazyImage(path="/path/to/image.jpg"))
        legacy_media = converter.convert_to_legacy_media(sample)

        assert isinstance(legacy_media, Image)
        assert getattr(legacy_media, "path", None) == "/path/to/image.jpg"

    def test_backward_mixed_media_converter_convert_lazy_video_frame(self):
        """Test converting a LazyVideoFrame back to legacy VideoFrame."""
        experimental_dataset = Dataset(MixedMediaSample)
        schema = experimental_dataset.schema

        converter = BackwardMixedMediaConverter.create_from_schema(schema)
        assert converter is not None

        sample = MixedMediaSample(
            media=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=42),
        )
        legacy_media = converter.convert_to_legacy_media(sample)

        assert isinstance(legacy_media, VideoFrame)
        assert legacy_media.video.path == "/path/to/video.mp4"
        assert legacy_media.index == 42

    def test_backward_mixed_media_converter_caches_video_objects(self):
        """Test that backward mixed converter caches Video objects for same path."""
        experimental_dataset = Dataset(MixedMediaSample)
        schema = experimental_dataset.schema

        converter = BackwardMixedMediaConverter.create_from_schema(schema)
        assert converter is not None

        sample1 = MixedMediaSample(
            media=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=0),
        )
        sample2 = MixedMediaSample(
            media=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=10),
        )

        media1 = converter.convert_to_legacy_media(sample1)
        media2 = converter.convert_to_legacy_media(sample2)

        assert isinstance(media1, VideoFrame)
        assert isinstance(media2, VideoFrame)
        assert media1.video is media2.video

    def test_backward_mixed_media_converter_handles_both_types(self):
        """Test that backward mixed converter can handle alternating image and video frame samples."""
        experimental_dataset = Dataset(MixedMediaSample)
        schema = experimental_dataset.schema

        converter = BackwardMixedMediaConverter.create_from_schema(schema)
        assert converter is not None

        img_sample = MixedMediaSample(media=LazyImage(path="/path/to/image.jpg"))
        vid_sample = MixedMediaSample(
            media=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=5),
        )

        img_media = converter.convert_to_legacy_media(img_sample)
        vid_media = converter.convert_to_legacy_media(vid_sample)

        assert isinstance(img_media, Image)
        assert getattr(img_media, "path", None) == "/path/to/image.jpg"

        assert isinstance(vid_media, VideoFrame)
        assert vid_media.video.path == "/path/to/video.mp4"
        assert vid_media.index == 5
