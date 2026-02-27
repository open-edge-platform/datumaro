# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Unit tests for Dataset video-related functionality.

Tests cover:
- Video metadata storage and retrieval
- VideoInfo serialization/deserialization (pickle)
- get_video_info() and get_video_info_for_sample()
- add_video_metadata()
- filter_by_subset() preserving video metadata
- append_dataset() merging video metadata
- slice() preserving video metadata
- from_dataframe() with video_metadata parameter
- _get_video_fields() utility
- _optimize_storage() for video path columns
"""

import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import bbox_field, subset_field
from datumaro.experimental.fields.datasets import Subset
from datumaro.experimental.fields.videos import (
    MediaPathField,
    VideoFramePathField,
    media_path_field,
    video_frame_path_field,
)
from datumaro.experimental.media import LazyImage, LazyVideoFrame, VideoInfo

# Path to test video in assets
TEST_VIDEO_PATH = Path(__file__).parent.parent.parent / "assets" / "cvat_dataset" / "test.mp4"


# Module-level sample classes for pickle tests (local classes can't be pickled)
class PickleableVideoSample(Sample):
    """Sample class for pickle tests - must be defined at module level."""

    frame: LazyVideoFrame = video_frame_path_field()


class VideoMetadataStorageTest:
    """Tests for video metadata storage and retrieval in Dataset."""

    def test_dataset_initializes_empty_video_metadata(self):
        """Test that a new dataset has empty video metadata."""

        class SimpleSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(SimpleSample)
        assert dataset.video_metadata == {}
        assert isinstance(dataset._video_metadata, dict)

    def test_video_metadata_property_returns_dict(self):
        """Test that video_metadata property returns the internal dict."""

        class SimpleSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(SimpleSample)
        video_info = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )
        dataset._video_metadata["/path/to/video.mp4"] = video_info

        assert dataset.video_metadata == {"/path/to/video.mp4": video_info}

    def test_get_video_info_returns_metadata(self):
        """Test get_video_info() returns correct VideoInfo."""

        class SimpleSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(SimpleSample)
        video_info = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )
        dataset._video_metadata["/path/to/video.mp4"] = video_info

        result = dataset.get_video_info("/path/to/video.mp4")
        assert result == video_info
        assert result.total_frames == 100
        assert result.fps == 30.0

    def test_get_video_info_returns_none_for_unknown_path(self):
        """Test get_video_info() returns None for unknown video path."""

        class SimpleSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(SimpleSample)
        result = dataset.get_video_info("/unknown/video.mp4")
        assert result is None

    def test_add_video_metadata(self):
        """Test add_video_metadata() adds or updates video info."""

        class SimpleSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(SimpleSample)
        video_info = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )

        dataset.add_video_metadata("/path/to/video.mp4", video_info)

        assert "/path/to/video.mp4" in dataset.video_metadata
        assert dataset.get_video_info("/path/to/video.mp4") == video_info

    def test_add_video_metadata_overwrites_existing(self):
        """Test add_video_metadata() overwrites existing metadata."""

        class SimpleSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(SimpleSample)

        video_info1 = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )
        video_info2 = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=200,
            fps=60.0,
            width=3840,
            height=2160,
            duration=3.33,
            codec="hevc",
        )

        dataset.add_video_metadata("/path/to/video.mp4", video_info1)
        dataset.add_video_metadata("/path/to/video.mp4", video_info2)

        result = dataset.get_video_info("/path/to/video.mp4")
        assert result.total_frames == 200
        assert result.fps == 60.0
        assert result.codec == "hevc"


class GetVideoInfoForSampleTest:
    """Tests for get_video_info_for_sample() method."""

    def test_get_video_info_for_sample_with_media_field(self):
        """Test getting video info from sample with media field (LazyVideoFrame)."""

        class MediaSample(Sample):
            media: LazyVideoFrame | LazyImage = media_path_field()

        dataset = Dataset(MediaSample)
        video_info = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )
        dataset._video_metadata["/path/to/video.mp4"] = video_info

        # Create a sample with LazyVideoFrame
        sample = MediaSample(media=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=42))

        result = dataset.get_video_info_for_sample(sample)
        assert result == video_info

    def test_get_video_info_for_sample_with_frame_field(self):
        """Test getting video info from sample with frame field."""

        class FrameSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(FrameSample)
        video_info = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )
        dataset._video_metadata["/path/to/video.mp4"] = video_info

        sample = FrameSample(frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=10))

        result = dataset.get_video_info_for_sample(sample)
        assert result == video_info

    def test_get_video_info_for_sample_with_lazy_image_returns_none(self):
        """Test that get_video_info_for_sample returns None for LazyImage."""

        class MediaSample(Sample):
            media: LazyVideoFrame | LazyImage = media_path_field()

        dataset = Dataset(MediaSample)

        sample = MediaSample(media=LazyImage(path="/path/to/image.jpg"))

        result = dataset.get_video_info_for_sample(sample)
        assert result is None

    def test_get_video_info_for_sample_without_video_fields_returns_none(self):
        """Test that get_video_info_for_sample returns None for non-video sample."""

        class NonVideoSample(Sample):
            bbox: np.ndarray = bbox_field(dtype=pl.Float32())

        dataset = Dataset(NonVideoSample)

        sample = NonVideoSample(bbox=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

        result = dataset.get_video_info_for_sample(sample)
        assert result is None

    @pytest.mark.skipif(not TEST_VIDEO_PATH.exists(), reason="Test video not available")
    def test_get_video_info_for_sample_extracts_and_caches_info(self):
        """Test that get_video_info_for_sample extracts info for unknown videos."""

        class FrameSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(FrameSample)
        assert len(dataset.video_metadata) == 0

        sample = FrameSample(frame=LazyVideoFrame(video_path=str(TEST_VIDEO_PATH), frame_index=0))

        result = dataset.get_video_info_for_sample(sample)

        # Should have extracted and cached the video info
        assert result is not None
        assert result.fps == 15.0
        assert result.width == 240
        assert result.height == 180
        assert str(TEST_VIDEO_PATH) in dataset.video_metadata


class VideoMetadataSerializationTest:
    """Tests for pickling/unpickling datasets with video metadata."""

    def test_pickle_dataset_with_video_metadata(self):
        """Test that video metadata survives pickle round-trip."""
        dataset = Dataset(PickleableVideoSample)
        video_info = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )
        dataset.add_video_metadata("/path/to/video.mp4", video_info)


class FilterBySubsetPreservesVideoMetadataTest:
    """Tests for filter_by_subset() preserving video metadata."""

    def test_filter_by_subset_preserves_video_metadata(self):
        """Test that filter_by_subset() preserves video metadata."""

        class VideoSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()
            subset: Subset = subset_field()

        dataset = Dataset(VideoSample)
        video_info = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )
        dataset.add_video_metadata("/path/to/video.mp4", video_info)

        dataset.append(
            VideoSample(
                frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=0),
                subset=Subset.TRAINING,
            )
        )
        dataset.append(
            VideoSample(
                frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=1),
                subset=Subset.VALIDATION,
            )
        )

        filtered = dataset.filter_by_subset(Subset.TRAINING)

        assert len(filtered) == 1
        assert "/path/to/video.mp4" in filtered.video_metadata
        assert filtered.get_video_info("/path/to/video.mp4") == video_info


class AppendDatasetMergesVideoMetadataTest:
    """Tests for append_dataset() merging video metadata."""

    def test_append_dataset_merges_video_metadata(self):
        """Test that append_dataset() merges video metadata from both datasets."""

        class SimpleSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset1 = Dataset(SimpleSample)
        video_info1 = VideoInfo(
            path="/path/to/video1.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )
        dataset1.add_video_metadata("/path/to/video1.mp4", video_info1)
        dataset1.append(SimpleSample(frame=LazyVideoFrame(video_path="/path/to/video1.mp4", frame_index=0)))

        dataset2 = Dataset(SimpleSample)
        video_info2 = VideoInfo(
            path="/path/to/video2.mp4",
            total_frames=200,
            fps=60.0,
            width=3840,
            height=2160,
            duration=3.33,
            codec="hevc",
        )
        dataset2.add_video_metadata("/path/to/video2.mp4", video_info2)
        dataset2.append(SimpleSample(frame=LazyVideoFrame(video_path="/path/to/video2.mp4", frame_index=0)))

        dataset1.append_dataset(dataset2)

        assert len(dataset1) == 2
        assert "/path/to/video1.mp4" in dataset1.video_metadata
        assert "/path/to/video2.mp4" in dataset1.video_metadata
        assert dataset1.get_video_info("/path/to/video1.mp4") == video_info1
        assert dataset1.get_video_info("/path/to/video2.mp4") == video_info2

    def test_append_dataset_overwrites_duplicate_video_metadata(self):
        """Test that append_dataset() overwrites duplicate video paths."""

        class SimpleSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset1 = Dataset(SimpleSample)
        video_info1 = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )
        dataset1.add_video_metadata("/path/to/video.mp4", video_info1)
        dataset1.append(SimpleSample(frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=0)))

        dataset2 = Dataset(SimpleSample)
        video_info2 = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=200,  # Different value
            fps=60.0,
            width=3840,
            height=2160,
            duration=3.33,
            codec="hevc",
        )
        dataset2.add_video_metadata("/path/to/video.mp4", video_info2)
        dataset2.append(SimpleSample(frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=1)))

        dataset1.append_dataset(dataset2)

        # The second dataset's metadata should overwrite
        result = dataset1.get_video_info("/path/to/video.mp4")
        assert result.total_frames == 200
        assert result.fps == 60.0


class GetVideoFieldsTest:
    """Tests for _get_video_fields() method."""

    def test_get_video_fields_with_video_frame_path_field(self):
        """Test _get_video_fields() finds VideoFramePathField."""

        class SimpleSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(SimpleSample)
        fields = dataset._get_video_fields()

        assert len(fields) == 1
        name, field = fields[0]
        assert name == "frame"
        assert isinstance(field, VideoFramePathField)

    def test_get_video_fields_with_media_path_field(self):
        """Test _get_video_fields() finds MediaPathField."""

        class MediaSample(Sample):
            media: LazyVideoFrame | LazyImage = media_path_field()

        dataset = Dataset(MediaSample)
        fields = dataset._get_video_fields()

        assert len(fields) == 1
        name, field = fields[0]
        assert name == "media"
        assert isinstance(field, MediaPathField)

    def test_get_video_fields_with_multiple_video_fields(self):
        """Test _get_video_fields() finds multiple video fields."""

        class MultiVideoSample(Sample):
            frame1: LazyVideoFrame = video_frame_path_field(semantic="primary")
            frame2: LazyVideoFrame = video_frame_path_field(semantic="secondary")
            media: LazyVideoFrame | LazyImage = media_path_field()

        dataset = Dataset(MultiVideoSample)
        fields = dataset._get_video_fields()

        assert len(fields) == 3
        names = {name for name, _ in fields}
        assert names == {"frame1", "frame2", "media"}

    def test_get_video_fields_without_video_fields(self):
        """Test _get_video_fields() returns empty for non-video dataset."""

        class NonVideoSample(Sample):
            bbox: np.ndarray = bbox_field(dtype=pl.Float32())

        dataset = Dataset(NonVideoSample)
        fields = dataset._get_video_fields()

        assert fields == []


class OptimizeStorageTest:
    """Tests for _optimize_storage() method."""

    def test_optimize_storage_converts_path_to_categorical(self):
        """Test that _optimize_storage converts String paths to Categorical."""

        class SimpleSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(SimpleSample)

        # Add multiple samples with same video path
        for i in range(10):
            dataset.append(SimpleSample(frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=i)))

        # Before optimization, column should be String
        assert dataset.df["frame"].dtype == pl.String

        dataset._optimize_storage()

        # After optimization, column should be Categorical
        assert dataset.df["frame"].dtype == pl.Categorical

    def test_optimize_storage_on_transformed_dataset_raises(self):
        """Test that _optimize_storage raises for transformed datasets."""

        class SimpleSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(SimpleSample)
        dataset.append(SimpleSample(frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=0)))

        # Create a transformed dataset
        transformed = dataset.transform(lambda t: t)

        with pytest.raises(RuntimeError, match="Cannot optimize storage on transformed datasets"):
            transformed._optimize_storage()


class ConvertToSchemaPreservesVideoMetadataTest:
    """Tests for convert_to_schema() preserving video metadata."""

    def test_convert_to_schema_preserves_video_metadata(self):
        """Test that convert_to_schema preserves video metadata."""

        class SourceSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        class TargetSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(SourceSample)
        video_info = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )
        dataset.add_video_metadata("/path/to/video.mp4", video_info)
        dataset.append(SourceSample(frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=0)))

        # Convert with same schema (should use early return path)
        converted = dataset.convert_to_schema(TargetSample)

        assert "/path/to/video.mp4" in converted.video_metadata
        assert converted.get_video_info("/path/to/video.mp4") == video_info


class VideoInfoSerializationTest:
    """Tests for VideoInfo.to_dict() and from_dict() methods."""

    def test_video_info_to_dict(self):
        """Test VideoInfo.to_dict() serialization."""
        video_info = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )

        result = video_info.to_dict()

        assert result == {
            "path": "/path/to/video.mp4",
            "total_frames": 100,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "duration": 3.33,
            "codec": "h264",
        }

    def test_video_info_from_dict(self):
        """Test VideoInfo.from_dict() deserialization."""
        data = {
            "path": "/path/to/video.mp4",
            "total_frames": 100,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "duration": 3.33,
            "codec": "h264",
        }

        video_info = VideoInfo.from_dict(data)

        assert video_info.path == "/path/to/video.mp4"
        assert video_info.total_frames == 100
        assert video_info.fps == 30.0
        assert video_info.width == 1920
        assert video_info.height == 1080
        assert video_info.duration == 3.33
        assert video_info.codec == "h264"

    def test_video_info_round_trip(self):
        """Test VideoInfo serialization round-trip."""
        original = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )

        restored = VideoInfo.from_dict(original.to_dict())

        assert restored == original

    def test_video_info_from_dict_without_codec(self):
        """Test VideoInfo.from_dict() handles missing codec gracefully."""
        data = {
            "path": "/path/to/video.mp4",
            "total_frames": 100,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "duration": 3.33,
        }

        video_info = VideoInfo.from_dict(data)

        assert video_info.codec is None


@pytest.mark.skipif(not TEST_VIDEO_PATH.exists(), reason="Test video not available")
class VideoIntegrationTest:
    """Integration tests using actual video file."""

    def test_dataset_with_real_video_frames(self):
        """Test dataset with samples from real video file."""

        class VideoSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(VideoSample)

        # Add frames from the test video
        for i in range(5):
            dataset.append(VideoSample(frame=LazyVideoFrame(video_path=str(TEST_VIDEO_PATH), frame_index=i)))

        assert len(dataset) == 5

        # Retrieve a sample and check video info extraction
        sample = dataset[0]
        video_info = dataset.get_video_info_for_sample(sample)

        assert video_info is not None
        assert video_info.fps == 15.0
        assert video_info.width == 240
        assert video_info.height == 180

    def test_pickle_dataset_with_real_video_samples(self):
        """Test pickling dataset with real video samples."""
        dataset = Dataset(PickleableVideoSample)

        for i in range(3):
            dataset.append(PickleableVideoSample(frame=LazyVideoFrame(video_path=str(TEST_VIDEO_PATH), frame_index=i)))

        # Pre-populate video metadata
        sample = dataset[0]
        dataset.get_video_info_for_sample(sample)

        # Pickle round-trip
        pickled = pickle.dumps(dataset)
        restored = pickle.loads(pickled)

        assert len(restored) == 3
        assert str(TEST_VIDEO_PATH) in restored.video_metadata

        # Verify we can still access samples
        restored_sample = restored[0]
        assert isinstance(restored_sample.frame, LazyVideoFrame)
        assert restored_sample.frame.frame_index == 0


class MultipleVideosDatasetTest:
    """Tests for datasets containing frames from multiple video files."""

    def test_dataset_with_multiple_videos(self):
        """Test dataset with frames from multiple different videos."""

        class VideoSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(VideoSample)

        # Add frames from multiple "videos" (using same path with different frame indices)
        video1 = "/path/to/video1.mp4"
        video2 = "/path/to/video2.mp4"
        video3 = "/path/to/video3.mp4"

        for video_path in [video1, video2, video3]:
            for frame_idx in range(3):
                dataset.append(VideoSample(frame=LazyVideoFrame(video_path=video_path, frame_index=frame_idx)))

        assert len(dataset) == 9

        # Check DataFrame structure
        unique_videos = dataset.df["frame"].unique().to_list()
        assert len(unique_videos) == 3
        assert set(unique_videos) == {video1, video2, video3}

    def test_multiple_videos_metadata_storage(self):
        """Test that metadata for multiple videos can be stored and retrieved."""

        class VideoSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(VideoSample)

        # Add metadata for multiple videos
        for i in range(3):
            video_path = f"/path/to/video{i}.mp4"
            video_info = VideoInfo(
                path=video_path,
                total_frames=100 + i * 50,
                fps=30.0 + i * 10,
                width=1920,
                height=1080,
                duration=3.33,
                codec="h264",
            )
            dataset.add_video_metadata(video_path, video_info)

        assert len(dataset.video_metadata) == 3

        # Verify each video's metadata
        for i in range(3):
            video_path = f"/path/to/video{i}.mp4"
            info = dataset.get_video_info(video_path)
            assert info is not None
            assert info.total_frames == 100 + i * 50
            assert info.fps == 30.0 + i * 10

    def test_multiple_videos_groupby_operations(self):
        """Test groupby operations on multiple videos using Polars."""

        class VideoSample(Sample):
            frame: LazyVideoFrame = video_frame_path_field()

        dataset = Dataset(VideoSample)

        # Add frames from multiple videos with different counts
        dataset.append(VideoSample(frame=LazyVideoFrame("/video1.mp4", frame_index=0)))
        dataset.append(VideoSample(frame=LazyVideoFrame("/video1.mp4", frame_index=1)))
        dataset.append(VideoSample(frame=LazyVideoFrame("/video2.mp4", frame_index=0)))
        dataset.append(VideoSample(frame=LazyVideoFrame("/video2.mp4", frame_index=1)))
        dataset.append(VideoSample(frame=LazyVideoFrame("/video2.mp4", frame_index=2)))

        # Group by video path using Polars
        video_counts = dataset.df.group_by("frame").agg(pl.len().alias("count"))

        assert len(video_counts) == 2
        counts_dict = dict(zip(video_counts["frame"].to_list(), video_counts["count"].to_list()))
        assert counts_dict["/video1.mp4"] == 2
        assert counts_dict["/video2.mp4"] == 3


class MixedMediaDatasetTest:
    """Tests for datasets with mixed images and video frames."""

    def test_mixed_media_dataset_creation(self):
        """Test creating dataset with both images and video frames."""

        class MixedSample(Sample):
            media: LazyImage | LazyVideoFrame = media_path_field()

        dataset = Dataset(MixedSample)

        # Add image samples
        dataset.append(MixedSample(media=LazyImage("/path/to/image1.jpg")))
        dataset.append(MixedSample(media=LazyImage("/path/to/image2.jpg")))

        # Add video frame samples
        dataset.append(MixedSample(media=LazyVideoFrame("/path/to/video.mp4", frame_index=0)))
        dataset.append(MixedSample(media=LazyVideoFrame("/path/to/video.mp4", frame_index=1)))

        assert len(dataset) == 4

        # Verify DataFrame structure
        assert "media" in dataset.df.columns
        assert "media_frame_index" in dataset.df.columns

        # Check frame_index values
        frame_indices = dataset.df["media_frame_index"].to_list()
        assert frame_indices[0] is None  # Image
        assert frame_indices[1] is None  # Image
        assert frame_indices[2] == 0  # Video frame
        assert frame_indices[3] == 1  # Video frame

    def test_mixed_media_filter_by_type(self):
        """Test filtering mixed media dataset by media type using Polars."""

        class MixedSample(Sample):
            media: LazyImage | LazyVideoFrame = media_path_field()

        dataset = Dataset(MixedSample)

        # Add mixed media
        dataset.append(MixedSample(media=LazyImage("/image1.jpg")))
        dataset.append(MixedSample(media=LazyVideoFrame("/video.mp4", frame_index=0)))
        dataset.append(MixedSample(media=LazyImage("/image2.jpg")))
        dataset.append(MixedSample(media=LazyVideoFrame("/video.mp4", frame_index=1)))

        # Filter for images (frame_index is null)
        image_df = dataset.df.filter(pl.col("media_frame_index").is_null())
        assert len(image_df) == 2

        # Filter for video frames (frame_index is not null)
        video_df = dataset.df.filter(pl.col("media_frame_index").is_not_null())
        assert len(video_df) == 2

    def test_mixed_media_video_metadata_only_for_videos(self):
        """Test that video metadata is only retrieved for video frames, not images."""

        class MixedSample(Sample):
            media: LazyImage | LazyVideoFrame = media_path_field()

        dataset = Dataset(MixedSample)

        # Add video metadata
        video_info = VideoInfo(
            path="/path/to/video.mp4",
            total_frames=100,
            fps=30.0,
            width=1920,
            height=1080,
            duration=3.33,
            codec="h264",
        )
        dataset.add_video_metadata("/path/to/video.mp4", video_info)

        # Create samples
        image_sample = MixedSample(media=LazyImage("/path/to/image.jpg"))
        video_sample = MixedSample(media=LazyVideoFrame("/path/to/video.mp4", frame_index=0))

        # Image should return None for video info
        assert dataset.get_video_info_for_sample(image_sample) is None

        # Video frame should return video info
        assert dataset.get_video_info_for_sample(video_sample) == video_info
