# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Video dataset support for the Datumaro experimental module.

This module provides a mixin class for adding video-specific functionality
to datasets, including video metadata storage, retrieval, and optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from datumaro.experimental.fields.videos import MediaPathField, VideoFramePathField
from datumaro.experimental.media import LazyVideoFrame, VideoInfo, extract_video_info

if TYPE_CHECKING:
    from datumaro.experimental.dataset import Sample
    from datumaro.experimental.schema import Schema


class VideoDatasetMixin:
    """
    Mixin class providing video-specific functionality for datasets.

    This mixin adds video metadata storage, retrieval, and optimization
    capabilities to dataset classes. It manages a dictionary mapping video
    paths to VideoInfo objects, avoiding redundant storage of video metadata
    per frame.

    Attributes:
        _video_metadata: Dictionary mapping video paths to VideoInfo objects
        _schema: The dataset schema (must be provided by the base class)
        df: The Polars DataFrame (must be provided by the base class)
    """

    _video_metadata: dict[str, VideoInfo]
    _schema: Schema
    df: pl.DataFrame
    _transforms: Any  # Transform | None

    def _init_video_metadata(self, video_metadata: dict[str, VideoInfo] | None = None) -> None:
        """
        Initialize video metadata storage.

        Args:
            video_metadata: Optional dictionary mapping video paths to VideoInfo
        """
        self._video_metadata = video_metadata if video_metadata is not None else {}

    @property
    def video_metadata(self) -> dict[str, VideoInfo]:
        """
        Get the video metadata dictionary.

        Returns:
            Dictionary mapping video paths to VideoInfo objects.
        """
        return self._video_metadata

    def get_video_info(self, video_path: str) -> VideoInfo | None:
        """
        Get video metadata by path.

        Args:
            video_path: Path to the video file

        Returns:
            VideoInfo for the video, or None if not found
        """
        return self._video_metadata.get(video_path)

    def get_video_info_for_sample(self, sample: Sample) -> VideoInfo | None:
        """
        Get video metadata for a sample, if it's a video frame.

        Searches through all video-related fields in the sample and returns
        the VideoInfo for the first LazyVideoFrame found. If the video info
        is not cached, it will be extracted from the video file.

        Args:
            sample: A Sample instance to get video info for

        Returns:
            VideoInfo for the sample's video, or None if not a video frame
        """
        for field_name, _field in self._get_video_fields():
            value = getattr(sample, field_name, None)
            if isinstance(value, LazyVideoFrame):
                path = str(value.video_path)
                if path not in self._video_metadata:
                    self._video_metadata[path] = extract_video_info(path)
                return self._video_metadata[path]

        return None

    def add_video_metadata(self, video_path: str, video_info: VideoInfo) -> None:
        """
        Add or update video metadata for a video path.

        Args:
            video_path: Path to the video file
            video_info: VideoInfo object for the video
        """
        self._video_metadata[video_path] = video_info

    def _get_video_fields(self) -> list[tuple[str, MediaPathField | VideoFramePathField]]:
        """
        Extract all video-related fields from the dataset schema.

        Returns:
            List of tuples containing field name and field instance
        """
        video_fields = []
        for name, attr_info in self._schema.attributes.items():
            if isinstance(attr_info.field, (MediaPathField, VideoFramePathField)):
                video_fields.append((name, attr_info.field))
        return video_fields

    def _optimize_storage(self) -> None:
        """
        Optimize DataFrame storage for video datasets.

        Converts path columns to Categorical type for deduplication,
        which is especially beneficial for video datasets where many
        frames share the same video path.

        Raises:
            RuntimeError: If called on a transformed dataset
        """
        if self._transforms is not None:
            raise RuntimeError("Cannot optimize storage on transformed datasets.")

        # Find all path columns from video fields
        video_fields = self._get_video_fields()
        path_columns = [name for name, _ in video_fields]

        # Also check for common media column
        if "media" in self.df.columns and self.df["media"].dtype == pl.String:
            path_columns.append("media")

        # Convert path columns to Categorical for deduplication
        for col in path_columns:
            if col in self.df.columns and self.df[col].dtype == pl.String:
                self.df = self.df.with_columns(pl.col(col).cast(pl.Categorical))

    def _serialize_video_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Serialize video metadata for pickling.

        Returns:
            Dictionary of video paths to serialized VideoInfo dicts
        """
        return {path: info.to_dict() for path, info in self._video_metadata.items()}

    def _deserialize_video_metadata(self, serialized: dict[str, dict[str, Any]]) -> None:
        """
        Deserialize video metadata after unpickling.

        Args:
            serialized: Dictionary of video paths to serialized VideoInfo dicts
        """
        self._video_metadata = {path: VideoInfo.from_dict(info_dict) for path, info_dict in serialized.items()}

    def _merge_video_metadata(self, other_metadata: dict[str, VideoInfo]) -> None:
        """
        Merge video metadata from another dataset.

        Args:
            other_metadata: Video metadata dictionary to merge
        """
        self._video_metadata.update(other_metadata)
