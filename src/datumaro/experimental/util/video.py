# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Video utilities for the Datumaro experimental module.

This module provides utility functions for working with video data,
including creating datasets from video files and extracting frames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields.videos import MediaPathField, VideoFramePathField, VideoInfoField
from datumaro.experimental.media import LazyVideoFrame, extract_video_info

if TYPE_CHECKING:
    from datumaro.experimental.categories import Categories

DType = TypeVar("DType", bound=Sample)


def _find_field_names(schema: Any) -> tuple[str | None, str | None, str | None]:
    """Find video-related field names from schema.

    Returns:
        Tuple of (frame_field_name, video_info_field_name, media_field_name)
    """
    frame_field_name = None
    video_info_field_name = None
    media_field_name = None

    for name, attr_info in schema.attributes.items():
        if isinstance(attr_info.field, VideoFramePathField):
            frame_field_name = name
        elif isinstance(attr_info.field, VideoInfoField):
            video_info_field_name = name
        elif isinstance(attr_info.field, MediaPathField):
            media_field_name = name

    return frame_field_name, video_info_field_name, media_field_name


def _compute_frame_indices(
    frame_indices: list[int] | None,
    start_frame: int,
    end_frame: int | None,
    frame_step: int,
    total_frames: int,
) -> list[int]:
    """Compute the list of frame indices to extract."""
    if frame_indices is not None:
        return frame_indices
    actual_end = end_frame if end_frame is not None else total_frames
    return list(range(start_frame, actual_end, frame_step))


def create_frame_samples(
    video_path: str | Path,
    sample_class: type[DType],
    frame_indices: list[int] | None = None,
    frame_step: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
    **extra_fields: Any,
) -> list[DType]:
    """
    Create sample instances for frames from a video.

    Args:
        video_path: Path to the video file
        sample_class: Sample class to instantiate for each frame
        frame_indices: Specific frame indices to extract (overrides step/start/end)
        frame_step: Extract every Nth frame (default: 1 = all frames)
        start_frame: First frame to extract (inclusive)
        end_frame: Last frame to extract (exclusive, None = end of video)
        **extra_fields: Additional field values to set on all samples

    Returns:
        List of Sample instances, one per extracted frame

    Examples:
        >>> # Extract every 10th frame
        >>> samples = create_frame_samples(
        ...     "video.mp4", VideoDetectionSample, frame_step=10
        ... )

        >>> # Extract specific frames
        >>> samples = create_frame_samples(
        ...     "video.mp4", VideoDetectionSample,
        ...     frame_indices=[0, 100, 200, 500]
        ... )
    """
    video_path_str = str(video_path)
    video_info = extract_video_info(video_path_str)

    # Determine which frames to extract
    indices_to_extract = _compute_frame_indices(
        frame_indices, start_frame, end_frame, frame_step, video_info.total_frames
    )

    # Infer the schema to determine field names
    schema = sample_class.infer_schema()
    frame_field_name, video_info_field_name, media_field_name = _find_field_names(schema)

    # Determine which field to use for the video frame
    target_field = frame_field_name or media_field_name

    # Create samples for each frame
    samples: list[DType] = []
    for frame_idx in indices_to_extract:
        sample_kwargs: dict[str, Any] = {}

        # Set the frame/media field
        if target_field is not None:
            sample_kwargs[target_field] = LazyVideoFrame(
                video_path=video_path_str,
                frame_index=frame_idx,
            )

        # Set video info if the field exists
        if video_info_field_name is not None:
            sample_kwargs[video_info_field_name] = video_info

        # Add extra fields
        sample_kwargs.update(extra_fields)

        samples.append(sample_class(**sample_kwargs))

    return samples


def iter_video_frames(
    video_path: str | Path,
    sample_class: type[DType],
    batch_size: int = 32,
    **kwargs: Any,
) -> Iterator[list[DType]]:
    """
    Iterate over video frames in batches for memory-efficient processing.

    Yields batches of samples without loading all frames into memory.

    Args:
        video_path: Path to the video file
        sample_class: Sample class to instantiate for each frame
        batch_size: Number of samples per batch
        **kwargs: Additional arguments passed to create_frame_samples

    Yields:
        List of Sample instances (batch_size samples per batch)

    Examples:
        >>> for batch in iter_video_frames("video.mp4", VideoSample, batch_size=32):
        ...     process_batch(batch)
    """
    video_path_str = str(video_path)
    video_info = extract_video_info(video_path_str)

    # Get frame step and calculate total frames to process
    frame_step = kwargs.get("frame_step", 1)
    start_frame = kwargs.get("start_frame", 0)
    end_frame = kwargs.get("end_frame")
    if end_frame is None:
        end_frame = video_info.total_frames
    frame_indices = kwargs.get("frame_indices")

    if frame_indices is not None:
        all_indices = frame_indices
    else:
        all_indices = list(range(start_frame, end_frame, frame_step))

    # Yield batches
    for i in range(0, len(all_indices), batch_size):
        batch_indices = all_indices[i : i + batch_size]
        excluded_keys = ("frame_step", "start_frame", "end_frame", "frame_indices")
        extra_fields = {k: v for k, v in kwargs.items() if k not in excluded_keys}
        batch_samples = create_frame_samples(
            video_path,
            sample_class,
            frame_indices=batch_indices,
            **extra_fields,
        )
        yield batch_samples


def dataset_from_video(
    video_path: str | Path,
    sample_class: type[DType],
    categories: dict[str, Categories] | None = None,
    frame_step: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_indices: list[int] | None = None,
    **extra_fields: Any,
) -> Dataset[DType]:
    """
    Create a Dataset from a video file.

    Each frame becomes a row in the dataset's DataFrame.

    Args:
        video_path: Path to the video file
        sample_class: Sample class defining the schema
        categories: Optional categories for label fields
        frame_step: Extract every Nth frame
        start_frame: First frame to extract
        end_frame: Last frame to extract
        frame_indices: Specific frame indices to extract (overrides step/start/end)
        **extra_fields: Additional field values to set on all samples

    Returns:
        Dataset with one row per extracted frame

    Examples:
        >>> dataset = dataset_from_video(
        ...     "video.mp4",
        ...     VideoDetectionSample,
        ...     categories={"labels": label_cats},
        ...     frame_step=5,  # Every 5th frame
        ... )
        >>> len(dataset)  # Number of frames
    """
    video_path_str = str(video_path)
    video_info = extract_video_info(video_path_str)

    # Create the dataset
    dataset = Dataset(sample_class, categories=categories)

    # Add video metadata
    dataset.add_video_metadata(video_path_str, video_info)

    # Create and append samples
    samples = create_frame_samples(
        video_path,
        sample_class,
        frame_indices=frame_indices,
        frame_step=frame_step,
        start_frame=start_frame,
        end_frame=end_frame,
        **extra_fields,
    )

    for sample in samples:
        dataset.append(sample)

    return dataset


def dataset_from_videos(
    video_paths: list[str | Path],
    sample_class: type[DType],
    categories: dict[str, Categories] | None = None,
    frame_step: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
    **extra_fields: Any,
) -> Dataset[DType]:
    """
    Create a Dataset from multiple video files.

    Concatenates frames from all videos into a single dataset.
    Video source is tracked via video_info field or can be queried via
    dataset.get_video_info_for_sample().

    Args:
        video_paths: List of paths to video files
        sample_class: Sample class defining the schema
        categories: Optional categories for label fields
        frame_step: Extract every Nth frame from each video
        start_frame: First frame to extract from each video
        end_frame: Last frame to extract from each video (None = all)
        **extra_fields: Additional field values to set on all samples

    Returns:
        Dataset with frames from all videos concatenated

    Examples:
        >>> dataset = dataset_from_videos(
        ...     ["video1.mp4", "video2.mp4"],
        ...     VideoDetectionSample,
        ...     frame_step=10,
        ... )
    """
    if not video_paths:
        return Dataset(sample_class, categories=categories)

    # Create dataset from first video
    dataset = dataset_from_video(
        video_paths[0],
        sample_class,
        categories=categories,
        frame_step=frame_step,
        start_frame=start_frame,
        end_frame=end_frame,
        **extra_fields,
    )

    # Append frames from remaining videos
    for video_path in video_paths[1:]:
        additional_dataset = dataset_from_video(
            video_path,
            sample_class,
            categories=categories,
            frame_step=frame_step,
            start_frame=start_frame,
            end_frame=end_frame,
            **extra_fields,
        )
        dataset.append_dataset(additional_dataset)

    return dataset


def get_video_paths_from_dataset(dataset: Dataset[DType]) -> set[str]:
    """
    Extract all unique video paths from a dataset.

    Args:
        dataset: Dataset to extract video paths from

    Returns:
        Set of unique video paths found in the dataset
    """
    video_paths: set[str] = set()

    # Check for video fields in schema
    for name, attr_info in dataset.schema.attributes.items():
        if isinstance(attr_info.field, VideoFramePathField):
            # Extract unique paths from the column
            if name in dataset.df.columns:
                paths = dataset.df[name].drop_nulls().unique().to_list()
                video_paths.update(str(p) for p in paths)
        elif isinstance(attr_info.field, MediaPathField):
            # For MediaPathField, check if frame_index is set to identify video frames
            frame_idx_col = f"{name}_frame_index"
            if name in dataset.df.columns and frame_idx_col in dataset.df.columns:
                # Filter to only rows with frame_index set (video frames)
                video_df = dataset.df.filter(pl.col(frame_idx_col).is_not_null())
                if len(video_df) > 0:
                    paths = video_df[name].drop_nulls().unique().to_list()
                    video_paths.update(str(p) for p in paths)

    return video_paths


__all__ = [
    "create_frame_samples",
    "dataset_from_video",
    "dataset_from_videos",
    "get_video_paths_from_dataset",
    "iter_video_frames",
]
