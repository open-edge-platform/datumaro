# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Video frame converters for the Datumaro experimental module.

This module provides converters for transforming video frame references
into image tensors, supporting both VideoFramePathField and MediaPathField.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from datumaro.experimental.converters.base import Converter
from datumaro.experimental.converters.registry import converter
from datumaro.experimental.fields.images import ImageCallableField, ImageField, ImageInfoField, ImagePathField
from datumaro.experimental.fields.videos import (
    MediaInfoField,
    MediaPathField,
    VideoFrameCallableField,
    VideoFramePathField,
)
from datumaro.experimental.media import LazyImage, LazyVideoFrame
from datumaro.experimental.schema import AttributeSpec

if TYPE_CHECKING:
    import numpy as np


@converter(lazy=True)
class VideoFramePathToImageConverter(Converter):
    """
    Lazy converter that loads video frames from VideoFramePathField to ImageField.

    Converts (video_path, frame_index) references to loaded image tensors.
    Uses VideoFrameCache for efficient caching.

    Input: VideoFramePathField (path + frame_index)
    Output: ImageField (tensor data + shape)

    Supported output formats:
        - RGB: 3-channel color image in Red-Green-Blue order
        - BGR: 3-channel color image in Blue-Green-Red order (OpenCV default)
        - RGBA: 4-channel color image with alpha channel
        - GRAY: Single-channel grayscale image
    """

    input_frame: AttributeSpec[VideoFramePathField]
    output_image: AttributeSpec[ImageField]

    # Supported output formats
    SUPPORTED_FORMATS = {"RGB", "BGR", "RGBA", "GRAY"}

    def filter_output_spec(self) -> bool:
        """Configure output image specification based on input."""
        output_format = self.output_image.field.format if self.output_image.field.format else "RGB"

        if output_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported output format '{output_format}' for VideoFramePathToImageConverter. "
                f"Supported formats are: {', '.join(sorted(self.SUPPORTED_FORMATS))}."
            )

        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_frame.field.semantic,
                dtype=pl.UInt8(),
                format=output_format,
                channels_first=self.output_image.field.channels_first,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert video frame paths to loaded image tensors.

        Args:
            df: DataFrame containing video path and frame_index columns

        Returns:
            DataFrame with loaded image data and shape information
        """
        input_path_col = self.input_frame.name
        input_frame_idx_col = f"{self.input_frame.name}_frame_index"
        output_col = self.output_image.name
        output_shape_col = f"{self.output_image.name}_shape"
        output_format = self.output_image.field.format

        image_data: list[Any] = []
        image_shapes: list[list[int]] = []

        for path, frame_idx in zip(df[input_path_col], df[input_frame_idx_col]):
            if path is None or frame_idx is None:
                image_data.append(None)
                image_shapes.append([])
                continue

            # Use LazyVideoFrame for loading (which uses VideoFrameCache)
            lazy_frame = LazyVideoFrame(
                video_path=str(path),
                frame_index=frame_idx,
                format=output_format,
                channels_first=self.output_image.field.channels_first,
            )

            try:
                img_array = lazy_frame.data
                image_data.append(img_array.flatten().tolist())
                image_shapes.append(list(img_array.shape))
            except Exception:
                image_data.append(None)
                image_shapes.append([])

        return df.with_columns(
            pl.Series(output_col, image_data, dtype=pl.List(self.output_image.field.dtype)),
            pl.Series(output_shape_col, image_shapes, dtype=pl.List(pl.Int32())),
        )


@converter
class VideoFrameToImageCallableConverter(Converter):
    """
    Converter that wraps VideoFramePathField as ImageCallableField.

    Creates a callable for each frame that loads data on demand.
    Useful for compatibility with existing image processing pipelines.

    Input: VideoFramePathField (path + frame_index)
    Output: ImageCallableField (callable that returns image data)
    """

    input_frame: AttributeSpec[VideoFramePathField]
    output_callable: AttributeSpec[ImageCallableField]

    def filter_output_spec(self) -> bool:
        """Configure output callable specification based on input."""
        self.output_callable = AttributeSpec(
            name=self.output_callable.name,
            field=ImageCallableField(
                semantic=self.input_frame.field.semantic,
                format=self.input_frame.field.format,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert video frame paths to image callables.

        Args:
            df: DataFrame containing video path and frame_index columns

        Returns:
            DataFrame with callable column
        """
        input_path_col = self.input_frame.name
        input_frame_idx_col = f"{self.input_frame.name}_frame_index"
        output_col = self.output_callable.name

        callables: list[Any] = []

        for path, frame_idx in zip(df[input_path_col], df[input_frame_idx_col]):
            if path is None or frame_idx is None:
                callables.append(None)
                continue

            # Create a closure that loads the frame on demand
            def make_loader(p: str, idx: int, fmt: str) -> callable:
                def load_frame() -> np.ndarray:
                    frame = LazyVideoFrame(
                        video_path=p,
                        frame_index=idx,
                        format=fmt,
                    )
                    return frame.data

                return load_frame

            callables.append(make_loader(str(path), frame_idx, self.input_frame.field.format))

        return df.with_columns(pl.Series(output_col, callables, dtype=pl.Object()))


@converter(lazy=True)
class MediaPathToImageConverter(Converter):
    """
    Lazy converter that loads images/video frames from MediaPathField to ImageField.

    Handles both standalone images and video frames uniformly:
    - For images (frame_index is None): loads using ImageCache
    - For video frames (frame_index is set): loads using VideoFrameCache

    This is the recommended converter for datasets using MediaPathField with
    the unified DetectionSample pattern.

    Input: MediaPathField (path + nullable frame_index)
    Output: ImageField (tensor data + shape)
    """

    input_media: AttributeSpec[MediaPathField]
    output_image: AttributeSpec[ImageField]

    # Supported output formats
    SUPPORTED_FORMATS = {"RGB", "BGR", "RGBA", "GRAY"}

    def filter_output_spec(self) -> bool:
        """Configure output image specification based on input."""
        output_format = self.output_image.field.format if self.output_image.field.format else "RGB"

        if output_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported output format '{output_format}' for MediaPathToImageConverter. "
                f"Supported formats are: {', '.join(sorted(self.SUPPORTED_FORMATS))}."
            )

        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_media.field.semantic,
                dtype=pl.UInt8(),
                format=output_format,
                channels_first=self.output_image.field.channels_first,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Load images or video frames and convert to image tensors.

        Checks frame_index to determine whether to load as image or video frame.

        Args:
            df: DataFrame containing path and frame_index columns

        Returns:
            DataFrame with loaded image data and shape information
        """
        input_path_col = self.input_media.name
        input_frame_idx_col = f"{self.input_media.name}_frame_index"
        output_col = self.output_image.name
        output_shape_col = f"{self.output_image.name}_shape"
        output_format = self.output_image.field.format

        image_data: list[Any] = []
        image_shapes: list[list[int]] = []

        for path, frame_idx in zip(df[input_path_col], df[input_frame_idx_col]):
            if path is None:
                image_data.append(None)
                image_shapes.append([])
                continue

            try:
                if frame_idx is not None:
                    # Video frame - use LazyVideoFrame
                    media = LazyVideoFrame(
                        video_path=str(path),
                        frame_index=frame_idx,
                        format=output_format,
                        channels_first=self.output_image.field.channels_first,
                    )
                else:
                    # Standalone image - use LazyImage
                    media = LazyImage(
                        path=str(path),
                        format=output_format,
                        channels_first=self.output_image.field.channels_first,
                    )

                img_array = media.data
                image_data.append(img_array.flatten().tolist())
                image_shapes.append(list(img_array.shape))
            except Exception:
                image_data.append(None)
                image_shapes.append([])

        return df.with_columns(
            pl.Series(output_col, image_data, dtype=pl.List(self.output_image.field.dtype)),
            pl.Series(output_shape_col, image_shapes, dtype=pl.List(pl.Int32())),
        )


@converter
class MediaPathToImageCallableConverter(Converter):
    """
    Converter that wraps MediaPathField as ImageCallableField.

    Creates a callable for each media item (image or video frame) that loads
    data on demand.

    Input: MediaPathField (path + nullable frame_index)
    Output: ImageCallableField (callable that returns image data)
    """

    input_media: AttributeSpec[MediaPathField]
    output_callable: AttributeSpec[ImageCallableField]

    def filter_output_spec(self) -> bool:
        """Configure output callable specification based on input."""
        self.output_callable = AttributeSpec(
            name=self.output_callable.name,
            field=ImageCallableField(
                semantic=self.input_media.field.semantic,
                format=self.input_media.field.format,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert media paths to image callables.

        Args:
            df: DataFrame containing path and frame_index columns

        Returns:
            DataFrame with callable column
        """
        input_path_col = self.input_media.name
        input_frame_idx_col = f"{self.input_media.name}_frame_index"
        output_col = self.output_callable.name

        callables: list[Any] = []

        for path, frame_idx in zip(df[input_path_col], df[input_frame_idx_col]):
            if path is None:
                callables.append(None)
                continue

            # Create a closure that loads the media on demand
            def make_loader(p: str, idx: int | None, fmt: str) -> callable:
                def load_media() -> np.ndarray:
                    if idx is not None:
                        media = LazyVideoFrame(video_path=p, frame_index=idx, format=fmt)
                    else:
                        media = LazyImage(path=p, format=fmt)
                    return media.data

                return load_media

            callables.append(make_loader(str(path), frame_idx, self.input_media.field.format))

        return df.with_columns(pl.Series(output_col, callables, dtype=pl.Object()))


@converter
class MediaPathToImagePathConverter(Converter):
    """
    Converter that converts MediaPathField to ImagePathField.

    Handles both standalone images and video frames:
    - For images (frame_index is None): casts the Categorical path to String.
    - For video frames (frame_index is set): extracts the frame from the video,
      saves it as a PNG image file, and outputs the saved image path.

    Extracted frames are saved to a deterministic path derived from the video
    file path and frame index, placed in a ``_extracted_frames`` directory next
    to the video file::

        /path/to/video_frames/video_frame000042.png

    If the extracted image already exists on disk it is not re-extracted,
    making repeated conversions cheap.

    Input: MediaPathField (path as Categorical + nullable frame_index)
    Output: ImagePathField (path as String)
    """

    input_media: AttributeSpec[MediaPathField]
    output_path: AttributeSpec[ImagePathField]

    def filter_output_spec(self) -> bool:
        """Configure output path specification based on input."""
        self.output_path = AttributeSpec(
            name=self.output_path.name,
            field=ImagePathField(
                semantic=self.input_media.field.semantic,
                format=self.input_media.field.format,
                channels_first=self.input_media.field.channels_first,
            ),
        )
        return True

    @staticmethod
    def _extract_frame(video_path: str, frame_index: int, output_format: str) -> str:
        """Extract a video frame and save it as a PNG image file.

        Args:
            video_path: Path to the video file.
            frame_index: Zero-based frame index to extract.
            output_format: Color format for loading (e.g. "RGB").

        Returns:
            Path to the saved image file.
        """
        from pathlib import Path as _Path

        from PIL import Image as _PILImage

        video_p = _Path(video_path)
        output_dir = video_p.parent / f"{video_p.stem}_frames"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_p.stem}_frame{frame_index:06d}.png"

        if output_path.exists():
            return str(output_path)

        frame = LazyVideoFrame(
            video_path=video_path,
            frame_index=frame_index,
            format=output_format,
        )
        img_array = frame.data
        _PILImage.fromarray(img_array).save(output_path)
        return str(output_path)

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert MediaPathField to ImagePathField.

        For image rows (null frame_index) the path is cast from Categorical to
        String.  For video frame rows the frame is extracted from the video,
        saved as a PNG file, and the resulting image path is used.

        Args:
            df: DataFrame containing MediaPathField columns (path + frame_index)

        Returns:
            DataFrame with a String path column for ImagePathField
        """
        input_col = self.input_media.name
        frame_idx_col = f"{input_col}_frame_index"
        output_col = self.output_path.name
        output_format = self.input_media.field.format

        has_frame_idx = frame_idx_col in df.columns
        has_video_frames = has_frame_idx and df.filter(pl.col(frame_idx_col).is_not_null()).height > 0

        if not has_video_frames:
            # Fast path — all rows are standalone images
            return df.with_columns(
                pl.col(input_col).cast(pl.String()).alias(output_col),
            )

        # Slow path — at least one video frame needs extraction
        output_paths: list[str | None] = []
        for path, frame_idx in zip(df[input_col], df[frame_idx_col]):
            if path is None:
                output_paths.append(None)
            elif frame_idx is not None:
                output_paths.append(self._extract_frame(str(path), int(frame_idx), output_format))
            else:
                output_paths.append(str(path))

        return df.with_columns(
            pl.Series(output_col, output_paths, dtype=pl.String()),
        )


@converter
class MediaInfoToImageInfoConverter(Converter):
    """
    Converter that extracts image info from MediaInfoField to ImageInfoField.

    MediaInfoField stores comprehensive metadata (width, height, source_path,
    fps, total_frames, duration, codec, frame_index) as a Polars struct.
    ImageInfoField stores only width and height as a simpler struct.

    This converter extracts the width and height fields from the MediaInfoField
    struct and produces an ImageInfoField struct.

    Input: MediaInfoField (struct with width, height, source_path, fps, ...)
    Output: ImageInfoField (struct with width, height)
    """

    input_info: AttributeSpec[MediaInfoField]
    output_info: AttributeSpec[ImageInfoField]

    def filter_output_spec(self) -> bool:
        """Configure output info specification based on input."""
        self.output_info = AttributeSpec(
            name=self.output_info.name,
            field=ImageInfoField(
                semantic=self.input_info.field.semantic,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract width and height from MediaInfoField struct to ImageInfoField struct.

        Args:
            df: DataFrame containing MediaInfoField column

        Returns:
            DataFrame with ImageInfoField column containing width and height
        """
        input_col = self.input_info.name
        output_col = self.output_info.name

        return df.with_columns(
            pl.struct(
                pl.col(input_col).struct.field("width").alias("width"),
                pl.col(input_col).struct.field("height").alias("height"),
            ).alias(output_col),
        )


@converter
class VideoFrameCallableToImageCallableConverter(Converter):
    """
    Converter that adapts VideoFrameCallableField to ImageCallableField.

    VideoFrameCallableField stores a callable that returns a video frame as a
    numpy array. This converter wraps it as an ImageCallableField, making video
    frame callables compatible with image processing pipelines.

    Input: VideoFrameCallableField (callable returning frame as numpy array)
    Output: ImageCallableField (callable returning image as numpy array)
    """

    input_callable: AttributeSpec[VideoFrameCallableField]
    output_callable: AttributeSpec[ImageCallableField]

    def filter_output_spec(self) -> bool:
        """Configure output callable specification based on input."""
        self.output_callable = AttributeSpec(
            name=self.output_callable.name,
            field=ImageCallableField(
                semantic=self.input_callable.field.semantic,
                format=self.input_callable.field.format,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Pass through video frame callables as image callables.

        Both field types store callables returning numpy arrays, so the
        data can be passed through directly.

        Args:
            df: DataFrame containing VideoFrameCallableField column

        Returns:
            DataFrame with ImageCallableField column
        """
        input_col = self.input_callable.name
        output_col = self.output_callable.name

        return df.with_columns(
            pl.col(input_col).alias(output_col),
        )


@converter
class ImagePathToMediaPathConverter(Converter):
    """
    Converter that promotes ImagePathField to MediaPathField.

    ImagePathField stores a simple String path. MediaPathField stores a
    Categorical path plus a nullable frame_index column. This converter
    casts the String path to Categorical and adds a null frame_index column,
    making image-only datasets compatible with the unified MediaPath format.

    Input: ImagePathField (path as String)
    Output: MediaPathField (path as Categorical + null frame_index)
    """

    input_path: AttributeSpec[ImagePathField]
    output_media: AttributeSpec[MediaPathField]

    def filter_output_spec(self) -> bool:
        """Configure output media path specification based on input."""
        self.output_media = AttributeSpec(
            name=self.output_media.name,
            field=MediaPathField(
                semantic=self.input_path.field.semantic,
                format=self.input_path.field.format,
                channels_first=self.input_path.field.channels_first,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Promote image path to media path with null frame_index.

        Args:
            df: DataFrame containing ImagePathField column

        Returns:
            DataFrame with MediaPathField columns (Categorical path + null frame_index)
        """
        input_col = self.input_path.name
        output_col = self.output_media.name
        frame_idx_col = f"{output_col}_frame_index"

        return df.with_columns(
            pl.col(input_col).cast(pl.Categorical()).alias(output_col),
            pl.lit(None, dtype=pl.UInt32()).alias(frame_idx_col),
        )


@converter
class ImageInfoToMediaInfoConverter(Converter):
    """
    Converter that promotes ImageInfoField to MediaInfoField.

    ImageInfoField stores a struct with {width, height}. MediaInfoField stores
    a richer struct with {width, height, source_path, fps, total_frames,
    duration, codec, frame_index}. This converter creates the MediaInfoField
    struct by copying width and height and filling video-specific fields with null.

    Input: ImageInfoField (struct with width, height)
    Output: MediaInfoField (struct with width, height, source_path=null, fps=null, ...)
    """

    input_info: AttributeSpec[ImageInfoField]
    output_info: AttributeSpec[MediaInfoField]

    def filter_output_spec(self) -> bool:
        """Configure output media info specification based on input."""
        self.output_info = AttributeSpec(
            name=self.output_info.name,
            field=MediaInfoField(
                semantic=self.input_info.field.semantic,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Promote image info to media info with null video-specific fields.

        Args:
            df: DataFrame containing ImageInfoField column

        Returns:
            DataFrame with MediaInfoField column
        """
        input_col = self.input_info.name
        output_col = self.output_info.name

        return df.with_columns(
            pl.struct(
                pl.col(input_col).struct.field("width").alias("width"),
                pl.col(input_col).struct.field("height").alias("height"),
                pl.lit(None, dtype=pl.String()).alias("source_path"),
                pl.lit(None, dtype=pl.Float32()).alias("fps"),
                pl.lit(None, dtype=pl.UInt32()).alias("total_frames"),
                pl.lit(None, dtype=pl.Float32()).alias("duration"),
                pl.lit(None, dtype=pl.String()).alias("codec"),
                pl.lit(None, dtype=pl.UInt32()).alias("frame_index"),
            ).alias(output_col),
        )
