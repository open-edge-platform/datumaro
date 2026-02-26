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

if TYPE_CHECKING:
    import numpy as np
from datumaro.experimental.converters.registry import converter
from datumaro.experimental.fields.images import ImageCallableField, ImageField
from datumaro.experimental.fields.videos import MediaPathField, VideoFramePathField
from datumaro.experimental.media import LazyImage, LazyVideoFrame
from datumaro.experimental.schema import AttributeSpec


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
