# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
from typing import Any

import cv2
import numpy as np
import polars as pl
from PIL import Image

from datumaro.experimental.converters.base import Converter, copy_columns_with_shape
from datumaro.experimental.converters.registry import converter
from datumaro.experimental.fields import ImageBytesField
from datumaro.experimental.fields.images import ImageCallableField, ImageField, ImageInfoField, ImagePathField
from datumaro.experimental.schema import AttributeSpec
from datumaro.experimental.type_registry import polars_to_numpy_dtype


@converter
class RedBlueColorConverter(Converter):
    """
    Converter that transforms RGB image format to BGR format and vice-versa.
    """

    input_image: AttributeSpec[ImageField]
    output_image: AttributeSpec[ImageField]

    def filter_output_spec(self) -> bool:
        """
        Check if input is RGB/BGR and configure output for BGR/RGB conversion.

        Returns:
            True if the converter should be applied (RGB to BGR/BGR to RGB), False otherwise
        """
        input_format = self.input_image.field.format
        output_format = self.output_image.field.format

        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_image.field.semantic,
                dtype=self.input_image.field.dtype,
                channels_first=self.output_image.field.channels_first,
                format=self.output_image.field.format,
            ),
        )

        return (input_format == "RGB" and output_format == "BGR") or (input_format == "BGR" and output_format == "RGB")

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        input_column_name = self.input_image.name
        output_column_name = self.output_image.name

        input_shape_column_name = self.input_image.name + "_shape"
        output_shape_column_name = self.output_image.name + "_shape"

        expected_dtype = polars_to_numpy_dtype(self.input_image.field.dtype)

        def red_blue_swap(flat_data: np.ndarray, shape: tuple) -> np.ndarray:
            """Fast channel swap using numpy views and advanced indexing."""
            if len(shape) == 3:
                h, w, c = shape
                if c == 3:
                    # RGB/BGR swap: swap channels 0 and 2
                    reshaped = flat_data.reshape(h, w, c)
                    swapped = np.empty_like(reshaped)
                    swapped[..., 0] = reshaped[..., 2]
                    swapped[..., 1] = reshaped[..., 1]
                    swapped[..., 2] = reshaped[..., 0]
                    return swapped.reshape(-1)
            # Fallback for non-standard shapes - reverse all channels
            reshaped = flat_data.reshape(shape)
            swapped = reshaped[..., ::-1].copy()
            return swapped.reshape(-1)

        result_data = []
        for i in range(len(df)):
            flat_list = df[input_column_name][i]
            shape_list = df[input_shape_column_name][i]
            flat_data = np.array(flat_list, dtype=expected_dtype)
            shape = tuple(shape_list)
            swapped = red_blue_swap(flat_data, shape)
            result_data.append(swapped.tolist())

        return df.with_columns(
            pl.Series(output_column_name, result_data, dtype=pl.List(self.input_image.field.dtype)),
            pl.col(input_shape_column_name).alias(output_shape_column_name),
        )


@converter
class UInt8ToFloat32Converter(Converter):
    """
    Convert image data from UInt8 to Float32 with normalization.

    This converter transforms 8-bit integer pixel values (0-255) to
    32-bit floating point values normalized to the range [0.0, 1.0].
    """

    input_image: AttributeSpec[ImageField]
    output_image: AttributeSpec[ImageField]

    def filter_output_spec(self) -> bool:
        """
        Check if input uses UInt8 dtype and configure Float32 output.

        Returns:
            True if the converter should be applied (UInt8 input), False otherwise
        """
        # Configure output specification for Float32 dtype
        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_image.field.semantic,
                dtype=pl.Float32(),
                format=self.input_image.field.format,
                channels_first=self.output_image.field.channels_first,
            ),
        )
        return self.input_image.field.dtype == pl.UInt8

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert image data from UInt8 to normalized Float32.

        Transforms pixel values from the range [0, 255] to [0.0, 1.0]
        by dividing by 255.0.

        Args:
            df: Input DataFrame containing UInt8 image data

        Returns:
            DataFrame with normalized Float32 image data
        """
        input_column_name = self.input_image.name
        output_column_name = self.output_image.name

        input_shape_column_name = self.input_image.name + "_shape"
        output_shape_column_name = self.output_image.name + "_shape"

        return df.with_columns(
            # Normalize UInt8 values (0-255) to Float32 (0.0-1.0)
            pl.col(input_column_name)
            .list.eval((pl.element() / 255.0).cast(self.output_image.field.dtype))
            .alias(output_column_name),
            pl.col(input_shape_column_name).alias(output_shape_column_name),
        )


@converter(lazy=True)
class ImagePathToImageConverter(Converter):
    """
    Lazy converter that loads images from file paths.

    This converter reads image files from disk and converts them to tensor format.
    It's marked as lazy to defer the expensive I/O operation until the data
    is actually accessed.

    Uses OpenCV for faster image loading, with PIL as fallback.
    """

    input_path: AttributeSpec[ImagePathField]
    output_image: AttributeSpec[ImageField]

    def filter_output_spec(self) -> bool:
        """Configure output image specification based on input."""
        # Configure output specification - preserve the requested format
        output_format = self.output_image.field.format if self.output_image.field.format else "RGB"
        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_path.field.semantic,
                dtype=pl.UInt8(),  # Default to UInt8 for loaded images
                format=output_format,
                channels_first=self.output_image.field.channels_first,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert image paths to loaded image tensors and image info.

        Args:
            df: DataFrame containing image path column

        Returns:
            DataFrame with loaded image data, shape information, and image info
        """
        input_col = self.input_path.name
        output_col = self.output_image.name
        output_format = self.output_image.field.format

        # Load images from paths
        image_data: list[Any] = []
        image_shapes: list[list[int]] = []

        for path in df[input_col]:
            img_array = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img_array is None:
                # Fallback to PIL for unsupported formats
                with Image.open(path) as img:
                    rgb_img = img if img.mode == "RGB" else img.convert("RGB")
                    img_array = np.array(rgb_img, dtype=np.uint8)
                    if output_format == "BGR":
                        img_array = img_array[..., ::-1].copy()
            elif output_format == "RGB":
                # cv2 loads as BGR, convert to RGB if needed
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            image_data.append(img_array.flatten())
            image_shapes.append(list(img_array.shape))

        # Create output DataFrame
        image_schema = self.output_image.field.to_polars_schema("image")

        result_df = df.clone()

        return result_df.with_columns(
            [
                pl.Series(output_col, image_data, dtype=image_schema["image"]),
                pl.Series(output_col + "_shape", image_shapes, dtype=image_schema["image_shape"]),
            ]
        )


@converter
class ImageToImageInfo(Converter):
    """
    Lazy converter that loads images from file paths using Pillow.

    This converter reads image files from disk and converts them to tensor format.
    It's marked as lazy to defer the expensive I/O operation until the data
    is actually accessed.
    """

    input_image: AttributeSpec[ImageField]
    output_info: AttributeSpec[ImageInfoField]

    def filter_output_spec(self) -> bool:
        """Configure output image specification based on input."""
        # Configure output info specification
        self.output_info = AttributeSpec(
            name=self.output_info.name,
            field=ImageInfoField(
                semantic=self.input_image.field.semantic,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert image paths to loaded image tensors and image info.

        Args:
            df: DataFrame containing image path column

        Returns:
            DataFrame with loaded image data, shape information, and image info
        """
        input_col = self.input_image.name + "_shape"
        output_col = self.output_info.name

        # Set image info
        return df.with_columns(
            pl.struct(
                pl.col(input_col).list.get(0).alias("height"),
                pl.col(input_col).list.get(1).alias("width"),
            ).alias(output_col)
        )


@converter(lazy=True)
class ImageBytesToImageConverter(Converter):
    """
    Lazy converter that decodes images from byte data.

    This converter takes encoded image bytes (PNG, JPEG, BMP, etc.) and decodes
    them to tensor format. It's marked as lazy to defer the expensive decoding
    operation until the data is actually accessed.
    """

    input_bytes: AttributeSpec[ImageBytesField]
    output_image: AttributeSpec[ImageField]

    def filter_output_spec(self) -> bool:
        """Configure output image specification based on input."""
        # Configure output specification with default RGB format
        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_bytes.field.semantic,
                dtype=pl.UInt8(),  # Default to UInt8 for decoded images
                format="RGB",  # Default to RGB format
                channels_first=self.output_image.field.channels_first,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert image bytes to decoded image tensors and image info.

        Args:
            df: DataFrame containing image bytes column

        Returns:
            DataFrame with decoded image data, shape information, and image info
        """
        input_col = self.input_bytes.name
        output_col = self.output_image.name

        # Decode images from bytes
        image_data: list[np.ndarray] = []
        image_shapes: list[list[int]] = []

        for image_bytes in df[input_col]:
            # Decode image using PIL
            from io import BytesIO

            with Image.open(BytesIO(image_bytes)) as img:
                # Convert to RGB if needed
                rgb_img = img if img.mode == "RGB" else img.convert("RGB")

                # Convert to numpy array
                img_array = np.array(rgb_img, dtype=np.uint8)
                image_data.append(img_array.reshape(-1))
                image_shapes.append(list(img_array.shape))

        # Create output DataFrame
        result_df = df.clone()
        return result_df.with_columns(
            [
                pl.Series(output_col, image_data),
                pl.Series(output_col + "_shape", image_shapes),
            ]
        )


@converter(lazy=True)
class ImageCallableToImageConverter(Converter):
    """
    Lazy converter that executes callables to generate image data.

    This converter takes a callable stored in an ImageCallableField,
    executes it to get image data as a numpy array, and produces both
    ImageField and ImageInfoField outputs.
    """

    input_callable: AttributeSpec[ImageCallableField]
    output_image: AttributeSpec[ImageField]

    def filter_output_spec(self) -> bool:
        """Configure output image and info specifications based on input."""
        # Configure output image specification
        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_callable.field.semantic,
                dtype=pl.UInt8(),  # Default to UInt8 for image data
                format=self.input_callable.field.format,  # Use format from callable field
                channels_first=self.output_image.field.channels_first,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Execute callables to generate image data and metadata.

        Args:
            df: DataFrame containing callable column

        Returns:
            DataFrame with image tensor data, shape information, and image info
        """
        input_col = self.input_callable.name
        output_col = self.output_image.name

        # Execute callables to generate image data
        image_data: list[Any] = []
        image_shapes: list[list[int]] = []

        for callable_obj in df[input_col]:
            try:
                # Execute the callable to get image array
                img_array = callable_obj()

                if img_array is None:
                    image_data.append(None)
                    image_shapes.append(None)
                    continue

                # Validate that we got a numpy array
                if not isinstance(img_array, np.ndarray):
                    raise TypeError(f"Callable must return numpy.ndarray, got {type(img_array)}")

                # Ensure the array has 3 dimensions for an image (height, width, channels)
                if len(img_array.shape) != 3:
                    raise ValueError(f"Image array must be 3D (height, width, channels), got shape {img_array.shape}")

                # Check that the array has the expected dtype (no conversion)
                expected_dtype = self.output_image.field.dtype
                expected_numpy_dtype = polars_to_numpy_dtype(expected_dtype)
                if img_array.dtype != expected_numpy_dtype:
                    raise TypeError(f"Expected {expected_numpy_dtype} image array, got {img_array.dtype}")
                # If no specific dtype checking needed, accept as-is

                # Store flattened image data and shape
                image_data.append(img_array.flatten())
                image_shapes.append(list(img_array.shape))

            except Exception as e:
                raise RuntimeError(f"Error executing callable for image generation: {e}") from e

        # Create output DataFrame
        result_df = df.clone()
        return result_df.with_columns(
            [
                pl.Series(output_col, image_data),
                pl.Series(output_col + "_shape", image_shapes),
            ]
        )


@converter
class ChannelsFirstConverter(Converter):
    """
    Converter that updates the channels_first metadata for image fields.

    This converter handles the transition between channels-first (C, H, W) and
    channels-last (H, W, C) format specifications. The actual data transposition
    is handled by the ImageField.from_polars() method when data is accessed,
    so this converter only needs to update the field metadata.

    Channels-first format: (C, H, W) - commonly used in PyTorch
    Channels-last format: (H, W, C) - commonly used in TensorFlow and image libraries
    """

    input_image: AttributeSpec[ImageField]
    output_image: AttributeSpec[ImageField]

    def filter_output_spec(self) -> bool:
        """
        Check if channels_first conversion is needed and configure output.

        Returns:
            True if the converter should be applied (channels_first differs), False otherwise
        """
        input_channels_first = self.input_image.field.channels_first
        output_channels_first = self.output_image.field.channels_first

        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_image.field.semantic,
                dtype=self.input_image.field.dtype,
                channels_first=output_channels_first,
                format=self.input_image.field.format,
            ),
        )

        # Apply converter only if channels_first status differs
        return input_channels_first != output_channels_first

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Update image field metadata for channels_first change.

        The actual transposition is handled by ImageField.from_polars(),
        so this just ensures the column names are properly mapped.
        """

        return copy_columns_with_shape(df, self.input_image.name, self.output_image.name)
