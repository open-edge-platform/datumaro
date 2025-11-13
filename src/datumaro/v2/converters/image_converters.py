# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
from typing import Any

import numpy as np
import polars as pl
from PIL import Image

from datumaro.v2 import ImageCallableField, ImageField, ImageInfoField, ImagePathField, converter
from datumaro.v2.converters import Converter
from datumaro.v2.fields import ImageBytesField
from datumaro.v2.schema import AttributeSpec
from datumaro.v2.type_registry import polars_to_numpy_dtype


@converter
class RGBToBGRConverter(Converter):
    """
    Converter that transforms RGB image format to BGR format.

    This converter swaps the red and blue channels of RGB images to produce
    BGR format images, commonly used for OpenCV compatibility.
    """

    input_image: AttributeSpec[ImageField]
    output_image: AttributeSpec[ImageField]

    def filter_output_spec(self) -> bool:
        """
        Check if input is RGB and configure output for BGR conversion.

        Returns:
            True if the converter should be applied (RGB to BGR), False otherwise
        """
        input_format = self.input_image.field.format
        output_format = self.output_image.field.format

        # Configure output specification for BGR format
        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_image.field.semantic,
                dtype=self.input_image.field.dtype,
                channels_first=self.output_image.field.channels_first,
                format="BGR",  # Set output format to BGR
            ),
        )

        # Only apply if input is RGB and output should be BGR
        return input_format == "RGB" and output_format == "BGR"

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert RGB image format to BGR using numpy channel swapping.

        Args:
            df: Input DataFrame containing RGB image data

        Returns:
            DataFrame with BGR image data in the output column
        """
        input_column_name = self.input_image.name
        output_column_name = self.output_image.name

        input_shape_column_name = self.input_image.name + "_shape"
        output_shape_column_name = self.output_image.name + "_shape"

        def rgb_to_bgr(tensor_data: pl.Series) -> Any:
            """Convert RGB tensor data to BGR by reversing the channel order."""
            data = tensor_data.to_numpy().copy()
            data = data.reshape(-1, 3)
            data = np.flip(data, 1)  # Flip along channel axis
            return data.reshape(-1)

        dtype = df.schema[input_column_name]
        # Apply the conversion using map_elements for efficient processing
        return df.with_columns(
            pl.col(input_column_name).map_elements(rgb_to_bgr, return_dtype=dtype).alias(output_column_name),
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
                dtype=pl.Float32,
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
    Lazy converter that loads images from file paths using Pillow.

    This converter reads image files from disk and converts them to tensor format.
    It's marked as lazy to defer the expensive I/O operation until the data
    is actually accessed.
    """

    input_path: AttributeSpec[ImagePathField]
    output_image: AttributeSpec[ImageField]

    def filter_output_spec(self) -> bool:
        """Configure output image specification based on input."""
        # Configure output specification with default RGB format
        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_path.field.semantic,
                dtype=pl.UInt8,  # Default to UInt8 for loaded images
                format="RGB",  # Default to RGB format
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

        # Load images from paths
        image_data: list[Any] = []
        image_shapes: list[list[int]] = []

        for path in df[input_col]:
            # Load image using PIL
            with Image.open(path) as img:
                # Convert to RGB if needed
                rgb_img = img if img.mode == "RGB" else img.convert("RGB")

                # Convert to numpy array
                img_array = np.array(rgb_img, dtype=np.uint8)
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
                dtype=pl.UInt8,  # Default to UInt8 for decoded images
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
                dtype=pl.UInt8,  # Default to UInt8 for image data
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
