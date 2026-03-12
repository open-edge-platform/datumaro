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
        for flat_list, shape_list in zip(df[input_column_name].to_list(), df[input_shape_column_name].to_list()):
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

    Supported output formats:
        - RGB: 3-channel color image in Red-Green-Blue order
        - BGR: 3-channel color image in Blue-Green-Red order (OpenCV default)
        - RGBA: 4-channel color image with alpha channel
        - GRAY: Single-channel grayscale image
    """

    input_path: AttributeSpec[ImagePathField]
    output_image: AttributeSpec[ImageField]

    # Supported output formats for this converter
    SUPPORTED_FORMATS = {"RGB", "BGR", "RGBA", "GRAY"}

    # OpenCV read flags for each format
    _CV2_READ_FLAGS = {
        "RGB": cv2.IMREAD_COLOR,
        "BGR": cv2.IMREAD_COLOR,
        "RGBA": cv2.IMREAD_UNCHANGED,
        "GRAY": cv2.IMREAD_GRAYSCALE,
    }

    # PIL target modes for each format
    _PIL_MODES = {
        "RGB": "RGB",
        "BGR": "RGB",  # Load as RGB, then swap channels
        "RGBA": "RGBA",
        "GRAY": "L",
    }

    def filter_output_spec(self) -> bool:
        """Configure output image specification based on input."""
        # Configure output specification - preserve the requested format
        output_format = self.output_image.field.format if self.output_image.field.format else "RGB"

        # Validate that the requested format is supported
        if output_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported output format '{output_format}' for ImagePathToImageConverter. "
                f"Supported formats are: {', '.join(sorted(self.SUPPORTED_FORMATS))}."
            )

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

    def _load_image_opencv(self, path: str, output_format: str) -> np.ndarray | None:
        """
        Load an image using OpenCV and convert to the requested format.

        Args:
            path: Path to the image file
            output_format: Desired output format (RGB, BGR, RGBA, GRAY)

        Returns:
            Image as numpy array in the requested format, or None if loading failed
        """
        read_flag = self._CV2_READ_FLAGS[output_format]
        img_array = cv2.imread(path, read_flag)

        if img_array is None:
            return None

        # Handle format conversions
        if output_format == "RGB":
            # OpenCV loads as BGR, convert to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif output_format == "BGR":
            # Already in BGR format, no conversion needed
            pass
        elif output_format == "RGBA":
            # Handle various source formats for RGBA output
            if len(img_array.shape) == 2:
                # Grayscale to RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGBA)
            elif img_array.shape[2] == 3:
                # BGR to RGBA (OpenCV loads as BGR)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGBA)
            elif img_array.shape[2] == 4:
                # BGRA to RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA)
        elif output_format == "GRAY" and len(img_array.shape) == 2:
            img_array = img_array[:, :, np.newaxis]

        return img_array

    def _load_image_pil(self, path: str, output_format: str) -> np.ndarray:
        """
        Load an image using PIL and convert to the requested format.

        Args:
            path: Path to the image file
            output_format: Desired output format (RGB, BGR, RGBA, GRAY)

        Returns:
            Image as numpy array in the requested format
        """
        target_mode = self._PIL_MODES[output_format]

        with Image.open(path) as img:
            # Convert to target mode if needed
            source_img = img if img.mode == target_mode else img.convert(target_mode)

            img_array = np.array(source_img, dtype=np.uint8)

            # Handle BGR format (swap R and B channels)
            if output_format == "BGR":
                img_array = img_array[..., ::-1].copy()

            # Handle grayscale - ensure 3D shape (H, W, 1)
            if output_format == "GRAY" and len(img_array.shape) == 2:
                img_array = img_array[:, :, np.newaxis]

        return img_array

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
            path_str = str(path)

            # Try OpenCV first (faster)
            img_array = self._load_image_opencv(path_str, output_format)

            if img_array is None:
                # Fallback to PIL for unsupported formats
                img_array = self._load_image_pil(path_str, output_format)

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


@converter
class ImagePathToImageInfoConverter(Converter):
    """
    Converter that reads image dimensions from file paths without loading pixel data.

    This is much more efficient than the alternative path of
    ImagePathField → ImageField → ImageInfoField, which requires loading
    the entire image into memory. This converter only reads image headers
    to extract width and height using PIL.

    Input: ImagePathField (path as String)
    Output: ImageInfoField (struct with width, height)
    """

    input_path: AttributeSpec[ImagePathField]
    output_info: AttributeSpec[ImageInfoField]

    def filter_output_spec(self) -> bool:
        """Configure output info specification based on input."""
        self.output_info = AttributeSpec(
            name=self.output_info.name,
            field=ImageInfoField(
                semantic=self.input_path.field.semantic,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Read image dimensions from file paths using PIL header-only reads.

        Args:
            df: DataFrame containing image path column

        Returns:
            DataFrame with ImageInfoField column containing width and height
        """
        input_col = self.input_path.name
        output_col = self.output_info.name

        widths: list[int | None] = []
        heights: list[int | None] = []

        for path in df[input_col]:
            if path is None:
                widths.append(None)
                heights.append(None)
                continue

            try:
                with Image.open(str(path)) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                widths.append(None)
                heights.append(None)

        return df.with_columns(
            pl.struct(
                pl.Series("width", widths, dtype=pl.Int32()),
                pl.Series("height", heights, dtype=pl.Int32()),
            ).alias(output_col),
        )


@converter
class ImagePathToImageCallableConverter(Converter):
    """
    Converter that wraps ImagePathField as ImageCallableField.

    Creates a callable for each image path that loads the image on demand.
    This is useful for pipeline compatibility where a callable interface
    is expected rather than a file path.

    Input: ImagePathField (path as String)
    Output: ImageCallableField (callable that returns image data as numpy array)
    """

    input_path: AttributeSpec[ImagePathField]
    output_callable: AttributeSpec[ImageCallableField]

    def filter_output_spec(self) -> bool:
        """Configure output callable specification based on input."""
        self.output_callable = AttributeSpec(
            name=self.output_callable.name,
            field=ImageCallableField(
                semantic=self.input_path.field.semantic,
                format=self.input_path.field.format,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Wrap image paths as lazy-loading callables.

        Args:
            df: DataFrame containing image path column

        Returns:
            DataFrame with callable column that loads images on demand
        """
        from datumaro.experimental.media import LazyImage

        input_col = self.input_path.name
        output_col = self.output_callable.name

        callables: list[Any] = []

        for path in df[input_col]:
            if path is None:
                callables.append(None)
                continue

            def make_loader(p: str, fmt: str) -> callable:
                def load_image() -> np.ndarray:
                    return LazyImage(path=p, format=fmt).data

                return load_image

            callables.append(make_loader(str(path), self.input_path.field.format))

        return df.with_columns(pl.Series(output_col, callables, dtype=pl.Object()))
