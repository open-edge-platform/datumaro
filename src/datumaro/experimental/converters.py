# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Converter implementations for data transformation between different schemas.

This module contains concrete converter implementations that handle various
data transformations such as format conversions, dtype conversions, and
multi-field transformations.
"""

from typing import Any, Callable

import numpy as np
import polars as pl
from PIL import Image

from .converter_registry import AttributeSpec, Converter, converter
from .fields import BBoxField, ImageField, ImagePathField


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

        def rgb_to_bgr(tensor_data: pl.Series) -> Any:
            """Convert RGB tensor data to BGR by reversing the channel order."""
            data = tensor_data.to_numpy().copy()
            data = data.reshape(-1, 3)
            data = np.flip(data, 1)  # Flip along channel axis
            return data.reshape(-1)

        dtype = df.schema[input_column_name]
        # Apply the conversion using map_elements for efficient processing
        return df.with_columns(
            pl.col(input_column_name)
            .map_elements(rgb_to_bgr, return_dtype=dtype)
            .alias(output_column_name)
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

        image_field = self.input_image.field
        print(f"Converting dtype from: {image_field.dtype}")
        print(f"Input attribute name: {self.input_image.name}")
        print(f"Output attribute name: {self.output_image.name}")

        return df.with_columns(
            # Normalize UInt8 values (0-255) to Float32 (0.0-1.0)
            pl.col(input_column_name)
            .list.eval((pl.element() / 255.0).cast(self.output_image.field.dtype))
            .alias(output_column_name)
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
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert image paths to loaded image tensors.

        Args:
            df: DataFrame containing image path column

        Returns:
            DataFrame with loaded image data and shape information
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
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Convert to numpy array
                img_array = np.array(img, dtype=np.uint8)
                image_data.append(img_array.flatten().tolist())
                image_shapes.append(list(img_array.shape))

        # Create output DataFrame
        result_df = df.clone()
        result_df = result_df.with_columns(
            [
                pl.Series(output_col, image_data),
                pl.Series(output_col + "_shape", image_shapes),
            ]
        )

        return result_df


def list_eval_ref(
    list_col: str,
    ref_col: str,
    op: Callable[[pl.Expr, pl.Expr], pl.Expr],
) -> pl.Expr:
    """
    Apply an operation element-wise between a list column and a reference column.

    This helper function enables operations between elements of a list column
    and values from a reference column, returning a new list column with
    the results.

    Args:
        list_col: Name of the list column
        ref_col: Name of the reference column
        op: Operation function to apply between list elements and reference values

    Returns:
        Polars expression for the computed list column

    Note:
        See https://github.com/pola-rs/polars/issues/7210 for implementation details
    """
    return pl.concat_list(pl.struct(list_col, ref_col)).list.eval(
        op(
            pl.element().struct.field(list_col).explode(),
            pl.element().struct.field(ref_col),
        )
    )


@converter
class BBoxCoordinateConverter(Converter):
    """
    Convert bounding box coordinates between normalized and absolute formats.

    This converter handles transformations between normalized coordinates
    (range [0,1]) and absolute pixel coordinates using image dimensions.
    """

    input_bbox: AttributeSpec[BBoxField]
    input_image: AttributeSpec[ImageField]
    output_bbox: AttributeSpec[BBoxField]

    def filter_output_spec(self) -> bool:
        """
        Check if bbox normalization conversion is needed and configure output.

        Returns:
            True if conversion is needed (normalization status differs), False otherwise
        """
        input_normalized = self.input_bbox.field.normalize
        output_normalized = self.output_bbox.field.normalize

        # Determine the target normalization from output specification
        target_normalized = output_normalized

        # Configure output specification with correct normalization
        self.output_bbox = AttributeSpec(
            name=self.output_bbox.name,
            field=BBoxField(
                semantic=self.input_bbox.field.semantic,
                dtype=self.input_bbox.field.dtype,
                format=self.input_bbox.field.format,
                normalize=target_normalized,
            ),
        )

        # Apply converter only if normalization status needs to change
        return input_normalized != target_normalized

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert bbox coordinates between normalized and absolute formats.

        Uses image dimensions to transform coordinates. For normalized to absolute:
        multiplies by image dimensions. For absolute to normalized: divides by
        image dimensions.

        Args:
            df: Input DataFrame containing bbox and image data

        Returns:
            DataFrame with converted bounding box coordinates
        """
        input_bbox_name = self.input_bbox.name
        input_image_name = self.input_image.name
        output_bbox_name = self.output_bbox.name

        # Get the image shape column name
        image_shape_name = f"{input_image_name}_shape"

        input_normalized = self.input_bbox.field.normalize

        # Create temporary column names for dimensions
        temp_width_col = f"{input_image_name}_width"
        temp_height_col = f"{input_image_name}_height"

        # Coordinate order for width/height mapping: [height, width, height, width]
        coordinates_order = [1, 0, 1, 0]

        def op(x: pl.Expr, y: pl.Expr) -> pl.Expr:
            """Choose operation based on conversion direction."""
            # FIXME: x.cast(pl.Float64) is a workaround for Polars bug
            # https://github.com/pola-rs/polars/issues/23924
            xy = x * y if input_normalized else x.cast(pl.Float64) / y
            return xy.cast(self.output_bbox.field.dtype)

        # Extract width and height from image shape
        df_with_temp = df.with_columns(
            [
                pl.col(image_shape_name).list.get(1).alias(temp_width_col),  # width
                pl.col(image_shape_name).list.get(0).alias(temp_height_col),  # height
            ]
        )

        # Apply coordinate transformation
        result_df = df_with_temp.with_columns(
            list_eval_ref(
                input_bbox_name,
                image_shape_name,
                lambda element, ref: pl.concat_arr(
                    op(element.arr.get(0), ref.list.get(coordinates_order[0])),  # x1
                    op(element.arr.get(1), ref.list.get(coordinates_order[1])),  # y1
                    op(element.arr.get(2), ref.list.get(coordinates_order[2])),  # x2
                    op(element.arr.get(3), ref.list.get(coordinates_order[3])),  # y2
                ),
            ).alias(output_bbox_name)
        )

        # Clean up temporary columns
        result_df = result_df.drop([temp_width_col, temp_height_col])

        return result_df
