# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Converter implementations for data transformation between different schemas.

This module contains concrete converter implementations that handle various
data transformations such as format conversions, dtype conversions, and
multi-field transformations.
"""

from typing import Any, Callable

import cv2
import numpy as np
import polars as pl
from PIL import Image

from datumaro.util.mask_tools import generate_colormap

from .categories import LabelCategories, MaskCategories, RgbColor
from .converter_registry import AttributeSpec, Converter, converter
from .fields import (
    BBoxField,
    ImageBytesField,
    ImageCallableField,
    ImageField,
    ImageInfo,
    ImageInfoField,
    ImagePathField,
    InstanceMaskField,
    LabelField,
    MaskField,
    PolygonField,
    RotatedBBoxField,
)
from .type_registry import polars_to_numpy_dtype


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
            pl.col(input_column_name)
            .map_elements(rgb_to_bgr, return_dtype=dtype)
            .alias(output_column_name),
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
    output_info: AttributeSpec[ImageInfoField]

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
        # Configure output info specification
        self.output_info = AttributeSpec(
            name=self.output_info.name,
            field=ImageInfoField(
                semantic=self.input_path.field.semantic,
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
        output_info_col = self.output_info.name

        # Load images from paths
        image_data: list[Any] = []
        image_shapes: list[list[int]] = []
        image_infos: list[ImageInfo] = []

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

                # Create image info with just width and height
                image_infos.append(ImageInfo(width=img_array.shape[1], height=img_array.shape[0]))

        # Create output DataFrame
        result_df = df.clone()
        result_df = result_df.with_columns(
            [
                pl.Series(output_col, image_data),
                pl.Series(output_col + "_shape", image_shapes),
                pl.Series(output_info_col, image_infos),
            ]
        )

        return result_df


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
    output_info: AttributeSpec[ImageInfoField]

    def filter_output_spec(self) -> bool:
        """Configure output image specification based on input."""
        # Configure output specification with default RGB format
        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_bytes.field.semantic,
                dtype=pl.UInt8,  # Default to UInt8 for decoded images
                format="RGB",  # Default to RGB format
            ),
        )
        # Configure output info specification
        self.output_info = AttributeSpec(
            name=self.output_info.name,
            field=ImageInfoField(
                semantic=self.input_bytes.field.semantic,
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
        output_info_col = self.output_info.name

        # Decode images from bytes
        image_data: list[np.ndarray] = []
        image_shapes: list[list[int]] = []
        image_infos: list[ImageInfo] = []

        for image_bytes in df[input_col]:
            # Decode image using PIL
            from io import BytesIO

            with Image.open(BytesIO(image_bytes)) as img:
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Convert to numpy array
                img_array = np.array(img, dtype=np.uint8)
                image_data.append(img_array.reshape(-1))
                image_shapes.append(list(img_array.shape))

                # Create image info with just width and height
                image_infos.append(ImageInfo(width=img_array.shape[1], height=img_array.shape[0]))

        # Create output DataFrame
        result_df = df.clone()
        result_df = result_df.with_columns(
            [
                pl.Series(output_col, image_data),
                pl.Series(output_col + "_shape", image_shapes),
                pl.Series(output_info_col, image_infos),
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


@converter(lazy=True)
class PolygonToMaskConverter(Converter):
    """
    Converts polygon annotations to rasterized masks.

    Transforms polygon coordinates into binary or indexed masks using
    OpenCV contour filling for efficient rasterization.
    """

    input_polygon: AttributeSpec[PolygonField]
    input_labels: AttributeSpec[LabelField]
    input_image_info: AttributeSpec[ImageInfoField]
    output_mask: AttributeSpec[MaskField]

    # Configuration options
    background_index: int = 0  # Background value

    def filter_output_spec(self) -> bool:
        """
        Configure mask output specification.

        Returns:
            True if the converter should be applied, False otherwise
        """

        # Copy label categories and create mask categories
        mask_categories = None
        if self.input_labels.categories is not None:
            # Create mask categories based on label categories
            if isinstance(self.input_labels.categories, LabelCategories):
                # Create a colormap for mask categories
                # Generate colors for all labels plus background
                num_classes = len(self.input_labels.categories) + 1  # +1 for background
                colormap_dict = generate_colormap(num_classes, include_background=True)

                # Create mask categories with the generated colormap
                mask_categories = MaskCategories()
                for index, color in colormap_dict.items():
                    if isinstance(color, tuple):
                        mask_categories.colormap[index] = RgbColor(*color)
                    else:
                        mask_categories.colormap[index] = color

                mask_categories.labels = ["background"] + self.input_labels.categories.labels

        # Configure output for mask format
        self.output_mask = AttributeSpec(
            name=self.output_mask.name,
            field=MaskField(
                semantic=self.input_polygon.field.semantic,
                dtype=self.output_mask.field.dtype,
            ),
            categories=mask_categories,
        )

        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Rasterize polygon coordinates into indexed masks.

        Args:
            df: DataFrame with polygon coordinates, labels, and image info

        Returns:
            DataFrame with mask data in output column
        """
        input_column_name = self.input_polygon.name
        labels_column_name = self.input_labels.name
        image_info_column_name = self.input_image_info.name
        output_column_name = self.output_mask.name
        output_shape_column_name = self.output_mask.name + "_shape"

        def polygons_to_mask(
            polygons_data: list, labels_data: list, img_info: dict
        ) -> tuple[list[int], list[int]]:
            """Rasterize polygons into indexed mask using OpenCV contour filling.

            The mask uses:
            - Index 0: Background (empty areas)
            - Index 1+: Polygon class labels (shifted by 1 to reserve 0 for background)
            """
            # Extract image dimensions
            image_width = img_info["width"]
            image_height = img_info["height"]

            # Initialize mask with background index
            numpy_dtype = polars_to_numpy_dtype(self.output_mask.field.dtype)
            mask = np.full(
                shape=(image_height, image_width),
                fill_value=self.background_index,
                dtype=numpy_dtype,
            )

            # Rasterize each polygon
            for i, polygon_data in enumerate(polygons_data):
                coords = polygon_data.to_numpy()
                class_index = labels_data[i]

                # Denormalize coordinates if needed
                if self.input_polygon.field.normalize:
                    coords = coords.copy()
                    coords[:, 0] *= image_width
                    coords[:, 1] *= image_height

                # Convert to OpenCV contour format
                contour = coords.astype(np.int32)

                # Fill polygon with class index + 1 (to reserve 0 for background)
                # This means label 0 becomes mask index 1, label 1 becomes mask index 2, etc.
                cv2.drawContours(
                    mask,
                    [contour],
                    0,
                    int(class_index) + 1,  # +1 to shift labels and reserve 0 for background
                    thickness=cv2.FILLED,
                )

            return mask.reshape(-1), [image_height, image_width]

        # Apply conversion using map_batches
        def apply_conversion_batch(batch_df: pl.DataFrame) -> pl.DataFrame:
            """Apply polygon-to-mask conversion for a batch."""
            batch_polygons = batch_df.struct["polygons"]
            batch_labels = batch_df.struct["labels"]
            batch_img_infos = batch_df.struct["img_info"]

            results_batch_polygons = []
            results_batch_shape = []
            for polygons, labels, img_infos in zip(batch_polygons, batch_labels, batch_img_infos):
                mask_data, shape_data = polygons_to_mask(polygons, labels, img_infos)
                results_batch_polygons.append(pl.Series(mask_data))
                results_batch_shape.append(shape_data)

            return pl.struct(
                pl.Series(results_batch_polygons).alias("mask"),
                pl.Series(results_batch_shape, dtype=pl.List(pl.Int32)).alias("shape"),
                eager=True,
            )

        mask_data = pl.struct(
            [
                pl.col(input_column_name).alias("polygons"),
                pl.col(labels_column_name).alias("labels"),
                pl.col(image_info_column_name).alias("img_info"),
            ]
        ).map_batches(
            apply_conversion_batch,
            return_dtype=pl.Struct({"mask": pl.List(pl.UInt8), "shape": pl.List(pl.Int32)}),
        )

        return df.with_columns(
            [
                mask_data.struct.field("mask").alias(output_column_name),
                mask_data.struct.field("shape").alias(output_shape_column_name),
            ]
        )


@converter(lazy=True)
class PolygonToInstanceMaskConverter(Converter):
    """
    Converts polygon annotations to instance masks.

    Transforms polygon coordinates into binary instance masks of shape (N, H, W)
    where N is the number of instances. Each mask represents a single instance
    without category information.
    """

    input_polygon: AttributeSpec[PolygonField]
    input_image_info: AttributeSpec[ImageInfoField]
    output_instance_mask: AttributeSpec[InstanceMaskField]

    def filter_output_spec(self) -> bool:
        """Configure output specification for instance mask format."""
        # Configure output for instance mask format
        self.output_instance_mask = AttributeSpec(
            name=self.output_instance_mask.name,
            field=InstanceMaskField(
                semantic=self.input_polygon.field.semantic,
                dtype=self.output_instance_mask.field.dtype,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Rasterize polygon coordinates into instance masks.

        Args:
            df: DataFrame with polygon coordinates and image info

        Returns:
            DataFrame with instance mask data in output column
        """
        input_column_name = self.input_polygon.name
        image_info_column_name = self.input_image_info.name
        output_column_name = self.output_instance_mask.name
        output_shape_column_name = self.output_instance_mask.name + "_shape"

        def polygons_to_instance_masks(
            polygons_data: list, img_info: dict
        ) -> tuple[list[bool], list[int]]:
            """Rasterize polygons into instance masks using OpenCV contour filling."""
            # Extract image dimensions
            image_width = img_info["width"]
            image_height = img_info["height"]

            # Convert dtype - use uint8 for OpenCV, then convert to bool
            numpy_dtype = polars_to_numpy_dtype(self.output_instance_mask.field.dtype)

            if len(polygons_data) == 0:
                # No polygons, return empty mask with shape (0, H, W)
                empty_mask = np.array([], dtype=numpy_dtype)
                return empty_mask.tolist(), [0, image_height, image_width]

            # Create instance masks for each polygon
            instance_masks = []

            for polygon_data in polygons_data:
                coords = polygon_data.to_numpy()

                # Initialize mask for this instance (use uint8 for OpenCV compatibility)
                mask = np.zeros((image_height, image_width), dtype=np.uint8)

                # Denormalize coordinates if needed
                if self.input_polygon.field.normalize:
                    coords = coords.copy()
                    coords[:, 0] *= image_width
                    coords[:, 1] *= image_height

                # Convert to OpenCV contour format
                contour = coords.astype(np.int32)

                # Fill polygon with 1 for instance mask
                cv2.drawContours(
                    mask,
                    [contour],
                    0,
                    1,  # Fill with 1 for binary instance mask
                    thickness=cv2.FILLED,
                )

                # Convert to the target dtype (e.g., bool)
                mask = mask.astype(numpy_dtype)
                instance_masks.append(mask)

            # Stack into (N, H, W) tensor
            stacked_masks = np.stack(instance_masks, axis=0)
            return stacked_masks.reshape(-1), list(stacked_masks.shape)

        # Apply conversion using map_batches
        def apply_conversion_batch(batch_df: pl.DataFrame, **kwargs) -> pl.DataFrame:
            """Apply polygon-to-instance-mask conversion for a batch."""
            batch_polygons = batch_df.struct["polygons"]
            batch_img_infos = batch_df.struct["img_info"]

            results_batch_mask = []
            results_batch_shape = []

            for polygons, img_info in zip(batch_polygons, batch_img_infos):
                mask_data, shape_data = polygons_to_instance_masks(polygons, img_info)
                results_batch_mask.append(pl.Series(mask_data))
                results_batch_shape.append(shape_data)

            return pl.struct(
                pl.Series(results_batch_mask).alias("mask"),
                pl.Series(results_batch_shape, dtype=pl.List(pl.Int32)).alias("shape"),
                eager=True,
            )

        mask_data = pl.struct(
            [
                pl.col(input_column_name).alias("polygons"),
                pl.col(image_info_column_name).alias("img_info"),
            ]
        ).map_batches(
            apply_conversion_batch,
            return_dtype=pl.Struct(
                {"mask": pl.List(self.output_instance_mask.field.dtype), "shape": pl.List(pl.Int32)}
            ),
        )

        return df.with_columns(
            [
                mask_data.struct.field("mask").alias(output_column_name),
                mask_data.struct.field("shape").alias(output_shape_column_name),
            ]
        )


@converter
class PolygonToBBoxConverter(Converter):
    """
    Converts polygon annotations to bounding boxes.

    Extracts the bounding box coordinates that enclose each polygon.
    """

    input_polygon: AttributeSpec[PolygonField]
    output_bbox: AttributeSpec[BBoxField]

    def filter_output_spec(self) -> bool:
        """Configure output specification for bounding box format."""
        # Configure output for bbox format
        self.output_bbox = AttributeSpec(
            name=self.output_bbox.name,
            field=BBoxField(
                semantic=self.input_polygon.field.semantic,
                dtype=self.input_polygon.field.dtype,
                format=self.output_bbox.field.format,
                normalize=self.input_polygon.field.normalize,  # Inherit normalization from polygon
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract bounding boxes from polygon coordinates.

        Args:
            df: DataFrame with polygon coordinates

        Returns:
            DataFrame with bounding box data in output column
        """
        input_column_name = self.input_polygon.name
        output_column_name = self.output_bbox.name

        df = df.with_columns(
            pl.col(input_column_name)
            .list.eval(
                pl.concat_arr(
                    [
                        pl.element().list.eval(pl.element().arr.get(0)).list.min(),
                        pl.element().list.eval(pl.element().arr.get(1)).list.min(),
                        pl.element().list.eval(pl.element().arr.get(0)).list.max(),
                        pl.element().list.eval(pl.element().arr.get(1)).list.max(),
                    ]
                )
            )
            .alias(output_column_name)
        )

        # Format according to output bbox format
        if self.output_bbox.field.format == "x1y1x2y2":
            # Already in this format
            pass
        elif self.output_bbox.field.format == "xywh":
            df = df.with_columns(
                pl.col(output_column_name).list.eval(
                    pl.concat_arr(
                        [
                            pl.element().arr.get(0),
                            pl.element().arr.get(1),
                            pl.element().arr.get(2) - pl.element().arr.get(0),
                            pl.element().arr.get(3) - pl.element().arr.get(1),
                        ]
                    )
                )
            )
        else:
            raise NotImplementedError(
                f"This conversion is not yet implemented "
                f"for the format {self.output_bbox.field.format}."
            )

        return df


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
    output_info: AttributeSpec[ImageInfoField]

    def filter_output_spec(self) -> bool:
        """Configure output image and info specifications based on input."""
        # Configure output image specification
        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_callable.field.semantic,
                dtype=pl.UInt8,  # Default to UInt8 for image data
                format=self.input_callable.field.format,  # Use format from callable field
            ),
        )
        # Configure output info specification
        self.output_info = AttributeSpec(
            name=self.output_info.name,
            field=ImageInfoField(
                semantic=self.input_callable.field.semantic,
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
        output_info_col = self.output_info.name

        # Execute callables to generate image data
        image_data: list[Any] = []
        image_shapes: list[list[int]] = []
        image_infos: list[ImageInfo] = []

        for callable_obj in df[input_col]:
            try:
                # Execute the callable to get image array
                img_array = callable_obj()

                # Validate that we got a numpy array
                if not isinstance(img_array, np.ndarray):
                    raise TypeError(f"Callable must return numpy.ndarray, got {type(img_array)}")

                # Ensure the array has 3 dimensions for an image (height, width, channels)
                if len(img_array.shape) != 3:
                    raise ValueError(
                        f"Image array must be 3D (height, width, channels), got shape {img_array.shape}"
                    )

                # Check that the array has the expected dtype (no conversion)
                expected_dtype = self.output_image.field.dtype
                expected_numpy_dtype = polars_to_numpy_dtype(expected_dtype)
                if img_array.dtype != expected_numpy_dtype:
                    raise TypeError(
                        f"Expected {expected_numpy_dtype} image array, got {img_array.dtype}"
                    )
                # If no specific dtype checking needed, accept as-is

                # Store flattened image data and shape
                image_data.append(img_array.flatten())
                image_shapes.append(list(img_array.shape))

                # Create image info with width and height from 3D array
                height, width = img_array.shape[:2]
                image_infos.append(ImageInfo(width=width, height=height))

            except Exception as e:
                raise RuntimeError(f"Error executing callable for image generation: {e}") from e

        # Create output DataFrame
        result_df = df.clone()
        result_df = result_df.with_columns(
            [
                pl.Series(output_col, image_data),
                pl.Series(output_col + "_shape", image_shapes),
                pl.Series(output_info_col, image_infos),
            ]
        )

        return result_df


@converter
class RotatedBBoxToPolygonConverter(Converter):
    """
    Converts rotated bounding boxes to polygon coordinates.

    Transforms rotated bounding box parameters (cx, cy, w, h, r) into
    polygon corner points by rotating the rectangle corners around the center.
    """

    input_rotated_bbox: AttributeSpec[RotatedBBoxField]
    output_polygon: AttributeSpec[PolygonField]

    def filter_output_spec(self) -> bool:
        """Configure output specification for polygon format."""
        # Configure output for polygon format
        self.output_polygon = AttributeSpec(
            name=self.output_polygon.name,
            field=PolygonField(
                semantic=self.input_rotated_bbox.field.semantic,
                dtype=self.input_rotated_bbox.field.dtype,
                format=self.output_polygon.field.format,
                normalize=self.input_rotated_bbox.field.normalize,  # Inherit normalization
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert rotated bounding boxes to polygon corner points.

        Args:
            df: DataFrame with rotated bounding box coordinates

        Returns:
            DataFrame with polygon data in output column
        """
        input_column_name = self.input_rotated_bbox.name
        output_column_name = self.output_polygon.name

        cx = pl.element().arr.get(0)
        cy = pl.element().arr.get(1)
        w = pl.element().arr.get(2)
        h = pl.element().arr.get(3)
        r = pl.element().arr.get(4)

        def rotate_corner(expr: pl.Expr):
            px = expr.arr.get(0)
            py = expr.arr.get(1)
            cos_theta = r.cos()
            sin_theta = r.sin()
            return pl.concat_arr(
                cos_theta * px - sin_theta * py + cx, sin_theta * px + cos_theta * py + cy
            )

        df = df.with_columns(
            pl.col(input_column_name)
            .list.eval(
                pl.concat_list(
                    rotate_corner(pl.concat_arr(-w / 2, -h / 2)),
                    rotate_corner(pl.concat_arr(w / 2, -h / 2)),
                    rotate_corner(pl.concat_arr(w / 2, h / 2)),
                    rotate_corner(pl.concat_arr(-w / 2, h / 2)),
                )
            )
            .alias(output_column_name)
        )

        return df
