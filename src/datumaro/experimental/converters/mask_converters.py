# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
from typing import Any

import cv2
import numpy as np
import polars as pl

from datumaro.experimental.categories import LabelCategories, MaskCategories, RgbColor
from datumaro.experimental.converters.base import Converter
from datumaro.experimental.converters.registry import converter
from datumaro.experimental.fields.annotations import LabelField, PolygonField
from datumaro.experimental.fields.images import ImageInfoField
from datumaro.experimental.fields.masks import (
    InstanceMaskCallableField,
    InstanceMaskField,
    MaskCallableField,
    MaskField,
)
from datumaro.experimental.schema import AttributeSpec
from datumaro.experimental.type_registry import polars_to_numpy_dtype
from datumaro.util.mask_tools import generate_colormap


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
        if self.input_labels.categories is not None and isinstance(self.input_labels.categories, LabelCategories):
            # Create mask categories based on label categories
            # Create a colormap for mask categories
            # Generate colors for all labels plus background
            num_classes = len(self.input_labels.categories) + 1  # +1 for background
            colormap_dict = generate_colormap(num_classes, include_background=True)
            colormap_struct_dict = {i: RgbColor(*color) for i, color in colormap_dict.items()}

            # Create mask categories with the generated colormap
            labels = ("background", *self.input_labels.categories.labels)
            mask_categories = MaskCategories(colormap=colormap_struct_dict, labels=labels)

        # Configure output for mask format
        self.output_mask = AttributeSpec(
            name=self.output_mask.name,
            field=MaskField(
                semantic=self.input_polygon.field.semantic,
                dtype=self.output_mask.field.dtype,
                channels_first=self.output_mask.field.channels_first,
                has_channels_dim=self.output_mask.field.has_channels_dim,
            ),
            categories=mask_categories,
        )

        return True

    def convert(self, df: pl.DataFrame()) -> pl.DataFrame:
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

        def polygons_to_mask(polygons_data: list, labels_data: list, img_info: dict) -> tuple[list[int], list[int]]:
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
        def apply_conversion_batch(batch_df: pl.DataFrame()) -> pl.DataFrame:
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
                pl.Series(results_batch_shape, dtype=pl.List(pl.Int32())).alias("shape"),
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
            return_dtype=pl.Struct({"mask": pl.List(pl.UInt8()), "shape": pl.List(pl.Int32)}),
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

    def convert(self, df: pl.DataFrame()) -> pl.DataFrame:
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

        def polygons_to_instance_masks(polygons_data: list, img_info: dict) -> tuple[list[bool], list[int]]:
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
        def apply_conversion_batch(batch_df: pl.DataFrame, **kwargs) -> pl.DataFrame:  # noqa: ARG001
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
                pl.Series(results_batch_shape, dtype=pl.List(pl.Int32())).alias("shape"),
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
                {"mask": pl.List(self.output_instance_mask.field.dtype), "shape": pl.List(pl.Int32())}
            ),
        )

        return df.with_columns(
            [
                mask_data.struct.field("mask").alias(output_column_name),
                mask_data.struct.field("shape").alias(output_shape_column_name),
            ]
        )


@converter(lazy=True)
class InstanceMaskCallableToInstanceMaskConverter(Converter):
    """
    Lazy converter that executes callables to generate instance mask data.

    This converter takes a callable stored in a InstanceMaskCallableField and
    executes it to get instance mask data as a 3D numpy array (N,H,W), producing
    an InstanceMaskField output. Each mask in the output is a binary mask
    representing a single instance.
    """

    input_callable: AttributeSpec[InstanceMaskCallableField]
    output_mask: AttributeSpec[InstanceMaskField]

    def filter_output_spec(self) -> bool:
        """Configure output mask specification based on input."""
        self.output_mask = AttributeSpec(
            name=self.output_mask.name,
            field=InstanceMaskField(
                semantic=self.input_callable.field.semantic,
                dtype=self.input_callable.field.dtype,  # Use dtype from callable field
            ),
        )
        return True

    def convert(self, df: pl.DataFrame()) -> pl.DataFrame:
        """
        Execute callables to generate instance mask data.

        Args:
            df: DataFrame containing callable column

        Returns:
            DataFrame with instance mask tensor data and shape information
        """
        input_col = self.input_callable.name
        output_col = self.output_mask.name

        # Execute callables to generate mask data
        mask_data: list[Any] = []
        mask_shapes: list[list[int]] = []

        for callable_obj in df[input_col]:
            # Execute the callable to get instance mask array
            mask_array = callable_obj()

            if mask_array is None:
                mask_data.append(None)
                mask_shapes.append(None)
                continue

            # Validate that we got a numpy array
            if not isinstance(mask_array, np.ndarray):
                raise TypeError(f"Callable must return numpy.ndarray, got {type(mask_array)}")

            # Check array shape - should be 3D for instance masks (N, height, width)
            if len(mask_array.shape) != 3:
                raise ValueError(f"Instance mask array must be 3D (N,H,W), got shape {mask_array.shape}")

            # Check that the array has the expected dtype
            expected_dtype = self.output_mask.field.dtype
            expected_numpy_dtype = polars_to_numpy_dtype(expected_dtype)
            if mask_array.dtype != expected_numpy_dtype:
                raise TypeError(f"Expected {expected_numpy_dtype} mask array, got {mask_array.dtype}")

            # Store flattened mask data and shape
            mask_data.append(mask_array.flatten())
            mask_shapes.append(list(mask_array.shape))

        # Create output columns
        return df.with_columns(
            [
                pl.Series(output_col, mask_data),
                pl.Series(f"{output_col}_shape", mask_shapes),
            ]
        ).drop(input_col)


@converter(lazy=True)
class MaskCallableToMaskConverter(Converter):
    """
    Lazy converter that executes callables to generate mask data.

    This converter takes a callable stored in a MaskCallableField and
    executes it to get mask data as a 2D numpy array (H,W), producing
    a MaskField output. The mask can be either a binary mask or a
    category mask.
    """

    input_callable: AttributeSpec[MaskCallableField]
    output_mask: AttributeSpec[MaskField]

    def filter_output_spec(self) -> bool:
        """Configure output mask specification based on input."""
        self.output_mask = AttributeSpec(
            name=self.output_mask.name,
            field=MaskField(
                semantic=self.input_callable.field.semantic,
                dtype=self.input_callable.field.dtype,  # Use dtype from callable field
                channels_first=self.output_mask.field.channels_first,
                has_channels_dim=self.output_mask.field.has_channels_dim,
            ),
            categories=self.input_callable.categories,
        )
        return True

    def convert(self, df: pl.DataFrame()) -> pl.DataFrame:
        """
        Execute callables to generate mask data.

        Args:
            df: DataFrame containing callable column

        Returns:
            DataFrame with mask tensor data and shape information
        """
        input_col = self.input_callable.name
        output_col = self.output_mask.name

        # Execute callables to generate mask data
        mask_data: list[Any] = []
        mask_shapes: list[list[int]] = []

        for callable_obj in df[input_col]:
            # Execute the callable to get mask array
            mask_array = callable_obj()

            if mask_array is None:
                mask_data.append(None)
                mask_shapes.append(None)
                continue

            # Validate that we got a numpy array
            if not isinstance(mask_array, np.ndarray):
                raise TypeError(f"Callable must return numpy.ndarray, got {type(mask_array)}")

            # Check array shape - should be 2D for masks (height, width)
            if len(mask_array.shape) != 2:
                raise ValueError(f"Mask array must be 2D (H,W), got shape {mask_array.shape}")

            # Check that the array has the expected dtype
            expected_dtype = self.output_mask.field.dtype
            expected_numpy_dtype = polars_to_numpy_dtype(expected_dtype)
            if mask_array.dtype != expected_numpy_dtype:
                raise TypeError(f"Expected {expected_numpy_dtype} mask array, got {mask_array.dtype}")

            # Store flattened mask data and shape
            mask_data.append(mask_array.flatten())
            mask_shapes.append(list(mask_array.shape))

        # Create output columns
        return df.with_columns(
            [
                pl.Series(output_col, mask_data),
                pl.Series(f"{output_col}_shape", mask_shapes),
            ]
        ).drop(input_col)
