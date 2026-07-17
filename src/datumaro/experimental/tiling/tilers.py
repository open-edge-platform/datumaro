# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
Implementations of tilers for specific field types.
"""

import operator
from typing import Any

import numpy as np
import polars as pl
from shapely import GeometryCollection, MultiPolygon, box, transform
from shapely import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry

from datumaro.experimental.fields import (
    BBoxField,
    ImageField,
    ImageInfoField,
    InstanceMaskField,
    LabelField,
    MaskField,
    PolygonField,
    SubsetField,
)
from datumaro.experimental.schema import AttributeSpec

from .tiler_registry import Tiler, TilerRegistry


@TilerRegistry.register(SubsetField)
class PassthroughTiler(Tiler):
    """Tiler for fields which do not require any changes (e.g. subset)."""

    field_spec: AttributeSpec[Any]

    def tile(self, df: pl.DataFrame, tiles_df: pl.DataFrame, slice_offset: int = 0) -> pl.DataFrame:
        """Process labels, adding keep column for list fields."""
        column_name = self.field_spec.name
        source_sample_idx = (
            tiles_df.select(pl.col("tile").struct["source_sample_idx"])["source_sample_idx"] - slice_offset
        )

        # Just a passthrough of the data
        return pl.DataFrame({column_name: df[column_name].gather(source_sample_idx)})


@TilerRegistry.register(MaskField)
class MaskTiler(Tiler):
    """Tiler for semantic segmentation masks.

    Extracts the corresponding region from the mask for each tile.
    The mask values (class labels) are preserved as is.
    """

    field_spec: AttributeSpec[MaskField]

    def tile(self, df: pl.DataFrame, tiles_df: pl.DataFrame, slice_offset: int = 0) -> pl.DataFrame:
        """Extract mask regions for each tile."""
        column_name = self.field_spec.name
        shape_column = f"{column_name}_shape"
        results_data = []
        results_shape = []

        for tile_row in tiles_df["tile"]:
            image_id = tile_row["source_sample_idx"] - slice_offset
            mask_data = df[column_name][image_id]
            mask_shape = df[shape_column][image_id]

            # Get tile coordinates
            x = tile_row["x"]
            y = tile_row["y"]
            width = tile_row["width"]
            height = tile_row["height"]

            # Reshape flattened data and extract tile
            mask = mask_data.reshape(mask_shape).to_numpy()
            tile_mask = mask[y : y + height, x : x + width]

            # Return flattened tile
            results_data.append(tile_mask.reshape(-1))
            results_shape.append(list(tile_mask.shape))

        return pl.DataFrame({column_name: results_data, shape_column: results_shape})


def _instance_intersects_tile(instance: np.ndarray, x: int, y: int, width: int, height: int) -> bool:
    """Return whether an instance mask's bounding box intersects the tile.

    The criterion mirrors :class:`BboxTiler` exactly: using the instance's
    full-image bounding box ``(x1, y1, x2, y2)`` (derived from its non-zero
    pixels, with ``x2``/``y2`` exclusive), the instance is kept iff::

        (x2 > tile_x) & (x1 < tile_x2) & (y2 > tile_y) & (y1 < tile_y2)

    An all-zero mask (no pixels) has no bounding box and is dropped.

    Args:
        instance: Full-image instance mask of shape ``(H, W)``.
        x: Tile left coordinate.
        y: Tile top coordinate.
        width: Tile width.
        height: Tile height.

    Returns:
        ``True`` if the instance overlaps the tile rectangle, else ``False``.
    """
    rows = np.nonzero(instance.any(axis=1))[0]
    cols = np.nonzero(instance.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return False
    y1, y2 = int(rows[0]), int(rows[-1]) + 1
    x1, x2 = int(cols[0]), int(cols[-1]) + 1
    return bool((x2 > x) and (x1 < x + width) and (y2 > y) and (y1 < y + height))


@TilerRegistry.register(InstanceMaskField)
class InstanceMaskTiler(Tiler):
    """Tiler for instance segmentation masks.

    Extracts the corresponding region from each instance mask and prunes
    instances that do not overlap the tile.

    Instance masks are stored as a flattened array plus a separate shape column,
    a format that the coordinated ``keep``-column filtering mechanism (used by
    :class:`BboxTiler` and :class:`LabelTiler`) cannot express. To keep masks
    aligned with the independently filtered bounding boxes and labels, this
    tiler self-filters instances using the **same tile-intersection criterion**
    as :class:`BboxTiler` (the instance's full-image mask bounding box must
    intersect the tile rectangle). Instances are kept in their original order so
    alignment with boxes/labels is preserved.
    """

    field_spec: AttributeSpec[InstanceMaskField]

    def tile(self, df: pl.DataFrame, tiles_df: pl.DataFrame, slice_offset: int = 0) -> pl.DataFrame:
        """Extract and filter instance mask regions for each tile."""
        column_name = self.field_spec.name
        shape_column = f"{column_name}_shape"
        results_data = []
        results_shape = []

        for tile_row in tiles_df["tile"]:
            image_id = tile_row["source_sample_idx"] - slice_offset
            instances_data = df[column_name][image_id]  # Flattened 3D array
            instances_shape = df[shape_column][image_id]  # (num_instances, height, width)

            # Get tile coordinates
            x = tile_row["x"]
            y = tile_row["y"]
            width = tile_row["width"]
            height = tile_row["height"]

            # Reshape flattened data
            instances = instances_data.reshape(instances_shape).to_numpy()  # (N, H, W)

            # Keep only instances whose full-image bounding box intersects the
            # tile, mirroring BboxTiler so masks stay aligned with boxes/labels.
            if instances.shape[0] > 0:
                keep = np.array(
                    [_instance_intersects_tile(inst, x, y, width, height) for inst in instances],
                    dtype=bool,
                )
                instances = instances[keep]

            # Extract tile region: shape (num_kept_instances, tile_height, tile_width)
            tile_result = instances[:, y : y + height, x : x + width]

            # Flatten result for storage
            results_data.append(tile_result.reshape(-1))
            results_shape.append(tile_result.shape)

        return pl.DataFrame(
            {
                column_name: results_data,
                shape_column: results_shape,
            }
        )


@TilerRegistry.register(BBoxField)
class BboxTiler(Tiler):
    """Tiler for bounding box annotations.

    Handles:
    - Adjusting bbox coordinates relative to tile origin
    - Filtering out boxes that don't intersect with tile
    - Adding keep flags for filtering
    """

    field_spec: AttributeSpec[BBoxField]

    def is_filterable(self) -> bool:
        return True

    def tile(self, df: pl.DataFrame, tiles_df: pl.DataFrame, slice_offset: int = 0) -> pl.DataFrame:
        """Process bounding boxes for each tile."""
        column_name = self.field_spec.name

        if self.field_spec.field.format != "x1y1x2y2":
            raise RuntimeError(f"The format {self.field_spec.field.format} is not supported.")

        results = []

        for tile_row in tiles_df["tile"]:
            image_id = tile_row["source_sample_idx"] - slice_offset
            boxes = df[image_id].select(column_name).explode(column_name)

            # Get tile coordinates
            tile_x = tile_row["x"]
            tile_y = tile_row["y"]
            tile_width = tile_row["width"]
            tile_height = tile_row["height"]
            tile_x2 = tile_x + tile_width
            tile_y2 = tile_y + tile_height

            # Process each bbox
            boxes = boxes.with_columns(
                x1=pl.col(column_name).arr.get(0),
                y1=pl.col(column_name).arr.get(1),
                x2=pl.col(column_name).arr.get(2),
                y2=pl.col(column_name).arr.get(3),
            )

            # Check if box intersects with tile
            boxes = boxes.with_columns(
                keep=(pl.col("x2") > tile_x)
                & (pl.col("x1") < tile_x2)
                & (pl.col("y2") > tile_y)
                & (pl.col("y1") < tile_y2)
            )

            # Calculate intersection
            boxes = boxes.with_columns(
                pl.col("x1").clip(lower_bound=tile_x) - tile_x,
                pl.col("y1").clip(lower_bound=tile_y) - tile_y,
                pl.col("x2").clip(upper_bound=tile_x2) - tile_x,
                pl.col("y2").clip(upper_bound=tile_y2) - tile_y,
            )

            boxes = boxes.with_columns(bboxes=pl.concat_arr("x1", "y1", "x2", "y2"))

            boxes = boxes.group_by(pl.lit(1)).agg(pl.col("bboxes", "keep")).drop("literal")

            results.append(boxes)

        return pl.concat(results)


@TilerRegistry.register(LabelField)
class LabelTiler(Tiler):
    """Tiler for label fields.

    For single labels, just passes through the data.
    For list fields, adds a keep column to mark all elements for inclusion
    in final filtering.
    """

    field_spec: AttributeSpec[LabelField]

    def is_filterable(self) -> bool:
        return self.field_spec.field.is_list

    def tile(self, df: pl.DataFrame, tiles_df: pl.DataFrame, slice_offset: int = 0) -> pl.DataFrame:
        """Process labels, adding keep column for list fields."""
        column_name = self.field_spec.name

        # For list fields, need to create keep flags for each element
        if self.field_spec.field.is_list:
            keeps = []
            labels = []
            for tile_row in tiles_df["tile"]:
                source_sample_idx = tile_row["source_sample_idx"] - slice_offset
                source_labels = df[source_sample_idx, column_name]
                # Create list of True values matching label list length
                keeps.append([True] * len(source_labels))
                labels.append(source_labels)

            # Return both the original labels and keep flags
            return pl.DataFrame({column_name: labels, "keep": keeps})

        # For non-list fields, just pass through the data
        return pl.DataFrame({column_name: df[column_name].take(tiles_df["source_sample_idx"])})


@TilerRegistry.register(ImageInfoField)
class ImageInfoTiler(Tiler):
    """Tiler for image info metadata.

    This tiler updates image dimensions and metadata for each tile.
    """

    field_spec: AttributeSpec[ImageInfoField]

    def tile(self, df: pl.DataFrame, tiles_df: pl.DataFrame, slice_offset: int = 0) -> pl.DataFrame:
        """Update image info for each tile."""
        results = []

        # Process each tile
        for tile_row in tiles_df["tile"]:
            # Get basic tile info
            source_sample_idx = tile_row["source_sample_idx"] - slice_offset
            tile_width = tile_row["width"]
            tile_height = tile_row["height"]

            # Create new image info for the tile
            tile_info = {
                "width": tile_width,
                "height": tile_height,
                "source_sample_idx": source_sample_idx,
            }

            # Add any additional info from original image
            original_info = df[source_sample_idx][self.field_spec.name]
            if isinstance(original_info, dict):
                # Copy relevant metadata but exclude size information
                for key, value in original_info.items():
                    if key not in ("width", "height"):
                        tile_info[key] = value

            results.append({self.field_spec.name: tile_info})

        return pl.DataFrame(results)


@TilerRegistry.register(ImageField)
class ImageTiler(Tiler):
    """Tiler for image data stored as numpy arrays."""

    field_spec: AttributeSpec[ImageField]

    def tile(self, df: pl.DataFrame, tiles_df: pl.DataFrame, slice_offset: int = 0) -> pl.DataFrame:
        """Tile images in the DataFrame."""

        column_name = self.field_spec.name
        shape_column = f"{column_name}_shape"

        def extract_tile(
            image_data: np.ndarray,
            image_shape: tuple[int, ...],
            tile_box: tuple[int, int, int, int],
        ) -> np.ndarray:
            """Extract a tile from flattened image data."""
            # Reshape the flattened data
            image = image_data.reshape(image_shape).to_numpy()

            # Extract coordinates
            y1, x1, h, w = tile_box
            y2, x2 = y1 + h, x1 + w

            # Extract tile
            tile = image[y1:y2, x1:x2]

            # Return flattened tile
            return tile.reshape(-1), tile.shape

        results_data = []
        results_shape = []
        for tile_row in tiles_df["tile"]:
            source_idx = tile_row["source_sample_idx"] - slice_offset

            # Get image data and shape
            image_data = df[source_idx, column_name]
            image_shape = df[source_idx, shape_column]

            # Extract tile
            tile_box = (tile_row["y"], tile_row["x"], tile_row["height"], tile_row["width"])
            tile_data, tile_shape = extract_tile(image_data, image_shape, tile_box)

            results_data.append(tile_data)
            results_shape.append(tile_shape)

        results = {column_name: pl.Series(results_data), f"{column_name}_shape": results_shape}

        return pl.DataFrame(results)


def _apply_offset(geom: BaseGeometry, offset_x: float, offset_y: float) -> BaseGeometry:
    """Apply offset to geometry."""
    return transform(geometry=geom, transformation=lambda x, y: (x - offset_x, y - offset_y), interleaved=False)


@TilerRegistry.register(PolygonField)
class PolygonTiler(Tiler):
    """Tiler for polygon annotations."""

    field_spec: AttributeSpec[PolygonField]
    threshold_drop_ann: float = 0.5  # Proportion of area below which to drop annotation

    def is_filterable(self) -> bool:
        return True

    def tile(self, df: pl.DataFrame, tiles_df: pl.DataFrame, slice_offset: int = 0) -> pl.DataFrame:
        """Tile polygon annotations in the DataFrame.

        Args:
            df: Input DataFrame containing polygon annotations
            tiles_df: DataFrame containing tile parameters
            slice_offset: Integer offset to subtract when accessing df based on
                        source_sample_idx. Used when df is a subset of another
                        DataFrame. Defaults to 0.

        Returns:
            DataFrame containing:
            - column_name: tiled polygon annotations
            - keep: List[bool] series indicating which polygons to keep
        """
        results = []
        keeps = []

        column_name = self.field_spec.name

        for tile_row in tiles_df["tile"]:
            source_idx = tile_row["source_sample_idx"] - slice_offset
            source_polygons = df[source_idx, column_name]

            # Create tile polygon
            tile_poly = box(
                tile_row["x"],
                tile_row["y"],
                tile_row["x"] + tile_row["width"],
                tile_row["y"] + tile_row["height"],
            )

            # Process each polygon
            tiled_polygons = []
            polygon_keeps = []  # Track which polygons to keep

            for poly_coords in source_polygons:
                polygon = ShapelyPolygon(poly_coords)

                # Get intersection and apply offset
                intersection = polygon.intersection(tile_poly)

                # NOTE: intersection may return a GeometryCollection or MultiPolygon
                if isinstance(intersection, GeometryCollection | MultiPolygon):
                    shapes = [(geom, geom.area) for geom in list(intersection.geoms) if geom.is_valid]
                    if not shapes:
                        tiled_polygons.append(None)  # Placeholder for dropped polygon
                        polygon_keeps.append(False)
                        continue

                    intersection, _ = max(shapes, key=operator.itemgetter(1))

                if not isinstance(intersection, ShapelyPolygon) or intersection.is_empty or not intersection.is_valid:
                    tiled_polygons.append(None)  # Placeholder for dropped polygon
                    polygon_keeps.append(False)
                    continue

                prop_area = intersection.area / polygon.area

                if prop_area < self.threshold_drop_ann:
                    tiled_polygons.append(None)  # Placeholder for dropped polygon
                    polygon_keeps.append(False)
                    continue

                offset_poly = _apply_offset(intersection, tile_row["x"], tile_row["y"])

                tiled_polygons.append(np.array(offset_poly.exterior.coords))
                polygon_keeps.append(True)

            # Always create output row
            results.append(
                pl.Series(
                    [pl.Series(polygon, dtype=pl.Array(self.field_spec.field.dtype, 2)) for polygon in tiled_polygons]
                )
            )
            keeps.append(polygon_keeps)

        # Create DataFrame with results and keep column as List[Boolean]
        return pl.DataFrame({column_name: results, "keep": pl.Series(keeps, dtype=pl.List(pl.Boolean()))})
