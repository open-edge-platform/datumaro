# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
Registry system for tiling operations on experimental datasets.

This module provides the foundation for tiling operations, including tiler
registration and configuration management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Type

import polars as pl

from ..fields import ImageInfoField, TileField, TileInfo
from ..schema import AttributeInfo, AttributeSpec, Field, Schema, Semantic


@dataclass
class TilingConfig:
    """Configuration for tiling operations."""

    tile_width: int
    tile_height: int
    overlap_x: int = 0
    overlap_y: int = 0


def calculate_tiles(
    df: pl.DataFrame,
    config: TilingConfig,
    image_info_spec: AttributeSpec[ImageInfoField],
    tile_info_spec: AttributeSpec[TileField],
) -> pl.DataFrame:
    """Calculate tile parameters for each image in the dataset.

    Args:
        df: Input DataFrame containing image info
        config: Tiling configuration
        image_info_spec: Specification for the input image info field
        tile_info_spec: Specification for the output tile info field

    Returns:
        DataFrame containing tile parameters using TileInfo structure
    """
    tiles = []
    field = tile_info_spec.field

    for idx, row in enumerate(df.iter_rows(named=True)):
        # Get image dimensions directly from row
        image_info = row[image_info_spec.name]
        if image_info is None:
            continue

        height = image_info["height"]
        width = image_info["width"]

        for y in range(0, height, config.tile_height - config.overlap_y):
            for x in range(0, width, config.tile_width - config.overlap_x):
                # Ensure tiles don't exceed image boundaries
                tile_width = min(config.tile_width, width - x)
                tile_height = min(config.tile_height, height - y)

                tile_info = TileInfo(
                    source_sample_idx=idx,
                    x=x,
                    y=y,
                    width=tile_width,
                    height=tile_height,
                )

                # Convert tile info to polars using the field
                tile_data = field.to_polars(tile_info_spec.name, tile_info)
                tiles.append(tile_data)

    # Combine all tile data into a single DataFrame
    return pl.concat([pl.DataFrame(t) for t in tiles])


class Tiler(ABC):
    """Base class for all tilers."""

    @abstractmethod
    def tile(self, df: pl.DataFrame, tiles_df: pl.DataFrame) -> pl.DataFrame:
        """Apply tiling operation to a DataFrame.

        Args:
            df: Input DataFrame containing the data to tile
            tiles_df: DataFrame containing tile parameters

        Returns:
            DataFrame with tiled data. The output will have the same
            number of rows as tiles_df.
        """
        pass


class TilerRegistry:
    """Registry for tiler implementations."""

    _tilers: Dict[Type[Field], Type[Tiler]] = {}

    @classmethod
    def register(cls, field_type: Type[Field]):
        """Decorator to register a tiler for a specific field type."""

        def wrapper(tiler_cls: Type[Tiler]):
            cls._tilers[field_type] = tiler_cls
            return tiler_cls

        return wrapper

    @classmethod
    def get_tiler(cls, field_type: Type[Field]) -> Optional[Type[Tiler]]:
        """Get the registered tiler for a field type."""
        return cls._tilers.get(field_type)


def create_tilers(schema: Schema, threshold_drop_ann: float) -> List[Tiler]:
    """Create tiler instances based on schema fields.

    Args:
        schema: Input schema defining the DataFrame structure
        config: Tiling configuration

    Returns:
        List of instantiated tilers for the given schema
    """
    tilers = []
    processed_fields: Set[str] = set()

    for field_name, field in schema.attributes.items():
        if field_name in processed_fields:
            continue

        tiler_cls = TilerRegistry.get_tiler(type(field.annotation))
        if tiler_cls:
            # Create field spec and tiler instance
            field_spec = AttributeSpec(name=field_name, field=field.annotation)
            tiler = tiler_cls()
            setattr(tiler, "field_spec", field_spec)  # Set the field spec
            setattr(tiler, "threshold_drop_ann", threshold_drop_ann)  # Set the threshold
            tilers.append(tiler)
            processed_fields.add(field_name)

    return tilers


from typing import Callable


def list_zip(
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
            pl.element().struct.field(ref_col).explode(),
        )
    )


def apply_tiling(
    df: pl.DataFrame, schema: Schema, config: TilingConfig, threshold_drop_ann: float = 0.8
) -> tuple[pl.DataFrame, Schema]:
    """Apply tiling operations to a DataFrame.

    Args:
        df: Input DataFrame to tile
        schema: Schema describing the DataFrame structure
        config: Tiling configuration

    Returns:
        Tuple of (tiled DataFrame, updated schema)
    """
    # Find image info field in schema
    image_info_field = None
    for field_name, field in schema.attributes.items():
        if isinstance(field.annotation, ImageInfoField):
            image_info_field = AttributeSpec(name=field_name, field=field)
            break

    if image_info_field is None:
        raise ValueError("Schema must contain an ImageInfoField")

    # Create tile info field spec
    tile_info_field = AttributeSpec(name="tile", field=TileField(semantic=Semantic.Default))

    # Calculate tile parameters
    tiles_df = calculate_tiles(df, config, image_info_field, tile_info_field)

    # Create tilers for each field type
    tilers = create_tilers(schema, threshold_drop_ann)

    # Apply each tiler and collect keep flags and columns to filter
    tiled_data = []
    keep_mask = None
    columns_to_filter = set()

    for tiler in tilers:
        tiled_df = tiler.tile(df, tiles_df)

        # If tiler provides a keep column, update our mask and track columns
        if "keep" in tiled_df.columns:
            # Initialize keep_mask if needed
            if keep_mask is None:
                keep_mask = tiled_df["keep"]
            else:
                # Update keep_mask at list element level
                keep_mask = tiled_df.with_columns(keep_mask=keep_mask).with_columns(
                    keep_mask=list_zip("keep", "keep_mask", lambda a, b: a & b)
                )["keep_mask"]

            tiled_df = tiled_df.drop("keep")

            # Track which columns from this tiler need filtering
            columns_to_filter.update(tiled_df.columns)

        tiled_data.append(tiled_df)

    # Combine all tiled data with tile parameters
    result = tiles_df
    for tiled_df in tiled_data:
        # Only keep columns that aren't in the result yet
        new_cols = [col for col in tiled_df.columns if col not in result.columns]
        if new_cols:
            result = result.with_columns([tiled_df.get_column(col).alias(col) for col in new_cols])

    # Apply list filtering if we have a keep mask
    if keep_mask is not None and columns_to_filter:
        result = result.with_columns(keep=keep_mask).with_columns(
            list_zip(col, "keep", lambda a, b: pl.struct(a, b))
            .list.filter(pl.element().struct["keep"])
            .list.eval(pl.element().struct[col])
            .alias(col)
            for col in columns_to_filter
        )
        result = result.drop("keep")

    # Update schema with tile information
    new_fields = dict(schema.attributes)
    new_fields[tile_info_field.name] = AttributeInfo(
        type=TileInfo, annotation=tile_info_field.field
    )
    new_schema = Schema(new_fields)

    return result, new_schema
