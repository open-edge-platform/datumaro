# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
Registry system for tiling operations on experimental datasets.

This module provides the foundation for tiling operations, including tiler
registration and configuration management.
"""

import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

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

    def is_filterable(self) -> bool:
        return False

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


def create_tilers(
    schema: Schema, threshold_drop_ann: float
) -> Dict[Semantic, List[tuple[str, Tiler]]]:
    """Create tiler instances based on schema fields.

    Args:
        schema: Input schema defining the DataFrame structure
        config: Tiling configuration

    Returns:
        List of instantiated tilers for the given schema
    """
    tilers = defaultdict(list)

    for field_name, field in schema.attributes.items():
        tiler_cls = TilerRegistry.get_tiler(type(field.annotation))
        if tiler_cls:
            # Create field spec and tiler instance
            field_spec = AttributeSpec(name=field_name, field=field.annotation)
            tiler = tiler_cls()
            setattr(tiler, "field_spec", field_spec)  # Set the field spec
            setattr(tiler, "threshold_drop_ann", threshold_drop_ann)  # Set the threshold
            tilers[field.annotation.semantic].append((field_name, tiler))

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


@dataclass
class TilingPlan:
    tilers_filter_group_id: dict[str, int]
    tilers_by_group_id: defaultdict[int, list[str]]
    tilers_instance_by_name: dict[str, Tiler]
    image_info_spec: AttributeSpec[ImageInfoField]
    tile_info_spec: AttributeSpec[ImageInfoField]
    target_schema: Schema
    config: TilingConfig


def create_tiling_plan(schema: Schema, config: TilingConfig, threshold_drop_ann: float = 0.8):
    # Find image info field in schema
    image_info_spec = None
    for field_name, field in schema.attributes.items():
        if isinstance(field.annotation, ImageInfoField):
            image_info_spec = AttributeSpec(name=field_name, field=field)
            break

    if image_info_spec is None:
        raise ValueError("Schema must contain an ImageInfoField")

    # Create tile info field spec
    tile_info_spec = AttributeSpec(name="tile", field=TileField(semantic=Semantic.Default))

    # Create tilers for each field type
    tilers = create_tilers(schema, threshold_drop_ann)

    tilers_filter_group_id = {}
    tilers_by_group_id = defaultdict(list)
    tilers_instance_by_name = {}

    # Group filterable tilers
    for semantic_tilers in tilers.values():
        filterable_tilers = [tiler for tiler in semantic_tilers if tiler[1].is_filterable()]

        group_id = id(filterable_tilers)
        tilers_filter_group_id.update((tiler[0], group_id) for tiler in filterable_tilers)
        tilers_by_group_id[group_id] = [tiler[0] for tiler in filterable_tilers]
        tilers_instance_by_name.update(semantic_tilers)

    # Update schema with tile information
    new_fields = dict(schema.attributes)
    new_fields[tile_info_spec.name] = AttributeInfo(type=TileInfo, annotation=tile_info_spec.field)
    target_schema = Schema(new_fields)

    return TilingPlan(
        tilers_filter_group_id,
        tilers_by_group_id,
        tilers_instance_by_name,
        image_info_spec,
        tile_info_spec,
        target_schema,
        config,
    )


def apply_tiling(
    input_df: pl.DataFrame, output_df: pl.DataFrame | None, plan: TilingPlan, fields: set[str]
) -> tuple[pl.DataFrame, set[str]]:
    """Apply tiling operations to a DataFrame.

    Args:
        df: Input DataFrame to tile
        schema: Schema describing the DataFrame structure
        config: Tiling configuration

    Returns:
        Tiled DataFrame
    """
    fields_set = set(fields)

    # Include all the other fields in the same groups
    for field in fields:
        group_id = plan.tilers_filter_group_id.get(field, None)
        if group_id is not None:
            fields_set.update(plan.tilers_by_group_id[group_id])

    if output_df is None:
        output_df = calculate_tiles(
            input_df, plan.config, plan.image_info_spec, plan.tile_info_spec
        )

    # Apply each tiler and collect keep flags and columns to filter
    tiled_data = []
    keep_mask_by_group_id = {}
    columns_to_filter_by_group_id = defaultdict(set)

    for field_name in fields_set:
        tiler = plan.tilers_instance_by_name[field_name]

        tiled_df = tiler.tile(input_df, output_df)

        # If tiler provides a keep column, update our mask and track columns
        if tiler.is_filterable():
            if "keep" not in tiled_df.columns:
                raise RuntimeError(
                    "Expected the 'keep' column to be present since the tiler is filterable."
                )

            group_id = plan.tilers_filter_group_id[field_name]

            # Initialize keep_mask if needed
            if group_id not in keep_mask_by_group_id:
                keep_mask_by_group_id[group_id] = tiled_df["keep"]
            else:
                # Update keep_mask at list element level
                keep_mask = keep_mask_by_group_id[group_id]
                keep_mask = tiled_df.with_columns(keep_mask=keep_mask).with_columns(
                    keep_mask=list_zip("keep", "keep_mask", lambda a, b: a & b)
                )["keep_mask"]
                keep_mask_by_group_id[group_id] = keep_mask

            tiled_df = tiled_df.drop("keep")

            # Track which columns from this tiler need filtering
            columns_to_filter_by_group_id[group_id].update(tiled_df.columns)

        tiled_data.append(tiled_df)

    # Combine all tiled data with tile parameters
    for tiled_df in tiled_data:
        # Only keep columns that aren't in the result yet
        new_cols = [col for col in tiled_df.columns if col not in output_df.columns]
        if new_cols:
            output_df = output_df.with_columns(
                [tiled_df.get_column(col).alias(col) for col in new_cols]
            )

    # Apply list filtering if we have a keep mask
    for group_id, keep_mask in keep_mask_by_group_id.items():
        columns_to_filter = keep_mask_by_group_id[group_id]

        output_df = output_df.with_columns(keep=keep_mask).with_columns(
            list_zip(col, "keep", lambda a, b: pl.struct(a, b))
            .list.filter(pl.element().struct["keep"])
            .list.eval(pl.element().struct[col])
            .alias(col)
            for col in columns_to_filter
        )
        output_df = output_df.drop("keep")

    return output_df, fields_set


from typing import Sequence

from datumaro.experimental.transform import Transform


class TilingTransform(Transform):
    def __init__(self, parent: Transform, tiling_plan: TilingPlan):
        super().__init__(tiling_plan.target_schema)

        # Tiling does not support a lazy image info field
        # because we need it at construction.
        lazy_outputs = parent.get_lazy_attributes()
        lazy_outputs.discard(tiling_plan.image_info_spec.name)

        self._lazy_outputs = lazy_outputs

        batch_outputs = self.get_batch_attributes()
        batch_outputs.add(tiling_plan.image_info_spec.name)

        self._parent = parent
        self._tiling_plan = tiling_plan
        self._df = None

        # Calculate tile parameters
        self._applied_tilers = set()

        self.apply(batch_outputs)

    def apply(self, fields: Sequence[str]) -> pl.DataFrame:
        fields_set = set(fields)
        fields_set.discard(self._tiling_plan.tile_info_spec.name)
        fields_set -= self._applied_tilers

        input_df = self._parent.apply(list(fields_set))

        self._df, fields_set = apply_tiling(input_df, self._df, self._tiling_plan, fields_set)

        self._applied_tilers.update(fields_set)
        return self._df

    def get_lazy_attributes(self) -> set[str]:
        return self._lazy_outputs

    def slice(self, offset: int, length: int | None = None) -> "Transform":
        instance = copy.copy(self)

        if self._df is None:
            raise RuntimeError("apply() should have been called in the constructor.")

        source_sample_offset = self._df[offset, self._tiling_plan.tile_info_spec.name][
            "source_sample_idx"
        ]

        if length is None:
            source_sample_length = None
        else:
            source_sample_last = self._df[
                offset + length - 1, self._tiling_plan.tile_info_spec.name
            ]["source_sample_idx"]
            source_sample_length = source_sample_last - source_sample_offset + 1

        instance._parent = self._parent.slice(source_sample_offset, source_sample_length)
        instance._applied_tilers = copy.copy(self._applied_tilers)
        instance._df = self._df.slice(offset, length)
        return instance

    def __len__(self):
        return len(self._df)


def create_tiling_transform(config: TilingConfig, threshold_drop_ann: float = 0.8):
    def factory(parent: Transform):
        plan = create_tiling_plan(parent.schema, config, threshold_drop_ann)
        return TilingTransform(parent, plan)

    return factory
