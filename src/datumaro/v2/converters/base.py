# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Converter implementations for data transformation between different schemas.

This module contains concrete converter implementations that handle various
data transformations such as format conversions, dtype conversions, and
multi-field transformations.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cache
from typing import TypeVar, dataclass_transform, Any, get_type_hints, cast, overload, Sequence

import cv2
import numpy as np
import polars as pl
from PIL import Image

from datumaro.v2 import Field, Schema
from datumaro.v2.categories import Categories
from datumaro.v2.converters import ConversionPaths
from datumaro.v2.converters.registry import ConversionPaths, _group_fields_by_semantic, _SchemaState, \
    _find_conversion_path_for_semantic, _separate_batch_and_lazy_converters
from datumaro.v2.schema import AttributeSpec
from datumaro.v2.transform import Transform


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


TField = TypeVar("TField", bound=Field)


@dataclass_transform()
class Converter(ABC):
    """
    Base class for data converters with input/output specifications.

    Converters transform data between different field representations by
    implementing the convert() method and optionally filtering their
    applicability through filter_output_spec().
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize converter with input and output AttributeSpec instances.

        Args:
            **kwargs: AttributeSpec instances for converter inputs/outputs
                     based on input_*/output_* class attributes
        """
        # Set all provided kwargs as instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    lazy: bool = False
    """
    Whether this converter performs lazy operations.

    Lazy converters defer expensive operations (like loading images from disk)
    until data is actually accessed. When a lazy converter is in the conversion
    path, all dependent converters must also be executed lazily.
    """

    @classmethod
    @cache
    def get_from_types(cls) -> dict[str, type[Field]]:
        """
        Extract input field types from input_* class attributes.

        Returns:
            Dictionary mapping input attribute names to their Field types
        """
        from_types: dict[str, type[Field]] = {}

        # Get type hints for the class
        hints = get_type_hints(cls)

        for attr_name, attr_type in hints.items():
            if attr_name.startswith("input_"):
                # Extract the Field type from AttributeSpec[FieldType] annotation
                if hasattr(attr_type, "__args__") and len(attr_type.__args__) > 0:
                    # Handle generic types like AttributeSpec[SomeField]
                    field_type = attr_type.__args__[0]
                else:
                    raise RuntimeError("Attributes must be annotated with AttributeSpec[FieldType]")

                from_types[attr_name] = field_type

        return from_types

    @classmethod
    @cache
    def get_to_types(cls) -> dict[str, type[Field]]:
        """
        Extract output field types from output_* class attributes.

        Returns:
            Dictionary mapping output attribute names to their Field types
        """
        to_types: dict[str, type[Field]] = {}

        # Get type hints for the class
        hints = get_type_hints(cls)

        for attr_name, attr_type in hints.items():
            if attr_name.startswith("output_"):
                # Extract the Field type from AttributeSpec[FieldType] annotation
                if hasattr(attr_type, "__args__") and len(attr_type.__args__) > 0:
                    # Handle generic types like AttributeSpec[SomeField]
                    field_type = attr_type.__args__[0]
                else:
                    raise RuntimeError("Attributes must be annotated with AttributeSpec[FieldType]")

                to_types[attr_name] = field_type

        return to_types

    @abstractmethod
    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert a DataFrame using the stored AttributeSpec instances.

        Args:
            df: Input DataFrame

        Returns:
            Converted DataFrame
        """

    def filter_output_spec(self) -> bool:
        """
        Filter and modify the converter's output specification in-place.

        This method allows converters to inspect and modify their output
        specifications based on input characteristics. It should return
        True if the converter can handle the given input/output combination.

        Returns:
            True if the converter is applicable, False otherwise
        """
        # Default implementation accepts all conversions
        # Subclasses should override for sophisticated filtering
        return True

    def get_input_attr_specs(self) -> list[AttributeSpec[Field]]:
        """
        Get the current input AttributeSpec instances from input_* attributes.

        Returns:
            List of input AttributeSpec instances currently configured on the converter
        """
        input_attr_specs: list[AttributeSpec[Field]] = []

        # Get the input attribute names from class type hints
        from_types = self.get_from_types()

        for attr_name in from_types:
            attr_spec = cast("AttributeSpec[Field]", getattr(self, attr_name))
            input_attr_specs.append(attr_spec)

        return input_attr_specs

    def get_output_attr_specs(self) -> list[AttributeSpec[Field]]:
        """
        Get the current output AttributeSpec instances from output_* attributes.

        Returns:
            List of output AttributeSpec instances currently configured on the converter
        """
        output_attr_specs: list[AttributeSpec[Field]] = []

        # Get the output attribute names from class type hints
        to_types = self.get_to_types()

        for attr_name in to_types:
            attr_spec = cast("AttributeSpec[Field]", getattr(self, attr_name))
            output_attr_specs.append(attr_spec)

        return output_attr_specs


class ConversionError(Exception):
    """Exception raised when conversion fails."""


class AttributeRemapperConverter(Converter):
    """
    Special converter for renaming/selecting attributes and dropping others.

    This converter is not registered with the converter registry but is used
    internally by find_conversion_path when attributes need to be renamed or deleted.
    It uses .select() to only keep the specified attributes with their new names,
    effectively handling both renaming and deletion in a single operation.
    """

    def __init__(self, attr_mappings: list[tuple[AttributeSpec, AttributeSpec]]):
        """
        Initialize the converter with a list of attribute mappings.

        Args:
            attr_mappings: List of tuples (from_attr_spec, to_attr_spec) defining
                          the attribute transformations. Only attributes in this
                          list will be kept in the output.
        """
        self.attr_mappings = attr_mappings

        # Calculate column mapping from attribute mappings
        self.column_map = {}
        for from_attr, to_attr in attr_mappings:
            # Get all column names for this field using to_polars_schema
            from_columns = list(from_attr.field.to_polars_schema(from_attr.name).keys())
            to_columns = list(to_attr.field.to_polars_schema(to_attr.name).keys())

            # Map each column from source to target
            for from_col, to_col in zip(from_columns, to_columns):
                self.column_map[from_col] = to_col

        # Dynamically set input_* and output_* attributes for get_from_types/get_to_types
        for i, (from_attr, to_attr) in enumerate(attr_mappings):
            setattr(self, f"input_{i}", from_attr)
            setattr(self, f"output_{i}", to_attr)

        super().__init__()

    @cache
    def get_from_types(self) -> dict[str, type[Field]]:
        """
        Extract input field types from input_* class attributes.

        Returns:
            Dictionary mapping input attribute names to their Field types
        """
        from_types: dict[str, type[Field]] = {}
        for i, (input_spec, _) in enumerate(self.attr_mappings):
            from_types[f"input_{i}"] = type(input_spec.field)

        return from_types

    @cache
    def get_to_types(self) -> dict[str, type[Field]]:
        """
        Extract output field types from output_* class attributes.

        Returns:
            Dictionary mapping output attribute names to their Field types
        """
        to_types: dict[str, type[Field]] = {}
        for i, (_, output_spec) in enumerate(self.attr_mappings):
            to_types[f"output_{i}"] = type(output_spec.field)

        return to_types

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Rename columns according to column_map and keep all other columns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with renamed columns
        """
        # Apply all renames
        return df.rename(self.column_map)

    def filter_output_spec(self) -> bool:
        """Always return True as renaming is always applicable."""
        return True


def find_conversion_path(from_schema: Schema, to_schema: Schema) -> tuple[ConversionPaths, dict[str, Categories]]:
    """
    Find an optimal sequence of converters using A* search, grouped by semantic.

    Fields with the same semantic can be converted between each other, but
    conversion across semantic boundaries is not allowed.

    Args:
        from_schema: Source schema
        to_schema: Target schema

    Returns:
        Tuple of (ConversionPaths with separated batch and lazy converter lists,
                 dictionary of attribute names to inferred categories)

    Raises:
        ConversionError: If no conversion path is found
    """
    # Group fields by semantic in both schemas
    start_groups = _group_fields_by_semantic(from_schema)
    target_groups = _group_fields_by_semantic(to_schema)

    # Collect all converters needed across all semantic groups
    all_converters: list[Converter] = []

    # Process each semantic group in the target schema
    for semantic, target_state in target_groups.items():
        # Get corresponding source state for this semantic (if any)
        start_state = start_groups.get(semantic, _SchemaState({}))

        # Find conversion path for this semantic group
        semantic_converters, updated_target_state = _find_conversion_path_for_semantic(
            start_state, target_state, semantic
        )

        # Update the target state with any inferred categories
        target_groups[semantic] = updated_target_state

        all_converters.extend(semantic_converters)

    # Reconstruct the updated schema with inferred categories
    # Use the list of attributes from to_schema rather than just the target_groups
    # because the target_groups may include attributes which are deleted in the final to_schema.
    # We do not want to include those attributes into the inferred_categories.
    inferred_categories: dict[str, Categories] = {}
    for attr_name, attr_info in to_schema.attributes.items():
        semantic = attr_info.field.semantic
        attr_spec = target_groups[semantic].field_to_attr_spec[type(attr_info.field)]
        if attr_spec.categories is not None:
            inferred_categories[attr_name] = attr_spec.categories

    # Separate batch and lazy converters
    conversion_paths = _separate_batch_and_lazy_converters(all_converters)

    return conversion_paths, inferred_categories


class ConverterTransform(Transform):
    def __init__(self, parent: Transform, schema: Schema, conversion_paths: ConversionPaths):
        super().__init__(schema)

        lazy_inputs = parent.get_lazy_attributes()

        lazy_outputs = set(conversion_paths.lazy_outputs)
        for input in lazy_inputs:
            lazy_outputs.update(conversion_paths.dependent_outputs_by_input[input])
        self._lazy_outputs = lazy_outputs

        batch_outputs = self.get_batch_attributes()

        self._parent = parent
        self._conversion_paths = conversion_paths
        self._df_input_columns = set()
        self._df = pl.DataFrame()
        self._applied_converters = set()

        self.apply(batch_outputs)

    def apply(self, fields: Sequence[str]) -> pl.DataFrame:
        required_inputs = set()
        for field in fields:
            if field in self._conversion_paths.converters:
                required_inputs.update(self._conversion_paths.required_inputs_by_output[field])

        parent_df = self._parent.apply(required_inputs)
        input_columns = set(parent_df.columns)
        new_columns = set(parent_df.columns) - self._df_input_columns

        self._df = self._df.with_columns(parent_df.select(new_columns))
        self._df_input_columns = input_columns

        for field in fields:
            converters = self._conversion_paths.converters.get(field, None)

            if converters is not None:
                for converter in converters:
                    if id(converter) not in self._applied_converters:
                        if not self._can_apply_converter(converter):
                            # Defer this converter; it will be attempted again on future apply() calls
                            # once the necessary input columns have been materialized.
                            continue

                        self._df = converter.convert(self._df)
                        self._applied_converters.add(id(converter))

        return self._df

    def _can_apply_converter(self, converter: Converter) -> bool:
        """
        Only apply the converter when all of its required input columns are present.
        This prevents race conditions when converters are evaluated lazily in
        multi-worker dataloaders, where some columns may not be materialized yet.
        """
        for attr_spec in converter.get_input_attr_specs():
            required_cols = attr_spec.field.to_polars_schema(attr_spec.name).keys()
            for col in required_cols:
                if col not in self._df.columns:
                    return False
        return True

    def get_lazy_attributes(self) -> set[str]:
        return self._lazy_outputs

    def slice(self, offset: int, length: int | None = None) -> Transform:
        instance = copy.copy(self)
        instance._parent = self._parent.slice(offset, length)
        instance._applied_converters = copy.copy(self._applied_converters)
        instance._df = self._df.slice(offset, length)
        return instance

    def __len__(self):
        return len(self._df)
