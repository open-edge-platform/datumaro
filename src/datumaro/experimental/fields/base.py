# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Base Field class implementations for various data types.

This module provides concrete field implementations that handle serialization
to/from Polars DataFrames for different data types commonly used in machine
learning and computer vision applications.
"""

from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from enum import Flag, auto
from typing import Any, TypeAlias, TypeVar

import numpy as np
import polars as pl

T = TypeVar("T")

PolarsDataType: TypeAlias = type[pl.DataType] | pl.DataType


class Semantic(Flag):
    """
    Used for disambiguation when multiple fields of the same type exist.
    Default is used for fields that don't need disambiguation.
    Left/Right are used for stereo vision scenarios.
    """

    Default = auto()
    Left = auto()
    Right = auto()
    Anomaly = auto()


class Field:
    """
    Base class for fields with semantic tags and Polars type mapping.

    This abstract base class defines the interface for all field types,
    providing methods for converting between Python objects and Polars
    DataFrame representations.

    Attributes:
        semantic: Semantic tags for disambiguation (Default, Left, Right)
    """

    semantic: Semantic

    def __post_init__(self):
        if hasattr(self, "dtype"):
            dtype = getattr(self, "dtype")
            if isinstance(dtype, type) and issubclass(dtype, pl.DataType):
                object.__setattr__(self, "dtype", dtype())
            if not isinstance(dtype, pl.DataType):
                raise TypeError(f"dtype must be a Polars DataType, got {type(dtype)}")

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """
        Generate Polars schema definition for this field.

        Args:
            name: The column name for this field

        Returns:
            Dictionary mapping column names to Polars data types

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the to_polars_type method.")

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """
        Convert the field value to Polars-compatible format.

        Args:
            name: The column name for this field
            value: The value to convert

        Returns:
            Dictionary mapping column names to Polars Series
        """
        return {name: pl.Series(name, [value])}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type) -> Any:
        """
        Convert from Polars-compatible format back to the field's value.

        Args:
            name: The column name for this field
            row_index: The row index to extract
            df: The source DataFrame
            target_type: The target type to convert to

        Returns:
            The converted value in the target type
        """
        return target_type(df[name][row_index])

    def __set_name__(self, _: Any, name: str):
        object.__setattr__(self, "_name", name)

    def __get__(self, instance: Any, _: Any):
        if instance is None:
            return self

        name = getattr(self, "_name")
        value = instance.evaluate_lazy_field(name)

        # Cache the value and set it as a real attribute
        setattr(instance, name, value)

        return value

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this Field to a JSON-compatible dictionary.

        Automatically serializes all dataclass fields by introspection.

        Returns:
            Dictionary containing field type and parameters
        """

        field_dict = {
            "type": self.__class__.__name__,
        }

        # Use dataclass introspection to get all fields
        if is_dataclass(self):
            for dc_field in dataclass_fields(self):
                field_name = dc_field.name
                field_value = getattr(self, field_name)

                # Handle semantic as special case (convert to string)
                if field_name == "semantic" and isinstance(field_value, Semantic):
                    field_dict[field_name] = field_value.name
                # Handle Polars data types
                elif field_name == "dtype" and isinstance(field_value, pl.DataType):
                    field_dict[field_name] = str(field_value)
                # Handle regular serializable values
                else:
                    field_dict[field_name] = field_value

        return field_dict

    @classmethod
    def from_dict(cls, field_dict: dict[str, Any]) -> "Field":
        """
        Deserialize a Field from a JSON dictionary.

        Automatically reconstructs all dataclass fields by introspection.

        Args:
            field_dict: Dictionary containing field type and parameters

        Returns:
            Reconstructed Field instance
        """
        import datumaro.experimental.fields as fields_module

        field_type = field_dict["type"]

        # Get the field class
        field_class = getattr(fields_module, field_type)

        # Prepare kwargs for field construction
        kwargs: dict[str, Any] = {}

        # Use dataclass introspection to get all expected fields
        if is_dataclass(field_class):
            for dc_field in dataclass_fields(field_class):
                field_name = dc_field.name

                # Skip if not in the serialized data
                if field_name not in field_dict:
                    continue

                field_value = field_dict[field_name]

                # Handle semantic reconstruction
                if field_name == "semantic" and isinstance(field_value, str):
                    kwargs[field_name] = Semantic[field_value]
                # Handle dtype reconstruction (Polars types)
                elif field_name == "dtype" and isinstance(field_value, str):
                    # Try to resolve Polars data types
                    dtype_str = field_value.replace("()", "")
                    if hasattr(pl, dtype_str):
                        dtype_obj = getattr(pl, dtype_str)
                        if callable(dtype_obj):
                            kwargs[field_name] = dtype_obj()
                        else:
                            kwargs[field_name] = dtype_obj
                    else:
                        # Keep as string if we can't resolve it
                        kwargs[field_name] = field_value
                # Handle regular values
                else:
                    kwargs[field_name] = field_value

        return field_class(**kwargs)


def convert_numpy_object_array_to_series(data: np.ndarray) -> pl.Series:
    """
    Convert ragged numpy object arrays to Polars Series recursively.

    Handles nested object arrays containing variable-length lists.

    Example:
        >>> import numpy as np
        >>> ragged = np.array([
        ...     np.array([1, 2, 3]),
        ...     np.array([4, 5]),
        ...     np.array([6, 7, 8, 9])
        ... ], dtype=object)
        >>> series = convert_numpy_object_array_to_series(ragged)
        >>> print(series)
        shape: (3,)
        Series: '' [list[i64]]
        [
                [1, 2, 3]
                [4, 5]
                [6, 7, … 9]
        ]

        # Compare with direct conversion which results
        # into an object Series instead of a list Series:
        >>> direct = pl.Series(ragged)
        >>> print(direct)
        shape: (3,)
        Series: '' [o][object]
        [
                [1 2 3]
                [4 5]
                [6 7 8 9]
        ]
    """
    if data is not None and data.dtype == object:
        return pl.Series([convert_numpy_object_array_to_series(elem) for elem in data])
    return pl.Series(data)
