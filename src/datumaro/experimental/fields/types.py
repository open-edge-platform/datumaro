# Copyright (C) 2022-2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from datumaro.experimental.fields.base import Field, T
from datumaro.experimental.type_registry import from_polars_data


@dataclass(frozen=True)
class NumericField(Field):
    """
    Represents a neutral numeric value.

    Use this for generic per-sample or per-instance numeric attributes such as
    area, depth, distance, score, etc. Supports scalar values or lists of values when
    ``is_list=True``.
    """

    semantic: str = "default"
    dtype: pl.DataType = field(default_factory=pl.Float32)
    is_list: bool = False

    @property
    def _pl_type(self) -> pl.DataType:
        pl_type = self.dtype
        if self.is_list:
            pl_type = pl.List(pl_type)
        return pl_type

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        return {name: self._pl_type}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        # Wrap the provided value in a Series with proper dtype/list-typing
        return {name: pl.Series(name, [value], dtype=self._pl_type)}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        data = df[name][row_index]
        if target_type is list:
            return list(data) if data is not None else None  # type: ignore[return-value]
        return from_polars_data(data, target_type)


def numeric_field(
    dtype: Any = pl.Float32(),
    is_list: bool = False,
    semantic: str = "default",
) -> Any:
    """
    Create a NumericField instance.

    Args:
        dtype: Polars data type for numeric values (defaults to pl.Float32())
        semantic: String tag describing the value purpose (optional)
        is_list: Whether this field should be treated as a list type

    Returns:
        NumericField configured with the given parameters
    """
    return NumericField(dtype=dtype, is_list=is_list, semantic=semantic)


@dataclass(frozen=True)
class BoolField(Field):
    """
    Represents a boolean value or a list of boolean values.

    Stored as Polars Boolean type and
    converts back to Python ``bool``/``list[bool]`` or numpy arrays as needed
    by the higher-level Dataset APIs.
    """

    semantic: str = "default"
    is_list: bool = False
    dtype: pl.DataType = field(default_factory=pl.Boolean, init=False)

    @property
    def _pl_type(self) -> pl.DataType:
        return pl.List(pl.Boolean) if self.is_list else pl.Boolean

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        return {name: self._pl_type}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        return {name: pl.Series(name, [value], dtype=self._pl_type)}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        data = df[name][row_index]
        if target_type is list:
            return list(data) if data is not None else None  # type: ignore[return-value]
        return from_polars_data(data, target_type)


def bool_field(
    semantic: str = "default",
    is_list: bool = False,
) -> Any:
    """
    Create a BoolField instance.

    Args:
        semantic: String tag describing the field purpose (optional)
        is_list: Whether this field should be treated as a list type

    Returns:
        BoolField configured with the given parameters
    """
    return BoolField(semantic=semantic, is_list=is_list)
