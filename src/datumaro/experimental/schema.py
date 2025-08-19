# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Schema definitions for the dataset system.
"""

import copy
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import TYPE_CHECKING, Any, Dict, Optional

import polars as pl

if TYPE_CHECKING:
    from .categories import Categories


class Semantic(Flag):
    """
    Used for disambiguation when multiple fields of the same type exist.
    Default is used for fields that don't need disambiguation.
    Left/Right are used for stereo vision scenarios.
    """

    Default = auto()
    Left = auto()
    Right = auto()


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


@dataclass
class AttributeInfo:
    """
    Container for attribute type and field annotation information.
    """

    type: type
    annotation: Field
    categories: Optional["Categories"] = None


@dataclass
class Schema:
    """
    Represents the schema of a dataset with attribute definitions.
    Enforces that only one field of each type exists per semantic context.
    """

    attributes: dict[str, AttributeInfo] = field(default_factory=dict[str, AttributeInfo])

    def __post_init__(self):
        """Validate that only one field of each type exists per semantic context."""
        seen: dict[tuple[type[Field], Semantic], str] = {}
        for name, attr in self.attributes.items():
            key = type(attr.annotation), attr.annotation.semantic
            if key in seen:
                raise ValueError(
                    f"Duplicate field type {key[0]} for semantic {key[1]} in schema. "
                    f"Fields '{seen[key]}' and '{name}' conflict."
                )
            seen[key] = name

    def with_categories(self, categories: Dict[str, "Categories"]) -> "Schema":
        """
        Create a new schema with categories applied to specific attributes.

        Args:
            categories: Dictionary mapping attribute names to categories

        Returns:
            A new Schema instance with categories applied

        Raises:
            ValueError: If an attribute name is not found in the schema
        """
        # Make a shallow copy of this schema
        new_schema = copy.copy(self)

        # Also copy the attributes dict to avoid modifying the original AttributeInfo objects
        new_schema.attributes = {
            name: copy.copy(attr_info) for name, attr_info in self.attributes.items()
        }

        # Add categories to specific attributes
        for attr_name, category in categories.items():
            if attr_name in new_schema.attributes:
                new_schema.attributes[attr_name].categories = category
            else:
                raise ValueError(f"Attribute '{attr_name}' not found in schema")

        return new_schema
