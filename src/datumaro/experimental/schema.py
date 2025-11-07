# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Schema definitions for the dataset system.
"""

import copy
import importlib
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from enum import Flag, auto
from typing import Any, Dict, Generic, Optional, TypeVar

import polars as pl

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

    def __set_name__(self, _, name):
        object.__setattr__(self, "_name", name)

    def __get__(self, instance, _):
        if instance is None:
            return self

        name = getattr(self, "_name")
        value = instance.evaluate_lazy_field(name)

        # Cache the value and set it as a real attribute
        setattr(instance, name, value)

        return value

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, field_dict: Dict[str, Any]) -> "Field":
        """
        Deserialize a Field from a JSON dictionary.

        Automatically reconstructs all dataclass fields by introspection.

        Args:
            field_dict: Dictionary containing field type and parameters

        Returns:
            Reconstructed Field instance
        """
        from . import fields as fields_module

        field_type = field_dict["type"]

        # Get the field class
        field_class = getattr(fields_module, field_type)

        # Prepare kwargs for field construction
        kwargs: Dict[str, Any] = {}

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


@dataclass
class AttributeInfo:
    """
    Container for attribute type and field annotation information.
    """

    type: type
    annotation: Field
    categories: Optional["Categories"] = None


TField = TypeVar("TField", bound=Field)


@dataclass(frozen=True)
class AttributeSpec(Generic[TField]):
    """
    Specification for an attribute used in converters, tilers, etc.

    This class is not part of the schema, but rather used a convenient container
    for passing attribute specifications around in transforms.

    Links an attribute name with its corresponding field type definition,
    providing the complete specification needed for converter operations.

    Args:
        TField: The specific Field type, defaults to Field

    Attributes:
        name: The attribute name
        field: The field type specification
        categories: Optional categories information (e.g., LabelCategories, MaskCategories)
    """

    name: str
    field: TField
    categories: Optional[Categories] = None


BUILTIN_TYPES = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
}


@dataclass
class Schema:
    """
    Represents the schema of a dataset with attribute definitions.
    Enforces that only one field of each type exists per semantic context.
    """

    attributes: dict[str, AttributeInfo] = dataclass_field(default_factory=dict[str, AttributeInfo])

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this Schema to a JSON-compatible dictionary.

        Returns:
            Dictionary representation of this Schema instance
        """

        attributes = {}
        categories = {}

        for name, attr_info in self.attributes.items():
            # Store both module and name for better type reconstruction
            type_info = (
                attr_info.type.__name__
                if hasattr(attr_info.type, "__name__")
                else str(attr_info.type)
            )
            type_module = (
                attr_info.type.__module__ if hasattr(attr_info.type, "__module__") else None
            )

            attributes[name] = {
                "type": type_info,
                "type_module": type_module,
                "field": attr_info.annotation.to_dict(),
            }
            if attr_info.categories is not None:
                categories[name] = attr_info.categories.to_dict()

        return {
            "attributes": attributes,
            "categories": categories,
        }

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]) -> "Schema":
        """
        Deserialize a Schema from a JSON dictionary.

        Args:
            schema_dict: Dictionary containing schema data

        Returns:
            Reconstructed Schema instance
        """

        attributes = {}

        for name, attr_dict in schema_dict["attributes"].items():
            field = Field.from_dict(attr_dict["field"])

            # Get categories if present
            categories = None
            if name in schema_dict.get("categories", {}):
                categories = Categories.from_dict(schema_dict["categories"][name])

            # Try to reconstruct the type
            type_name = attr_dict.get("type", "object")
            type_module = attr_dict.get("type_module")

            # Attempt to reconstruct the actual type
            if type_module and type_module != "builtins":
                try:
                    module = importlib.import_module(type_module)
                    attr_type = getattr(module, type_name, object)
                except (ImportError, AttributeError):
                    attr_type = object
            elif type_name in BUILTIN_TYPES:
                # Handle built-in types
                attr_type = BUILTIN_TYPES.get(type_name, object)
            else:
                # Default to object as placeholder
                attr_type = object

            attributes[name] = AttributeInfo(
                type=attr_type,
                annotation=field,
                categories=categories,
            )

        return cls(attributes=attributes)
