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
from typing import Any, Generic, Optional, TypeVar

from datumaro.experimental.categories import Categories
from datumaro.experimental.fields.base import Field


@dataclass
class AttributeInfo:
    """
    Container for attribute type and field annotation information.
    """

    type: type
    field: Field
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
    categories: Categories | None = None


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
        seen: dict[tuple[type[Field], str], str] = {}
        for name, attr in self.attributes.items():
            key = type(attr.field), attr.field.semantic
            if key in seen:
                raise ValueError(
                    f"Duplicate field type {key[0]} for semantic {key[1]} in schema. "
                    f"Fields '{seen[key]}' and '{name}' conflict."
                )
            seen[key] = name

        self._fields_with_categories: dict[str, Categories] | None = None
        self._required_columns_cache: dict[str, set[str]] | None = None

    def get_required_columns(self, attr_name: str) -> set[str]:
        """
        Get the required Polars column names for a given attribute.

        This method caches the results to avoid repeated computation of
        to_polars_schema() on hot paths like __getitem__.

        Args:
            attr_name: The attribute name to get required columns for

        Returns:
            Set of column names required for this attribute
        """
        if self._required_columns_cache is None:
            self._required_columns_cache = {}

        if attr_name not in self._required_columns_cache:
            attr_info = self.attributes[attr_name]
            self._required_columns_cache[attr_name] = set(attr_info.field.to_polars_schema(attr_name).keys())

        return self._required_columns_cache[attr_name]

    def with_categories(self, categories: dict[str, "Categories"]) -> "Schema":
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

        # Reset caches to avoid sharing references with the original schema
        new_schema._fields_with_categories = None
        new_schema._required_columns_cache = None

        # Also copy the attributes dict to avoid modifying the original AttributeInfo objects
        new_schema.attributes = {name: copy.copy(attr_info) for name, attr_info in self.attributes.items()}

        # Add categories to specific attributes
        for attr_name, category in categories.items():
            if attr_name in new_schema.attributes:
                new_schema.attributes[attr_name].categories = category
            else:
                raise ValueError(f"Attribute '{attr_name}' not found in schema")
        return new_schema

    def get_fields_with_required_categories(self) -> dict[str, Categories]:
        """
        Get the attributes that semantically require a category.

        Returns:
            dict mapping attribute names to the respective required categories.
            Attributes that do not strictly require categories are not returned.
        """
        if self._fields_with_categories is None:
            self._fields_with_categories = {}
            for attribute, attribute_info in self.attributes.items():
                if expected_categories_type := attribute_info.field.get_expected_categories_type():
                    categories = attribute_info.categories
                    if categories is None or not isinstance(attribute_info.categories, expected_categories_type):
                        raise ValueError(
                            f"Expected schema attribute '{attribute}' to have categories defined of type "
                            f"'{expected_categories_type}', found '{categories}' instead. "
                            f"Note that schemas (including ones that are attached to datasets) can be created without "
                            f"categories for fields that require them, but for certain operations like appending data "
                            f"categories have to be defined if fields require them in order to validate the data."
                        )
                    self._fields_with_categories[attribute] = categories

        return self._fields_with_categories

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this Schema to a JSON-compatible dictionary.

        Returns:
            Dictionary representation of this Schema instance
        """

        attributes = {}
        categories = {}

        for name, attr_info in self.attributes.items():
            # Store both module and name for better type reconstruction
            type_info = attr_info.type.__name__ if hasattr(attr_info.type, "__name__") else str(attr_info.type)
            type_module = attr_info.type.__module__ if hasattr(attr_info.type, "__module__") else None

            attributes[name] = {
                "type": type_info,
                "type_module": type_module,
                "field": attr_info.field.to_dict(),
            }
            if attr_info.categories is not None:
                categories[name] = attr_info.categories.to_dict()

        return {
            "attributes": attributes,
            "categories": categories,
        }

    @classmethod
    def from_dict(cls, schema_dict: dict[str, Any]) -> "Schema":
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
                field=field,
                categories=categories,
            )

        return cls(attributes=attributes)
