# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Converter system for transforming data between different field representations.

This module provides the foundation for data transformation pipelines,
including converter registration, schema mapping, and automatic conversion
path discovery using graph algorithms.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    get_type_hints,
    overload,
)

import polars as pl
from typing_extensions import cast, dataclass_transform

from .categories import Categories
from .schema import Field, Schema, Semantic

TField = TypeVar("TField", bound=Field)


class ConversionPaths(NamedTuple):
    """
    Container for separated batch and lazy conversion paths.

    The batch converters can be applied immediately to the entire DataFrame,
    while lazy converters must be deferred and applied at sample access time.
    """

    batch_converters: List["Converter"]
    lazy_converters: List["Converter"]


@dataclass(frozen=True)
class AttributeSpec(Generic[TField]):
    """
    Specification for an attribute used in converters.

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
    def get_from_types(cls) -> dict[str, Type[Field]]:
        """
        Extract input field types from input_* class attributes.

        Returns:
            Dictionary mapping input attribute names to their Field types
        """
        from_types: dict[str, Type[Field]] = {}

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
    def get_to_types(cls) -> dict[str, Type[Field]]:
        """
        Extract output field types from output_* class attributes.

        Returns:
            Dictionary mapping output attribute names to their Field types
        """
        to_types: dict[str, Type[Field]] = {}

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
        pass

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

    def get_input_attr_specs(self) -> List[AttributeSpec[Field]]:
        """
        Get the current input AttributeSpec instances from input_* attributes.

        Returns:
            List of input AttributeSpec instances currently configured on the converter
        """
        input_attr_specs: List[AttributeSpec[Field]] = []

        # Get the input attribute names from class type hints
        from_types = self.get_from_types()

        for attr_name in from_types.keys():
            attr_spec = cast(AttributeSpec[Field], getattr(self, attr_name))
            input_attr_specs.append(attr_spec)

        return input_attr_specs

    def get_output_attr_specs(self) -> List[AttributeSpec[Field]]:
        """
        Get the current output AttributeSpec instances from output_* attributes.

        Returns:
            List of output AttributeSpec instances currently configured on the converter
        """
        output_attr_specs: List[AttributeSpec[Field]] = []

        # Get the output attribute names from class type hints
        to_types = self.get_to_types()

        for attr_name in to_types.keys():
            attr_spec = cast(AttributeSpec[Field], getattr(self, attr_name))
            output_attr_specs.append(attr_spec)

        return output_attr_specs


class ConverterRegistry:
    """
    Registry for managing and discovering data converters.

    This class maintains a global registry of converter classes and provides
    functionality for finding and instantiating appropriate converters for
    schema transformations.
    """

    _converter_registry: List[Type[Converter]] = []

    @classmethod
    def add_converter(cls, converter: Type[Converter]):
        """Add a converter class to the registry."""
        cls._converter_registry.append(converter)

    @classmethod
    def remove_converter(cls, converter: Type[Converter]) -> None:
        """Remove a converter class from the registry.

        Args:
            converter: The converter class to remove

        Raises:
            ValueError: If the converter is not found in the registry
        """
        cls._converter_registry.remove(converter)

    @classmethod
    def list_converters(cls) -> Sequence[Type[Converter]]:
        """List all registered converter classes as an immutable sequence."""
        return cls._converter_registry


@overload
def converter(cls: Type[Converter], /) -> Type[Converter]:
    """Overload for @converter (no parentheses)."""
    ...


@overload
def converter(*, lazy: bool = False) -> Callable[[Type[Converter]], Type[Converter]]:
    """Overload for @converter() or @converter(lazy=True)."""
    ...


def converter(
    cls: Optional[Type[Converter]] = None, /, *, lazy: bool = False
) -> Type[Converter] | Callable[[Type[Converter]], Type[Converter]]:
    """Register a converter class and configure its lazy loading behavior.

    This decorator automatically registers converter classes with the global
    converter registry and sets their lazy evaluation mode. The converter
    class must define at least one output_* attribute with type hints.

    Args:
        lazy: If True, this converter will only be applied during lazy
              evaluation in Dataset.__getitem__. If False, it will be
              applied during batch conversion operations. Lazy converters
              automatically make all dependent converters lazy as well.

    Usage:
        @converter
        class ImageToTensorConverter(Converter):
            input_image: AttributeSpec
            output_tensor: AttributeSpec

            def convert(self, df: pl.DataFrame) -> pl.DataFrame:
                # conversion logic
                return df

        @converter(lazy=True)
        class ImagePathToImageConverter(Converter):
            input_path: AttributeSpec
            output_image: AttributeSpec

            def convert(self, df: pl.DataFrame) -> pl.DataFrame:
                # lazy conversion logic
                return df
    """

    def decorator(cls: Type[Converter]) -> Type[Converter]:
        # Validate converter class by checking for required attributes
        hints = get_type_hints(cls)

        # Ensure at least one output attribute is defined
        output_attrs = [name for name in hints if name.startswith("output_")]
        if not output_attrs:
            raise TypeError(f"{cls.__name__} must define at least one 'output_*' attribute")

        # Set the lazy attribute directly on the class
        cls.lazy = lazy

        # Register with the global converter registry for discovery
        ConverterRegistry.add_converter(cls)

        return cls

    # Handle both @converter and @converter() syntax patterns
    if cls is None:
        # Called with parentheses: @converter() or @converter(lazy=True)
        return decorator

    # Called without parentheses: @converter
    return decorator(cls)


class ConversionError(Exception):
    """Exception raised when conversion fails."""

    pass


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
    def get_from_types(self) -> dict[str, Type[Field]]:
        """
        Extract input field types from input_* class attributes.

        Returns:
            Dictionary mapping input attribute names to their Field types
        """
        from_types: dict[str, Type[Field]] = {}
        for i, (input_spec, _) in enumerate(self.attr_mappings):
            from_types[f"input_{i}"] = type(input_spec.field)

        return from_types

    @cache
    def get_to_types(self) -> dict[str, Type[Field]]:
        """
        Extract output field types from output_* class attributes.

        Returns:
            Dictionary mapping output attribute names to their Field types
        """
        to_types: dict[str, Type[Field]] = {}
        for i, (_, output_spec) in enumerate(self.attr_mappings):
            to_types[f"output_{i}"] = type(output_spec.field)

        return to_types

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Select and rename columns according to column_map.
        Columns not in the mapping are dropped.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with only the selected/renamed columns
        """
        # Build selection expressions for all columns we want to keep
        select_exprs = {}

        for old_name, new_name in self.column_map.items():
            select_exprs[new_name] = pl.col(old_name)

        return df.select(**select_exprs)

    def filter_output_spec(self) -> bool:
        """Always return True as renaming is always applicable."""
        return True


@dataclass(frozen=True)
class _SchemaState:
    """Represents a schema state during A* search."""

    field_to_attr_spec: dict[
        Type[Field], AttributeSpec[Field]
    ]  # Map field types to their AttributeSpec

    def __hash__(self):
        # Hash only field types and their properties, not names
        field_items = []
        for field_type, attr_spec in self.field_to_attr_spec.items():
            # Hash field type and field properties, but not the attribute name
            field_items.append((field_type, attr_spec.field))
        return hash(tuple(field_items))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _SchemaState):
            return False

        # Compare only field types and their properties, not names
        if set(self.field_to_attr_spec.keys()) != set(other.field_to_attr_spec.keys()):
            return False

        for field_type in self.field_to_attr_spec.keys():
            if (
                self.field_to_attr_spec[field_type].field
                != other.field_to_attr_spec[field_type].field
            ):
                return False

        return True

    def get_attr_spec_for_field_type(
        self, field_type: Type[Field]
    ) -> Optional[AttributeSpec[Field]]:
        """Get AttributeSpec for a specific field type."""
        return self.field_to_attr_spec.get(field_type)


@dataclass
class _SearchNode:
    """Node in the A* search tree."""

    state: _SchemaState
    path: List[Converter]  # Now stores Converter instances directly
    g_cost: int  # Actual cost from start
    h_cost: int  # Heuristic cost to goal

    @property
    def f_cost(self) -> int:
        """Total cost (g + h)."""
        return self.g_cost + self.h_cost

    def __lt__(self, other: "_SearchNode") -> bool:
        return self.f_cost < other.f_cost


def _heuristic_cost(current_state: _SchemaState, target_state: _SchemaState) -> int:
    """
    Heuristic function for A* search.
    Returns the number of missing target fields plus field differences as a heuristic.

    This counts both:
    1. Missing field types that need to be created
    2. Field differences where the type exists but properties differ (dtype, format, semantic, etc.)

    Note: Attribute names are ignored in the heuristic as they can be fixed in post-processing.
    """
    cost = 0

    current_field_types = set(current_state.field_to_attr_spec.keys())
    target_field_types = set(target_state.field_to_attr_spec.keys())

    # Count missing field types
    missing_field_types = target_field_types - current_field_types
    cost += len(missing_field_types)

    # Count field differences for types that exist in both states
    common_field_types = current_field_types & target_field_types
    for field_type in common_field_types:
        current_attr_spec = current_state.field_to_attr_spec[field_type]
        target_attr_spec = target_state.field_to_attr_spec[field_type]

        # Compare field properties (ignoring names) - if they differ, we need conversion
        if current_attr_spec.field != target_attr_spec.field:
            cost += 1

    return cost


def _get_applicable_converters(
    semantic: Semantic, state: _SchemaState, target_state: _SchemaState, iteration: int = 0
) -> List[Tuple[Converter, _SchemaState]]:
    """Get all converters that can be applied to the current state along with their resulting states."""
    applicable: List[Tuple[Converter, _SchemaState]] = []

    # Get available field types
    available_field_types = set(state.field_to_attr_spec.keys())

    for converter_class in ConverterRegistry.list_converters():
        # Check if all required input types are available
        from_types = converter_class.get_from_types()

        # Collect available input AttributeSpec instances
        converter_kwargs = {}
        all_inputs_available = True
        for attr_name, field_type in from_types.items():
            if field_type not in available_field_types:
                all_inputs_available = False
                break

            # Add the input attribute to kwargs for the converter constructor
            attr_spec = state.field_to_attr_spec[field_type]
            converter_kwargs[attr_name] = attr_spec

        # Check if we have the required input types
        if not all_inputs_available:
            continue

        # Collect desired output AttributeSpec instances
        to_types = converter_class.get_to_types()
        to_attr_specs: List[AttributeSpec[Field]] = []
        for field_type in to_types.values():
            if field_type in target_state.field_to_attr_spec:
                attr_spec = target_state.field_to_attr_spec[field_type]
                to_attr_specs.append(attr_spec)

        for attr_name, field_type in to_types.items():
            # First, check if target state has a matching field type and use its name/field
            if field_type in target_state.field_to_attr_spec:
                target_attr_spec = target_state.field_to_attr_spec[field_type]
                output_name = target_attr_spec.name
                output_field = target_attr_spec.field
                output_categories = target_attr_spec.categories
            else:
                # The field does not exist, use a temporary name
                output_name = field_type.__name__.lower()
                # and create a new instance of the field
                output_field = field_type(semantic=semantic)
                output_categories = None

            # Add the iteration count at the end to ensure uniqueness
            # and avoid any conflict with existing attribute names
            output_name = f"{output_name}_temp_{iteration}"

            output_attr_spec = AttributeSpec(
                name=output_name, field=output_field, categories=output_categories
            )
            converter_kwargs[attr_name] = output_attr_spec

        # Create converter instance with all AttributeSpec instances as kwargs
        converter_instance = converter_class(**converter_kwargs)
        if not converter_instance.filter_output_spec():
            continue

        # Fetch the updated output AttributeSpec instances after filter_output_spec()
        updated_output_attr_specs = converter_instance.get_output_attr_specs()

        # Apply converter to get new state
        new_field_to_attr_spec = dict(state.field_to_attr_spec)

        # Keep old attributes instead of removing them
        # (Later we can work on cleanup/rename logic separately)

        # Add produced output types using the updated output_attr_specs
        for attr_spec in updated_output_attr_specs:
            field_type = type(attr_spec.field)
            new_field_to_attr_spec[field_type] = attr_spec

        new_state = _SchemaState(new_field_to_attr_spec)

        applicable.append((converter_instance, new_state))

    return applicable


def _group_fields_by_semantic(schema: Schema) -> dict[Semantic, _SchemaState]:
    """
    Group schema attributes by their semantic tags and return as SchemaState objects.

    Args:
        schema: Schema to group

    Returns:
        Dictionary mapping semantic tags to SchemaState objects
    """
    groups: dict[Semantic, dict[Type[Field], AttributeSpec[Field]]] = defaultdict(dict)

    for attr_name, attr_info in schema.attributes.items():
        semantic = attr_info.annotation.semantic

        field_type = type(attr_info.annotation)
        attr_spec = AttributeSpec(
            name=attr_name, field=attr_info.annotation, categories=attr_info.categories
        )
        groups[semantic][field_type] = attr_spec

    # Convert to SchemaState objects
    return {
        semantic: _SchemaState(field_to_attr_spec)
        for semantic, field_to_attr_spec in groups.items()
    }


def _create_post_processing_for_semantic(
    final_state: _SchemaState, target_state: _SchemaState
) -> Tuple[List[Converter], _SchemaState]:
    """
    Create post-processing converters for a single semantic group.

    Args:
        final_state: Final state reached after conversions
        target_state: Target state for this semantic group

    Returns:
        Tuple of (list of post-processing converters, updated_state_after_processing)
        where updated_state_after_processing reflects the state after renaming/deletion
    """
    # Build attribute mappings: include only attributes that should be kept
    attr_mappings = []

    # Determine if a converter is needed (i.e. at least one attribute has been renamed or deleted)
    converter_needed = False

    for field_type, target_attr_spec in target_state.field_to_attr_spec.items():
        if field_type in final_state.field_to_attr_spec:
            final_attr_spec = final_state.field_to_attr_spec[field_type]

            if final_attr_spec.name != target_attr_spec.name:
                converter_needed = True

            # Add the mapping from final to target attribute spec
            attr_mappings.append((final_attr_spec, target_attr_spec))

    # Check if any fields need to be deleted (exist in final but not in target)
    for field_type, final_attr_spec in final_state.field_to_attr_spec.items():
        if field_type not in target_state.field_to_attr_spec:
            # If the field is not in the target state, it should be deleted
            converter_needed = True

    # Create a single remapper converter that handles both renaming and deletion
    if converter_needed:
        # Create the updated state after processing by applying the attr_mappings to final_state
        # This preserves the categories from final_state but with the target names/structure
        updated_field_to_attr_spec = {}

        for final_attr_spec, target_attr_spec in attr_mappings:
            # Use the target name and field type, but preserve categories from final state
            updated_field_to_attr_spec[type(target_attr_spec.field)] = AttributeSpec(
                name=target_attr_spec.name,
                field=target_attr_spec.field,
                categories=final_attr_spec.categories,  # Preserve inferred categories
            )

        updated_state_after_processing = _SchemaState(updated_field_to_attr_spec)
        return [
            AttributeRemapperConverter(attr_mappings=attr_mappings)
        ], updated_state_after_processing

    # No converter needed, return the final state as-is
    return [], final_state


def _find_conversion_path_for_semantic(
    start_state: _SchemaState, target_state: _SchemaState, semantic: Semantic
) -> Tuple[List[Converter], _SchemaState]:
    """
    Find conversion path for fields with a specific semantic tag.

    Args:
        start_state: Source state for this semantic
        target_state: Target state for this semantic
        semantic: The semantic tag being processed

    Returns:
        Tuple of (list of converters needed for this semantic group, updated target state)

    Raises:
        ConversionError: If no conversion path is found for this semantic
    """
    # If we already have all required fields, check if we need renaming/deletion
    if start_state == target_state:
        return _create_post_processing_for_semantic(start_state, target_state)

    # Initialize A* search
    open_set: List[_SearchNode] = []
    closed_set: Set[_SchemaState] = set()

    start_node = _SearchNode(
        state=start_state,
        path=[],
        g_cost=0,
        h_cost=_heuristic_cost(start_state, target_state),
    )

    heapq.heappush(open_set, start_node)

    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.state in closed_set:
            continue

        closed_set.add(current_node.state)

        # Check if we've reached the goal - all target fields must match exactly
        if _heuristic_cost(current_node.state, target_state) == 0:
            # Add post-processing converters for final renaming and deletion
            post_processing, final_state = _create_post_processing_for_semantic(
                current_node.state, target_state
            )
            return current_node.path + post_processing, final_state

        # Explore neighbors
        for converter, new_state in _get_applicable_converters(
            semantic,
            current_node.state,
            target_state,
            current_node.g_cost,
        ):
            if new_state in closed_set:
                continue

            new_path = current_node.path + [converter]
            new_g_cost = current_node.g_cost + 1  # Each converter has cost 1
            new_h_cost = _heuristic_cost(new_state, target_state)

            new_node = _SearchNode(
                state=new_state, path=new_path, g_cost=new_g_cost, h_cost=new_h_cost
            )

            heapq.heappush(open_set, new_node)

    # No path found
    missing_fields = set(target_state.field_to_attr_spec.keys()) - set(
        start_state.field_to_attr_spec.keys()
    )
    raise ConversionError(
        f"No conversion path found for semantic {semantic}. " f"Missing fields: {missing_fields}"
    )


def find_conversion_path(
    from_schema: Schema, to_schema: Schema
) -> Tuple[ConversionPaths, Dict[str, Categories]]:
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
    all_converters: List[Converter] = []

    attr_mappings = []

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

        # Merge all the attribute remappers into a single one. If any, the remapper is always the last step.
        if semantic_converters and isinstance(semantic_converters[-1], AttributeRemapperConverter):
            attr_mappings += semantic_converters[-1].attr_mappings
            semantic_converters = semantic_converters[:-1]

        all_converters.extend(semantic_converters)

    if attr_mappings:
        all_converters.append(AttributeRemapperConverter(attr_mappings=attr_mappings))

    # Reconstruct the updated schema with inferred categories
    inferred_categories: dict[str, Categories] = {}
    for semantic, updated_target_state in target_groups.items():
        for attr_spec in updated_target_state.field_to_attr_spec.values():
            if attr_spec.categories is not None:
                inferred_categories[attr_spec.name] = attr_spec.categories

    # Separate batch and lazy converters
    conversion_paths = _separate_batch_and_lazy_converters(all_converters)

    return conversion_paths, inferred_categories


def _separate_batch_and_lazy_converters(
    conversion_path: List[Converter],
) -> ConversionPaths:
    """
    Separate converters into batch and lazy lists based on dependencies.

    If a converter is lazy, all converters that depend on its output must also be lazy.

    Args:
        conversion_path: The complete conversion path from A* search

    Returns:
        ConversionPaths with separated batch and lazy converter lists
    """
    if not conversion_path:
        return ConversionPaths(batch_converters=[], lazy_converters=[])

    # Track which converters must be lazy
    lazy_indices: Set[int] = set()

    lazy_fields: dict[str, bool] = defaultdict(
        bool
    )  # Maps fields whether they were produced lazily

    for i, converter in enumerate(conversion_path):
        lazy = False

        if converter.lazy:
            # Mark all intrinsically lazy converters as lazy
            lazy = True
        else:
            # Check whether the converter depends on a lazy converter
            input_specs = converter.get_input_attr_specs()
            for attr_spec in input_specs:
                if attr_spec.name in lazy_fields:
                    lazy = True
                    break

        if lazy:
            lazy_indices.add(i)

            # Mark all output fields as lazy
            output_specs = converter.get_output_attr_specs()
            for attr_spec in output_specs:
                lazy_fields[attr_spec.name] = True

    # Separate into batch and lazy lists
    batch_converters: List[Converter] = []
    lazy_converters: List[Converter] = []

    for i, converter in enumerate(conversion_path):
        if i in lazy_indices:
            lazy_converters.append(converter)
        else:
            batch_converters.append(converter)

    return ConversionPaths(batch_converters=batch_converters, lazy_converters=lazy_converters)
