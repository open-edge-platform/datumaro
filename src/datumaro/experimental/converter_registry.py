# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Converter system for transforming data between different field representations.

This module provides the foundation for data transformation pipelines,
including converter registration, schema mapping, and automatic conversion
path discovery using graph algorithms.
"""

import heapq
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from typing import (
    Any,
    Callable,
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
    """

    name: str
    field: TField


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


@dataclass(frozen=True)
class _SchemaState:
    """Represents a schema state during A* search."""

    field_to_attr_spec: dict[
        Type[Field], AttributeSpec[Field]
    ]  # Map field types to their AttributeSpec

    def __hash__(self):
        # Hash the items of the dict as a frozenset for immutability
        return hash(frozenset(self.field_to_attr_spec.items()))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _SchemaState) and self.field_to_attr_spec == other.field_to_attr_spec
        )

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

        # Compare field properties - if they differ, we need conversion
        if current_attr_spec.field != target_attr_spec.field:
            cost += 1

    return cost


def _get_applicable_converters(
    state: _SchemaState, target_state: _SchemaState
) -> List[Tuple[Converter, _SchemaState]]:
    """Get all converters that can be applied to the current state along with their resulting states."""
    applicable: List[Tuple[Converter, _SchemaState]] = []

    # Get available field types
    available_field_types = set(state.field_to_attr_spec.keys())

    for converter_class in ConverterRegistry.list_converters():
        # Check if all required input types are available
        from_types = converter_class.get_from_types()
        if (
            not all(field_type in available_field_types for field_type in from_types.values())
            and len(from_types) > 0
        ):
            continue

        # Collect available input AttributeSpec instances
        from_attr_specs: List[AttributeSpec[Field]] = []
        for field_type in from_types.values():
            if field_type in available_field_types:
                attr_spec = state.field_to_attr_spec[field_type]
                from_attr_specs.append(attr_spec)

        # Collect desired output AttributeSpec instances
        to_types = converter_class.get_to_types()
        to_attr_specs: List[AttributeSpec[Field]] = []
        for field_type in to_types.values():
            if field_type in target_state.field_to_attr_spec:
                attr_spec = target_state.field_to_attr_spec[field_type]
                to_attr_specs.append(attr_spec)

        # Initialize the converter with AttributeSpec instances
        # Check if we have the required input types
        if len(from_attr_specs) < len(from_types):
            continue

        # Create a mapping from input types to input specs
        input_type_to_spec = {}
        for attr_spec in from_attr_specs:
            field_type = type(attr_spec.field)
            input_type_to_spec[field_type] = attr_spec

        # Verify all required input types are available
        all_inputs_available = True
        converter_kwargs = {}
        for attr_name, field_type in from_types.items():
            if field_type not in input_type_to_spec:
                all_inputs_available = False
                break
            # Add the input attribute to kwargs for the converter constructor
            converter_kwargs[attr_name] = input_type_to_spec[field_type]

        if not all_inputs_available:
            continue

        # Create output AttributeSpec instances and add to kwargs
        output_attr_specs: List[AttributeSpec[Field]] = []
        for i, (attr_name, field_type) in enumerate(to_types.items()):
            # Create a default output AttributeSpec
            # Use desired target name if specified, otherwise generate default
            if i < len(to_attr_specs):
                output_name = to_attr_specs[i].name
                # Optionally use the target field if it matches the type
                if isinstance(to_attr_specs[i].field, field_type):
                    output_field = to_attr_specs[i].field
                else:
                    output_field = field_type()
            else:
                field_hash = abs(hash(str(field_type))) % 10000
                output_name = f"{field_type.__name__.lower()}_{field_hash}"
                output_field = field_type()

            output_attr_spec = AttributeSpec(name=output_name, field=output_field)
            converter_kwargs[attr_name] = output_attr_spec
            output_attr_specs.append(output_attr_spec)

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
        attr_spec = AttributeSpec(name=attr_name, field=attr_info.annotation)
        groups[semantic][field_type] = attr_spec

    # Convert to SchemaState objects
    return {
        semantic: _SchemaState(field_to_attr_spec)
        for semantic, field_to_attr_spec in groups.items()
    }


def _find_conversion_path_for_semantic(
    start_state: _SchemaState, target_state: _SchemaState, semantic: Semantic
) -> List[Converter]:
    """
    Find conversion path for fields with a specific semantic tag.

    Args:
        start_state: Source state for this semantic
        target_state: Target state for this semantic
        semantic: The semantic tag being processed

    Returns:
        List of converters needed for this semantic group

    Raises:
        ConversionError: If no conversion path is found for this semantic
    """
    # If we already have all required fields, no conversion needed
    if start_state == target_state:
        return []

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
            return current_node.path

        # Explore neighbors
        for converter, new_state in _get_applicable_converters(
            current_node.state,
            target_state,
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


def find_conversion_path(from_schema: Schema, to_schema: Schema) -> ConversionPaths:
    """
    Find an optimal sequence of converters using A* search, grouped by semantic.

    Fields with the same semantic can be converted between each other, but
    conversion across semantic boundaries is not allowed.

    Args:
        from_schema: Source schema
        to_schema: Target schema

    Returns:
        ConversionPaths with separated batch and lazy converter lists

    Raises:
        ConversionError: If no conversion path is found
    """
    # Group fields by semantic in both schemas
    start_groups = _group_fields_by_semantic(from_schema)
    target_groups = _group_fields_by_semantic(to_schema)

    # Collect all converters needed across all semantic groups
    all_converters: List[Converter] = []

    # Process each semantic group in the target schema
    for semantic, target_state in target_groups.items():
        # Get corresponding source state for this semantic (if any)
        start_state = start_groups.get(semantic, _SchemaState({}))

        # Find conversion path for this semantic group
        semantic_converters = _find_conversion_path_for_semantic(
            start_state, target_state, semantic
        )

        all_converters.extend(semantic_converters)

    # Separate batch and lazy converters
    return _separate_batch_and_lazy_converters(all_converters)


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

    # First pass: mark all intrinsically lazy converters
    for i, converter in enumerate(conversion_path):
        if converter.lazy:
            lazy_indices.add(i)

    # Second pass: mark converters that depend on lazy converter outputs
    # Build dependency graph
    converter_outputs: dict[
        Type[Field], int
    ] = {}  # Maps field types to converter indices that produce them

    for i, converter in enumerate(conversion_path):
        output_specs = converter.get_output_attr_specs()
        for attr_spec in output_specs:
            field_type = type(attr_spec.field)
            converter_outputs[field_type] = i

    # Mark dependent converters as lazy
    changed = True
    while changed:
        changed = False
        for i, converter in enumerate(conversion_path):
            if i in lazy_indices:
                continue

            # Check if this converter depends on any lazy converter output
            from_types = converter.get_from_types()
            for field_type in from_types.values():
                if field_type in converter_outputs:
                    producer_index: int = converter_outputs[field_type]
                    if producer_index in lazy_indices and producer_index < i:
                        lazy_indices.add(i)
                        changed = True
                        break

    # Separate into batch and lazy lists
    batch_converters: List[Converter] = []
    lazy_converters: List[Converter] = []

    for i, converter in enumerate(conversion_path):
        if i in lazy_indices:
            lazy_converters.append(converter)
        else:
            batch_converters.append(converter)

    return ConversionPaths(batch_converters=batch_converters, lazy_converters=lazy_converters)
