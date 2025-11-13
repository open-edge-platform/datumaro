# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, get_type_hints, overload

from datumaro.v2.converters.base import AttributeRemapperConverter, ConversionError, Converter
from datumaro.v2.fields.base import Field, Semantic
from datumaro.v2.schema import AttributeSpec, Schema


class ConversionPaths(NamedTuple):
    """
    Container for separated batch and lazy conversion paths.

    The batch converters can be applied immediately to the entire DataFrame,
    while lazy converters must be deferred and applied at sample access time.
    """

    converters: dict[str, list[Converter]]
    lazy_outputs: dict[str, list[Converter]]
    required_inputs_by_output: dict[str, set[str]]
    dependent_outputs_by_input: dict[str, set[str]]


class ConverterRegistry:
    """
    Registry for managing and discovering data converters.

    This class maintains a global registry of converter classes and provides
    functionality for finding and instantiating appropriate converters for
    schema transformations.
    """

    _converter_registry: list[type[Converter]] = []

    @classmethod
    def add_converter(cls, converter: type[Converter]):
        """Add a converter class to the registry."""
        cls._converter_registry.append(converter)

    @classmethod
    def remove_converter(cls, converter: type[Converter]) -> None:
        """Remove a converter class from the registry.

        Args:
            converter: The converter class to remove

        Raises:
            ValueError: If the converter is not found in the registry
        """
        cls._converter_registry.remove(converter)

    @classmethod
    def list_converters(cls) -> Sequence[type[Converter]]:
        """List all registered converter classes as an immutable sequence."""
        return cls._converter_registry


@dataclass(frozen=True)
class _SchemaState:
    """Represents a schema state during A* search."""

    field_to_attr_spec: dict[type[Field], AttributeSpec[Field]]  # Map field types to their AttributeSpec

    def __hash__(self):
        # Hash only field types and their properties, not names
        field_items = []
        for field_type, attr_spec in self.field_to_attr_spec.items():
            # Hash field type, field properties, and categories,
            # but not the attribute name
            field_items.append((field_type, attr_spec.field, attr_spec.categories))
        return hash(tuple(field_items))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _SchemaState):
            return False

        # Compare only field types and their properties, not names
        if set(self.field_to_attr_spec.keys()) != set(other.field_to_attr_spec.keys()):
            return False

        for field_type in self.field_to_attr_spec:
            self_attr_spec = self.field_to_attr_spec[field_type]
            other_attr_spec = other.field_to_attr_spec[field_type]

            # Compare field properties
            if self_attr_spec.field != other_attr_spec.field:
                return False

            # Compare categories only if both are not None (per requirements)
            # This is because None categories mean "don't care" in this context
            if (
                self_attr_spec.categories is not None
                and other_attr_spec.categories is not None
                and self_attr_spec.categories != other_attr_spec.categories
            ):
                return False

        return True

    def get_attr_spec_for_field_type(self, field_type: type[Field]) -> AttributeSpec[Field] | None:
        """Get AttributeSpec for a specific field type."""
        return self.field_to_attr_spec.get(field_type)


@dataclass
class _SearchNode:
    """Node in the A* search tree."""

    state: _SchemaState
    path: list[Converter]  # Now stores Converter instances directly
    g_cost: int  # Actual cost from start
    h_cost: int  # Heuristic cost to goal

    @property
    def f_cost(self) -> int:
        """Total cost (g + h)."""
        return self.g_cost + self.h_cost

    def __lt__(self, other: _SearchNode) -> bool:
        return self.f_cost < other.f_cost


def _heuristic_cost(current_state: _SchemaState, target_state: _SchemaState) -> int:
    """
    Heuristic function for A* search.
    Returns the number of missing target fields plus field differences as a heuristic.

    This counts both:
    1. Missing field types that need to be created
    2. Field differences where the type exists but properties differ (dtype, format, semantic, etc.)
    3. Category differences where both input and output categories are not None but differ

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
        if current_attr_spec.field != target_attr_spec.field or (
            current_attr_spec.categories is not None
            and target_attr_spec.categories is not None
            and current_attr_spec.categories != target_attr_spec.categories
        ):
            cost += 1

    return cost


def _get_applicable_converters(
    semantic: Semantic, state: _SchemaState, target_state: _SchemaState, iteration: int = 0
) -> list[tuple[Converter, _SchemaState]]:
    """Get all converters that can be applied to the current state along with their resulting states."""
    applicable: list[tuple[Converter, _SchemaState]] = []

    # Get available field types
    available_field_types = set(state.field_to_attr_spec.keys())

    for converter_class in ConverterRegistry.list_converters():
        # Check if all required input types are available
        from_types = converter_class.get_from_types()

        # Check if we have the required input types
        if not available_field_types.issuperset(from_types.values()):
            continue

        # Collect available input AttributeSpec instances
        converter_kwargs = {
            attr_name: state.field_to_attr_spec[field_type] for attr_name, field_type in from_types.items()
        }

        # Collect desired output AttributeSpec instances
        to_types = converter_class.get_to_types()
        to_attr_specs: list[AttributeSpec[Field]] = []
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

            output_attr_spec = AttributeSpec(name=output_name, field=output_field, categories=output_categories)
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
    groups: dict[Semantic, dict[type[Field], AttributeSpec[Field]]] = defaultdict(dict)

    for attr_name, attr_info in schema.attributes.items():
        semantic = attr_info.field.semantic

        field_type = type(attr_info.field)
        attr_spec = AttributeSpec(name=attr_name, field=attr_info.field, categories=attr_info.categories)
        groups[semantic][field_type] = attr_spec

    # Convert to SchemaState objects
    return {semantic: _SchemaState(field_to_attr_spec) for semantic, field_to_attr_spec in groups.items()}


def _create_initial_renaming_converter(
    start_state: _SchemaState, target_state: _SchemaState
) -> tuple[AttributeRemapperConverter | None, _SchemaState]:
    """
    Create an initial AttributeRemapperConverter to handle renaming at the beginning.

    Args:
        start_state: Starting state of the schema
        target_state: Target state for this semantic group

    Returns:
        Tuple of (optional AttributeRemapperConverter, updated_start_state after renaming)
    """

    attr_mappings = []
    converter_needed = False
    updated_field_to_attr_spec = dict(start_state.field_to_attr_spec)

    # Used to check for conflicts when renaming
    used_names = {updated_field_to_attr_spec[field_type].name for field_type in updated_field_to_attr_spec}

    for field_type, start_attr_spec in start_state.field_to_attr_spec.items():
        if field_type in target_state.field_to_attr_spec:
            target_attr_spec = target_state.field_to_attr_spec[field_type]
            if start_attr_spec.name != target_attr_spec.name:
                converter_needed = True
                new_name = target_attr_spec.name
                # If the new name would conflict with another attribute in the target_state, use a temporary name
                if new_name in used_names:
                    new_name = f"{new_name}_temp_{id(start_attr_spec)}"
                renamed_attr_spec = AttributeSpec(
                    name=new_name,
                    field=start_attr_spec.field,
                    categories=start_attr_spec.categories,
                )
                attr_mappings.append((start_attr_spec, renamed_attr_spec))
                updated_field_to_attr_spec[field_type] = renamed_attr_spec

    if converter_needed:
        converter = AttributeRemapperConverter(attr_mappings=attr_mappings)
        updated_start_state = _SchemaState(updated_field_to_attr_spec)
        return converter, updated_start_state
    return None, start_state


def _can_lazy_converter_handle_conversion(from_field_type: type[Field], to_field_type: type[Field]) -> bool:
    """
    Check if any lazy converter can handle the conversion from one field type to another.

    Args:
        from_field_type: Source field type
        to_field_type: Target field type
        semantic: The semantic context

    Returns:
        True if a lazy converter can handle this conversion, False otherwise
    """
    for converter_class in ConverterRegistry.list_converters():
        # Only consider lazy converters
        if not getattr(converter_class, "lazy", False):
            continue

        from_types = converter_class.get_from_types()
        to_types = converter_class.get_to_types()

        # Check if this converter takes the from_field_type as input and produces to_field_type as output
        if from_field_type in from_types.values() and to_field_type in to_types.values():
            return True

    return False


def _create_conversion_error_message(
    effective_start_state: _SchemaState,
    target_state: _SchemaState,
) -> str:
    """
    Create a detailed error message when no conversion path is found.

    Args:
        effective_start_state: The effective source state after initial renaming
        target_state: Target state for this semantic
        semantic: The semantic tag being processed

    Returns:
        Formatted error message string
    """
    available_source_fields = {
        field_type: attr_spec.field for field_type, attr_spec in effective_start_state.field_to_attr_spec.items()
    }
    required_target_fields = {
        field_type: attr_spec.field for field_type, attr_spec in target_state.field_to_attr_spec.items()
    }

    if available_source_fields:
        available_msg = f"Available source fields: {list(available_source_fields.values())}"
    else:
        available_msg = "No source fields available"

    required_msg = f"Required target fields: {list(required_target_fields.values())}"

    # Check if fields exist by name but need type/dtype conversion
    source_field_names = {attr_spec.name for attr_spec in effective_start_state.field_to_attr_spec.values()}

    type_conversion_issues = []
    truly_missing_fields = []

    for _, target_attr_spec in target_state.field_to_attr_spec.items():
        if target_attr_spec.name in source_field_names:
            # Find the source field with same name
            for _, src_attr_spec in effective_start_state.field_to_attr_spec.items():
                if src_attr_spec.name == target_attr_spec.name:
                    if src_attr_spec.field != target_attr_spec.field and not _can_lazy_converter_handle_conversion(
                        type(src_attr_spec.field),
                        type(target_attr_spec.field),
                    ):
                        # Check if a lazy converter can handle this field property conversion
                        type_conversion_issues.append(
                            f"'{target_attr_spec.name}': {src_attr_spec.field} → {target_attr_spec.field}"
                        )
                    break
        else:
            truly_missing_fields.append(target_attr_spec.name)

    # Create appropriate error message based on the type of issue
    if type_conversion_issues and not truly_missing_fields:
        missing_section = "\nMissing converters for type/dtype conversions:\n" + "\n".join(
            f"  - {issue}" for issue in type_conversion_issues
        )
    elif truly_missing_fields and not type_conversion_issues:
        missing_section = "\nMissing field types:\n" + "\n".join(f"  - {field}" for field in truly_missing_fields)
    elif type_conversion_issues and truly_missing_fields:
        missing_section = (
            "\nMissing field types:\n"
            + "\n".join(f"  - {field}" for field in truly_missing_fields)
            + "\n\nMissing converters for type/dtype conversions:\n"
            + "\n".join(f"  - {issue}" for issue in type_conversion_issues)
        )
    else:
        missing_section = "\nAll required field types are available but conversion failed"

    # Format the complete error message with clear sections
    return f"""No conversion path found.
    
    {available_msg}
    {required_msg}{missing_section}"""


def _find_conversion_path_for_semantic(
    start_state: _SchemaState, target_state: _SchemaState, semantic: Semantic
) -> tuple[list[Converter], _SchemaState]:
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
    # Apply initial renaming at the beginning if needed
    initial_converter, effective_start_state = _create_initial_renaming_converter(start_state, target_state)
    initial_converters = [initial_converter] if initial_converter else []

    # If we already have all required fields after initial renaming, we might be done
    if effective_start_state == target_state:
        return initial_converters, target_state

    # Initialize A* search from the effective start state
    open_set: list[_SearchNode] = []
    closed_set: set[_SchemaState] = set()

    start_node = _SearchNode(
        state=effective_start_state,
        path=initial_converters,  # Add initial converters to the start node path
        g_cost=len(initial_converters),  # Account for initial converters in cost
        h_cost=_heuristic_cost(effective_start_state, target_state),
    )

    heapq.heappush(open_set, start_node)

    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.state in closed_set:
            continue

        closed_set.add(current_node.state)

        # Check if we've reached the goal - all target fields must match exactly
        if _heuristic_cost(current_node.state, target_state) == 0:
            # We've reached the goal, return the path
            return current_node.path, current_node.state

        # Explore neighbors
        for converter, new_state in _get_applicable_converters(
            semantic,
            current_node.state,
            target_state,
            current_node.g_cost,
        ):
            if new_state in closed_set:
                continue

            new_path = [*current_node.path, converter]
            new_g_cost = current_node.g_cost + 1  # Each converter has cost 1
            new_h_cost = _heuristic_cost(new_state, target_state)

            new_node = _SearchNode(state=new_state, path=new_path, g_cost=new_g_cost, h_cost=new_h_cost)

            heapq.heappush(open_set, new_node)

    # No path found - create a more informative error message
    error_msg = _create_conversion_error_message(effective_start_state, target_state)
    raise ConversionError(error_msg)


def _is_converter_lazy(
    converter: Converter,
    lazy_fields: dict[str, bool],
    input_specs: list[AttributeSpec[Field]],
    output_specs: list[AttributeSpec[Field]],
) -> None:
    if converter.lazy or any(attr_spec.name in lazy_fields for attr_spec in input_specs):
        # Mark all output fields as lazy
        for attr_spec in output_specs:
            lazy_fields[attr_spec.name] = True


def _separate_batch_and_lazy_converters(
    conversion_path: list[Converter],
) -> ConversionPaths:
    """
    Separate converters into batch and lazy lists based on dependencies.

    If a converter is lazy, all converters that depend on its output must also be lazy.
    Also tracks which lazy converters are required for each output attribute.

    Args:
        conversion_path: The complete conversion path from A* search

    Returns:
        ConversionPaths with separated batch and lazy converter lists
    """
    if not conversion_path:
        return ConversionPaths(
            converters={},
            lazy_outputs={},
            required_inputs_by_output={},
            dependent_outputs_by_input={},
        )

    # Track which outputs must be lazy
    lazy_fields: dict[str, bool] = defaultdict(bool)  # Maps fields whether they were produced lazily

    required_inputs_by_output: dict[str, set[str]] = defaultdict(set)

    for converter in conversion_path:
        input_specs = converter.get_input_attr_specs()
        output_specs = converter.get_output_attr_specs()
        _is_converter_lazy(
            converter=converter,
            lazy_fields=lazy_fields,
            input_specs=input_specs,
            output_specs=output_specs,
        )

        required_inputs = [required_inputs_by_output.get(attr_spec.name, {attr_spec.name}) for attr_spec in input_specs]
        flattened_required_inputs = set(itertools.chain(*required_inputs))
        for attr_spec in output_specs:
            required_inputs_by_output[attr_spec.name] = flattened_required_inputs

    # Collect lazy converters by output attribute
    converters_by_output: dict[str, list[Converter]] = defaultdict(list)

    # Iterate through converters in reverse to propagate output dependencies
    dependents_by_output: dict[str, set[str]] = defaultdict(set)

    for converter in reversed(conversion_path):
        # This is a lazy converter - track its outputs
        dependents = set()

        output_specs = converter.get_output_attr_specs()
        for attr_spec in output_specs:
            dependents.update(dependents_by_output.get(attr_spec.name, []))
            dependents.add(attr_spec.name)

        for dependent in dependents:
            converters_by_output[dependent].append(converter)

        # Propagate dependencies from outputs to inputs
        input_specs = converter.get_input_attr_specs()
        for input_spec in input_specs:
            dependents_by_output[input_spec.name].update(dependents)

    # Reverse all chains to get dependencies-first order
    for output_name, chain in converters_by_output.items():
        converters_by_output[output_name] = list(reversed(chain))

    return ConversionPaths(
        converters=converters_by_output,
        lazy_outputs=lazy_fields,
        required_inputs_by_output=required_inputs_by_output,
        dependent_outputs_by_input=dependents_by_output,
    )


@overload
def converter(cls: type[Converter], /) -> type[Converter]:
    """Overload for @converter (no parentheses)."""


@overload
def converter(*, lazy: bool = False) -> Callable[[type[Converter]], type[Converter]]:
    """Overload for @converter() or @converter(lazy=True)."""


def converter(
    cls: type[Converter] | None = None, /, *, lazy: bool = False
) -> type[Converter] | Callable[[type[Converter]], type[Converter]]:
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

    def decorator(cls: type[Converter]) -> type[Converter]:
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
