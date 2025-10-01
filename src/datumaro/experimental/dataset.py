# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import sys
import types
from functools import cache
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

import polars as pl
from typing_extensions import TypeGuard, TypeVar, dataclass_transform

from .converter_registry import Converter, find_conversion_path
from .schema import AttributeInfo, Field, Schema

if TYPE_CHECKING:
    from .categories import Categories

from typing import Sequence

from .converter_registry import ConversionPaths
from .transform import IdentityTransform, Transform


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
                        self._df = converter.convert(self._df)
                        self._applied_converters.add(id(converter))

        return self._df

    def get_lazy_attributes(self) -> set[str]:
        return self._lazy_outputs

    def slice(self, offset: int, length: int | None = None) -> "Transform":
        instance = copy.copy(self)
        instance._parent = self._parent.slice(offset, length)
        instance._applied_converters = copy.copy(self._applied_converters)
        instance._df = self._df.slice(offset, length)
        return instance

    def __len__(self):
        return len(self._df)


@dataclass_transform()
class Sample:
    """
    Base class for all samples in a dataset.

    This class provides a foundation for creating sample objects with
    schema inference capabilities and flexible attribute assignment.
    """

    def __init__(self, **kwargs: Any):
        """Initialize sample with provided attributes."""
        # Store dataframe and lazy converters for dynamic property loading
        self._dataframe: Optional[pl.DataFrame] = kwargs.pop("_dataframe", None)
        self._lazy_converters: Dict[str, List["Converter"]] = kwargs.pop("_lazy_converters", {})
        self._applied_converters: Set[int] = set()  # Track which converters have been applied
        self._schema: Optional["Schema"] = kwargs.pop(
            "_schema", None
        )  # Store schema for lazy attributes

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    def __repr__(self):
        """Return a string representation of the sample."""
        fields = ", ".join(
            f"{key}={getattr(self, key)}" for key in self.__dict__ if not key.startswith("_")
        )
        return f"{self.__class__.__name__}({fields})"

    @classmethod
    @cache
    def infer_schema(cls) -> Schema:
        """
        Infer schema from this Sample class definition.

        Returns:
            Schema: The inferred schema containing attribute information

        Raises:
            TypeError: If attributes don't have proper Field annotations
        """

        attributes: dict[str, AttributeInfo] = {}
        for name, annotation in cls.__annotations__.items():
            # Resolve string annotations to actual type objects
            # This handles cases where `from __future__ import annotations` is used
            if isinstance(annotation, str):
                try:
                    # Get the module where the class is defined to resolve annotations
                    module = sys.modules[cls.__module__]
                    annotation = eval(annotation, module.__dict__)
                except Exception as e:
                    raise TypeError(
                        f"Failed to resolve type annotation '{annotation}' for attribute '{name}': {e}"
                    )

            origin = get_origin(annotation)
            if origin is Annotated:
                # Handle Annotated[Type, Field] approach
                annotation, *annotations = get_args(annotation)
                field_annotation = annotations[0] if annotations else None
            else:
                # Handle Type = field(...) approach
                field_annotation = getattr(cls, name, None)

            if not isinstance(field_annotation, Field):
                raise TypeError(f"Attribute '{name}' must have a Field annotation.")

            # Extract base class from generic types like MyClass[A, B, C] -> MyClass
            type_origin = get_origin(annotation)

            # For Union types, keep the original annotation (the Union instance)
            # instead of the origin (which is just the UnionType class)
            if (
                sys.version_info >= (3, 10) and isinstance(annotation, types.UnionType)
            ) or type_origin is Union:
                final_type = annotation
            else:
                final_type = type_origin if type_origin is not None else annotation
            attributes[name] = AttributeInfo(type=final_type, annotation=field_annotation)
        return Schema(attributes=attributes)


class LazyDescriptor:
    def __init__(self, attr_name, transforms: Transform):
        self._attr_name = attr_name
        self._transforms = transforms

    def __get__(self, instance, _):
        """
        Create a lazy property that applies converters on demand.

        Args:
            attr_name: Name of the attribute
            attr_info: AttributeInfo for the attribute

        Returns:
            The computed value for the attribute
        """

        row_df = self._transforms.apply([self._attr_name])

        # Now extract the value from the converted dataframe
        attr_info = self._transforms.schema.attributes[self._attr_name]
        value = attr_info.annotation.from_polars(self._attr_name, 0, row_df, attr_info.type)

        # Cache the value and set it as a real attribute
        setattr(instance, self._attr_name, value)

        return value


DType = TypeVar("DType", bound=Sample)
DTargetType = TypeVar("DTargetType", bound=Sample)


class Dataset(Generic[DType]):
    """
    Represents a typed dataset with schema validation and conversion capabilities.

    This class provides a strongly-typed container for tabular data with support
    for complex field types, schema inference, and automatic conversions between
    different schema representations.

    Args:
        DType: The sample type this dataset contains
    """

    def __init__(
        self,
        dtype_or_schema: Union[Schema, Type[DType]],
        categories: dict[str, Categories] = None,
    ):
        """
        Initialize dataset with either a schema or sample type.

        Args:
            dtype_or_schema: Either a Schema instance or a Sample class type
            categories: Optional dictionary mapping attribute names to categories
        """
        if isinstance(dtype_or_schema, Schema):
            self._schema = dtype_or_schema
            self._dtype = cast(Type[DType], Sample)
        else:
            self._schema = dtype_or_schema.infer_schema()
            self._dtype = dtype_or_schema

        # Apply categories if provided
        if categories is not None:
            self._schema = self._schema.with_categories(categories)

        self.df = pl.DataFrame(schema=self._generate_polars_schema())
        self._transforms: Transform | None = None

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        dtype_or_schema: Union[Schema, Type[DTargetType]],
        transforms: Transform | None = None,
        categories: Dict[str, Categories] = None,
    ) -> "Dataset[DTargetType]":
        """
        Create a Dataset from an existing DataFrame and lazy converters.

        Args:
            df: The Polars DataFrame containing the data
            dtype_or_schema: Either a Schema instance or a Sample class type
            transforms: Optional Transform instance to apply during sample access
            categories: Optional dictionary mapping attribute names to categories

        Returns:
            A new Dataset instance with the provided DataFrame and converters
        """
        dataset = Dataset(dtype_or_schema, categories)
        dataset.df = df
        dataset._transforms = transforms
        return dataset

    @property
    def schema(self) -> Schema:
        """Get the schema of this dataset."""
        return self._schema

    def _generate_polars_schema(self) -> pl.Schema:
        """Generate a Polars schema from the dataset's field definitions."""
        schema: dict[str, pl.DataType] = {}
        for key, attr_info in self._schema.attributes.items():
            schema.update(attr_info.annotation.to_polars_schema(key))
        return pl.Schema(schema)

    def append(self, sample: DType):
        """
        Add a new sample to the dataset.

        Args:
            sample: The sample instance to add to the dataset
        """
        if self._transforms is not None:
            raise RuntimeError("Transformed dataset are immutable.")

        series_data: dict[str, pl.Series] = {}
        for key, attr_info in self._schema.attributes.items():
            series_data.update(attr_info.annotation.to_polars(key, getattr(sample, key)))

        new_row = pl.DataFrame(series_data).cast(dict(self.df.schema))  # type: ignore

        # Use vstack instead of extend for object columns since extend doesn't support them
        if any(dtype == pl.Object for dtype in self.df.schema.values()):
            self.df = self.df.vstack(new_row)
        else:
            self.df.extend(new_row)

    def slice(self, offset: int, length: int | None = None) -> Dataset[DType]:
        """
        Create a new dataset that is a slice of this dataset.

        Args:
            offset: The starting index of the slice
            length: The number of samples to include in the slice
        """
        if self._transforms is None:
            slice_df = self.df.slice(offset, length)
            transforms = None
        else:
            transforms = self._transforms.slice(offset, length)
            slice_df = pl.DataFrame()

        dataset = Dataset.from_dataframe(
            slice_df,
            self._dtype,
            transforms,
        )
        dataset._dtype = self._dtype

        return dataset

    def __getitem__(self, row_idx: int) -> DType:
        """
        Retrieve a sample from the dataset by index.

        Args:
            row_idx: The index of the sample to retrieve

        Returns:
            The sample instance at the specified index
        """
        # Extract the row as a single-row DataFrame
        if self._transforms is None:
            row_df = self.df.slice(row_idx, 1)
            lazy_attributes = set()
            transforms = None
        else:
            transforms = self._transforms.slice(row_idx, 1)
            row_df = transforms.apply(transforms.get_batch_attributes())
            lazy_attributes = transforms.get_lazy_attributes()

        # Separate attributes into those available directly and those requiring lazy conversion
        direct_attributes = {}

        for key, attr_info in self._schema.attributes.items():
            if key not in lazy_attributes:
                # This attribute is directly available
                direct_attributes[key] = attr_info.annotation.from_polars(
                    key, 0, row_df, attr_info.type
                )

        # If there are lazy converters, create a dynamic class with descriptors
        dtype = self._dtype

        if lazy_attributes:
            attrs = {}
            for lazy_attr in lazy_attributes:
                attrs[lazy_attr] = LazyDescriptor(lazy_attr, transforms)

            # Create a new dynamic class inheriting from dtype
            dtype = type(dtype.__name__, (dtype,), attrs)

        sample = dtype(
            **direct_attributes,
        )
        return sample

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            The number of samples (rows) in the dataset
        """
        return len(self.df) if self._transforms is None else len(self._transforms)

    def __iter__(self):
        """
        Return an iterator over the samples in the dataset.

        Yields:
            Sample instances from the dataset in order
        """
        for i in range(len(self)):
            yield self[i]

    def __delitem__(self, row_idx: int):
        """
        Delete a sample from the dataset at the specified index.

        Args:
            row_idx: The index of the sample to delete

        Raises:
            IndexError: If the row index is out of bounds
        """
        if self._transforms is not None:
            raise RuntimeError("Transformed dataset are immutable.")

        if row_idx < 0 or row_idx >= len(self.df):
            raise IndexError("Row index out of bounds.")

        # Create a filter to exclude the row at the specified index
        self.df = self.df.with_row_index().filter(pl.col("index") != row_idx).drop("index")

    def __setitem__(self, row_idx: int, sample: DType):
        """
        Update the dataset at the specified index with the given sample.

        Args:
            row_idx: The index to update
            sample: The sample instance to set at the specified index

        Raises:
            IndexError: If the row index is out of bounds
        """
        if self._transforms is not None:
            raise RuntimeError("Transformed dataset are immutable.")

        if row_idx < 0 or row_idx >= len(self.df):
            raise IndexError("Row index out of bounds.")

        series_data: dict[str, pl.Series] = {}
        for key, attr_info in self._schema.attributes.items():
            series_data.update(attr_info.annotation.to_polars(key, getattr(sample, key)))

        updated_row = pl.DataFrame(series_data).cast(dict(self.df.schema))  # type: ignore

        # Update the dataframe by replacing the row at the specified index
        self.df = self.df.with_row_index().select(
            pl.when(pl.col("index") == row_idx).then(updated_row[c]).otherwise(pl.col(c)).alias(c)
            for c in self.df.columns
        )

    def transform(self, transform_factory: Callable[[Transform], Transform]) -> Dataset:
        transforms = self._transforms
        if transforms is None:
            transforms = IdentityTransform(self.df, self.schema)

        transforms = transform_factory(transforms)

        return Dataset.from_dataframe(
            self.df,
            transforms.schema,
            transforms,
        )

    def convert_to_schema(
        self, target_dtype_or_schema: Union[Schema, Type[DTargetType]]
    ) -> "Dataset[DTargetType]":
        """
        Convert this dataset to a new schema using registered converters.

        Args:
            target_dtype_or_schema: The target schema or sample type to convert to

        Returns:
            A new Dataset instance with the converted schema
        """
        # Import the converter implementations to register them
        import datumaro.experimental.converters  # type: ignore[import]  # noqa: F401

        # Determine target schema
        if isinstance(target_dtype_or_schema, Schema):
            target_schema = target_dtype_or_schema
        else:
            target_schema = target_dtype_or_schema.infer_schema()

        # Early return if schemas are already compatible
        if has_schema(self, target_dtype_or_schema):
            # Same schema but mismatching dtype.
            return Dataset.from_dataframe(self.df, target_dtype_or_schema)

        # Find the optimal conversion path using A* search
        conversion_paths, inferred_categories = find_conversion_path(self._schema, target_schema)

        # Create a converter transform
        transforms = self._transforms
        if transforms is None:
            transforms = IdentityTransform(self.df, self.schema)

        transforms = ConverterTransform(transforms, target_schema, conversion_paths)

        # Create new dataset with converted data and inferred categories
        return Dataset.from_dataframe(
            self.df,
            target_dtype_or_schema,
            transforms,
            categories=inferred_categories,
        )


def convert_sample_to_schema(
    sample: Sample,
    source_schema: Schema,
    target_dtype_or_schema: Union[Schema, Type[DTargetType]],
) -> DTargetType:
    """
    Convert a sample to a new schema using registered converters.

    This function creates a temporary dataset, converts it, and returns the
    converted sample. It's useful for one-off conversions without creating
    a full dataset.

    Args:
        sample: The sample instance to convert
        source_schema: The source schema of the sample
        target_schema: The target schema to convert to

    Returns:
        A new Sample instance with the converted schema
    """
    # Create temporary dataset with single sample
    temp_dataset = Dataset(source_schema)
    temp_dataset.append(sample)

    # Convert the dataset
    converted_dataset = temp_dataset.convert_to_schema(target_dtype_or_schema)

    # Return the converted sample
    return converted_dataset[0]


def has_schema(
    dataset: "Dataset[Any]", target_dtype_or_schema: Union[Schema, Type[DTargetType]]
) -> TypeGuard["Dataset[DTargetType]"]:
    """
    Check if a dataset has the specified schema.

    This function performs schema compatibility checking and serves as a
    type guard for type narrowing.

    Args:
        dataset: The dataset to check
        target_dtype_or_schema: The target schema or sample type to check against

    Returns:
        True if the dataset has the specified schema, False otherwise
    """
    if isinstance(target_dtype_or_schema, Schema):
        target_schema = target_dtype_or_schema
    else:
        # For sample type input, infer the schema
        target_schema = target_dtype_or_schema.infer_schema()

    return dataset.schema == target_schema
