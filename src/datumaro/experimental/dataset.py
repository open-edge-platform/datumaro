# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
import types
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, Generic, Type, Union, cast, get_args, get_origin

import polars as pl
from typing_extensions import TypeGuard, TypeVar, dataclass_transform

from .categories import Categories
from .converter_registry import ConverterTransform, find_conversion_path
from .io import (
    MediaCopyMode,
    PathStyle,
    build_sidecar_dict,
    create_zip_archive,
    deserialize_categories_map,
    detect_file_path_fields,
    extract_zip_to_dir,
    package_and_rewrite_paths,
    read_sidecar_file,
    rebase_media_paths,
    write_sidecar_file,
)
from .schema import AttributeInfo, Field, Schema
from .transform import IdentityTransform, Transform


@dataclass_transform()
class Sample:
    """
    Base class for all samples in a dataset.

    This class provides a foundation for creating sample objects with
    schema inference capabilities and flexible attribute assignment.
    """

    def __init__(self, **kwargs: Any):
        """Initialize sample with provided attributes."""
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

    def evaluate_lazy_field(self, name: str) -> Any:
        row_df = self._transforms.apply([name])

        # Now extract the value from the converted dataframe
        attr_info = self._transforms.schema.attributes[name]
        value = attr_info.annotation.from_polars(name, 0, row_df, attr_info.type)

        return value


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
        schema: Schema | None = None,
    ):
        """
        Initialize dataset with either a schema or sample type.

        Args:
            dtype_or_schema: Either a Schema instance or a Sample class type
            categories: Optional dictionary mapping attribute names to categories
            schema: Optional schema if a dtype is provided
        """
        if isinstance(dtype_or_schema, Schema):
            self._schema = dtype_or_schema
            self._dtype = cast(Type[DType], Sample)
        else:
            self._schema = dtype_or_schema.infer_schema() if schema is None else schema
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
        schema: Schema | None = None,
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
        dataset = Dataset(dtype_or_schema, categories, schema)
        dataset.df = df
        dataset._transforms = transforms
        return dataset

    @property
    def schema(self) -> Schema:
        """Get the schema of this dataset."""
        return self._schema

    @property
    def dtype(self) -> Type[DType]:
        """Get the sample type of this dataset."""
        return self._dtype

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
            direct_attributes["_transforms"] = transforms
            # attrs = {}
            # for lazy_attr in lazy_attributes:
            #    attrs[lazy_attr] = LazyDescriptor(lazy_attr, transforms)

            ## Create a new dynamic class inheriting from dtype
            # dtype = type(dtype.__name__, (dtype,), attrs)
            # dtype.__annotations__ = self._dtype.__annotations__

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

    def transform(
        self,
        transform_factory: Callable[[Transform], Transform],
        dtype: Type[DTargetType] | None = None,
    ) -> Dataset[DTargetType]:
        transforms = self._transforms
        if transforms is None:
            transforms = IdentityTransform(self.df, self.schema)

        transforms = transform_factory(transforms)

        if dtype is None:
            return Dataset.from_dataframe(
                self.df,
                transforms.schema,
                transforms,
            )
        else:
            return Dataset.from_dataframe(
                self.df,
                dtype,
                transforms,
                schema=transforms.schema,
            )

    def convert_to_schema(
        self,
        target_dtype_or_schema: Union[Schema, Type[DTargetType]],
        target_categories: Dict[str, Categories] = None,
    ) -> "Dataset[DTargetType]":
        """
        Convert this dataset to a new schema using registered converters.

        Args:
            target_dtype_or_schema: The target schema or sample type to convert to

        Returns:
            A new Dataset instance with the converted schema
        """
        # Determine target schema
        if isinstance(target_dtype_or_schema, Schema):
            target_schema = target_dtype_or_schema
        else:
            target_schema = target_dtype_or_schema.infer_schema()

        if target_categories is not None:
            target_schema = target_schema.with_categories(target_categories)

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

    def save_parquet(
        self,
        path: str,
        *,
        save_media: bool = False,
        media_dir: str = "media",
        copy_mode: MediaCopyMode = MediaCopyMode.COPY,
        path_style: PathStyle = PathStyle.RELATIVE,
        archive: bool = False,
        archive_path: str | None = None,
    ) -> None:
        """
        Save the dataset to a Parquet file on disk with an adjacent sidecar and optional media packaging/archiving.

        What gets written:
        - The internal Polars DataFrame to a `.parquet` file at `path`.
        - A sidecar JSON (`<path_without_ext>.json`) that records schema and categories, and
          optionally a `media` block describing how image paths were stored.
        - If requested, a zip archive that bundles the parquet, sidecar, and media directory.

        Args:
            path: Destination path to write the Parquet file to. The sidecar JSON will be
                created next to this file using the same stem and the `.json` extension.
            save_media: When True, any columns declared as `ImagePathField` in the schema are
                treated as file-system paths. Their referenced files are copied/symlinked/moved
                into `media_dir` placed next to the Parquet, and the column values are rewritten
                to portable paths according to `path_style`.
            media_dir: Name of the directory (created next to the Parquet) that will contain
                packaged media files when `save_media=True`. Defaults to "media".
            copy_mode: How to materialize media into `media_dir`. One of
                `MediaCopyMode.COPY`, `MediaCopyMode.SYMLINK`, or `MediaCopyMode.MOVE`.
                The chosen mode is recorded in the sidecar `media.copy_mode` as a string.
                Note: `SYMLINK` may fall back to copy on platforms that disallow symlinks.
            path_style: How to store the rewritten path values in the Parquet table for
                `ImagePathField` columns. Use `PathStyle.RELATIVE` to store POSIX-style
                relative paths like `media/xxx.png` (recommended for portability), or
                `PathStyle.ABSOLUTE` to store absolute OS paths to the packaged files.
            archive: When True, create a `.zip` archive next to the Parquet containing the
                Parquet file, the sidecar JSON, and the `media_dir` if it exists.
            archive_path: Optional explicit output path for the `.zip` archive. If provided,
                the archive is written here regardless of `archive`. If `None` and `archive`
                is True, the archive is created next to the Parquet with the same stem.

        Notes:
            - Columns treated as media paths are auto-detected from the schema by looking for
              `ImagePathField` attributes; there is no need to specify them manually.
            - Filenames in `media_dir` are collision-safe; if two sources share the same
              basename, numeric suffixes are applied.
            - The sidecar's `media` block contains `{ root, path_fields, style, copy_mode }`.
            - Path strings written to Parquet are standard POSIX strings when relative.

        Raises:
            OSError: If the Parquet or sidecar cannot be written due to I/O errors.
            FileNotFoundError: If `save_media=True` and a referenced source file does not exist.
            ValueError: If unsupported enum values are somehow provided (should not occur when
                using the provided enums).

        Examples:
            Save with packaged media and relative paths:
            >>> ds.save_parquet(
            ...     "out/dataset.parquet",
            ...     save_media=True,
            ...     media_dir="media",
            ...     copy_mode=MediaCopyMode.COPY,
            ...     path_style=PathStyle.RELATIVE,
            ... )

            Save and also bundle everything into a `.zip` next to the parquet:
            >>> ds.save_parquet("out/data.parquet", save_media=True, archive=True)

            Save and write the archive to a custom location:
            >>> ds.save_parquet("out/data.parquet", save_media=True, archive_path="/share/pkg.zip")
        """
        # 1) Prepare path fields via IO helper
        file_path_fields = detect_file_path_fields(self._schema)

        # 2) Optionally package media and rewrite path columns
        df_to_write = self.df
        media_meta: dict[str, Any] | None = None
        if save_media and file_path_fields:
            df_to_write, media_meta = package_and_rewrite_paths(
                df=df_to_write,
                out_parquet_path=Path(path),
                path_fields=file_path_fields,
                media_dir=media_dir,
                copy_mode=copy_mode,
                path_style=path_style,
            )

        # 3) Write parquet
        df_to_write.write_parquet(path)

        # 4) Build and write sidecar JSON
        sidecar = build_sidecar_dict(self._schema, media_meta)
        write_sidecar_file(Path(path), sidecar)

        # 5) Optionally create an archive
        if archive or archive_path is not None:
            create_zip_archive(
                out_parquet_path=Path(path),
                media_dir=media_dir,
                archive_path=Path(archive_path) if archive_path is not None else None,
            )

    @classmethod
    def from_parquet(
        cls,
        path: str,
        dtype_or_schema: Union[Schema, Type[DTargetType]],
        categories: Dict[str, Categories] | None = None,
        schema: Schema | None = None,
        *,
        rebase_media_root: str | None = None,
        make_paths_absolute: bool = False,
        extract_dir: str | None = None,
    ) -> "Dataset[DTargetType]":
        """
        Load a dataset from a Parquet file or from a zip archive produced by `save_parquet`.

        Behavior overview:
        - If `path` points to a `.zip`, the archive is extracted to `extract_dir` (if provided)
          or to a temporary directory. The contained Parquet is then located and loaded.
          When loading from a zip, `rebase_media_root` defaults to the extraction directory
          so that relative media paths resolve to the extracted `media/` folder.
        - If the sidecar JSON (`<parquet>.json`) contains a `media` block and either
          `rebase_media_root` or `make_paths_absolute` is requested, columns that were
          declared as `ImagePathField` are rewritten accordingly.

        Args:
            path: Path to a `.parquet` file or a `.zip` produced by `save_parquet(archive=True)`.
            dtype_or_schema: The target Sample class (dtype) or a `Schema` instance describing
                the structure of rows to construct. Parquet itself does not carry the custom
                Datumaro schema, so this is required for typed access.
            categories: Optional categories to attach to the loaded schema. When `None`, the
                loader attempts to read categories from the sidecar JSON. If both are provided,
                the explicit `categories` argument takes precedence.
            schema: Optional explicit `Schema` to use instead of inferring from `dtype_or_schema`.
                This is rarely needed; pass only if you manage schemas manually.
            rebase_media_root: Optional directory that becomes the new base for resolving stored
                relative media paths (e.g., `media/img.png`). If `None`, relative paths are
                resolved relative to the Parquet file location. When loading from zip, the
                default is the extraction directory unless you override it.
            make_paths_absolute: When True, after optional rebasing, all media path values are
                converted to absolute OS paths. When False, stored values are preserved except
                for possible rebasing of relative paths.
            extract_dir: Directory to extract the zip to when `path` is a `.zip`. If `None`, a
                temporary directory is used. The directory is not automatically cleaned up.

        Returns:
            A `Dataset` instance of type `dtype_or_schema` (or using the provided `Schema`).

        Notes:
            - Path-field columns are auto-detected from the sidecar schema by selecting
              attributes with `field == "ImagePathField"`.
            - Categories are restored from the sidecar if not passed explicitly.
            - When `path` is a `.zip`, the loader selects the Parquet inside by preferring a
              file whose name matches the archive stem; otherwise it chooses a top-level Parquet
              if present, or the first Parquet found.

        Raises:
            FileNotFoundError: If a `.zip` is provided but no Parquet is found inside.
            OSError: If the Parquet file cannot be read.
            ValueError: If the provided arguments are inconsistent.

        Examples:
            Load a plain Parquet written earlier:
            >>> ds = Dataset.from_parquet("out/data.parquet", PathSample)

            Load from a zip and resolve media paths to absolute on this machine:
            >>> ds = Dataset.from_parquet(
            ...     "out/data.zip",
            ...     PathSample,
            ...     make_paths_absolute=True,
            ... )

            Load and rebase relative media paths under a new directory:
            >>> ds = Dataset.from_parquet(
            ...     "out/data.parquet",
            ...     PathSample,
            ...     rebase_media_root="/mnt/datasets/shared_copy",
            ... )
        """

        # Handle zip archive inputs by extracting and delegating to inner parquet
        try:
            import zipfile

            is_zip = path.lower().endswith(".zip") and zipfile.is_zipfile(path)
        except Exception:
            is_zip = False

        if is_zip:
            candidate, extract_root = extract_zip_to_dir(path, extract_dir)
            effective_rebase = (
                rebase_media_root if rebase_media_root is not None else str(extract_root)
            )
            return cls.from_parquet(
                str(candidate),
                dtype_or_schema,
                categories=categories,
                schema=schema,
                rebase_media_root=effective_rebase,
                make_paths_absolute=make_paths_absolute,
            )

        df: pl.DataFrame = pl.read_parquet(path)

        # Load categories from sidecar JSON if available
        loaded_categories: Dict[str, Categories] | None = None
        sidecar: dict[str, Any] | None = None
        if categories is None or rebase_media_root or make_paths_absolute:
            sidecar = read_sidecar_file(Path(path))
            if categories is None and isinstance(sidecar, dict):
                cats = sidecar.get("categories")
                if isinstance(cats, dict):
                    try:
                        loaded_categories = deserialize_categories_map(cats)
                    except Exception:
                        loaded_categories = None

        # Optionally rebase/resolve media paths using sidecar info
        if sidecar is not None and (rebase_media_root or make_paths_absolute):
            media_meta = sidecar.get("media") if isinstance(sidecar, dict) else None
            if isinstance(media_meta, dict):
                # Discover ImagePathField columns from sidecar schema
                detected: list[str] = []
                schema_info = sidecar.get("schema", {})
                attrs = schema_info.get("attributes", []) if isinstance(schema_info, dict) else []
                for a in attrs:
                    try:
                        if a.get("field") == "ImagePathField":
                            detected.append(a.get("name"))
                    except Exception:
                        pass
                if detected:
                    df = rebase_media_paths(
                        df=df,
                        path_fields=detected,
                        sidecar_media_meta=media_meta,
                        parquet_path=Path(path),
                        rebase_media_root=rebase_media_root,
                        make_paths_absolute=make_paths_absolute,
                    )

        categories_final: Dict[str, Categories] | None = categories or loaded_categories

        if isinstance(dtype_or_schema, Schema):
            # When caller passed a Schema, pass the adjusted schema directly
            return cls.from_dataframe(df, dtype_or_schema, transforms=None)
        return cls.from_dataframe(
            df,
            dtype_or_schema,
            transforms=None,
            categories=categories_final,
            schema=schema,
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
