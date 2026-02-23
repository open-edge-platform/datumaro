# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
Export and import functionality for datasets.

This module provides comprehensive export/import capabilities for Datumaro datasets,
including support for:
- Parquet format for efficient DataFrame storage
- JSON metadata for schema and categories
- Image export for callable and path-based image fields
- ZIP archive format for complete dataset packages
- Automatic dtype detection when importing without an explicit ``dtype``
"""

from __future__ import annotations

import json
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile, is_zipfile

import numpy as np
import polars as pl
from PIL import Image

from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields.images import ImageCallableField, ImagePathField
from datumaro.experimental.fields.masks import InstanceMaskCallableField, MaskCallableField
from datumaro.experimental.schema import Schema

if TYPE_CHECKING:
    from .dataset import DType

# Registry sample classes that can automatically be detected when importing datasets
_sample_registry: set[type[Sample]] = set()


def register_sample(cls: type[Sample]) -> type[Sample]:
    """Register a Sample subclass for automatic dtype detection during import.

    Use this to ensure that custom Sample subclasses defined outside of Datumaro
    are discoverable by :func:`import_dataset` when ``dtype`` is not provided.
    Can be used as a decorator or called directly.

    Args:
        cls: A Sample subclass to register

    Returns:
        The same class, unmodified (allows use as a decorator)

    Example::

        @register_sample
        class MySample(Sample):
            image: Annotated[np.ndarray, ImageField()]
            label: Annotated[int, ScalarField()]
    """
    # Validate that only proper Sample subclasses (excluding Sample itself)
    # can be registered. This prevents invalid entries that could later cause
    # failures during schema inference.
    if not isinstance(cls, type) or not issubclass(cls, Sample) or cls is Sample:
        raise TypeError(f"register_sample expects a subclass of Sample (excluding Sample itself), got {cls!r}")
    _sample_registry.add(cls)
    return cls


# Constants for export structure
METADATA_FILE = "metadata.json"
DATAFRAME_FILE = "data.parquet"
IMAGES_DIR = "images"
try:
    VERSION = _pkg_version("datumaro")
except PackageNotFoundError:
    VERSION = "0.0.0"


def _get_image_fields(dataset: Dataset[DType]) -> list[tuple[str, object]]:
    """Extract all image-related fields from the dataset schema."""
    image_fields = []
    for name, attr_info in dataset.schema.attributes.items():
        if isinstance(
            attr_info.field,
            ImageCallableField | ImagePathField | InstanceMaskCallableField | MaskCallableField,
        ):
            image_fields.append((name, attr_info.field))
    return image_fields


def _get_field_value(dataset: Dataset[DType], idx: int, field_name: str) -> object:
    """Get the value of a field for a specific row index."""
    if dataset._transforms is None:
        row_df = dataset.df.slice(idx, 1)
    else:
        transforms = dataset._transforms.slice(idx, 1)
        row_df = transforms.apply([field_name])
    return row_df[field_name][0]


def _export_image_path_field(
    value: object,
    field_name: str,
    idx: int,
    output_dir: Path,
) -> str | None:
    """Export an ImagePathField by copying the file from the filesystem."""
    try:
        source_path = Path(str(value))
        if not source_path.exists():
            print(f"Warning: Image file not found: {value}")
            return None

        extension = source_path.suffix if source_path.suffix else ".png"
        rel_path = f"{field_name}_{idx:06d}{extension}"
        abs_path = output_dir / rel_path

        shutil.copy2(source_path, abs_path)
        return str(rel_path)
    except Exception as e:
        print(f"Warning: Failed to copy image from {value}: {e}")
        return None


def _array_to_pil_image(img_data: np.ndarray, field_name: str, idx: int) -> Image.Image | None:
    """Convert a numpy array to a PIL Image."""
    if len(img_data.shape) == 2:
        return Image.fromarray(img_data.astype(np.uint8))
    if len(img_data.shape) == 3:
        if img_data.shape[2] == 1:
            return Image.fromarray(img_data[:, :, 0].astype(np.uint8))
        return Image.fromarray(img_data.astype(np.uint8))
    print(f"Warning: Unsupported image shape {img_data.shape} for field {field_name}, idx {idx}")
    return None


def _export_callable_field(
    value: object,
    field_name: str,
    idx: int,
    output_dir: Path,
    is_mask: bool = False,
) -> str | None:
    """Export an ImageCallableField or MaskCallableField by calling it and saving as PNG."""
    if not callable(value):
        return None

    try:
        img_data = value()
    except Exception as e:
        field_type = "mask" if is_mask else "image"
        print(f"Warning: Failed to call {field_type} callable for field {field_name}, idx {idx}: {e}")
        return None

    if img_data is None:
        return None

    try:
        pil_img = _array_to_pil_image(img_data, field_name, idx)
        if pil_img is None:
            return None

        rel_path = f"{field_name}_{idx:06d}.png"
        abs_path = output_dir / rel_path
        pil_img.save(abs_path)
        return str(rel_path)
    except Exception as e:
        field_type = "mask" if is_mask else "image"
        print(f"Warning: Failed to save {field_type} for field {field_name}, idx {idx}: {e}")
        return None


def _export_single_field(
    dataset: Dataset[DType],
    field_name: str,
    field: object,
    output_dir: Path,
    skip_missing_images: bool = False,
) -> dict[int, str]:
    """Export images for a single field across all rows."""
    field_paths: dict[int, str] = {}

    for idx in range(len(dataset)):
        value = _get_field_value(dataset, idx, field_name)
        if value is None:
            if not skip_missing_images:
                print(f"Warning: No value set for field {field_name}, idx {idx}.")
            continue

        rel_path = None
        if isinstance(field, ImagePathField):
            rel_path = _export_image_path_field(value, field_name, idx, output_dir)
        elif isinstance(field, ImageCallableField):
            rel_path = _export_callable_field(value, field_name, idx, output_dir, is_mask=False)
        elif isinstance(field, InstanceMaskCallableField | MaskCallableField):
            rel_path = _export_callable_field(value, field_name, idx, output_dir, is_mask=True)

        if rel_path is not None:
            field_paths[idx] = rel_path
        elif not skip_missing_images:
            raise ValueError(
                f"Value was set for field {field_name}, index {idx}, value {value}, but no image could be obtained "
                f"(ImagePathField) or no image could be generated and saved (Image/MaskCallableField)."
            )

    return field_paths


def _export_images_from_dataset(
    dataset: Dataset[DType],
    output_dir: Path,
    skip_missing_images: bool = False,
) -> pl.DataFrame:
    """
    Export images from callable or path fields in the dataset.

    For ImagePathField: Copies images directly from filesystem (preserving format)
    For ImageCallableField: Saves as PNG (lossless for arrays)
    For InstanceMaskCallableField/MaskCallableField: Saves as PNG (best for masks)

    Args:
        dataset: The dataset to export images from
        output_dir: Directory to save images to
        skip_missing_images: Boolean indicating if to raise errors or skip when images are missing

    Returns:
        Dictionary mapping field names to dictionaries of row_idx -> relative path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_paths: dict[str, dict[int, str]] = {}

    image_fields = _get_image_fields(dataset)
    if not image_fields:
        return dataset.df

    for field_name, field in image_fields:
        images_paths[field_name] = _export_single_field(
            dataset=dataset,
            field_name=field_name,
            field=field,
            output_dir=output_dir,
            skip_missing_images=skip_missing_images,
        )

    df_to_export = dataset.df.with_row_index("__idx")
    for field_name, path_map in images_paths.items():
        if not path_map:
            continue

        # Create a mapping DataFrame from the exported paths
        mapping_df = pl.DataFrame(
            {"__idx": list(path_map.keys()), "__new_path": list(path_map.values())},
            schema={"__idx": pl.UInt32, "__new_path": pl.String},
        )

        # Left join to apply updates; rows not in path_map will have null paths
        df_to_export = (
            df_to_export.join(mapping_df, on="__idx", how="left")
            .with_columns(pl.col("__new_path").alias(field_name))
            .drop("__new_path")
        )

    # Check for missing images
    image_fields = list(images_paths.keys())
    missing_images_mask = pl.all_horizontal(pl.col(image_fields).is_null())

    if skip_missing_images:
        df_to_export = df_to_export.filter(~missing_images_mask)
    else:
        bad_rows = df_to_export.filter(missing_images_mask).select("__idx")
        if not bad_rows.is_empty():
            raise ValueError(f"Missing images for indices {bad_rows.to_series().to_list()}")

    return df_to_export.drop("__idx")


def export_dataset(
    dataset: Dataset[DType],
    output_path: str | Path,
    export_images: bool = True,
    as_zip: bool = False,
    skip_missing_images: bool = False,
) -> None:
    """
    Export a dataset to disk in a structured format.

    The dataset is exported with the following structure:
    - data.parquet: The DataFrame in Parquet format
    - metadata.json: Schema, categories, and image paths
    - images/: Directory containing exported images (if export_images=True)

    Image format is automatically determined:
    - ImagePathField: Preserves original format (copied directly)
    - ImageCallableField: Saved as PNG (lossless)
    - InstanceMaskCallableField/MaskCallableField: Saved as PNG (best for masks)

    Args:
        dataset: The dataset to export
        output_path: Path to export to (directory or .zip file)
        export_images: Whether to export images from callable/path fields
        as_zip: Whether to package everything as a ZIP file
        skip_missing_images: Boolean indicating if to raise errors or skip when images are missing
    """
    output_path = Path(output_path)

    # Create temporary directory if using zip, otherwise use output_path directly
    temp_dir = None
    if as_zip:
        if output_path.suffix != ".zip":
            output_path.mkdir(parents=True, exist_ok=True)
            output_path /= "dataset.zip"
        temp_dir = Path(tempfile.mkdtemp())
        work_dir = temp_dir
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        work_dir = output_path

    try:
        # Export images if requested
        if export_images:
            images_dir = work_dir / IMAGES_DIR
            df_to_export = _export_images_from_dataset(
                dataset=dataset, output_dir=images_dir, skip_missing_images=skip_missing_images
            )
        else:
            df_to_export = dataset.df

        # Export DataFrame to Parquet
        # Filter out Object columns (like callables) as they can't be serialized to Parquet
        columns_to_export = [col for col, dtype in df_to_export.schema.items() if dtype != pl.Object]
        parquet_path = work_dir / DATAFRAME_FILE
        df_to_export.select(columns_to_export).write_parquet(parquet_path)

        # Track which columns are Object types (excluded from parquet)
        object_columns = [col for col, dtype in df_to_export.schema.items() if dtype == pl.Object]

        # Create metadata
        metadata = {
            "version": VERSION,
            "schema": dataset.schema.to_dict(),
            "object_columns": object_columns,
        }

        # Write metadata
        metadata_path = work_dir / METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create ZIP if requested
        if as_zip:
            with ZipFile(output_path, "w", ZIP_DEFLATED) as zipf:
                for file_path in work_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(work_dir)
                        zipf.write(file_path, arcname)

    finally:
        # Clean up temporary directory if used
        if as_zip and temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)


def import_dataset(
    input_path: str | Path,
    dtype: type[DType] | None = None,
    extract_dir: str | Path | None = None,
) -> Dataset[DType]:
    """
    Import a dataset from an exported format.

    When dtype is None the function tries to automatically determine the correct Sample subclass by matching the stored
    schema against all registered subclasses (discovered via the :func:`register_sample` registry).  If no
    match is found, it falls back to the base ``Sample`` class.

    Args:
        input_path: Path to the exported dataset (directory or .zip file)
        dtype: Optional Sample class to use for the dataset.  When provided,
            the dataset is typed with this class directly.  When ``None``,
            automatic dtype detection is attempted (see above).
        extract_dir: Optional directory to extract zip contents to. If not provided,
            the zip will be extracted to a directory next to the zip file with the
            same name (excluding the .zip extension). This parameter is ignored
            when input_path is a directory.

    Returns:
        The imported Dataset instance

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If the dataset format is invalid
    """
    input_path = Path(input_path)

    if is_zipfile(input_path):
        if extract_dir is not None:
            extract_dir = Path(extract_dir)
        else:
            # Default: extract to a directory next to the zip with the same name (minus .zip)
            extract_dir = input_path.with_suffix("")

        extract_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(input_path) as zipf:
            zipf.extractall(extract_dir)
        return _import_dataset_from_dir(extract_dir, dtype)
    return _import_dataset_from_dir(input_path, dtype)


def _load_metadata(input_dir: Path) -> dict:
    """Load and validate metadata file."""
    metadata_path = input_dir / METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    if metadata.get("version") != VERSION:
        print(
            f"Warning: Dataset version {metadata.get('version')} may not be fully "
            f"compatible with current version {VERSION}"
        )

    return metadata


def _load_dataframe(input_dir: Path) -> pl.DataFrame:
    """Load DataFrame from parquet file."""
    parquet_path = input_dir / DATAFRAME_FILE
    if not parquet_path.exists():
        raise FileNotFoundError(f"DataFrame file not found: {parquet_path}")

    return pl.read_parquet(parquet_path)


def _make_image_loader(path: Path):
    """Create a closure that loads an image from a path."""

    def load_image():
        return np.array(Image.open(path))

    return load_image


def _reconstruct_image_fields(
    df: pl.DataFrame,
    schema: Schema,
    images_base_dir: Path,
) -> pl.DataFrame:
    """Reconstruct image-related fields from the images directory."""
    if not images_base_dir.exists():
        return df

    updates: dict[str, pl.Series] = {}
    for field_name, attr_info in schema.attributes.items():
        field = getattr(attr_info, "field", None)
        if field is None or field_name not in df.columns:
            continue

        is_path_field = isinstance(field, ImagePathField)
        is_callable_field = isinstance(field, ImageCallableField | InstanceMaskCallableField | MaskCallableField)

        if not (is_path_field or is_callable_field):
            continue

        file_names = df.get_column(field_name)
        values: list[object | None] = []
        if is_path_field:
            values = [str(images_base_dir / file_name) if file_name is not None else None for file_name in file_names]
        else:
            values = [
                _make_image_loader(images_base_dir / file_name) if file_name is not None else None
                for file_name in file_names
            ]

        field_dtype = pl.String() if is_path_field else pl.Object()
        updates[field_name] = pl.Series(field_name, values, dtype=field_dtype)

    if updates:
        df = df.with_columns(list(updates.values()))
    return df


def _add_missing_object_columns(
    df: pl.DataFrame,
    object_columns: list[str],
) -> pl.DataFrame:
    """Add back any object columns that weren't reconstructed from images."""
    for col_name in object_columns:
        if col_name not in df.columns:
            df = df.with_columns(pl.Series(col_name, [None] * len(df), dtype=pl.Object()))
    return df


def _get_registered_samples() -> list[type[Sample]]:
    """Get all Sample subclasses that have been explicitly registered.

    Only returns classes that were registered via :func:`register_sample`.

    Returns:
        List of explicitly registered Sample subclasses
    """
    return list(_sample_registry)


def _match_dtype_from_schema(schema: Schema) -> type[Sample]:
    """Try to match a schema against registered Sample subclasses.

    Compares the serialized schema attributes (including field configurations like
    dtype, semantic, is_list, format, etc.) between the loaded schema and each
    registered Sample subclass's inferred schema. Categories are not compared
    since they may differ between loaded and inferred schemas.

    Args:
        schema: The schema loaded from the dataset metadata

    Returns:
        The matching Sample subclass, or base Sample if no match is found
    """
    schema_dict = schema.to_dict()
    schema_attributes = schema_dict.get("attributes", {})

    for subclass in _get_registered_samples():
        candidate_schema = subclass.infer_schema()
        candidate_dict = candidate_schema.to_dict()
        candidate_attributes = candidate_dict.get("attributes", {})
        if schema_attributes == candidate_attributes:
            return subclass

    return Sample


def _import_dataset_from_dir(
    input_dir: Path,
    dtype: type[DType] | None = None,
) -> Dataset[DType]:
    """
    Import dataset from a directory.

    Args:
        input_dir: Directory containing the exported dataset
        dtype: Optional Sample class to use

    Returns:
        The imported Dataset instance
    """
    metadata = _load_metadata(input_dir)
    df = _load_dataframe(input_dir)  # Check DataFrame exists before processing schema
    schema = Schema.from_dict(metadata["schema"])

    object_columns = metadata.get("object_columns", [])
    images_base_dir = input_dir / IMAGES_DIR

    df = _reconstruct_image_fields(df, schema, images_base_dir)
    df = _add_missing_object_columns(df, object_columns)

    if dtype is None:
        dtype = _match_dtype_from_schema(schema)

    return Dataset.from_dataframe(df, dtype, schema=schema)
