# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
Export and import functionality for experimental datasets.

This module provides comprehensive export/import capabilities for Datumaro datasets,
including support for:
- Parquet format for efficient DataFrame storage
- JSON metadata for schema and categories
- Image export for callable and path-based image fields
- ZIP archive format for complete dataset packages
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Type, Union
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import polars as pl
from PIL import Image

from .dataset import Dataset, Sample
from .fields import ImageCallableField, ImagePathField, InstanceMaskCallableField
from .schema import Schema

if TYPE_CHECKING:
    from .dataset import DType

# Constants for export structure
METADATA_FILE = "metadata.json"
DATAFRAME_FILE = "data.parquet"
IMAGES_DIR = "images"
VERSION = "1.0"


def _export_images_from_dataset(
    dataset: "Dataset[DType]",
    output_dir: Path,
) -> Dict[str, Dict[int, str]]:
    """
    Export images from callable or path fields in the dataset.

    For ImagePathField: Copies images directly from filesystem (preserving format)
    For ImageCallableField: Saves as PNG (lossless for arrays)
    For InstanceMaskCallableField: Saves as PNG (best for masks)

    Args:
        dataset: The dataset to export images from
        output_dir: Directory to save images to

    Returns:
        Dictionary mapping field names to dictionaries of row_idx -> relative path
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: Dict[str, Dict[int, str]] = {}

    # Find all image-related fields
    image_fields = []
    for name, attr_info in dataset.schema.attributes.items():
        if isinstance(
            attr_info.annotation, (ImageCallableField, ImagePathField, InstanceMaskCallableField)
        ):
            image_fields.append((name, attr_info.annotation))

    if not image_fields:
        return image_paths

    # Export images for each field
    for field_name, field_annotation in image_fields:
        field_dir = output_dir / field_name
        field_dir.mkdir(exist_ok=True)
        image_paths[field_name] = {}

        for idx in range(len(dataset)):
            # Get the raw dataframe value
            if dataset._transforms is None:
                row_df = dataset.df.slice(idx, 1)
            else:
                # For transformed datasets, we need to access the original df
                transforms = dataset._transforms.slice(idx, 1)
                row_df = transforms.apply([field_name])

            value = row_df[field_name][0]

            if value is None:
                continue

            # Handle different field types
            if isinstance(field_annotation, ImagePathField):
                # For ImagePathField: Copy image directly from filesystem
                try:
                    source_path = Path(value)
                    if not source_path.exists():
                        print(f"Warning: Image file not found: {value}")
                        continue

                    # Preserve original extension/format
                    extension = source_path.suffix if source_path.suffix else ".png"
                    rel_path = f"{field_name}/{idx:06d}{extension}"
                    abs_path = output_dir / rel_path.replace("/", "_")

                    # Direct file copy - no loading into memory
                    shutil.copy2(source_path, abs_path)
                    image_paths[field_name][idx] = str(rel_path)
                except Exception as e:
                    print(f"Warning: Failed to copy image from {value}: {e}")
                    continue

            elif isinstance(field_annotation, ImageCallableField):
                # Call the callable to get the image data
                if not callable(value):
                    continue

                try:
                    img_data = value()
                except Exception as e:
                    print(
                        f"Warning: Failed to call image callable for field {field_name}, idx {idx}: {e}"
                    )
                    continue

                # Convert to PIL Image and save as PNG (lossless)
                if img_data is not None:
                    try:
                        # Handle different image formats
                        if len(img_data.shape) == 2:
                            # Grayscale
                            pil_img = Image.fromarray(img_data.astype(np.uint8))
                        elif len(img_data.shape) == 3:
                            if img_data.shape[2] == 1:
                                # Single channel
                                pil_img = Image.fromarray(img_data[:, :, 0].astype(np.uint8))
                            else:
                                # RGB or RGBA
                                pil_img = Image.fromarray(img_data.astype(np.uint8))
                        else:
                            print(
                                f"Warning: Unsupported image shape {img_data.shape} for field {field_name}, idx {idx}"
                            )
                            continue

                        # Save as PNG (lossless)
                        rel_path = f"{field_name}/{idx:06d}.png"
                        abs_path = output_dir / rel_path.replace("/", "_")
                        pil_img.save(abs_path)
                        image_paths[field_name][idx] = str(rel_path)
                    except Exception as e:
                        print(
                            f"Warning: Failed to save image for field {field_name}, idx {idx}: {e}"
                        )
                        continue

            elif isinstance(field_annotation, InstanceMaskCallableField):
                # Call the callable to get the mask data
                if not callable(value):
                    continue

                try:
                    img_data = value()
                except Exception as e:
                    print(
                        f"Warning: Failed to call mask callable for field {field_name}, idx {idx}: {e}"
                    )
                    continue

                # Convert to PIL Image and save as PNG (best for masks)
                if img_data is not None:
                    try:
                        # Handle different image formats
                        if len(img_data.shape) == 2:
                            # Grayscale mask
                            pil_img = Image.fromarray(img_data.astype(np.uint8))
                        elif len(img_data.shape) == 3:
                            if img_data.shape[2] == 1:
                                # Single channel mask
                                pil_img = Image.fromarray(img_data[:, :, 0].astype(np.uint8))
                            else:
                                # Multi-channel mask
                                pil_img = Image.fromarray(img_data.astype(np.uint8))
                        else:
                            print(
                                f"Warning: Unsupported mask shape {img_data.shape} for field {field_name}, idx {idx}"
                            )
                            continue

                        # Save as PNG (best for masks - lossless)
                        rel_path = f"{field_name}/{idx:06d}.png"
                        abs_path = output_dir / rel_path.replace("/", "_")
                        pil_img.save(abs_path)
                        image_paths[field_name][idx] = str(rel_path)
                    except Exception as e:
                        print(
                            f"Warning: Failed to save mask for field {field_name}, idx {idx}: {e}"
                        )
                        continue

    return image_paths


def export_dataset(
    dataset: "Dataset[DType]",
    output_path: Union[str, Path],
    export_images: bool = True,
    as_zip: bool = False,
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
    - InstanceMaskCallableField: Saved as PNG (best for masks)

    Args:
        dataset: The dataset to export
        output_path: Path to export to (directory or .zip file)
        export_images: Whether to export images from callable/path fields
        as_zip: Whether to package everything as a ZIP file

    Raises:
        ValueError: If the dataset has transforms (not yet supported)
    """
    output_path = Path(output_path)

    # Check if dataset has transforms
    if dataset._transforms is not None:
        raise ValueError(
            "Exporting datasets with transforms is not yet supported. "
            "Please materialize the dataset first by converting it."
        )

    # Create temporary directory if using zip, otherwise use output_path directly
    temp_dir = None
    if as_zip:
        if output_path.suffix != ".zip":
            output_path = output_path.with_suffix(".zip")
        temp_dir = Path(tempfile.mkdtemp())
        work_dir = temp_dir
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        work_dir = output_path

    try:
        # Export DataFrame to Parquet
        # Filter out Object columns (like callables) as they can't be serialized to Parquet
        columns_to_export = [col for col, dtype in dataset.df.schema.items() if dtype != pl.Object]
        parquet_path = work_dir / DATAFRAME_FILE
        dataset.df.select(columns_to_export).write_parquet(parquet_path)

        # Export images if requested
        image_paths = {}
        if export_images:
            images_dir = work_dir / IMAGES_DIR
            image_paths = _export_images_from_dataset(dataset, images_dir)

        # Track which columns are Object types (excluded from parquet)
        object_columns = [col for col, dtype in dataset.df.schema.items() if dtype == pl.Object]

        # Create metadata
        metadata = {
            "version": VERSION,
            "schema": dataset.schema.to_dict(),
            "dtype": dataset.dtype.__name__ if dataset.dtype != Sample else None,
            "image_paths": image_paths,
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
    input_path: Union[str, Path],
    dtype: Type["DType"] | None = None,
) -> "Dataset[DType]":
    """
    Import a dataset from an exported format.

    Args:
        input_path: Path to the exported dataset (directory or .zip file)
        dtype: Optional Sample class to use for the dataset. If None, uses generic Sample.

    Returns:
        The imported Dataset instance

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If the dataset format is invalid
    """
    input_path = Path(input_path)

    # Handle ZIP files
    if input_path.suffix == ".zip":
        temp_dir = Path(tempfile.mkdtemp())
        try:
            with ZipFile(input_path) as zipf:
                zipf.extractall(temp_dir)
            return _import_dataset_from_dir(temp_dir, dtype)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    else:
        return _import_dataset_from_dir(input_path, dtype)


def _import_dataset_from_dir(
    input_dir: Path,
    dtype: Type["DType"] | None = None,
) -> "Dataset[DType]":
    """
    Import dataset from a directory.

    Args:
        input_dir: Directory containing the exported dataset
        dtype: Optional Sample class to use

    Returns:
        The imported Dataset instance
    """
    metadata_path = input_dir / METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    parquet_path = input_dir / DATAFRAME_FILE
    if not parquet_path.exists():
        raise FileNotFoundError(f"DataFrame file not found: {parquet_path}")

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Check version
    if metadata.get("version") != VERSION:
        print(
            f"Warning: Dataset version {metadata.get('version')} may not be fully "
            f"compatible with current version {VERSION}"
        )

    # Load schema
    schema = Schema.from_dict(metadata["schema"])

    # Load DataFrame
    df = pl.read_parquet(parquet_path)

    # Get object columns that were excluded from parquet
    object_columns = metadata.get("object_columns", [])

    # Handle image paths if present
    if metadata.get("image_paths"):
        images_base_dir = input_dir / IMAGES_DIR

        for field_name, paths in metadata["image_paths"].items():
            # JSON serialization converts int keys to strings, so convert back
            paths = {int(k): v for k, v in paths.items()}

            # Check if this is an ImagePathField in the schema
            field_annotation = schema.attributes.get(field_name)
            is_path_field = field_annotation and isinstance(
                field_annotation.annotation, ImagePathField
            )

            if is_path_field:
                # For ImagePathField: Update paths to point to images in IMAGES_DIR
                # Create a list of updated paths in the correct order
                path_list = []
                for idx in range(len(df)):
                    if idx in paths:
                        rel_path = paths[idx]
                        abs_path = images_base_dir / rel_path.replace("/", "_")
                        path_list.append(str(abs_path))
                    else:
                        path_list.append(None)

                # Update or add the path column
                if field_name in df.columns:
                    df = df.drop(field_name)
                df = df.with_columns(pl.Series(field_name, path_list, dtype=pl.String))
            else:
                # For ImageCallableField/InstanceMaskCallableField: Create lazy-loading callables

                callable_list = []
                for idx in range(len(df)):
                    if idx in paths:
                        rel_path = paths[idx]
                        abs_path = images_base_dir / rel_path.replace("/", "_")

                        # Create a closure that lazily loads the image
                        def make_loader(path: Path):
                            def load_image():
                                return np.array(Image.open(path))

                            return load_image

                        callable_list.append(make_loader(abs_path))
                    else:
                        callable_list.append(None)

                # Update or add the callable column
                if field_name in df.columns:
                    df = df.with_columns(pl.Series(field_name, callable_list))
                else:
                    df = df.with_columns(pl.Series(field_name, callable_list, dtype=pl.Object))

    # Add back any other object columns that weren't reconstructed from images
    for col_name in object_columns:
        if col_name not in df.columns:
            # Add a column of Nones
            df = df.with_columns(pl.Series(col_name, [None] * len(df), dtype=pl.Object))

    # Determine dtype
    if dtype is None:
        dtype = Sample  # type: ignore

    # Create and return dataset
    return Dataset.from_dataframe(df, dtype, schema=schema)
