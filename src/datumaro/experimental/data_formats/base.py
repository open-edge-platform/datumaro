# Copyright (C) 2022-2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import shutil
import tempfile
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class DataFormat(Enum):
    """Supported data formats for load/save."""

    DATUMARO = "DATUMARO"
    COCO = "COCO"
    YOLO = "YOLO"
    YOLO_ULTRALYTICS = "YOLO_ULTRALYTICS"
    UNKNOWN = "UNKNOWN"


def load_dataset(
    data_format: DataFormat,
    images_dir_path: str | dict[str, str] | None = None,
    annotations_path: str | list[str] | dict[str, str | list[str]] | None = None,
    root_dir: str | None = None,
) -> Dataset:
    """
    Load a dataset in the specified format.

    Args:
        data_format: The format of the dataset to load.
        images_dir_path: Path to image directory (str) or dict mapping subset names to paths.
        annotations_path: Path to annotations file (str) or dict mapping subset names to paths.
        root_dir: Root directory for YOLO format datasets.

    Returns:
        Dataset containing the loaded samples.

    Raises:
        ValueError: If the data format is not supported or required arguments are missing.
    """
    if data_format == DataFormat.COCO:
        from datumaro.experimental.data_formats.coco.io import load_coco_dataset

        if images_dir_path is None or annotations_path is None:
            raise ValueError("images_dir_path and annotations_path are required for COCO format")

        return load_coco_dataset(
            images_dir_path=images_dir_path,
            annotations_path=annotations_path,
        )
    if data_format in (DataFormat.YOLO, DataFormat.YOLO_ULTRALYTICS):
        from datumaro.experimental.data_formats.yolo.io import load_yolo_dataset

        if root_dir is None:
            raise ValueError("root_dir is required for YOLO format")

        # Set format hint based on enum value
        format_hint = "ultralytics" if data_format == DataFormat.YOLO_ULTRALYTICS else "auto"
        return load_yolo_dataset(root_dir=root_dir, format=format_hint)
    raise ValueError(f"Unsupported data format: {data_format}")


def save_dataset(
    dataset: Dataset,
    data_format: DataFormat,
    output_path: str,
    as_zip: bool = False,
) -> None:
    """
    Save a dataset in the specified format.

    Args:
        dataset: Dataset to save.
        data_format: The format to save the dataset in.
        output_path: Output location. When as_zip=False, this is the output directory.
            When as_zip=True, this is the zip file path (with or without .zip extension).
        as_zip: If True, save as a zip archive.

    Raises:
        ValueError: If the data format is not supported.
    """
    if data_format not in DataFormat:
        raise ValueError(f"Unsupported data format: {data_format}")

    output_dir = os.path.dirname(output_path) if as_zip else output_path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if as_zip:
        _save_dataset_as_zip(
            dataset=dataset,
            data_format=data_format,
            output_path=output_path,
        )
    else:
        _save_dataset_to_dir(
            dataset=dataset,
            data_format=data_format,
            output_dir=output_path,
        )


def _save_dataset_as_zip(
    dataset: Dataset,
    data_format: DataFormat,
    output_path: str,
) -> None:
    """Save dataset as a zip archive."""
    # Strip .zip extension if present since shutil.make_archive adds it automatically
    output_path, _ = os.path.splitext(output_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        _save_dataset_to_dir(
            dataset=dataset,
            data_format=data_format,
            output_dir=temp_dir,
        )

        shutil.make_archive(
            base_name=output_path,
            format="zip",
            root_dir=temp_dir,
        )


def _save_dataset_to_dir(
    dataset: Dataset,
    data_format: DataFormat,
    output_dir: str,
) -> None:
    """Internal helper to save dataset to a directory."""
    if data_format == DataFormat.COCO:
        from datumaro.experimental.data_formats.coco.io import save_coco_dataset
        from datumaro.experimental.data_formats.coco.sample import CocoSample

        dataset = dataset.convert_to_schema(CocoSample.infer_schema())
        save_coco_dataset(
            dataset,
            images_dir_path=os.path.join(output_dir, "images"),
            annotations_path=os.path.join(output_dir, "annotations.json"),
        )
    elif data_format in (DataFormat.YOLO, DataFormat.YOLO_ULTRALYTICS):
        from datumaro.experimental.data_formats.yolo.io import save_yolo_dataset
        from datumaro.experimental.data_formats.yolo.sample import YoloSample

        dataset = dataset.convert_to_schema(YoloSample.infer_schema())
        save_yolo_dataset(dataset, root_dir=output_dir, format=data_format)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
