# Copyright (C) 2022-2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum

from datumaro.experimental import Dataset
from datumaro.experimental.data_formats.coco.sample import CocoSample
from datumaro.experimental.data_formats.yolo.sample import YoloSample


class DataFormat(Enum):
    """Supported data formats for load/save."""

    COCO = "COCO"
    YOLO = "YOLO"
    YOLO_ULTRALYTICS = "YOLO_ULTRALYTICS"


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
    images_dir_path: str | dict[str, str] | None = None,
    annotations_path: str | dict[str, str] | None = None,
    root_dir: str | None = None,
) -> None:
    """
    Save a dataset in the specified format.

    Args:
        dataset: Dataset to save.
        data_format: The format to save the dataset in.
        images_dir_path: Path to image directory (str) or dict mapping subset names to paths.
        annotations_path: Path to annotations file (str) or dict mapping subset names to paths.
        root_dir: Root directory for YOLO format datasets.

    Raises:
        ValueError: If the data format is not supported or required arguments are missing.
    """
    if data_format == DataFormat.COCO:
        from datumaro.experimental.data_formats.coco.io import save_coco_dataset

        if images_dir_path is None or annotations_path is None:
            raise ValueError("images_dir_path and annotations_path are required for COCO format")

        dataset = dataset.convert_to_schema(CocoSample.infer_schema())
        save_coco_dataset(
            dataset,
            images_dir_path=images_dir_path,
            annotations_path=annotations_path,
        )
    elif data_format in (DataFormat.YOLO, DataFormat.YOLO_ULTRALYTICS):
        from datumaro.experimental.data_formats.yolo.io import save_yolo_dataset

        if root_dir is None:
            raise ValueError("root_dir is required for YOLO format")

        dataset = dataset.convert_to_schema(YoloSample.infer_schema())
        save_yolo_dataset(dataset, root_dir=root_dir, format=data_format)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
