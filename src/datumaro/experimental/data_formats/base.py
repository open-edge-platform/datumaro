# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE
from enum import Enum

from datumaro.experimental import Dataset
from datumaro.experimental.data_formats.coco.sample import CocoSample
from datumaro.experimental.data_formats.yolo.sample import YoloSample


class DataFormat(Enum):
    """Supported data formats for load/save."""

    COCO = "COCO"
    YOLO = "YOLO"
    YOLO_ULTRALYTICS = "YOLO_ULTRALYTICS"


def load_dataset(root_dir: str, data_format: DataFormat, **kwargs) -> Dataset:
    """Load a dataset from the given directory in the specified format."""
    if data_format == DataFormat.COCO:
        from datumaro.experimental.data_formats.coco.io import load_coco_dataset

        return load_coco_dataset(root_dir=root_dir, **kwargs)
    if data_format in (DataFormat.YOLO, DataFormat.YOLO_ULTRALYTICS):
        from datumaro.experimental.data_formats.yolo.io import load_yolo_dataset

        # Set format hint based on enum value
        format_hint = "ultralytics" if data_format == DataFormat.YOLO_ULTRALYTICS else "auto"
        return load_yolo_dataset(root_dir=root_dir, format=format_hint, **kwargs)
    raise ValueError(f"Unsupported data format: {data_format}")


def save_dataset(dataset: Dataset, root_dir: str, data_format: DataFormat, **kwargs) -> None:
    """Save a dataset to the given directory in the specified format."""
    if data_format == DataFormat.COCO:
        from datumaro.experimental.data_formats.coco.io import save_coco_dataset

        dataset = dataset.convert_to_schema(CocoSample.infer_schema())
        save_coco_dataset(dataset, root_dir=root_dir, **kwargs)
    elif data_format in (DataFormat.YOLO, DataFormat.YOLO_ULTRALYTICS):
        from datumaro.experimental.data_formats.yolo.io import save_yolo_dataset

        dataset = dataset.convert_to_schema(YoloSample.infer_schema())
        save_yolo_dataset(dataset, root_dir=root_dir, format=data_format, **kwargs)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
