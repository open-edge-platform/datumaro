# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE
from enum import Enum

from datumaro.experimental import Dataset
from datumaro.experimental.data_formats.coco.sample import CocoSample


class DataFormat(Enum):
    """Supported data formats for load/save."""

    COCO = "COCO"


def load_dataset(root_dir: str, data_format: DataFormat, **kwargs) -> Dataset:
    """Load a dataset from the given directory in the specified format."""
    if data_format == DataFormat.COCO:
        from datumaro.experimental.data_formats.coco.io import load_coco_dataset

        return load_coco_dataset(root_dir=root_dir, **kwargs)
    raise ValueError(f"Unsupported data format: {data_format}")


def save_dataset(dataset: Dataset, root_dir: str, data_format: DataFormat, **kwargs) -> None:
    """Save a dataset to the given directory in the specified format."""
    if data_format == DataFormat.COCO:
        from datumaro.experimental.data_formats.coco.io import save_coco_dataset

        dataset = dataset.convert_to_schema(CocoSample.infer_schema())
        save_coco_dataset(dataset, root_dir=root_dir, **kwargs)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
