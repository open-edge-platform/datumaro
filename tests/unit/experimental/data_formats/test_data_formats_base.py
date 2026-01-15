from enum import Enum

import pytest

from datumaro.experimental.data_formats.base import DataFormat, load_dataset, save_dataset
from datumaro.experimental.data_formats.coco.sample import CocoSample


class FakeFormat(Enum):
    OTHER = "OTHER"


class DummyDataset:
    def __init__(self):
        self.converted_schema = None

    def convert_to_schema(self, schema):
        self.converted_schema = schema
        return self


def test_load_dataset_delegates_to_coco(monkeypatch):
    captured = {}

    def fake_loader(images_dir_path, annotations_path):
        captured["images_dir_path"] = images_dir_path
        captured["annotations_path"] = annotations_path
        return "dataset"

    monkeypatch.setattr("datumaro.experimental.data_formats.coco.io.load_coco_dataset", fake_loader)

    result = load_dataset(
        DataFormat.COCO,
        images_dir_path="/tmp/images",
        annotations_path="/tmp/annotations.json",
    )

    assert result == "dataset"
    assert captured["images_dir_path"] == "/tmp/images"
    assert captured["annotations_path"] == "/tmp/annotations.json"


def test_load_dataset_requires_paths_for_coco():
    with pytest.raises(ValueError, match="images_dir_path and annotations_path are required"):
        load_dataset(DataFormat.COCO)


def test_load_dataset_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported data format"):
        load_dataset(
            FakeFormat.OTHER,
            images_dir_path="/tmp/images",
            annotations_path="/tmp/annotations.json",
        )


def test_save_dataset_delegates_to_coco_with_converted_schema(monkeypatch):
    dummy_dataset = DummyDataset()
    captured = {}

    def fake_saver(dataset, images_dir_path, annotations_path):
        captured["dataset"] = dataset
        captured["images_dir_path"] = images_dir_path
        captured["annotations_path"] = annotations_path

    monkeypatch.setattr("datumaro.experimental.data_formats.coco.io.save_coco_dataset", fake_saver)

    save_dataset(
        dummy_dataset,
        DataFormat.COCO,
        images_dir_path="/tmp/images",
        annotations_path="/tmp/annotations.json",
    )

    assert captured["dataset"] is dummy_dataset
    assert captured["images_dir_path"] == "/tmp/images"
    assert captured["annotations_path"] == "/tmp/annotations.json"
    assert dummy_dataset.converted_schema == CocoSample.infer_schema()


def test_save_dataset_requires_paths_for_coco():
    with pytest.raises(ValueError, match="images_dir_path and annotations_path are required"):
        save_dataset(DummyDataset(), DataFormat.COCO)


def test_save_dataset_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported data format"):
        save_dataset(
            DummyDataset(),
            FakeFormat.OTHER,
            images_dir_path="/tmp/images",
            annotations_path="/tmp/annotations.json",
        )
