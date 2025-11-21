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

    def fake_loader(root_dir: str, **kwargs):
        captured["root_dir"] = root_dir
        captured["kwargs"] = kwargs
        return "dataset"

    monkeypatch.setattr("datumaro.experimental.data_formats.coco.io.load_coco_dataset", fake_loader)

    result = load_dataset("/tmp/root", DataFormat.COCO, option=1)

    assert result == "dataset"
    assert captured["root_dir"] == "/tmp/root"
    assert captured["kwargs"] == {"option": 1}


def test_load_dataset_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported data format"):
        load_dataset("/tmp/root", FakeFormat.OTHER)


def test_save_dataset_delegates_to_coco_with_converted_schema(monkeypatch):
    dummy_dataset = DummyDataset()
    captured = {}

    def fake_saver(dataset, root_dir: str, **kwargs):
        captured["dataset"] = dataset
        captured["root_dir"] = root_dir
        captured["kwargs"] = kwargs

    monkeypatch.setattr("datumaro.experimental.data_formats.coco.io.save_coco_dataset", fake_saver)

    save_dataset(dummy_dataset, "/tmp/out", DataFormat.COCO, extra=True)

    assert captured["dataset"] is dummy_dataset
    assert captured["root_dir"] == "/tmp/out"
    assert captured["kwargs"] == {"extra": True}
    assert dummy_dataset.converted_schema == CocoSample.infer_schema()


def test_save_dataset_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported data format"):
        save_dataset(DummyDataset(), "/tmp/out", FakeFormat.OTHER)
