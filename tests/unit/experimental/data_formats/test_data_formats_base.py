import os
import tempfile
import zipfile
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dataset(
            dummy_dataset,
            DataFormat.COCO,
            tmp_dir,
        )

        assert captured["dataset"] is dummy_dataset
        assert captured["images_dir_path"] == os.path.join(tmp_dir, "images")
        assert captured["annotations_path"] == os.path.join(tmp_dir, "annotations.json")
        assert dummy_dataset.converted_schema == CocoSample.infer_schema()


def test_save_dataset_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported data format"):
        save_dataset(
            DummyDataset(),
            FakeFormat.OTHER,
            "/tmp/output",
        )


# Tests for the as_zip functionality in save_dataset


def test_save_as_zip_creates_archive_with_correct_files(monkeypatch):
    """Test that as_zip=True creates a zip archive containing the expected files."""
    dummy_dataset = DummyDataset()

    def fake_saver(dataset, images_dir_path, annotations_path):
        os.makedirs(images_dir_path, exist_ok=True)
        with open(os.path.join(images_dir_path, "test_image.jpg"), "w") as f:
            f.write("fake image data")
        with open(annotations_path, "w") as f:
            f.write('{"images": [], "annotations": []}')

    monkeypatch.setattr("datumaro.experimental.data_formats.coco.io.save_coco_dataset", fake_saver)

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "my_dataset.zip")

        save_dataset(
            dummy_dataset,
            DataFormat.COCO,
            zip_path,
            as_zip=True,
        )

        assert os.path.exists(zip_path)
        assert zipfile.is_zipfile(zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "annotations.json" in names
            assert "images/test_image.jpg" in names


def test_save_as_zip_adds_extension_if_missing(monkeypatch):
    """Test that .zip extension is added if not present in output_path."""
    dummy_dataset = DummyDataset()

    def fake_saver(dataset, images_dir_path, annotations_path):
        os.makedirs(images_dir_path, exist_ok=True)
        with open(annotations_path, "w") as f:
            f.write("{}")

    monkeypatch.setattr("datumaro.experimental.data_formats.coco.io.save_coco_dataset", fake_saver)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # No .zip extension
        output_path = os.path.join(tmp_dir, "my_dataset")

        save_dataset(
            dummy_dataset,
            DataFormat.COCO,
            output_path,
            as_zip=True,
        )

        # Should have added .zip
        zip_path = os.path.join(tmp_dir, "my_dataset.zip")
        assert os.path.exists(zip_path)


def test_save_as_zip_unsupported_format_raises():
    """Test that unsupported format raises ValueError when as_zip=True."""
    with pytest.raises(ValueError, match="Unsupported data format"):
        save_dataset(
            DummyDataset(),
            FakeFormat.OTHER,
            "/tmp/test.zip",
            as_zip=True,
        )


def test_save_as_zip_cleans_up_on_error(monkeypatch):
    """Test that temp directory is cleaned up even if an error occurs."""
    dummy_dataset = DummyDataset()
    created_temp_dir = None

    original_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        nonlocal created_temp_dir
        result = original_mkdtemp(*args, **kwargs)
        created_temp_dir = result
        return result

    # Import the base module and patch tempfile.mkdtemp on its imported tempfile
    import datumaro.experimental.data_formats.base as base_module

    monkeypatch.setattr(base_module.tempfile, "mkdtemp", tracking_mkdtemp)

    def failing_saver(dataset, images_dir_path, annotations_path):
        raise RuntimeError("Simulated save failure")

    monkeypatch.setattr("datumaro.experimental.data_formats.coco.io.save_coco_dataset", failing_saver)

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "test.zip")

        with pytest.raises(RuntimeError, match="Simulated save failure"):
            save_dataset(
                dummy_dataset,
                DataFormat.COCO,
                zip_path,
                as_zip=True,
            )

        # Verify temp directory was still cleaned up
        assert created_temp_dir is not None
        assert not os.path.exists(created_temp_dir)
