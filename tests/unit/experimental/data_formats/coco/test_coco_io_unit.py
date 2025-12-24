import json
from pathlib import Path

import numpy as np
import pytest

from datumaro.experimental import Dataset
from datumaro.experimental.data_formats.coco.io import load_coco_dataset, save_coco_dataset
from datumaro.experimental.data_formats.coco.sample import CocoCategories, CocoSample
from datumaro.experimental.fields import ImageInfo, Subset


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _make_sample(image_path: Path) -> CocoSample:
    return CocoSample(
        image=str(image_path),
        image_info=ImageInfo(height=2, width=3),
        bboxes=np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        polygons=None,
        labels=np.array([0], dtype=np.int32),
        areas=np.array([1.0], dtype=np.float32),
        iscrowd=np.array([0], dtype=np.int32),
        subset=Subset.TRAINING,
        image_id=1,
        caption_group_ids=None,
        captions=None,
        keypoints=None,
    )


def test_load_coco_dataset_minimal(tmp_path: Path):
    root = tmp_path / "coco"
    annotations = root / "annotations"
    train_dir = root / "train2017"
    val_dir = root / "val2017"
    train_dir.mkdir(parents=True)
    val_dir.mkdir()

    (train_dir / "img.jpg").write_bytes(b"img")

    _write_json(
        annotations / "instances_train2017.json",
        {
            "images": [{"id": 1, "file_name": "img.jpg", "height": 10, "width": 8}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 2, 2],
                    "segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]],
                    "area": 4,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "cat"}],
        },
    )

    ds = load_coco_dataset(str(root), version="2017")

    assert len(ds) == 1
    sample = ds[0]
    assert sample.subset == Subset.TRAINING
    assert sample.labels.tolist() == [0]
    assert sample.bboxes.shape == (1, 4)


def test_load_coco_dataset_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_coco_dataset(str(tmp_path / "missing"))

    root = tmp_path / "no_ann"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        load_coco_dataset(str(root))


def test_save_coco_dataset_writes_expected_structure(tmp_path: Path):
    dataset = Dataset(CocoSample, categories={"labels": CocoCategories(), "caption_group_ids": CocoCategories()})
    img_path = tmp_path / "src.jpg"
    img_path.write_bytes(b"img")
    dataset.append(_make_sample(img_path))

    written = save_coco_dataset(dataset, str(tmp_path / "export"), version="2017")

    inst_path = tmp_path / "export" / "annotations" / "instances_train2017.json"
    assert written["instances_train"] == inst_path
    result = json.loads(inst_path.read_text())
    assert result["annotations"]
    assert (tmp_path / "export" / "train2017" / "src.jpg").exists()
