import json
from pathlib import Path

import numpy as np
import pytest

from datumaro.experimental import Dataset
from datumaro.experimental.categories import KeypointCategories
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

    ds = load_coco_dataset(
        images_dir_path={"training": str(train_dir)},
        annotations_path={"training": str(annotations / "instances_train2017.json")},
    )

    assert len(ds) == 1
    sample = ds[0]
    assert sample.subset == Subset.TRAINING
    assert sample.labels.tolist() == [0]
    assert sample.bboxes.shape == (1, 4)


def test_load_coco_dataset_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_coco_dataset(
            images_dir_path=str(tmp_path / "missing"),
            annotations_path=str(tmp_path / "missing.json"),
        )

    # Test with mismatched types (str vs dict)
    with pytest.raises(ValueError):
        load_coco_dataset(
            images_dir_path=str(tmp_path),
            annotations_path={"train": str(tmp_path / "train.json")},
        )


def test_save_coco_dataset_writes_expected_structure(tmp_path: Path):
    dataset = Dataset(CocoSample, categories={"labels": CocoCategories()})
    img_path = tmp_path / "src.jpg"
    img_path.write_bytes(b"img")
    dataset.append(_make_sample(img_path))

    export_dir = tmp_path / "export"
    images_dir = export_dir / "images"
    annotations_path = export_dir / "annotations.json"

    save_coco_dataset(
        dataset,
        images_dir_path=str(images_dir),
        annotations_path=str(annotations_path),
    )

    assert annotations_path.exists()
    result = json.loads(annotations_path.read_text())
    assert result["annotations"]
    assert (images_dir / "src.jpg").exists()


def test_load_coco_dataset_simple_layout(tmp_path: Path):
    """Test loading a dataset with simple COCOAPI layout (single folder)."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img1.jpg").write_bytes(b"img1")
    (images_dir / "img2.jpg").write_bytes(b"img2")

    annotations_file = tmp_path / "annotations.json"
    _write_json(
        annotations_file,
        {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "height": 10, "width": 8},
                {"id": 2, "file_name": "img2.jpg", "height": 12, "width": 10},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [1, 1, 3, 3], "area": 9, "iscrowd": 0},
            ],
            "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        },
    )

    ds = load_coco_dataset(
        images_dir_path=str(images_dir),
        annotations_path=str(annotations_file),
    )

    assert len(ds) == 2
    # Simple layout should assign UNASSIGNED subset
    assert ds[0].subset == Subset.UNASSIGNED
    assert ds[1].subset == Subset.UNASSIGNED


def test_load_coco_dataset_split_layout(tmp_path: Path):
    """Test loading a dataset with split layout (multiple subsets)."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    (train_dir / "train_img.jpg").write_bytes(b"train")
    (val_dir / "val_img.jpg").write_bytes(b"val")

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()

    _write_json(
        annotations_dir / "train.json",
        {
            "images": [{"id": 1, "file_name": "train_img.jpg", "height": 10, "width": 8}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "cat"}],
        },
    )
    _write_json(
        annotations_dir / "val.json",
        {
            "images": [{"id": 2, "file_name": "val_img.jpg", "height": 12, "width": 10}],
            "annotations": [{"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 1, 3, 3], "area": 9, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "cat"}],
        },
    )

    ds = load_coco_dataset(
        images_dir_path={
            "training": str(train_dir),
            "validation": str(val_dir),
        },
        annotations_path={
            "training": str(annotations_dir / "train.json"),
            "validation": str(annotations_dir / "val.json"),
        },
    )

    assert len(ds) == 2
    # Check subsets are correctly assigned
    subsets = {s.subset for s in ds}
    assert Subset.TRAINING in subsets
    assert Subset.VALIDATION in subsets


def test_save_coco_dataset_split_layout(tmp_path: Path):
    """Test saving a dataset with split layout (multiple subsets)."""
    dataset = Dataset(CocoSample, categories={"labels": CocoCategories()})

    # Create training sample
    train_img = tmp_path / "train_src.jpg"
    train_img.write_bytes(b"train")
    train_sample = CocoSample(
        image=str(train_img),
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
    dataset.append(train_sample)

    # Create validation sample
    val_img = tmp_path / "val_src.jpg"
    val_img.write_bytes(b"val")
    val_sample = CocoSample(
        image=str(val_img),
        image_info=ImageInfo(height=4, width=5),
        bboxes=np.array([[1.0, 1.0, 2.0, 2.0]], dtype=np.float32),
        polygons=None,
        labels=np.array([1], dtype=np.int32),
        areas=np.array([4.0], dtype=np.float32),
        iscrowd=np.array([0], dtype=np.int32),
        subset=Subset.VALIDATION,
        image_id=2,
        caption_group_ids=None,
        captions=None,
        keypoints=None,
    )
    dataset.append(val_sample)

    export_dir = tmp_path / "export"
    train_export_dir = export_dir / "train"
    val_export_dir = export_dir / "val"
    annotations_dir = export_dir / "annotations"

    save_coco_dataset(
        dataset,
        images_dir_path={
            "training": str(train_export_dir),
            "validation": str(val_export_dir),
        },
        annotations_path={
            "training": str(annotations_dir / "train.json"),
            "validation": str(annotations_dir / "val.json"),
        },
    )

    # Check training subset
    assert (annotations_dir / "train.json").exists()
    train_result = json.loads((annotations_dir / "train.json").read_text())
    assert len(train_result["images"]) == 1
    assert (train_export_dir / "train_src.jpg").exists()

    # Check validation subset
    assert (annotations_dir / "val.json").exists()
    val_result = json.loads((annotations_dir / "val.json").read_text())
    assert len(val_result["images"]) == 1
    assert (val_export_dir / "val_src.jpg").exists()


def test_load_coco_dataset_mismatched_keys(tmp_path: Path):
    """Test that mismatched subset keys between images and annotations raise an error."""
    with pytest.raises(ValueError) as exc_info:
        load_coco_dataset(
            images_dir_path={"train": str(tmp_path)},
            annotations_path={"val": str(tmp_path / "val.json")},
        )
    assert "Subset keys must match" in str(exc_info.value)


def test_load_coco_dataset_attaches_keypoint_categories(tmp_path: Path):
    """When annotation file contains keypoint names, the dataset should have KeypointCategories."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img.jpg").write_bytes(b"img")

    annotations_file = tmp_path / "person_keypoints.json"
    _write_json(
        annotations_file,
        {
            "images": [{"id": 1, "file_name": "img.jpg", "height": 10, "width": 8}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "keypoints": [100, 200, 2, 150, 250, 2, 180, 220, 1],
                    "num_keypoints": 3,
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "keypoints": ["nose", "left_eye", "right_eye"],
                    "skeleton": [[0, 1], [0, 2]],
                }
            ],
        },
    )

    ds = load_coco_dataset(
        images_dir_path=str(images_dir),
        annotations_path=str(annotations_file),
    )

    kp_categories = ds.schema.get_categories_for_field("keypoints")
    assert isinstance(kp_categories, KeypointCategories)
    assert kp_categories.labels == ("nose", "left_eye", "right_eye")


def test_load_coco_dataset_no_keypoint_categories_when_absent(tmp_path: Path):
    """When annotation file has no keypoint names, keypoint categories should not be attached."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img.jpg").write_bytes(b"img")

    annotations_file = tmp_path / "instances.json"
    _write_json(
        annotations_file,
        {
            "images": [{"id": 1, "file_name": "img.jpg", "height": 10, "width": 8}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "cat"}],
        },
    )

    ds = load_coco_dataset(
        images_dir_path=str(images_dir),
        annotations_path=str(annotations_file),
    )

    kp_categories = ds.schema.get_categories_for_field("keypoints")
    assert kp_categories is None


def test_load_coco_dataset_keypoint_categories_from_second_file(tmp_path: Path):
    """KeypointCategories should be detected even when only the second annotation file has them."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img.jpg").write_bytes(b"img")

    instances_file = tmp_path / "instances.json"
    _write_json(
        instances_file,
        {
            "images": [{"id": 1, "file_name": "img.jpg", "height": 10, "width": 8}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "person"}],
        },
    )
    keypoints_file = tmp_path / "person_keypoints.json"
    _write_json(
        keypoints_file,
        {
            "images": [{"id": 1, "file_name": "img.jpg", "height": 10, "width": 8}],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "keypoints": ["left_hip", "right_hip"],
                }
            ],
        },
    )

    ds = load_coco_dataset(
        images_dir_path=str(images_dir),
        annotations_path=[str(instances_file), str(keypoints_file)],
    )

    kp_categories = ds.schema.get_categories_for_field("keypoints")
    assert isinstance(kp_categories, KeypointCategories)
    assert kp_categories.labels == ("left_hip", "right_hip")
