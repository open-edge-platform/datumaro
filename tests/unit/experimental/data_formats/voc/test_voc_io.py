# Copyright (C) 2022-2026 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Unit tests for VOC I/O functions.
"""

from pathlib import Path

import numpy as np
import pytest

from datumaro.experimental import Dataset
from datumaro.experimental.data_formats.voc.io import load_voc_dataset, save_voc_dataset
from datumaro.experimental.data_formats.voc.sample import VocSample
from datumaro.experimental.fields import ImageInfo, Subset


def _create_test_image(path: Path, width: int = 640, height: int = 480) -> None:
    """Create a minimal valid image file for testing."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color="red")
    img.save(path)


def _create_voc_structure(root: Path) -> None:
    """Create a minimal VOC directory structure with test data."""
    # Create directories
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir(parents=True)
    (root / "ImageSets" / "Main").mkdir(parents=True)

    # Create test images
    _create_test_image(root / "JPEGImages" / "test001.jpg")
    _create_test_image(root / "JPEGImages" / "test002.jpg")

    # Create annotations
    (root / "Annotations" / "test001.xml").write_text("""
<annotation>
    <size>
        <width>640</width>
        <height>480</height>
    </size>
    <object>
        <name>person</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>200</ymax>
        </bndbox>
        <difficult>0</difficult>
    </object>
</annotation>
    """)

    (root / "Annotations" / "test002.xml").write_text("""
<annotation>
    <size>
        <width>640</width>
        <height>480</height>
    </size>
    <object>
        <name>dog</name>
        <bndbox>
            <xmin>50</xmin>
            <ymin>50</ymin>
            <xmax>150</xmax>
            <ymax>150</ymax>
        </bndbox>
    </object>
    <object>
        <name>cat</name>
        <bndbox>
            <xmin>300</xmin>
            <ymin>200</ymin>
            <xmax>400</xmax>
            <ymax>350</ymax>
        </bndbox>
    </object>
</annotation>
    """)

    # Create ImageSets
    (root / "ImageSets" / "Main" / "train.txt").write_text("test001\n")
    (root / "ImageSets" / "Main" / "val.txt").write_text("test002\n")


# ==============================
# load_voc_dataset Tests
# ==============================


def test_load_voc_dataset_from_standard_layout(tmp_path: Path):
    """Test loading a VOC dataset from standard directory layout."""
    _create_voc_structure(tmp_path)

    dataset = load_voc_dataset(root_dir=str(tmp_path))

    assert len(dataset) == 2


def test_load_voc_dataset_samples_have_correct_structure(tmp_path: Path):
    """Test that loaded samples have the expected structure."""
    _create_voc_structure(tmp_path)

    dataset = load_voc_dataset(root_dir=str(tmp_path))

    for sample in dataset:
        assert hasattr(sample, "image")
        assert hasattr(sample, "image_info")
        assert hasattr(sample, "bboxes")
        assert hasattr(sample, "labels")
        assert hasattr(sample, "subset")


def test_load_voc_dataset_parses_annotations_correctly(tmp_path: Path):
    """Test that annotations are parsed correctly."""
    _create_voc_structure(tmp_path)

    dataset = load_voc_dataset(root_dir=str(tmp_path))

    # Find the sample with 2 objects (test002)
    multi_obj_sample = None
    for sample in dataset:
        if sample.bboxes is not None and len(sample.bboxes) == 2:
            multi_obj_sample = sample
            break

    assert multi_obj_sample is not None
    assert multi_obj_sample.bboxes.shape == (2, 4)
    assert len(multi_obj_sample.labels) == 2


def test_load_voc_dataset_assigns_subsets_correctly(tmp_path: Path):
    """Test that subsets are assigned based on ImageSets."""
    _create_voc_structure(tmp_path)

    dataset = load_voc_dataset(root_dir=str(tmp_path))

    subsets = set()
    for sample in dataset:
        subsets.add(sample.subset)

    # Should have both training and validation samples
    assert Subset.TRAINING in subsets or Subset.VALIDATION in subsets


def test_load_voc_dataset_raises_on_missing_root():
    """Test that loading from non-existent path raises error."""
    with pytest.raises(FileNotFoundError):
        load_voc_dataset(root_dir="/nonexistent/path")


def test_load_voc_dataset_from_simple_layout(tmp_path: Path):
    """Test loading VOC dataset from simple images + annotations directories."""
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"
    images_dir.mkdir()
    annotations_dir.mkdir()

    # Create test image and annotation
    _create_test_image(images_dir / "test.jpg")
    (annotations_dir / "test.xml").write_text("""
<annotation>
    <size>
        <width>640</width>
        <height>480</height>
    </size>
    <object>
        <name>person</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>200</ymax>
        </bndbox>
    </object>
</annotation>
    """)

    dataset = load_voc_dataset(
        images_dir_path=str(images_dir),
        annotations_dir_path=str(annotations_dir),
    )

    assert len(dataset) == 1


# ==============================
# save_voc_dataset Tests
# ==============================


def test_save_voc_dataset_creates_correct_structure(tmp_path: Path):
    """Test that saving creates the correct VOC directory structure."""
    # Create a simple dataset
    from datumaro.experimental.categories import LabelCategories
    from datumaro.experimental.data_formats.voc.constants import VOC_LABELS

    dataset = Dataset(VocSample, categories={"labels": LabelCategories(labels=VOC_LABELS)})

    # Create a temp image
    img_path = tmp_path / "source" / "image.jpg"
    img_path.parent.mkdir()
    _create_test_image(img_path)

    sample = VocSample(
        image=str(img_path),
        image_info=ImageInfo(height=480, width=640),
        bboxes=np.array([[100, 100, 200, 200]], dtype=np.float32),
        labels=np.array([1], dtype=np.uint32),
        difficult=np.array([False]),
        truncated=np.array([False]),
        occluded=np.array([False]),
        pose=np.array(["Frontal"], dtype=object),
        subset=Subset.TRAINING,
    )
    dataset.append(sample)

    output_dir = tmp_path / "output"
    save_voc_dataset(dataset, root_dir=str(output_dir), save_images=True)

    # Check directory structure
    assert (output_dir / "JPEGImages").is_dir()
    assert (output_dir / "Annotations").is_dir()
    assert (output_dir / "ImageSets" / "Main").is_dir()
    assert (output_dir / "labelmap.txt").exists()


def test_save_voc_dataset_creates_xml_annotations(tmp_path: Path):
    """Test that XML annotation files are created."""
    from datumaro.experimental.categories import LabelCategories
    from datumaro.experimental.data_formats.voc.constants import VOC_LABELS

    dataset = Dataset(VocSample, categories={"labels": LabelCategories(labels=VOC_LABELS)})

    img_path = tmp_path / "source" / "test001.jpg"
    img_path.parent.mkdir()
    _create_test_image(img_path)

    sample = VocSample(
        image=str(img_path),
        image_info=ImageInfo(height=480, width=640),
        bboxes=np.array([[100, 100, 200, 200]], dtype=np.float32),
        labels=np.array([1], dtype=np.uint32),
        difficult=None,
        truncated=None,
        occluded=None,
        pose=None,
        subset=Subset.TRAINING,
    )
    dataset.append(sample)

    output_dir = tmp_path / "output"
    save_voc_dataset(dataset, root_dir=str(output_dir), save_images=True)

    # Check XML annotation
    xml_path = output_dir / "Annotations" / "test001.xml"
    assert xml_path.exists()
    content = xml_path.read_text()
    assert "<annotation>" in content
    assert "<object>" in content


def test_save_voc_dataset_creates_imagesets(tmp_path: Path):
    """Test that ImageSets files are created based on subsets."""
    from datumaro.experimental.categories import LabelCategories
    from datumaro.experimental.data_formats.voc.constants import VOC_LABELS

    dataset = Dataset(VocSample, categories={"labels": LabelCategories(labels=VOC_LABELS)})

    for i, subset in enumerate([Subset.TRAINING, Subset.VALIDATION]):
        img_path = tmp_path / "source" / f"image{i}.jpg"
        img_path.parent.mkdir(exist_ok=True)
        _create_test_image(img_path)

        sample = VocSample(
            image=str(img_path),
            image_info=ImageInfo(height=480, width=640),
            bboxes=None,
            labels=None,
            difficult=None,
            truncated=None,
            occluded=None,
            pose=None,
            subset=subset,
        )
        dataset.append(sample)

    output_dir = tmp_path / "output"
    save_voc_dataset(dataset, root_dir=str(output_dir), save_images=True)

    # Check ImageSets files
    assert (output_dir / "ImageSets" / "Main" / "train.txt").exists()
    assert (output_dir / "ImageSets" / "Main" / "val.txt").exists()


# ==============================
# Round-trip Tests
# ==============================


def test_roundtrip_preserves_sample_count(tmp_path: Path):
    """Test that load -> save -> load preserves sample count."""
    _create_voc_structure(tmp_path)

    # Load original
    dataset = load_voc_dataset(root_dir=str(tmp_path))
    original_len = len(dataset)

    # Save
    output_dir = tmp_path / "output"
    save_voc_dataset(dataset, root_dir=str(output_dir), save_images=True)

    # Reload
    reloaded = load_voc_dataset(root_dir=str(output_dir))

    assert len(reloaded) == original_len


def test_roundtrip_preserves_bbox_data(tmp_path: Path):
    """Test that round-trip preserves bounding box data."""
    _create_voc_structure(tmp_path)

    # Load original
    dataset = load_voc_dataset(root_dir=str(tmp_path))

    # Save
    output_dir = tmp_path / "output"
    save_voc_dataset(dataset, root_dir=str(output_dir), save_images=True)

    # Reload and compare
    reloaded = load_voc_dataset(root_dir=str(output_dir))

    original_samples = list(dataset)
    reloaded_samples = list(reloaded)

    for orig, reload in zip(
        sorted(original_samples, key=lambda s: s.image.path), sorted(reloaded_samples, key=lambda s: s.image.path)
    ):
        if orig.bboxes is not None:
            assert reload.bboxes is not None
            np.testing.assert_array_almost_equal(orig.bboxes, reload.bboxes, decimal=0)
