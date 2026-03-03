# Copyright (C) 2022-2026 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Unit tests for VOC helper functions.
"""

from pathlib import Path
from xml.etree.ElementTree import Element

import numpy as np

from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.data_formats.voc.constants import VOC_LABELS
from datumaro.experimental.data_formats.voc.helpers import (
    _create_voc_xml_annotation,
    _detect_voc_subsets,
    _find_image_file,
    _load_voc_categories,
    _parse_subset_list,
    _parse_voc_annotation,
    _parse_voc_labelmap,
    _write_labelmap,
    _write_voc_xml,
)
from datumaro.experimental.data_formats.voc.sample import VocSample
from datumaro.experimental.fields import ImageInfo, Subset


def _create_test_image(path: Path, width: int = 640, height: int = 480) -> None:
    """Create a minimal valid image file for testing."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color="red")
    img.save(path)


# ==========================
# _find_image_file Tests
# ==========================


def test_find_image_file_finds_jpg_image(tmp_path: Path):
    """Test finding a .jpg image file."""
    img_path = tmp_path / "test.jpg"
    _create_test_image(img_path)

    result = _find_image_file(tmp_path, "test")
    assert result == img_path


def test_find_image_file_finds_png_image(tmp_path: Path):
    """Test finding a .png image file."""
    img_path = tmp_path / "test.png"
    _create_test_image(img_path)

    result = _find_image_file(tmp_path, "test")
    assert result == img_path


def test_find_image_file_returns_none_for_missing_image(tmp_path: Path):
    """Test returning None when no image is found."""
    result = _find_image_file(tmp_path, "nonexistent")
    assert result is None


def test_find_image_file_ignores_non_image_files(tmp_path: Path):
    """Test that non-image files are ignored."""
    (tmp_path / "test.txt").write_text("not an image")

    result = _find_image_file(tmp_path, "test")
    assert result is None


# ==============================
# _parse_voc_labelmap Tests
# ==============================


def test_parse_voc_labelmap_parses_valid_labelmap(tmp_path: Path):
    """Test parsing a valid VOC labelmap file."""
    labelmap_path = tmp_path / "labelmap.txt"
    labelmap_path.write_text("# comment\ndog:255,0,0::\ncat:0,255,0::\n")

    labels = _parse_voc_labelmap(labelmap_path)

    assert labels == ["dog", "cat"]


def test_parse_voc_labelmap_skips_comments_and_empty_lines(tmp_path: Path):
    """Test that comments and empty lines are skipped."""
    labelmap_path = tmp_path / "labelmap.txt"
    labelmap_path.write_text("# comment\n\nperson:::\n\n# another comment\ncar:::\n")

    labels = _parse_voc_labelmap(labelmap_path)

    assert labels == ["person", "car"]


def test_parse_voc_labelmap_handles_empty_file(tmp_path: Path):
    """Test handling an empty labelmap file."""
    labelmap_path = tmp_path / "labelmap.txt"
    labelmap_path.write_text("")

    labels = _parse_voc_labelmap(labelmap_path)

    assert labels == []


# ==============================
# _load_voc_categories Tests
# ==============================


def test_load_voc_categories_uses_defaults_when_no_labelmap(tmp_path: Path):
    """Test that default VOC labels are used when no labelmap exists."""
    categories = _load_voc_categories(tmp_path)

    assert categories.labels == VOC_LABELS


def test_load_voc_categories_loads_from_labelmap(tmp_path: Path):
    """Test loading categories from a labelmap file."""
    labelmap_path = tmp_path / "labelmap.txt"
    labelmap_path.write_text("custom1:::\ncustom2:::\n")

    categories = _load_voc_categories(tmp_path)

    assert categories.labels == ("custom1", "custom2")


def test_load_voc_categories_loads_from_meta_file(tmp_path: Path):
    """Test loading categories from a dataset_meta.json file."""
    import json

    meta_path = tmp_path / "dataset_meta.json"
    meta_path.write_text(json.dumps({"labels": ["label1", "label2", "label3"]}))

    categories = _load_voc_categories(tmp_path)

    assert categories.labels == ("label1", "label2", "label3")


# ==============================
# _parse_subset_list Tests
# ==============================


def test_parse_subset_list_parses_image_ids(tmp_path: Path):
    """Test parsing a simple subset file with image IDs."""
    subset_file = tmp_path / "train.txt"
    subset_file.write_text("image1\nimage2\nimage3\n")

    image_ids = _parse_subset_list(subset_file)

    assert image_ids == ["image1", "image2", "image3"]


def test_parse_subset_list_handles_classification_format(tmp_path: Path):
    """Test parsing subset file with classification labels (image_id label format)."""
    subset_file = tmp_path / "train.txt"
    subset_file.write_text("image1 1\nimage2 -1\nimage3 0\n")

    image_ids = _parse_subset_list(subset_file)

    # Should only extract image IDs, not the labels
    assert image_ids == ["image1", "image2", "image3"]


def test_parse_subset_list_skips_comments_and_empty_lines(tmp_path: Path):
    """Test that comments and empty lines are skipped."""
    subset_file = tmp_path / "train.txt"
    subset_file.write_text("# comment\n\nimage1\n\nimage2\n")

    image_ids = _parse_subset_list(subset_file)

    assert image_ids == ["image1", "image2"]


# ==============================
# _detect_voc_subsets Tests
# ==============================


def test_detect_voc_subsets_finds_subsets(tmp_path: Path):
    """Test detecting VOC subsets from ImageSets/Main directory."""
    imagesets_main = tmp_path / "ImageSets" / "Main"
    imagesets_main.mkdir(parents=True)
    (imagesets_main / "train.txt").write_text("image1\nimage2\n")
    (imagesets_main / "val.txt").write_text("image3\n")

    subsets = _detect_voc_subsets(tmp_path)

    assert "train" in subsets
    assert "val" in subsets
    assert subsets["train"] == imagesets_main / "train.txt"
    assert subsets["val"] == imagesets_main / "val.txt"


def test_detect_voc_subsets_returns_empty_when_no_imagesets(tmp_path: Path):
    """Test that empty dict is returned when ImageSets/Main doesn't exist."""
    subsets = _detect_voc_subsets(tmp_path)

    assert subsets == {}


# ==============================
# _parse_voc_annotation Tests
# ==============================


def test_parse_voc_annotation_parses_valid_xml(tmp_path: Path):
    """Test parsing a valid VOC XML annotation file."""
    anno_path = tmp_path / "test.xml"
    anno_path.write_text("""
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
        <truncated>1</truncated>
        <occluded>0</occluded>
        <pose>Frontal</pose>
    </object>
</annotation>
    """)

    categories = LabelCategories(labels=VOC_LABELS)
    result = _parse_voc_annotation(anno_path, categories)

    assert result["width"] == 640
    assert result["height"] == 480
    assert len(result["bboxes"]) == 1
    assert result["bboxes"][0] == [100.0, 100.0, 200.0, 200.0]
    assert result["labels"][0] == VOC_LABELS.index("person")
    assert result["difficult"][0] is False
    assert result["truncated"][0] is True
    assert result["occluded"][0] is False
    assert result["pose"][0] == "Frontal"


def test_parse_voc_annotation_handles_missing_file(tmp_path: Path):
    """Test that missing annotation file returns empty result."""
    anno_path = tmp_path / "nonexistent.xml"
    categories = LabelCategories(labels=VOC_LABELS)

    result = _parse_voc_annotation(anno_path, categories)

    assert result["width"] == 0
    assert result["height"] == 0
    assert result["bboxes"] == []
    assert result["labels"] == []


def test_parse_voc_annotation_handles_multiple_objects(tmp_path: Path):
    """Test parsing annotation with multiple objects."""
    anno_path = tmp_path / "test.xml"
    anno_path.write_text("""
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
    <object>
        <name>dog</name>
        <bndbox>
            <xmin>300</xmin>
            <ymin>300</ymin>
            <xmax>400</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>
    """)

    categories = LabelCategories(labels=VOC_LABELS)
    result = _parse_voc_annotation(anno_path, categories)

    assert len(result["bboxes"]) == 2
    assert len(result["labels"]) == 2


# ==============================
# _write_labelmap Tests
# ==============================


def test_write_labelmap_writes_valid_format(tmp_path: Path):
    """Test writing a VOC labelmap file."""
    output_path = tmp_path / "labelmap.txt"
    categories = LabelCategories(labels=("dog", "cat", "bird"))

    _write_labelmap(categories, output_path)

    content = output_path.read_text()
    assert "# label:color_rgb:parts:actions" in content
    assert "dog:::" in content
    assert "cat:::" in content
    assert "bird:::" in content


# ==============================
# _create_voc_xml_annotation Tests
# ==============================


def test_create_voc_xml_annotation_creates_valid_xml():
    """Test creating a VOC XML annotation element."""
    sample = VocSample(
        image="/path/to/image.jpg",
        image_info=ImageInfo(height=480, width=640),
        bboxes=np.array([[100, 100, 200, 200]], dtype=np.float32),
        labels=np.array([1], dtype=np.uint32),
        difficult=np.array([False]),
        truncated=np.array([True]),
        occluded=np.array([False]),
        pose=np.array(["Frontal"], dtype=object),
        subset=Subset.TRAINING,
    )
    categories = LabelCategories(labels=VOC_LABELS)

    root = _create_voc_xml_annotation(sample, "image.jpg", categories)

    assert root.tag == "annotation"
    assert root.find("filename").text == "image.jpg"
    assert root.find("size/width").text == "640"
    assert root.find("size/height").text == "480"

    obj = root.find("object")
    assert obj is not None
    assert obj.find("name").text == VOC_LABELS[1]  # Label index 1
    assert obj.find("bndbox/xmin").text == "100"
    assert obj.find("bndbox/ymin").text == "100"
    assert obj.find("bndbox/xmax").text == "200"
    assert obj.find("bndbox/ymax").text == "200"


def test_create_voc_xml_annotation_handles_no_objects():
    """Test creating XML annotation with no objects."""
    sample = VocSample(
        image="/path/to/image.jpg",
        image_info=ImageInfo(height=480, width=640),
        bboxes=None,
        labels=None,
        difficult=None,
        truncated=None,
        occluded=None,
        pose=None,
        subset=Subset.TRAINING,
    )
    categories = LabelCategories(labels=VOC_LABELS)

    root = _create_voc_xml_annotation(sample, "image.jpg", categories)

    assert root.tag == "annotation"
    assert root.find("object") is None


# ==============================
# _write_voc_xml Tests
# ==============================


def test_write_voc_xml_creates_file(tmp_path: Path):
    """Test writing VOC XML to a file."""
    root = Element("annotation")
    filename = Element("filename")
    filename.text = "test.jpg"
    root.append(filename)

    output_path = tmp_path / "test.xml"
    _write_voc_xml(root, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "<annotation>" in content
    assert "<filename>test.jpg</filename>" in content
