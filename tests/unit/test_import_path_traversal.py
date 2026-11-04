# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp

import pytest

from datumaro.components.dataset import Dataset


def _write(path, text):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _voc_xml(filename="a.jpg"):
    return f"<annotation><filename>{filename}</filename></annotation>"


def _labelme_xml(filename="a.png", folder="", mask=None):
    obj = ""
    if mask is not None:
        obj = f"<object><id>0</id><name>a</name><segm><mask>{mask}</mask></segm></object>"
    return f"<annotation><filename>{filename}</filename><folder>{folder}</folder>{obj}</annotation>"


def _voc_image_filename(root, secret):
    _write(osp.join(root, "ImageSets", "Main", "train.txt"), "a\n")
    _write(osp.join(root, "Annotations", "a.xml"), _voc_xml("../../secret.jpg"))
    return "voc_detection"


def _voc_absolute_image_filename(root, secret):
    _write(osp.join(root, "ImageSets", "Main", "train.txt"), "a\n")
    _write(osp.join(root, "Annotations", "a.xml"), _voc_xml(secret))
    return "voc_detection"


def _voc_item_id(root, secret):
    _write(osp.join(root, "ImageSets", "Main", "train.txt"), "../../secret\n")
    return "voc_detection"


def _voc_mask_item_id(root, secret):
    _write(osp.join(root, "ImageSets", "Segmentation", "train.txt"), "../../secret\n")
    return "voc_segmentation"


def _voc_label_name(root, secret):
    _write(osp.join(root, "labelmap.txt"), "# label:color_rgb:parts:actions\n../../../secret:0,0,0::\n")
    _write(osp.join(root, "ImageSets", "Main", "train.txt"), "a\n")
    return "voc_classification"


def _camvid_image(root, secret):
    _write(osp.join(root, "train.txt"), "/train/../../secret.jpg\n")
    return "camvid"


def _camvid_mask(root, secret):
    _write(osp.join(root, "train.txt"), "/train/a.png /trainannot/../../secret.jpg\n")
    return "camvid"


def _labelme_image(root, secret):
    _write(osp.join(root, "Annotations", "a.xml"), _labelme_xml(filename="../../secret.jpg"))
    return "label_me"


def _labelme_folder(root, secret):
    _write(osp.join(root, "Annotations", "a.xml"), _labelme_xml(folder="../.."))
    return "label_me"


def _labelme_mask(root, secret):
    _write(osp.join(root, "Annotations", "a.xml"), _labelme_xml(mask="../../secret.jpg"))
    return "label_me"


def _widerface_image(root, secret):
    _write(osp.join(root, "wider_face_split", "wider_face_train_bbx_gt.txt"), "../../../secret.jpg\n0\n")
    return "wider_face"


def _market1501_image(root, secret):
    _write(osp.join(root, "images_train.txt"), "../secret.jpg\n")
    return "market1501"


@pytest.mark.parametrize(
    "make_dataset",
    [
        _voc_image_filename,
        _voc_absolute_image_filename,
        _voc_item_id,
        _voc_mask_item_id,
        _voc_label_name,
        _camvid_image,
        _camvid_mask,
        _labelme_image,
        _labelme_folder,
        _labelme_mask,
        _widerface_image,
        _market1501_image,
    ],
)
def test_rejects_annotation_paths_escaping_dataset_dir(tmp_path, make_dataset):
    root = str(tmp_path / "dataset")
    secret = str(tmp_path / "secret.jpg")
    _write(secret, "")

    fmt = make_dataset(root, secret)

    with pytest.raises(Exception) as exc_info:
        list(Dataset.import_from(root, fmt))

    error = exc_info.value
    while error is not None and not isinstance(error, ValueError):
        error = error.__cause__
    assert error is not None, f"Expected a path confinement error, got {exc_info.value!r}"
    assert "escapes base directory" in str(error) or "parent directory" in str(error)
