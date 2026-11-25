# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import numpy as np
import pytest

from datumaro.components.annotation import Bbox
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image


@pytest.mark.parametrize(
    ("export_format", "import_format"),
    [
        ("yolo_ultralytics", "yolo"),
        ("voc_detection", "voc_detection"),
        ("coco_instances", "coco_instances"),
        ("kitti_detection", "kitti_detection"),
        ("label_me", "label_me"),
        ("cvat", "cvat"),
        ("datumaro", "datumaro"),
    ],
)
@pytest.mark.parametrize("dirname", ["data [v2]", "data[1-3]"])
def test_can_import_from_path_with_glob_characters(test_dir, export_format, import_format, dirname):
    # A dataset path is literal text, also when it contains glob special characters
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id=item_id,
                subset=subset,
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[Bbox(2, 3, 10, 5, label=0, id=0, group=0)],
            )
            for item_id, subset in (("a", "train"), ("b", "val"))
        ],
        categories=["label_0", "label_1"],
    )
    dataset_dir = osp.join(test_dir, dirname)
    dataset.export(dataset_dir, export_format, save_media=True)

    imported = Dataset.import_from(dataset_dir, import_format)

    assert len(imported) == 2
    assert all(len(item.annotations) == 1 for item in imported)
