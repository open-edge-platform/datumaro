# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
import os.path as osp

import numpy as np
import pytest

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from tests.utils.test_utils import IGNORE_ALL, TestCaseHelper, compare_datasets
from tests.utils.test_utils import run_datum as run


@pytest.fixture()
def fxt_dataset():
    h = w = 8
    n_labels = 5
    n_items = 5

    return Dataset.from_iterable(
        [
            DatasetItem(
                id=f"img_{item_id}",
                subset=subset,
                media=Image.from_numpy(data=np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8), ext=".png"),
                annotations=[
                    Bbox(
                        *np.random.randint(0, h, size=(4,)).tolist(),
                        id=item_id,
                        label=np.random.randint(0, n_labels),
                        group=item_id,
                        z_order=0,
                        attributes={},
                    )
                ],
            )
            for subset in ["Test", "Train", "Validation"]
            for item_id in range(n_items)
        ],
        categories={AnnotationType.label: LabelCategories.from_iterable([f"label_{idx}" for idx in range(n_labels)])},
    )


@pytest.mark.parametrize("input_format", ["coco", "yolo", "datumaro", "datumaro_binary"], ids=lambda x: f"[if:{x}]")
@pytest.mark.parametrize("output_format", ["coco", "yolo", "datumaro", "datumaro_binary"], ids=lambda x: f"[of:{x}]")
@pytest.mark.parametrize("give_input_format", [True, False])
def test_convert_object_detection(
    fxt_dataset: Dataset,
    input_format: str,
    output_format: str,
    give_input_format: bool,
    test_dir: str,
    helper_tc: TestCaseHelper,
    caplog: pytest.LogCaptureFixture,
):
    src_dir = osp.join(test_dir, "src")
    dst_dir = osp.join(test_dir, "dst")

    fxt_dataset.export(src_dir, format=input_format, save_media=True)

    cmd = [
        "convert",
        "-f",
        output_format,
        "-i",
        src_dir,
        "-o",
        dst_dir,
    ]
    if give_input_format:
        cmd += [
            "-if",
            input_format,
        ]
    cmd += [
        "--",
        "--save-media",
    ]

    run(
        helper_tc,
        *cmd,
        expected_code=0,
    )

    if not give_input_format:
        # If no input_format => detect msg
        matched = [msg for msg in caplog.messages if msg == f"Source dataset format detected as {input_format}"]
        assert len(matched) == 1

    # All conversions now succeed
    actual = Dataset.import_from(dst_dir, format=output_format)

    # COCO and YOLO force reindex annotation's "id" and "group"
    # because they do not exist in their schemas.
    # Therefore, we should ignore them when comparison.
    data_formats_forcing_reindex = {"coco", "yolo"}
    if input_format in data_formats_forcing_reindex or output_format in data_formats_forcing_reindex:
        ignore_ann_id, ignore_ann_group = True, True
    else:
        ignore_ann_id, ignore_ann_group = False, False

    compare_datasets(
        helper_tc,
        fxt_dataset,
        actual,
        require_media=True,
        ignored_attrs=IGNORE_ALL,
        ignore_ann_id=ignore_ann_id,
        ignore_ann_group=ignore_ann_group,
    )
