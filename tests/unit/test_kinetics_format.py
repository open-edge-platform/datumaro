# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp

import pytest

from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Video
from datumaro.plugins.data_formats.kinetics import KineticsImporter
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

KINETICS_DATASET_DIR = get_test_asset_path("kinetics_dataset")


@pytest.fixture
def fxt_kinetics_dataset(test_dir):
    def make_video(fname, frame_size=(4, 6), frames=4):
        src_path = osp.join(KINETICS_DATASET_DIR, fname)
        dst_path = osp.join(test_dir, fname)
        if not osp.exists(osp.dirname(dst_path)):
            os.makedirs(osp.dirname(dst_path))
        os.symlink(src_path, dst_path)
        return Video(dst_path)

    return Dataset.from_iterable(
        [
            DatasetItem(
                id="1",
                subset="test",
                annotations=[Label(0, attributes={"time_start": 0, "time_end": 2})],
                media=make_video("video_1.avi"),
            ),
            DatasetItem(
                id="2",
                subset="test",
                annotations=[Label(0, attributes={"time_start": 5, "time_end": 7})],
            ),
            DatasetItem(
                id="4",
                subset="test",
                annotations=[Label(1, attributes={"time_start": 10, "time_end": 15})],
            ),
            DatasetItem(
                id="3",
                subset="train",
                annotations=[Label(2, attributes={"time_start": 0, "time_end": 2})],
                media=make_video("train/3.avi"),
            ),
        ],
        categories=["label_0", "label_1", "label_2"],
        media_type=Video,
    )


class KineticsImporterTest:
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(KINETICS_DATASET_DIR)
        assert detected_formats == [KineticsImporter.NAME]

    def test_can_import_with_video(self, helper_tc, fxt_kinetics_dataset):
        expected_dataset = fxt_kinetics_dataset
        imported_dataset = Dataset.import_from(KINETICS_DATASET_DIR, "kinetics")

        compare_datasets(helper_tc, expected_dataset, imported_dataset, require_media=True)

    def test_can_convert_to_datumaro_and_export_it(self, helper_tc, test_dir):
        imported_dataset = Dataset.import_from(KINETICS_DATASET_DIR, "kinetics")
        export_dir = osp.join(test_dir, "dst")
        imported_dataset.export(export_dir, "datumaro", save_media=True)

        exported_dataset = Dataset.import_from(export_dir, "datumaro")

        compare_datasets(helper_tc, imported_dataset, exported_dataset, require_media=True)
