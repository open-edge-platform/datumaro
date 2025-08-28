# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from unittest.case import TestCase

import numpy as np

from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.mars import MarsImporter

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

DUMMY_MARS_DATASET = get_test_asset_path("mars_dataset")


class MarsImporterTest(TestCase):
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0001C1T0001F001",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    subset="train",
                    annotations=[Label(label=2)],
                    attributes={"person_id": "0001", "camera_id": 1, "track_id": 1, "frame_id": 1},
                ),
                DatasetItem(
                    id="0000C6T0101F001",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    subset="train",
                    annotations=[Label(label=1)],
                    attributes={
                        "person_id": "0000",
                        "camera_id": 6,
                        "track_id": 101,
                        "frame_id": 1,
                    },
                ),
                DatasetItem(
                    id="00-1C2T0081F201",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    subset="test",
                    annotations=[Label(label=0)],
                    attributes={
                        "person_id": "00-1",
                        "camera_id": 2,
                        "track_id": 81,
                        "frame_id": 201,
                    },
                ),
            ],
            categories=["00-1", "0000", "0001"],
        )

        imported_dataset = Dataset.import_from(DUMMY_MARS_DATASET, "mars")
        compare_datasets(self, expected_dataset, imported_dataset, require_media=True)

    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_MARS_DATASET)
        self.assertEqual([MarsImporter.NAME], detected_formats)
