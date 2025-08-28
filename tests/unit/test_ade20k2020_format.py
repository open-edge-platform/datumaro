# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from unittest.case import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, LabelCategories, Mask, Polygon
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.ade20k2020 import Ade20k2020Importer

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

DUMMY_DATASET_DIR = get_test_asset_path("ade20k2020_dataset", "dataset")

DUMMY_DATASET_DIR_META_FILE = get_test_asset_path("ade20k2020_dataset", "dataset_with_meta_file")


class Ade20k2020ImporterTest(TestCase):
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([Ade20k2020Importer.NAME], detected_formats)

    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="street/1",
                    subset="training",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Polygon(
                            [1, 0, 1, 1, 1, 2, 1, 3, 1, 4],
                            group=1,
                            z_order=0,
                            id=1,
                            label=1,
                            attributes={"walkin": True},
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 1, 1]] * 5),
                            label=0,
                            group=401,
                            z_order=0,
                            id=401,
                        ),
                        Mask(
                            image=np.array([[0, 1, 0, 0, 0]] * 5),
                            label=1,
                            group=1831,
                            z_order=0,
                            id=1831,
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]] * 5),
                            label=2,
                            id=774,
                            group=774,
                            z_order=1,
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 1, 1]] * 5), label=0, group=0, z_order=0, id=0
                        ),
                        Mask(
                            image=np.array([[0, 1, 0, 0, 0]] * 5),
                            label=1,
                            group=1,
                            z_order=0,
                            id=1,
                            attributes={"walkin": True},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]] * 5), label=2, group=2, z_order=1, id=2
                        ),
                    ],
                ),
                DatasetItem(
                    id="2",
                    subset="validation",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[0, 0, 1, 1, 1]] * 5),
                            label=0,
                            group=401,
                            z_order=0,
                            id=401,
                        ),
                        Mask(
                            image=np.array([[0, 1, 0, 0, 0]] * 5),
                            label=1,
                            group=1831,
                            z_order=0,
                            id=1831,
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 0, 1]] * 5),
                            label=3,
                            group=2122,
                            z_order=2,
                            id=2122,
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]] * 5),
                            label=2,
                            group=774,
                            z_order=1,
                            id=774,
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 1, 1]] * 5), label=0, group=0, z_order=0, id=0
                        ),
                        Mask(
                            image=np.array([[0, 1, 0, 0, 0]] * 5),
                            label=1,
                            group=1,
                            z_order=0,
                            id=1,
                            attributes={"walkin": True},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]] * 5), label=2, group=2, z_order=1, id=2
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 0, 1]] * 5), label=3, group=3, z_order=2, id=3
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    ["car", "person", "door", "rim"]
                )
            },
        )

        imported_dataset = Dataset.import_from(DUMMY_DATASET_DIR, "ade20k2020")
        compare_datasets(self, expected_dataset, imported_dataset, require_media=True)

    def test_can_import_with_meta_file(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="street/1",
                    subset="training",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Polygon(
                            [1, 0, 1, 1, 1, 2, 1, 3, 1, 4],
                            group=1,
                            z_order=0,
                            id=1,
                            label=1,
                            attributes={"walkin": True},
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 1, 1]] * 5),
                            label=0,
                            group=401,
                            z_order=0,
                            id=401,
                        ),
                        Mask(
                            image=np.array([[0, 1, 0, 0, 0]] * 5),
                            label=1,
                            group=1831,
                            z_order=0,
                            id=1831,
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]] * 5),
                            label=2,
                            id=774,
                            group=774,
                            z_order=1,
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 1, 1]] * 5), label=0, group=0, z_order=0, id=0
                        ),
                        Mask(
                            image=np.array([[0, 1, 0, 0, 0]] * 5),
                            label=1,
                            group=1,
                            z_order=0,
                            id=1,
                            attributes={"walkin": True},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]] * 5), label=2, group=2, z_order=1, id=2
                        ),
                    ],
                )
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    ["car", "person", "door", "rim"]
                )
            },
        )

        imported_dataset = Dataset.import_from(DUMMY_DATASET_DIR_META_FILE, "ade20k2020")
        compare_datasets(self, expected_dataset, imported_dataset, require_media=True)
