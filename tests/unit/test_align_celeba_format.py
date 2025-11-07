from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Label, LabelCategories, Points, PointsCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.celeba import AlignCelebaImporter
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

DUMMY_ALIGN_DATASET_DIR = get_test_asset_path("align_celeba_dataset", "dataset")
DUMMY_ALIGN_DATASET_DIR_WITH_META_FILE = get_test_asset_path("align_celeba_dataset", "dataset_with_meta_file")


class AlignCelebaImporterTest(TestCase):
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="000001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[
                        Label(12),
                        Points([69, 109, 106, 113, 77, 142, 73, 152, 108, 154], label=12),
                    ],
                    attributes={
                        "5_o_Clock_Shadow": False,
                        "Arched_Eyebrows": True,
                        "Attractive": True,
                        "Bags_Under_Eyes": False,
                        "Bald": False,
                        "Bangs": False,
                        "Big_Lips": False,
                        "Big_Nose": False,
                    },
                ),
                DatasetItem(
                    id="000002",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[
                        Label(5),
                        Points([69, 110, 107, 112, 81, 135, 70, 151, 108, 153], label=5),
                    ],
                ),
                DatasetItem(
                    id="000003",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[
                        Label(2),
                        Points([76, 112, 104, 106, 108, 128, 74, 156, 98, 158], label=2),
                    ],
                    attributes={
                        "5_o_Clock_Shadow": False,
                        "Arched_Eyebrows": False,
                        "Attractive": False,
                        "Bags_Under_Eyes": True,
                        "Bald": False,
                        "Bangs": False,
                        "Big_Lips": False,
                        "Big_Nose": True,
                    },
                ),
                DatasetItem(
                    id="000004",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[
                        Label(10),
                        Points([72, 113, 108, 108, 101, 138, 71, 155, 101, 151], label=10),
                    ],
                ),
                DatasetItem(
                    id="000005",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[
                        Label(7),
                        Points([66, 114, 112, 112, 86, 119, 71, 147, 104, 150], label=7),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(f"class-{i}" for i in range(13)),
                AnnotationType.points: PointsCategories.from_iterable(
                    [
                        (0, ["lefteye_x"]),
                        (1, ["lefteye_y"]),
                        (2, ["righteye_x"]),
                        (3, ["righteye_y"]),
                        (4, ["nose_x"]),
                        (5, ["nose_y"]),
                        (6, ["leftmouth_x"]),
                        (7, ["leftmouth_y"]),
                        (8, ["rightmouth_x"]),
                        (9, ["rightmouth_y"]),
                    ]
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_ALIGN_DATASET_DIR, "align_celeba")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_import_with_meta_file(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="000001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[Label(1)],
                ),
                DatasetItem(
                    id="000002",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[Label(3)],
                ),
                DatasetItem(
                    id="000003",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="000004",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[Label(2)],
                ),
                DatasetItem(
                    id="000005",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[Label(6)],
                ),
            ],
            categories=[f"class-{i}" for i in range(7)],
        )

        dataset = Dataset.import_from(DUMMY_ALIGN_DATASET_DIR_WITH_META_FILE, "align_celeba")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_detect_align_dataset(self):
        detected_formats = Environment().detect_dataset(DUMMY_ALIGN_DATASET_DIR)
        self.assertEqual([AlignCelebaImporter.NAME], detected_formats)
