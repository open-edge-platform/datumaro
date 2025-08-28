from unittest import TestCase

import numpy as np

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Label,
    LabelCategories,
    Points,
    PointsCategories,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.celeba import CelebaImporter

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

DUMMY_DATASET_DIR = get_test_asset_path("celeba_dataset", "dataset")
DUMMY_DATASET_DIR_WITH_META_FILE = get_test_asset_path("celeba_dataset", "dataset_with_meta_file")


class CelebaImporterTest(TestCase):
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="000001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Label(12),
                        Bbox(95, 71, 226, 313, label=12),
                        Points([165, 184, 244, 176, 196, 249, 194, 271, 266, 260], label=12),
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
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Label(5),
                        Bbox(72, 94, 221, 306, label=5),
                        Points([140, 204, 220, 204, 168, 254, 146, 289, 226, 289], label=5),
                    ],
                ),
                DatasetItem(
                    id="000003",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Label(2),
                        Bbox(216, 59, 91, 126, label=2),
                        Points([244, 104, 264, 105, 263, 121, 235, 134, 251, 140], label=2),
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
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Label(10),
                        Bbox(622, 257, 564, 781, label=10),
                        Points([796, 539, 984, 539, 930, 687, 762, 756, 915, 756], label=10),
                    ],
                ),
                DatasetItem(
                    id="000005",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Label(7),
                        Bbox(236, 109, 120, 166, label=7),
                        Points([273, 169, 328, 161, 298, 172, 283, 208, 323, 207], label=7),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    f"class-{i}" for i in range(13)
                ),
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

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "celeba")

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

        dataset = Dataset.import_from(DUMMY_DATASET_DIR_WITH_META_FILE, "celeba")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([CelebaImporter.NAME], detected_formats)
