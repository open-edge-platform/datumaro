from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Bbox
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.vott_json import VottJsonImporter
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

DUMMY_DATASET_DIR = get_test_asset_path("vott_json_dataset", "dataset")
DUMMY_DATASET_DIR_WITH_META_FILE = get_test_asset_path("vott_json_dataset", "dataset_with_meta_file")


class VottJsonImporterTest(TestCase):
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="img0001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    attributes={"id": "0d3de147f"},
                    annotations=[Bbox(5, 10, 10, 2, label=0, attributes={"id": "BsO3zj9bn"})],
                ),
                DatasetItem(
                    id="img0002",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    attributes={"id": "b482849bc"},
                    annotations=[
                        Bbox(11.5, 12, 10.2, 20.5, label=0, attributes={"id": "mosw0b97K"}),
                        Bbox(11.5, 12, 10.2, 20.5, label=1, attributes={"id": "mosw0b97K"}),
                    ],
                ),
                DatasetItem(
                    id="img0003",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    attributes={"id": "50fef05a8"},
                    annotations=[
                        Bbox(6.7, 10.3, 3.3, 4.7, attributes={"id": "35t9mf-Zr"}),
                        Bbox(13.7, 20.2, 31.9, 43.4, label=2, attributes={"id": "sO4m1DtTZ"}),
                    ],
                ),
            ],
            categories=["animal", "dog", "person"],
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "vott_json")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_import_with_meta_file(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="img0001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    attributes={"id": "0d3de147f"},
                    annotations=[Bbox(5, 10, 10, 2, label=0, attributes={"id": "BsO3zj9bn"})],
                ),
                DatasetItem(
                    id="img0002",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    attributes={"id": "b482849bc"},
                    annotations=[Bbox(11.5, 12, 10.2, 20.5, label=1, attributes={"id": "mosw0b97K"})],
                ),
                DatasetItem(
                    id="img0003",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    attributes={"id": "50fef05a8"},
                    annotations=[
                        Bbox(6.7, 10.3, 3.3, 4.7, attributes={"id": "35t9mf-Zr"}),
                        Bbox(13.7, 20.2, 31.9, 43.4, label=2, attributes={"id": "sO4m1DtTZ"}),
                    ],
                ),
            ],
            categories=["animal", "dog", "person"],
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR_WITH_META_FILE, "vott_json")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([VottJsonImporter.NAME], detected_formats)

    def test_can_detect_with_meta_file(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR_WITH_META_FILE)
        self.assertEqual([VottJsonImporter.NAME], detected_formats)
