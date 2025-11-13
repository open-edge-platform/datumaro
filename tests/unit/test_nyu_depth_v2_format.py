from unittest import TestCase

import numpy as np

from datumaro.components.annotation import DepthAnnotation
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.nyu_depth_v2 import NyuDepthV2Importer
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

DUMMY_DATASET_DIR = get_test_asset_path("nyu_depth_v2_dataset")


class NyuDepthV2ImporterTest(TestCase):
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([NyuDepthV2Importer.NAME], detected_formats)

    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    media=Image.from_numpy(data=np.ones((6, 4, 3))),
                    annotations=[DepthAnnotation(Image.from_numpy(data=np.ones((6, 4))))],
                ),
                DatasetItem(
                    id="2",
                    media=Image.from_numpy(data=np.ones((4, 3, 3))),
                    annotations=[DepthAnnotation(Image.from_numpy(data=np.ones((4, 3))))],
                ),
            ],
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "nyu_depth_v2")

        compare_datasets(self, expected_dataset, dataset, require_media=True)
