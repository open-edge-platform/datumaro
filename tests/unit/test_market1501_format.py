from unittest import TestCase

import numpy as np

from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.market1501 import Market1501Exporter, Market1501Importer
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets


class Market1501FormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0001_c2s3_000001_00",
                    subset="query",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    attributes={
                        "camera_id": 1,
                        "person_id": "0001",
                        "track_id": 3,
                        "frame_id": 1,
                        "bbox_id": 0,
                        "query": True,
                    },
                ),
                DatasetItem(
                    id="0002_c4s2_000002_00",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    attributes={
                        "camera_id": 3,
                        "person_id": "0002",
                        "track_id": 2,
                        "frame_id": 2,
                        "bbox_id": 0,
                        "query": False,
                    },
                ),
                DatasetItem(
                    id="0001_c1s1_000003_00",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    attributes={
                        "camera_id": 0,
                        "person_id": "0001",
                        "track_id": 1,
                        "frame_id": 3,
                        "bbox_id": 0,
                        "query": False,
                    },
                ),
            ],
        )

        with TestDir() as test_dir:
            Market1501Exporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "market1501")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_no_subsets(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0001_c2s3_000001_00",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    attributes={
                        "camera_id": 1,
                        "person_id": "0001",
                        "track_id": 3,
                        "frame_id": 1,
                        "bbox_id": 0,
                        "query": False,
                    },
                ),
            ],
        )

        with TestDir() as test_dir:
            Market1501Exporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "market1501")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    attributes={"camera_id": 0, "person_id": "0001", "query": False},
                ),
            ],
        )

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0001_c1s1_000000_00",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    attributes={
                        "camera_id": 0,
                        "person_id": "0001",
                        "track_id": 1,
                        "frame_id": 0,
                        "bbox_id": 0,
                        "query": False,
                    },
                ),
            ],
        )

        with TestDir() as test_dir:
            Market1501Exporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "market1501")

            compare_datasets(self, expected_dataset, parsed_dataset, require_media=True)

    def test_can_save_dataset_with_no_save_media(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0001_c2s3_000001_00",
                    subset="query",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    attributes={
                        "camera_id": 1,
                        "person_id": "0001",
                        "track_id": 3,
                        "frame_id": 1,
                        "bbox_id": 0,
                        "query": True,
                    },
                ),
            ],
        )

        with TestDir() as test_dir:
            Market1501Exporter.convert(source_dataset, test_dir, save_media=False)
            parsed_dataset = Dataset.import_from(test_dir, "market1501")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id="c/0001_c1s1_000000_00",
                    media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG"),
                    attributes={
                        "camera_id": 0,
                        "person_id": "0001",
                        "track_id": 1,
                        "frame_id": 0,
                        "bbox_id": 0,
                        "query": False,
                    },
                ),
                DatasetItem(
                    id="a/b/0002_c2s2_000001_00",
                    media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp"),
                    attributes={
                        "camera_id": 1,
                        "person_id": "0002",
                        "track_id": 2,
                        "frame_id": 1,
                        "bbox_id": 0,
                        "query": False,
                    },
                ),
            ],
        )

        with TestDir() as test_dir:
            Market1501Exporter.convert(expected, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "market1501")

            compare_datasets(self, expected, parsed_dataset, require_media=True)

    def test_can_save_dataset_with_no_attributes(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="test1",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                ),
            ],
        )

        with TestDir() as test_dir:
            Market1501Exporter.convert(source_dataset, test_dir, save_media=False)
            parsed_dataset = Dataset.import_from(test_dir, "market1501")

            compare_datasets(self, source_dataset, parsed_dataset)


DUMMY_DATASET_DIR = get_test_asset_path("market1501_dataset")


class Market1501ImporterTest(TestCase):
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([Market1501Importer.NAME], detected_formats)

    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0001_c2s3_000111_00",
                    subset="query",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    attributes={
                        "camera_id": 1,
                        "person_id": "0001",
                        "track_id": 3,
                        "frame_id": 111,
                        "bbox_id": 0,
                        "query": True,
                    },
                ),
                DatasetItem(
                    id="0001_c1s1_001051_00",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    attributes={
                        "camera_id": 0,
                        "person_id": "0001",
                        "track_id": 1,
                        "frame_id": 1051,
                        "bbox_id": 0,
                        "query": False,
                    },
                ),
                DatasetItem(
                    id="0002_c1s3_000151_00",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    attributes={
                        "camera_id": 0,
                        "person_id": "0002",
                        "track_id": 3,
                        "frame_id": 151,
                        "bbox_id": 0,
                        "query": False,
                    },
                ),
            ],
        )
        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "market1501")

        compare_datasets(self, expected_dataset, dataset)
