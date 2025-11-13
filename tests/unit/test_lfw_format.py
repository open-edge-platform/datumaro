import os
import os.path as osp
import shutil
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Label, Points
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.lfw import LfwExporter, LfwImporter
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets


class LfwFormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="name0_0001",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[Label(0, attributes={"positive_pairs": ["name0/name0_0002"]})],
                ),
                DatasetItem(
                    id="name0_0002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(
                            0,
                            attributes={
                                "positive_pairs": ["name0/name0_0001"],
                                "negative_pairs": ["name1/name1_0001"],
                            },
                        )
                    ],
                ),
                DatasetItem(
                    id="name1_0001",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[Label(1, attributes={"positive_pairs": ["name1/name1_0002"]})],
                ),
                DatasetItem(
                    id="name1_0002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(
                            1,
                            attributes={
                                "positive_pairs": ["name1/name1_0002"],
                                "negative_pairs": ["name0/name0_0001"],
                            },
                        )
                    ],
                ),
            ],
            categories=["name0", "name1"],
        )

        with TestDir() as test_dir:
            LfwExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "lfw")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_with_no_save_media(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="name0_0001",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[Label(0, attributes={"positive_pairs": ["name0/name0_0002"]})],
                ),
                DatasetItem(
                    id="name0_0002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(
                            0,
                            attributes={
                                "positive_pairs": ["name0/name0_0001"],
                                "negative_pairs": ["name1/name1_0001"],
                            },
                        )
                    ],
                ),
                DatasetItem(
                    id="name1_0001",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[Label(1, attributes={})],
                ),
            ],
            categories=["name0", "name1"],
        )

        with TestDir() as test_dir:
            LfwExporter.convert(source_dataset, test_dir, save_media=False)
            parsed_dataset = Dataset.import_from(test_dir, "lfw")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_and_load_with_landmarks(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="name0_0001",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(0, attributes={"positive_pairs": ["name0/name0_0002"]}),
                        Points([0, 4, 3, 3, 2, 2, 1, 0, 3, 0], label=0),
                    ],
                ),
                DatasetItem(
                    id="name0_0002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(0),
                        Points([0, 5, 3, 5, 2, 2, 1, 0, 3, 0], label=0),
                    ],
                ),
            ],
            categories=["name0"],
        )

        with TestDir() as test_dir:
            LfwExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "lfw")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_and_load_with_no_subsets(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="name0_0001",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[Label(0, attributes={"positive_pairs": ["name0/name0_0002"]})],
                ),
                DatasetItem(
                    id="name0_0002",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[Label(0)],
                ),
            ],
            categories=["name0"],
        )

        with TestDir() as test_dir:
            LfwExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "lfw")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_and_load_with_no_format_names(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/1",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(
                            0,
                            attributes={"positive_pairs": ["name0/b/2"], "negative_pairs": ["d/4"]},
                        )
                    ],
                ),
                DatasetItem(
                    id="b/2",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="c/3",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[Label(1)],
                ),
                DatasetItem(
                    id="d/4",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                ),
            ],
            categories=["name0", "name1"],
        )

        with TestDir() as test_dir:
            LfwExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "lfw")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id="кириллица с пробелом", media=Image.from_numpy(data=np.ones((2, 5, 3)))),
                DatasetItem(
                    id="name0_0002",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[Label(0, attributes={"negative_pairs": ["кириллица с пробелом"]})],
                ),
            ],
            categories=["name0"],
        )

        with TestDir() as test_dir:
            LfwExporter.convert(dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "lfw")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/1",
                    media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG"),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="b/c/d/2",
                    media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp"),
                    annotations=[Label(1)],
                ),
            ],
            categories=["name0", "name1"],
        )

        with TestDir() as test_dir:
            LfwExporter.convert(dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "lfw")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_with_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="name0_0001",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[Label(0, attributes={"positive_pairs": ["name0/name0_0002"]})],
                ),
                DatasetItem(
                    id="name0_0002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(
                            0,
                            attributes={
                                "positive_pairs": ["name0/name0_0001"],
                                "negative_pairs": ["name1/name1_0001"],
                            },
                        )
                    ],
                ),
                DatasetItem(
                    id="name1_0001",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[Label(1, attributes={"positive_pairs": ["name1/name1_0002"]})],
                ),
                DatasetItem(
                    id="name1_0002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(
                            1,
                            attributes={
                                "positive_pairs": ["name1/name1_0002"],
                                "negative_pairs": ["name0/name0_0001"],
                            },
                        )
                    ],
                ),
            ],
            categories=["name0", "name1"],
        )

        with TestDir() as test_dir:
            LfwExporter.convert(source_dataset, test_dir, save_media=True, save_dataset_meta=True)
            parsed_dataset = Dataset.import_from(test_dir, "lfw")

            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))
            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)


DUMMY_DATASET_DIR = get_test_asset_path("lfw_dataset")


class LfwImporterTest(TestCase):
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([LfwImporter.NAME], detected_formats)

    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="name0_0001",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(
                            0,
                            attributes={"negative_pairs": ["name1/name1_0001", "name1/name1_0002"]},
                        ),
                        Points([0, 4, 3, 3, 2, 2, 1, 0, 3, 0], label=0),
                    ],
                ),
                DatasetItem(
                    id="name1_0001",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(
                            1,
                            attributes={
                                "positive_pairs": ["name1/name1_0002"],
                            },
                        ),
                        Points([1, 6, 4, 6, 3, 3, 2, 1, 4, 1], label=1),
                    ],
                ),
                DatasetItem(
                    id="name1_0002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(1),
                        Points([0, 5, 3, 5, 2, 2, 1, 0, 3, 0], label=1),
                    ],
                ),
            ],
            categories=["name0", "name1"],
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "lfw")

        compare_datasets(self, expected_dataset, dataset)

    def test_can_import_without_people_file(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="name0_0001",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(
                            0,
                            attributes={"negative_pairs": ["name1/name1_0001", "name1/name1_0002"]},
                        ),
                        Points([0, 4, 3, 3, 2, 2, 1, 0, 3, 0], label=0),
                    ],
                ),
                DatasetItem(
                    id="name1_0001",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(
                            1,
                            attributes={
                                "positive_pairs": ["name1/name1_0002"],
                            },
                        ),
                        Points([1, 6, 4, 6, 3, 3, 2, 1, 4, 1], label=1),
                    ],
                ),
                DatasetItem(
                    id="name1_0002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Label(1),
                        Points([0, 5, 3, 5, 2, 2, 1, 0, 3, 0], label=1),
                    ],
                ),
            ],
            categories=["name0", "name1"],
        )

        with TestDir() as test_dir:
            dataset_path = osp.join(test_dir, "dataset")
            shutil.copytree(DUMMY_DATASET_DIR, dataset_path)
            os.remove(osp.join(dataset_path, "test", "annotations", "people.txt"))

            dataset = Dataset.import_from(DUMMY_DATASET_DIR, "lfw")

            compare_datasets(self, expected_dataset, dataset)
