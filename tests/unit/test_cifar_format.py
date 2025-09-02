# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import pickle  # nosec B403
import shutil
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.errors import DatasetImportError
from datumaro.components.media import Image
from datumaro.plugins.data_formats.cifar import CifarExporter, CifarImporter

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets

DUMMY_10_DATASET_DIR = get_test_asset_path("cifar10_dataset")
DUMMY_100_DATASET_DIR = get_test_asset_path("cifar100_dataset")


class CifarFormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="image_2",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="image_3", subset="test", media=Image.from_numpy(data=np.ones((32, 32, 3)))
                ),
                DatasetItem(
                    id="image_4",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(1)],
                ),
            ],
            categories=["label_0", "label_1"],
        )

        with TestDir() as test_dir:
            CifarExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "cifar")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_without_saving_images(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="a", subset="train_1", annotations=[Label(0)]),
                DatasetItem(id="b", subset="train_first", annotations=[Label(1)]),
            ],
            categories=["x", "y"],
        )

        with TestDir() as test_dir:
            CifarExporter.convert(source_dataset, test_dir, save_media=False)
            parsed_dataset = Dataset.import_from(test_dir, "cifar")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_with_different_image_size(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="image_1",
                    media=Image.from_numpy(data=np.ones((10, 8, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="image_2",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(1)],
                ),
            ],
            categories=["dog", "cat"],
        )

        with TestDir() as test_dir:
            CifarExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "cifar")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(0)],
                ),
            ],
            categories=["label_0"],
        )

        with TestDir() as test_dir:
            CifarExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "cifar")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="q/1", media=Image.from_numpy(data=np.zeros((32, 32, 3)), ext=".JPEG")
                ),
                DatasetItem(
                    id="a/b/c/2", media=Image.from_numpy(data=np.zeros((32, 32, 3)), ext=".bmp")
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            CifarExporter.convert(dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "cifar")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_empty_image(self):
        dataset = Dataset.from_iterable(
            [DatasetItem(id="a", annotations=[Label(0)]), DatasetItem(id="b")],
            categories=["label_0"],
        )

        with TestDir() as test_dir:
            CifarExporter.convert(dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "cifar")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)

    def test_inplace_save_writes_only_updated_data(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((2, 1, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    2,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((3, 2, 3))),
                    annotations=[Label(1)],
                ),
                DatasetItem(
                    2,
                    subset="b",
                    media=Image.from_numpy(data=np.ones((2, 2, 3))),
                    annotations=[Label(1)],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((2, 1, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    2,
                    subset="b",
                    media=Image.from_numpy(data=np.ones((2, 2, 3))),
                    annotations=[Label(1)],
                ),
                DatasetItem(
                    3,
                    subset="c",
                    media=Image.from_numpy(data=np.ones((2, 3, 3))),
                    annotations=[Label(2)],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        with TestDir() as path:
            dataset.export(path, "cifar", save_media=True)

            dataset.put(
                DatasetItem(
                    2,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((3, 2, 3))),
                    annotations=[Label(1)],
                )
            )
            dataset.remove(3, "c")
            dataset.save(save_media=True)

            self.assertEqual({"a", "b", "batches.meta"}, set(os.listdir(path)))
            compare_datasets(self, expected, Dataset.import_from(path, "cifar"), require_media=True)

    def test_can_save_and_load_cifar100(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="image_2",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="image_3", subset="test", media=Image.from_numpy(data=np.ones((32, 32, 3)))
                ),
                DatasetItem(
                    id="image_4",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(1)],
                ),
            ],
            categories=[["class_0", "superclass_0"], ["class_1", "superclass_0"]],
        )

        with TestDir() as test_dir:
            CifarExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "cifar")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_cifar100_without_saving_images(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="a", subset="train_1", annotations=[Label(0)]),
                DatasetItem(id="b", subset="train_1", annotations=[Label(1)]),
            ],
            categories=[["class_0", "superclass_0"], ["class_1", "superclass_0"]],
        )

        with TestDir() as test_dir:
            CifarExporter.convert(source_dataset, test_dir, save_media=False)
            parsed_dataset = Dataset.import_from(test_dir, "cifar")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_catch_pickle_exception(self):
        with TestDir() as test_dir:
            # Create dummy CIFAR100 dataset by copy
            dst_dir = osp.join(test_dir, "cifar100")
            shutil.copytree(DUMMY_100_DATASET_DIR, dst_dir)
            anno_file = osp.join(dst_dir, "test")
            # Create a malformed annotation file, "test"
            with open(anno_file, "wb") as file:
                pickle.dump(enumerate([1, 2, 3]), file)
            with self.assertRaisesRegex(pickle.UnpicklingError, "Global"):
                try:
                    Dataset.import_from(dst_dir, "cifar")
                except Exception as e:
                    if isinstance(e, DatasetImportError) and e.__cause__:
                        raise e.__cause__
                    raise

    def test_can_save_and_load_with_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="image_2",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="image_3", subset="test", media=Image.from_numpy(data=np.ones((32, 32, 3)))
                ),
                DatasetItem(
                    id="image_4",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(1)],
                ),
            ],
            categories=["label_0", "label_1"],
        )

        with TestDir() as test_dir:
            CifarExporter.convert(source_dataset, test_dir, save_media=True, save_dataset_meta=True)
            parsed_dataset = Dataset.import_from(test_dir, "cifar")

            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))
            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)


class CifarImporterTest(TestCase):
    def test_can_import_10(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="image_1",
                    subset="data_batch_1",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="image_2",
                    subset="test_batch",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(1)],
                ),
                DatasetItem(
                    id="image_3",
                    subset="test_batch",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(3)],
                ),
                DatasetItem(
                    id="image_4",
                    subset="test_batch",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(2)],
                ),
                DatasetItem(
                    id="image_5",
                    subset="test_batch",
                    media=Image.from_numpy(
                        data=np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
                    ),
                    annotations=[Label(3)],
                ),
            ],
            categories=["airplane", "automobile", "bird", "cat"],
        )

        dataset = Dataset.import_from(DUMMY_10_DATASET_DIR, "cifar")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_detect_10(self):
        detected_formats = Environment().detect_dataset(DUMMY_10_DATASET_DIR)
        self.assertEqual(detected_formats, [CifarImporter.NAME])

    def test_can_import_100(self):
        # Unless simple dataset merge can't overlap labels and add parent
        # information, the datasets must contain all the possible labels.
        # This should be normal on practice.
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="image_1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((7, 8, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="image_2",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 5, 3))),
                    annotations=[Label(1)],
                ),
                DatasetItem(
                    id="image_3",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 5, 3))),
                    annotations=[Label(2)],
                ),
                DatasetItem(
                    id="image_1",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="image_2",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((32, 32, 3))),
                    annotations=[Label(1)],
                ),
                DatasetItem(
                    id="image_3",
                    subset="test",
                    media=Image.from_numpy(
                        data=np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
                    ),
                    annotations=[Label(2)],
                ),
            ],
            categories=[
                ["airplane", "air_object"],
                ["automobile", "ground_object"],
                ["bird", "air_object"],
            ],
        )

        dataset = Dataset.import_from(DUMMY_100_DATASET_DIR, "cifar")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_detect_100(self):
        detected_formats = Environment().detect_dataset(DUMMY_100_DATASET_DIR)
        self.assertEqual(detected_formats, [CifarImporter.NAME])
