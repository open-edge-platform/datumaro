import os.path as osp
from unittest import TestCase

import numpy as np

from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, save_image
from datumaro.plugins.data_formats.image_zip import ImageZipExporter, ImageZipPath

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets


class ImageZipExporterTest(TestCase):
    def _test_can_save_and_load(self, source_dataset, test_dir, **kwargs):
        archive_path = osp.join(test_dir, kwargs.get("name", ImageZipPath.DEFAULT_ARCHIVE_NAME))
        ImageZipExporter.convert(source_dataset, test_dir, **kwargs)
        parsed_dataset = Dataset.import_from(archive_path, "image_zip")

        compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.ones((10, 6, 3)))),
                DatasetItem(id="2", media=Image.from_numpy(data=np.ones((5, 4, 3)))),
            ]
        )

        with TestDir() as test_dir:
            self._test_can_save_and_load(source_dataset, test_dir)

    def test_can_save_and_load_with_custom_archive_name(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="img_1", media=Image.from_numpy(data=np.ones((10, 10, 3)))),
            ]
        )

        with TestDir() as test_dir:
            self._test_can_save_and_load(source_dataset, test_dir, name="my_archive.zip")

    def test_relative_paths(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.ones((10, 10, 3)))),
                DatasetItem(id="a/2", media=Image.from_numpy(data=np.ones((4, 5, 3)))),
                DatasetItem(id="a/b/3", media=Image.from_numpy(data=np.ones((20, 10, 3)))),
            ]
        )

        with TestDir() as test_dir:
            self._test_can_save_and_load(source_dataset, test_dir)

    def test_can_save_and_load_custom_compresion_method(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.ones((5, 5, 3)))),
                DatasetItem(id="2", media=Image.from_numpy(data=np.ones((4, 3, 3)))),
            ]
        )

        with TestDir() as test_dir:
            self._test_can_save_and_load(source_dataset, test_dir, compression="ZIP_DEFLATED")

    def test_can_save_and_load_with_arbitrary_extensions(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="subset/1", media=Image.from_numpy(data=np.ones((10, 10, 3)))),
                DatasetItem(id="2", media=Image.from_numpy(data=np.ones((4, 5, 3)))),
            ]
        )

        with TestDir() as test_dir:
            save_image(osp.join(test_dir, "2.jpg"), source_dataset.get("2").media.data)
            save_image(
                osp.join(test_dir, "subset", "1.png"),
                source_dataset.get("subset/1").media.data,
                create_dir=True,
            )

            self._test_can_save_and_load(source_dataset, test_dir)


DUMMY_DATASET_DIR = get_test_asset_path("image_zip_dataset")


class ImageZipImporterTest(TestCase):
    def test_can_import(self):
        source_dataset = Dataset.from_iterable(
            [DatasetItem(id="1", media=Image.from_numpy(data=np.ones((10, 10, 3))))]
        )

        zip_path = osp.join(DUMMY_DATASET_DIR, "1.zip")
        parsed_dataset = Dataset.import_from(zip_path, format="image_zip")
        compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_import_from_directory(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.ones((10, 10, 3)))),
                DatasetItem(id="2", media=Image.from_numpy(data=np.ones((5, 10, 3)))),
            ]
        )

        parsed_dataset = Dataset.import_from(DUMMY_DATASET_DIR, format="image_zip")
        compare_datasets(self, source_dataset, parsed_dataset)
