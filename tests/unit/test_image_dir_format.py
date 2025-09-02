import numpy as np
import pytest

from datumaro.components.dataset import Dataset, StreamDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.data_formats.image_dir import ImageDirExporter

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, check_save_and_load, compare_datasets

DUMMY_DATASET_DIR = get_test_asset_path("image_dir_dataset")
FORMAT_NAME = "image_dir"


class ImageDirFormatTest:
    def test_can_load(self, helper_tc):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, media=Image.from_numpy(data=np.ones((10, 6, 3)))),
                DatasetItem(id=2, media=Image.from_numpy(data=np.ones((5, 4, 3)))),
            ]
        )

        with TestDir() as test_dir:
            check_save_and_load(
                helper_tc,
                dataset,
                ImageDirExporter.convert,
                test_dir,
                importer=FORMAT_NAME,
                require_media=True,
            )

    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_can_import(self, dataset_cls, is_stream, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG")),
                DatasetItem(id="2", media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp")),
            ]
        )

        actual_dataset = dataset_cls.import_from(DUMMY_DATASET_DIR, FORMAT_NAME)

        assert actual_dataset.is_stream == is_stream
        compare_datasets(helper_tc, expected_dataset, actual_dataset)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self, helper_tc):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом", media=Image.from_numpy(data=np.ones((4, 2, 3)))
                ),
            ]
        )

        with TestDir() as test_dir:
            check_save_and_load(
                helper_tc, dataset, ImageDirExporter.convert, test_dir, importer=FORMAT_NAME
            )

    def test_can_save_and_load_image_with_arbitrary_extension(self, helper_tc):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG")),
                DatasetItem(id="2", media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp")),
            ]
        )

        with TestDir() as test_dir:
            check_save_and_load(
                helper_tc,
                dataset,
                ImageDirExporter.convert,
                test_dir,
                importer=FORMAT_NAME,
                require_media=True,
            )
