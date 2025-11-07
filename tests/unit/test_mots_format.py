import os.path as osp
from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Mask
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.mots import MotsImporter, MotsPngExporter
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, check_save_and_load, compare_datasets

DUMMY_DATASET_DIR = get_test_asset_path("mots_dataset")


class MotsPngExporterTest(TestCase):
    def _test_save_and_load(
        self, source_dataset, converter, test_dir, target_dataset=None, importer_args=None, **kwargs
    ):
        return check_save_and_load(
            self,
            source_dataset,
            converter,
            test_dir,
            importer="mots",
            target_dataset=target_dataset,
            importer_args=importer_args,
            **kwargs,
        )

    def test_can_save_masks(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((5, 1))),
                    annotations=[
                        # overlapping masks, the first should be truncated
                        # the first and third are different instances
                        Mask(
                            np.array([[0, 0, 0, 1, 0]]),
                            label=3,
                            z_order=3,
                            attributes={"track_id": 1},
                        ),
                        Mask(
                            np.array([[0, 1, 1, 1, 0]]),
                            label=2,
                            z_order=1,
                            attributes={"track_id": 2},
                        ),
                        Mask(
                            np.array([[1, 1, 0, 0, 0]]),
                            label=3,
                            z_order=2,
                            attributes={"track_id": 3},
                        ),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((5, 1))),
                    annotations=[
                        Mask(np.array([[1, 0, 0, 0, 0]]), label=3, attributes={"track_id": 2}),
                    ],
                ),
                DatasetItem(
                    id=3,
                    subset="b",
                    media=Image.from_numpy(data=np.ones((5, 1))),
                    annotations=[
                        Mask(np.array([[0, 1, 0, 0, 0]]), label=0, attributes={"track_id": 1}),
                    ],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        target = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((5, 1))),
                    annotations=[
                        Mask(np.array([[0, 0, 0, 1, 0]]), label=3, attributes={"track_id": 1}),
                        Mask(np.array([[0, 0, 1, 0, 0]]), label=2, attributes={"track_id": 2}),
                        Mask(np.array([[1, 1, 0, 0, 0]]), label=3, attributes={"track_id": 3}),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((5, 1))),
                    annotations=[
                        Mask(np.array([[1, 0, 0, 0, 0]]), label=3, attributes={"track_id": 2}),
                    ],
                ),
                DatasetItem(
                    id=3,
                    subset="b",
                    media=Image.from_numpy(data=np.ones((5, 1))),
                    annotations=[
                        Mask(np.array([[0, 1, 0, 0, 0]]), label=0, attributes={"track_id": 1}),
                    ],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source,
                partial(MotsPngExporter.convert, save_media=True),
                test_dir,
                target_dataset=target,
            )

    def test_can_save_and_load_with_no_save_media(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((5, 1))),
                    annotations=[
                        Mask(np.array([[1, 1, 0, 0, 0]]), label=0, attributes={"track_id": 3}),
                        Mask(np.array([[0, 0, 1, 1, 1]]), label=1, attributes={"track_id": 3}),
                    ],
                ),
            ],
            categories=["label_0", "label_1"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset, partial(MotsPngExporter.convert, save_media=False), test_dir)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    subset="a",
                    media=Image.from_numpy(data=np.ones((5, 1))),
                    annotations=[
                        Mask(np.array([[1, 0, 0, 0, 0]]), label=0, attributes={"track_id": 2}),
                    ],
                ),
            ],
            categories=["a"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source,
                partial(MotsPngExporter.convert, save_media=True),
                test_dir,
                require_media=True,
            )

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    "q/1",
                    media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG"),
                    annotations=[
                        Mask(np.array([[0, 1, 0, 0, 0]]), label=0, attributes={"track_id": 1}),
                    ],
                ),
                DatasetItem(
                    "a/b/c/2",
                    media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp"),
                    annotations=[
                        Mask(np.array([[0, 1, 0, 0, 0]]), label=0, attributes={"track_id": 1}),
                    ],
                ),
            ],
            categories=["a"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected,
                partial(MotsPngExporter.convert, save_media=True),
                test_dir,
                require_media=True,
            )

    def test_can_save_and_load_with_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((5, 1))),
                    annotations=[
                        Mask(np.array([[1, 1, 0, 0, 0]]), label=0, attributes={"track_id": 3}),
                        Mask(np.array([[0, 0, 1, 1, 1]]), label=1, attributes={"track_id": 3}),
                    ],
                ),
            ],
            categories=["label_0", "label_1"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(MotsPngExporter.convert, save_media=True, save_dataset_meta=True),
                test_dir,
                require_media=True,
            )
            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))


class MotsImporterTest(TestCase):
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([MotsImporter.NAME], detected_formats)

    def test_can_import(self):
        target = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 1))),
                    annotations=[
                        Mask(np.array([[0, 0, 0, 1, 0]]), label=3, attributes={"track_id": 1}),
                        Mask(np.array([[0, 0, 1, 0, 0]]), label=2, attributes={"track_id": 2}),
                        Mask(np.array([[1, 1, 0, 0, 0]]), label=3, attributes={"track_id": 3}),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 1))),
                    annotations=[
                        Mask(np.array([[1, 0, 0, 0, 0]]), label=3, attributes={"track_id": 2}),
                    ],
                ),
                DatasetItem(
                    id=3,
                    subset="val",
                    media=Image.from_numpy(data=np.ones((5, 1))),
                    annotations=[
                        Mask(np.array([[0, 1, 0, 0, 0]]), label=0, attributes={"track_id": 1}),
                    ],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        parsed = Dataset.import_from(DUMMY_DATASET_DIR, "mots")
        compare_datasets(self, expected=target, actual=parsed)
