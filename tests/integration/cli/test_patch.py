import os.path as osp
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Bbox
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image

from ...requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir, compare_datasets
from tests.utils.test_utils import run_datum as run


class PatchTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_run_patch(self):
        dataset = Dataset.from_iterable(
            [
                # Must be overridden
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0),
                    ],
                ),
                # Must be kept
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1)],
                ),
            ],
            categories=["a", "b"],
        )

        patch = Dataset.from_iterable(
            [
                # Must override
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=0),  # Label must be remapped
                        Bbox(5, 6, 2, 3, label=1),  # Label must be remapped
                        Bbox(2, 2, 2, 3, label=2),  # Will be dropped due to label
                    ],
                ),
                # Must be added
                DatasetItem(
                    id=2,
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[Bbox(1, 2, 3, 2, label=1)],  # Label must be remapped
                ),
            ],
            categories=["b", "a", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=1, id=1, group=1),
                        Bbox(5, 6, 2, 3, label=0, id=2, group=2),
                    ],
                ),
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1, id=1, group=1)],
                ),
                DatasetItem(
                    id=2,
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[Bbox(1, 2, 3, 2, label=0, id=2, group=2)],
                ),
            ],
            categories=["a", "b"],
        )

        with TestDir() as test_dir:
            dataset_url = osp.join(test_dir, "dataset1")
            patch_url = osp.join(test_dir, "dataset2")

            dataset.export(dataset_url, "coco", save_media=True)
            patch.export(patch_url, "voc", save_media=True)

            run(
                self,
                "patch",
                "--overwrite",
                dataset_url + ":coco",
                patch_url + ":voc",
                "--",
                "--reindex",
                "1",
                "--save-media",
            )

            compare_datasets(
                self,
                expected,
                Dataset.import_from(dataset_url, format="coco"),
                require_media=True,
                ignored_attrs="*",
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_patch_fails_on_inplace_update_without_overwrite(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((3, 5, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1)],
                ),
            ],
            categories=["a", "b"],
        )

        patch = Dataset.from_iterable(
            [
                DatasetItem(
                    id=2,
                    media=Image.from_numpy(data=np.zeros((3, 4, 3))),
                    annotations=[Bbox(1, 2, 3, 2, label=1)],
                ),
            ],
            categories=["b", "a", "c"],
        )

        with TestDir() as test_dir:
            dataset_url = osp.join(test_dir, "dataset1")
            patch_url = osp.join(test_dir, "dataset2")

            dataset.export(dataset_url, "coco", save_media=True)
            patch.export(patch_url, "coco", save_media=True)

            run(self, "patch", dataset_url + ":coco", patch_url + ":coco", expected_code=1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_patch_fails_on_inplace_update_of_stage(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((3, 5, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=1)],
                ),
            ],
            categories=["a", "b"],
        )

        patch = Dataset.from_iterable(
            [
                DatasetItem(
                    id=2,
                    media=Image.from_numpy(data=np.zeros((3, 4, 3))),
                    annotations=[Bbox(1, 2, 3, 2, label=1)],
                ),
            ],
            categories=["b", "a", "c"],
        )

        with TestDir() as test_dir:
            dataset_url = osp.join(test_dir, "dataset1")
            patch_url = osp.join(test_dir, "dataset2")

            dataset.export(dataset_url, "coco", save_media=True)
            patch.export(patch_url, "coco", save_media=True)

            run(self, "patch", dataset_url + ":coco", patch_url + ":coco", expected_code=1)
