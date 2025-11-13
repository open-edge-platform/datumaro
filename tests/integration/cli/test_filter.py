# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from unittest import TestCase

from datumaro.components.annotation import Bbox, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.util.scope import scope_add, scoped
from tests.utils.test_utils import TestDir, compare_datasets
from tests.utils.test_utils import run_datum as run


class FilterTest(TestCase):
    @scoped
    def test_can_filter_dataset_inplace(self):
        test_dir = scope_add(TestDir())
        Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ],
            categories=["a", "b"],
        ).export(test_dir, "coco")

        run(self, "filter", "-e", '/item[id = "1"]', "--overwrite", test_dir + ":coco")

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(1, annotations=[Label(0, id=1, group=1)]),
            ],
            categories=["a", "b"],
        )
        compare_datasets(self, expected_dataset, Dataset.import_from(test_dir, "coco"), ignored_attrs="*")

    def test_filter_fails_on_inplace_update_without_overwrite(self):
        with TestDir() as test_dir:
            Dataset.from_iterable(
                [
                    DatasetItem(id=1, annotations=[Bbox(1, 2, 3, 4, label=1)]),
                ],
                categories=["a", "b"],
            ).export(test_dir, "coco")

            run(self, "filter", "-e", "/item", test_dir + ":coco", expected_code=1)
