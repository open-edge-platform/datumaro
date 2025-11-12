# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest

from datumaro import Dataset, DatasetItem, HLOps
from datumaro.components.annotation import Bbox, Ellipse, Label, Polygon
from tests.utils.test_utils import TestCaseHelper, TestDir
from tests.utils.test_utils import compare_datasets as _compare_datasets


def compare_datasets(_, expected, actual):
    _compare_datasets(TestCaseHelper(), expected, actual)


class HLOpsTest:
    def test_can_transform(self):
        expected = Dataset.from_iterable([DatasetItem(0, subset="train")], categories=["cat", "dog"])

        dataset = Dataset.from_iterable([DatasetItem(10, subset="train")], categories=["cat", "dog"])

        actual = HLOps.transform(dataset, "reindex", start=0)

        compare_datasets(self, expected, actual)

    @pytest.mark.parametrize(
        "expr_or_filter_func",
        ["/item[id=0]", lambda item: str(item.id) == "0"],
        ids=["xpath", "pyfunc"],
    )
    def test_can_filter_items(self, expr_or_filter_func):
        expected = Dataset.from_iterable([DatasetItem(0, subset="train")], categories=["cat", "dog"])

        dataset = Dataset.from_iterable(
            [DatasetItem(0, subset="train"), DatasetItem(1, subset="train")],
            categories=["cat", "dog"],
        )

        actual = HLOps.filter(dataset, expr_or_filter_func)

        compare_datasets(self, expected, actual)

    @pytest.mark.parametrize(
        "expr_or_filter_func",
        ["/item/annotation[id=1]", lambda item, ann: str(ann.id) == "1"],
        ids=["xpath", "pyfunc"],
    )
    def test_can_filter_annotations(self, expr_or_filter_func):
        expected = Dataset.from_iterable(
            [DatasetItem(0, subset="train", annotations=[Label(0, id=1)])],
            categories=["cat", "dog"],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    subset="train",
                    annotations=[
                        Label(0, id=0),
                        Label(0, id=1),
                    ],
                ),
                DatasetItem(1, subset="train"),
            ],
            categories=["cat", "dog"],
        )

        actual = HLOps.filter(dataset, expr_or_filter_func, filter_annotations=True, remove_empty=True)

        compare_datasets(self, expected, actual)

    def test_can_filter_by_annotation_types(self):
        annotations = [
            Label(0, id=0),
            Bbox(0, 0, 1, 1, id=1),
            Polygon([0, 0, 0, 1, 1, 1], id=2),
            Ellipse(0, 0, 1, 1, id=3),
        ]

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    subset="train",
                    annotations=annotations,
                )
            ],
        )

        types = {ann.type.name for ann in annotations}

        for t in types:
            allowed_types = types - {t}
            cmd = " or ".join([f"type='{allowed_type}'" for allowed_type in allowed_types])
            actual = HLOps.filter(
                dataset,
                f"/item/annotation[{cmd}]",
                filter_annotations=True,
                remove_empty=True,
            )
            actual_anns = next(iter(actual)).annotations
            assert len(actual_anns) == len(allowed_types)

    def test_can_merge(self):
        expected = Dataset.from_iterable(
            [DatasetItem(0, subset="train"), DatasetItem(1, subset="train")],
            categories=["cat", "dog"],
        )

        dataset_a = Dataset.from_iterable(
            [
                DatasetItem(0, subset="train"),
            ],
            categories=["cat", "dog"],
        )

        dataset_b = Dataset.from_iterable([DatasetItem(1, subset="train")], categories=["cat", "dog"])

        actual = HLOps.merge(dataset_a, dataset_b)

        compare_datasets(self, expected, actual)

    def test_can_export(self):
        expected = Dataset.from_iterable(
            [DatasetItem(0, subset="train"), DatasetItem(1, subset="train")],
            categories=["cat", "dog"],
        )

        dataset = Dataset.from_iterable(
            [DatasetItem(0, subset="train"), DatasetItem(1, subset="train")],
            categories=["cat", "dog"],
        )

        with TestDir() as test_dir:
            HLOps.export(dataset, test_dir, "datumaro")
            actual = Dataset.load(test_dir)

            compare_datasets(self, expected, actual)

    def test_aggregate(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(0, subset="default"),
                DatasetItem(1, subset="default"),
                DatasetItem(2, subset="default"),
            ],
            categories=["cat", "dog"],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(0, subset="train"),
                DatasetItem(1, subset="val"),
                DatasetItem(2, subset="test"),
            ],
            categories=["cat", "dog"],
        )

        actual = HLOps.aggregate(dataset, from_subsets=["train", "val", "test"], to_subset="default")

        compare_datasets(self, expected, actual)
