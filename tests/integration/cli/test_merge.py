# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from unittest import TestCase

import numpy as np
import pytest

import datumaro.plugins.data_formats.voc.format as VOC
from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories, MaskCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from tests.utils.test_utils import TestDir, compare_datasets
from tests.utils.test_utils import run_datum as run


class MergeTest:
    @pytest.fixture()
    def fxt_homogenous(self, test_dir):
        export_dirs = []
        n_datasets = 3
        for n in range(n_datasets):
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id=f"dset_{n}_{idx}",
                        subset="train",
                        media=Image.from_numpy(data=np.ones((10, 6, 3))),
                        annotations=[
                            Bbox(1, 2, 3, 3, label=0),
                        ],
                    )
                    for idx in range(10)
                ],
                categories=["a", "b"],
            )
            dir_path = osp.join(test_dir, f"dataset_{n}")
            dataset.export(dir_path, format="datumaro", save_media=True)
            export_dirs += [dir_path]

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=f"dset_{n}_{idx}",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0),
                    ],
                )
                for idx in range(10)
                for n in range(n_datasets)
            ],
            categories=["a", "b"],
        )

        return export_dirs, expected

    @pytest.fixture()
    def fxt_heterogenous(self, test_dir):
        export_dirs = []
        n_datasets = 3
        for n in range(n_datasets):
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id=f"dset_{idx}",
                        subset="train",
                        media=Image.from_numpy(data=np.ones((10, 6, 3))),
                        annotations=[
                            Bbox(1, 2, 3, 3, label=0),
                            Bbox(1, 2, 3, 3, label=1),
                        ],
                    )
                    for idx in range(10)
                ],
                categories=["a", f"{n}"],
            )
            dir_path = osp.join(test_dir, f"dataset_{n}")
            dataset.export(dir_path, format="datumaro", save_media=True)
            export_dirs += [dir_path]

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=f"dset_{idx}-{n}",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0),
                        Bbox(1, 2, 3, 3, label=1 + n),
                    ],
                )
                for idx in range(10)
                for n in range(n_datasets)
            ],
            categories=["a"] + [f"{n}" for n in range(n_datasets)],
        )

        return export_dirs, expected

    @pytest.fixture()
    def fxt_heterogenous_mixed_categories(self, test_dir):
        datasets = []
        datasets.append(
            Dataset.from_iterable(
                [
                    DatasetItem(
                        id="dset_0",
                        subset="train",
                        media=Image.from_numpy(data=np.ones((10, 6, 3))),
                        annotations=[
                            Bbox(1, 2, 3, 3, label=0),
                        ],
                    ),
                    DatasetItem(
                        id="dset_1",
                        subset="train",
                        media=Image.from_numpy(data=np.ones((10, 6, 3))),
                        annotations=[
                            Bbox(1, 2, 3, 3, label=0),
                        ],
                    ),
                ],
                categories=["good"],
            )
        )
        datasets.append(
            Dataset.from_iterable(
                [
                    DatasetItem(
                        id="dset_0",
                        subset="train",
                        media=Image.from_numpy(data=np.ones((10, 6, 3))),
                        annotations=[
                            Bbox(1, 2, 3, 3, label=0),
                        ],
                    ),
                    DatasetItem(
                        id="dset_1",
                        subset="train",
                        media=Image.from_numpy(data=np.ones((10, 6, 3))),
                        annotations=[
                            Bbox(1, 2, 3, 3, label=1),
                        ],
                    ),
                    DatasetItem(
                        id="dset_2",
                        subset="train",
                        media=Image.from_numpy(data=np.ones((10, 6, 3))),
                        annotations=[
                            Bbox(1, 2, 3, 3, label=2),
                        ],
                    ),
                ],
                categories=["combined", "good", "missing"],
            )
        )

        export_dirs = []
        for n in range(len(datasets)):
            dir_path = osp.join(test_dir, f"dataset_{n}")
            datasets[n].export(dir_path, format="datumaro", save_media=True)
            export_dirs += [dir_path]

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id="dset_0-0",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0),
                    ],
                ),
                DatasetItem(
                    id="dset_0-1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=1),
                    ],
                ),
                DatasetItem(
                    id="dset_1-0",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0),
                    ],
                ),
                DatasetItem(
                    id="dset_1-1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0),
                    ],
                ),
                DatasetItem(
                    id="dset_2",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=2),
                    ],
                ),
            ],
            categories=["good", "combined", "missing"],
        )
        return export_dirs, expected

    @pytest.fixture
    def test_case(self, request):
        return request.getfixturevalue(request.param)

    @pytest.mark.parametrize(
        "test_case,merge_policy",
        [
            ("fxt_homogenous", "exact"),
            ("fxt_heterogenous", "union"),
            ("fxt_heterogenous_mixed_categories", "union"),
        ],
        indirect=["test_case"],
    )
    def test_merge(self, test_case, merge_policy, test_dir, helper_tc):
        export_dirs, expected = test_case
        result_dir = osp.join(test_dir, "result")
        cmds = [
            "merge",
            "-m",
            merge_policy,
            "-o",
            result_dir,
        ]
        for export_dir in export_dirs:
            cmds += [export_dir]
        cmds += ["--", "--save-media"]

        run(helper_tc, *cmds)
        actual = Dataset.import_from(result_dir)

        compare_datasets(helper_tc, expected, actual, require_media=True)


class IntersectMergeTest(TestCase):
    def test_can_run_self_merge(self):
        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        dataset2 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=1),
                        Bbox(5, 6, 2, 3, label=2),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=2,
                            id=0,
                            group=0,
                            attributes={
                                "score": 0.5,
                                "occluded": False,
                                "difficult": False,
                                "truncated": False,
                            },
                        ),
                        Bbox(
                            5,
                            6,
                            2,
                            3,
                            label=3,
                            id=1,
                            group=1,
                            attributes={
                                "score": 0.5,
                                "occluded": False,
                                "difficult": False,
                                "truncated": False,
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            3,
                            label=1,
                            id=1,
                            group=1,
                            attributes={"score": 0.5, "is_crowd": False},
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["background", "a", "b", "c"]),
                AnnotationType.mask: MaskCategories(VOC.generate_colormap(4)),
            },
        )

        with TestDir() as test_dir:
            dataset1_url = osp.join(test_dir, "dataset1")
            dataset2_url = osp.join(test_dir, "dataset2")

            dataset1.export(dataset1_url, "coco", save_media=True)
            dataset2.export(dataset2_url, "voc", save_media=True)

            result_dir = osp.join(test_dir, "result")
            run(
                self,
                "merge",
                "-m",
                "intersect",
                "-o",
                result_dir,
                dataset2_url + ":voc",
                dataset1_url + ":coco",
            )

            compare_datasets(self, expected, Dataset.load(result_dir), require_media=True)

    def test_can_run_multimerge(self):
        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        dataset2 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=1),
                        Bbox(5, 6, 2, 3, label=2),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=2,
                            id=0,
                            group=0,
                            attributes={
                                "score": 0.5,
                                "occluded": False,
                                "difficult": False,
                                "truncated": False,
                            },
                        ),
                        Bbox(
                            5,
                            6,
                            2,
                            3,
                            label=3,
                            id=1,
                            group=1,
                            attributes={
                                "score": 0.5,
                                "occluded": False,
                                "difficult": False,
                                "truncated": False,
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            3,
                            label=1,
                            id=1,
                            group=1,
                            attributes={"score": 0.5, "is_crowd": False},
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["background", "a", "b", "c"]),
                AnnotationType.mask: MaskCategories(VOC.generate_colormap(4)),
            },
        )

        with TestDir() as test_dir:
            dataset1_url = osp.join(test_dir, "dataset1")
            dataset2_url = osp.join(test_dir, "dataset2")

            dataset1.export(dataset1_url, "coco", save_media=True)
            dataset2.export(dataset2_url, "voc", save_media=True)

            result_dir = osp.join(test_dir, "result")
            run(
                self,
                "merge",
                "-m",
                "intersect",
                "-o",
                result_dir,
                dataset2_url + ":voc",
                dataset1_url + ":coco",
            )

            compare_datasets(self, expected, Dataset.load(result_dir), require_media=True)

    def test_can_save_in_another_format(self):
        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0, id=0, group=0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        dataset2 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=1, id=0, group=0),
                        Bbox(5, 6, 2, 3, label=2, id=1, group=1),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=2, id=0, group=0),
                        Bbox(5, 6, 2, 3, label=3, id=1, group=1),
                        Bbox(1, 2, 3, 3, label=1, id=2, group=2),
                    ],
                ),
            ],
            categories=["background", "a", "b", "c"],
        )

        with TestDir() as test_dir:
            dataset1_url = osp.join(test_dir, "dataset1")
            dataset2_url = osp.join(test_dir, "dataset2")

            dataset1.export(dataset1_url, "coco", save_media=True)
            dataset2.export(dataset2_url, "voc", save_media=True)

            result_dir = osp.join(test_dir, "result")
            run(
                self,
                "merge",
                "-m",
                "intersect",
                "-o",
                result_dir,
                "-f",
                "yolo",
                dataset2_url + ":voc",
                dataset1_url + ":coco",
                "--",
                "--save-media",
            )

            compare_datasets(self, expected, Dataset.import_from(result_dir, "yolo"), require_media=True)
