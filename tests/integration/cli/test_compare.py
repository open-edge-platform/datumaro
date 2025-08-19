# Copyright (C) 2021-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
import os
import os.path as osp

import numpy as np

from datumaro.cli.util.compare import DistanceCompareVisualizer
from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
)
from datumaro.components.comparator import DistanceComparator
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image

from ...requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir
from tests.utils.test_utils import run_datum as run


class CompareTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_compare_projects(self, capsys):  # just a smoke test
        label_categories1 = LabelCategories.from_iterable(["x", "a", "b", "y", "z"])
        mask_categories1 = MaskCategories.generate(len(label_categories1))

        point_categories1 = PointsCategories()
        for index, _ in enumerate(label_categories1.items):
            point_categories1.add(index, ["cat1", "cat2"], joints=[[0, 1]])

        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Caption("hello", id=1),
                        Caption("world", id=2, group=5),
                        Label(
                            2,
                            id=3,
                            attributes={
                                "x": 1,
                                "y": "2",
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=0,
                            id=4,
                            z_order=1,
                            attributes={
                                "score": 1.0,
                            },
                        ),
                        Bbox(5, 6, 7, 8, id=5, group=5),
                        Points([1, 2, 2, 0, 1, 1], label=0, id=5, z_order=4),
                        Mask(label=3, id=5, z_order=2, image=np.ones((2, 3))),
                    ],
                ),
                DatasetItem(
                    id=21,
                    subset="train",
                    annotations=[
                        Caption("test"),
                        Label(2),
                        Bbox(1, 2, 3, 4, label=2, id=42, group=42),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="val",
                    annotations=[
                        PolyLine([1, 2, 3, 4, 5, 6, 7, 8], id=11, z_order=1),
                        Polygon([1, 2, 3, 4, 5, 6, 7, 8], id=12, z_order=4),
                    ],
                ),
                DatasetItem(id=42, subset="test", attributes={"a1": 5, "a2": "42"}),
                DatasetItem(id=42),
                DatasetItem(id=43, media=Image.from_file(path="1/b/c.qq", size=(2, 4))),
            ],
            categories={
                AnnotationType.label: label_categories1,
                AnnotationType.mask: mask_categories1,
                AnnotationType.points: point_categories1,
            },
        )

        label_categories2 = LabelCategories.from_iterable(["a", "b", "c", "x", "y"])
        mask_categories2 = MaskCategories.generate(len(label_categories2))

        point_categories2 = PointsCategories()
        for index, _ in enumerate(label_categories2.items):
            point_categories2.add(index, ["cat1", "cat2"], joints=[[0, 1]])

        dataset2 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Caption("hello", id=1),
                        Caption("world", id=2, group=5),
                        Label(
                            2,
                            id=3,
                            attributes={
                                "x": 1,
                                "y": "2",
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=1,
                            id=4,
                            z_order=1,
                            attributes={
                                "score": 1.0,
                            },
                        ),
                        Bbox(5, 6, 7, 8, id=5, group=5),
                        Points([1, 2, 2, 0, 1, 1], label=0, id=5, z_order=4),
                        Mask(label=3, id=5, z_order=2, image=np.ones((2, 3))),
                    ],
                ),
                DatasetItem(
                    id=21,
                    subset="train",
                    annotations=[
                        Caption("test"),
                        Label(2),
                        Bbox(1, 2, 3, 4, label=3, id=42, group=42),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="val",
                    annotations=[
                        PolyLine([1, 2, 3, 4, 5, 6, 7, 8], id=11, z_order=1),
                        Polygon([1, 2, 3, 4, 5, 6, 7, 8], id=12, z_order=4),
                    ],
                ),
                DatasetItem(id=42, subset="test", attributes={"a1": 5, "a2": "42"}),
                DatasetItem(id=42),
                DatasetItem(id=43, media=Image.from_file(path="1/b/c.qq", size=(2, 4))),
            ],
            categories={
                AnnotationType.label: label_categories2,
                AnnotationType.mask: mask_categories2,
                AnnotationType.points: point_categories2,
            },
        )

        with TestDir() as test_dir:
            with DistanceCompareVisualizer(
                save_dir=test_dir,
                comparator=DistanceComparator(iou_threshold=0.8),
            ) as visualizer:
                visualizer.save(dataset1, dataset2)

            expected_output1 = "> z"
            expected_output2 = "< c"
            captured = capsys.readouterr()
            assert expected_output1 in captured.out
            assert expected_output2 in captured.out
            assert 0 != os.listdir(osp.join(test_dir))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_run_distance_diff(self, helper_tc):
        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=0),
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
                        Bbox(5, 6, 7, 8, label=2),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        with TestDir() as test_dir:
            dataset1_url = osp.join(test_dir, "dataset1")
            dataset2_url = osp.join(test_dir, "dataset2")

            dataset1.export(dataset1_url, "coco", save_media=True)
            dataset2.export(dataset2_url, "voc", save_media=True)

            result_dir = osp.join(test_dir, "cmp_result")
            run(
                helper_tc,
                "compare",
                dataset1_url + ":coco",
                dataset2_url + ":voc",
                "-m",
                "distance",
                "-o",
                result_dir,
            )
            assert {"bbox_confusion.png", "train"} == set(os.listdir(result_dir))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_run_equality_diff(self, helper_tc):
        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=0),
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
                        Bbox(5, 6, 7, 8, label=2),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        with TestDir() as test_dir:
            dataset1_url = osp.join(test_dir, "dataset1")
            dataset2_url = osp.join(test_dir, "dataset2")

            dataset1.export(dataset1_url, "coco", save_media=True)
            dataset2.export(dataset2_url, "voc", save_media=True)

            result_dir = osp.join(test_dir, "cmp_result")
            run(
                helper_tc,
                "compare",
                dataset1_url + ":coco",
                dataset2_url + ":voc",
                "-m",
                "equality",
                "-o",
                result_dir,
            )

            assert {"equality_compare.json"} == set(os.listdir(result_dir))
