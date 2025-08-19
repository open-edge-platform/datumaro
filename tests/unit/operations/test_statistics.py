# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from datumaro.components.annotation import (
    Bbox,
    Caption,
    Cuboid2D,
    Ellipse,
    Label,
    Mask,
    Points,
    RotatedBbox,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image, PointCloud
from datumaro.components.operations import (
    IMAGE_STATS_SCHEMA,
    compute_ann_statistics,
    compute_image_statistics,
)

from tests.requirements import Requirements, mark_requirement


@pytest.fixture
def fxt_image_dataset_expected_mean_std():
    """Expected image mean and std (RGB)"""
    np.random.seed(3003)
    expected_mean = [100, 50, 150]
    expected_std = [2, 1, 3]

    return expected_mean, expected_std


@pytest.fixture
def fxt_image_dataset(fxt_image_dataset_expected_mean_std: Tuple[List[int], List[int]], tmpdir):
    np.random.seed(3003)

    expected_mean, expected_std = fxt_image_dataset_expected_mean_std

    fpaths = []
    for i, (w, h) in enumerate([(300, 10), (80, 60), (40, 20), (70, 30)]):
        fpath = str(Path(tmpdir) / f"img_{i}.png")
        # NOTE: cv2 use BGR so that we flip expected_mean and expected_std which are RGB
        img_data = np.random.normal(expected_mean[::-1], expected_std[::-1], size=(h, w, 3))
        cv2.imwrite(fpath, img_data)
        fpaths.append(fpath)

    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id=str(i),
                media=Image.from_file(path=str(fpath)),
            )
            for i, fpath in enumerate(fpaths)
        ]
    )
    dataset.put(dataset.get("1"), id="5", subset="train")
    dataset.put(
        DatasetItem(
            id="invalid",
            media=Image.from_file(path="invalid.path"),
        )
    )
    yield dataset


@pytest.fixture
def fxt_point_cloud_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id=i,
                media=PointCloud.from_file(path="dummy.pcd"),
            )
            for i in range(5)
        ],
        media_type=PointCloud,
    )
    return dataset


class ImageStatisticsTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_image_stats(
        self,
        fxt_image_dataset: Dataset,
        fxt_image_dataset_expected_mean_std: Tuple[List[int], List[int]],
    ):
        expected_mean, expected_std = fxt_image_dataset_expected_mean_std
        actual = compute_image_statistics(fxt_image_dataset)

        assert actual["dataset"] == {
            "images count": 5,
            "unique images count": 5,
            "repeated images count": 1,
            "repeated images": [[("1", "default"), ("5", "train")]],
        }
        assert actual["subsets"]["default"]["images count"] == 4
        assert actual["subsets"]["train"]["images count"] == 1

        actual_mean = actual["subsets"]["default"]["image mean (RGB)"]
        actual_std = actual["subsets"]["default"]["image std (RGB)"]

        for em, am in zip(expected_mean, actual_mean):
            assert am == pytest.approx(em, 5e-1)
        for estd, astd in zip(expected_std, actual_std):
            assert astd == pytest.approx(estd, 1e-1)

    @mark_requirement(Requirements.DATUM_BUG_873)
    def test_invalid_media_type(
        self,
        fxt_point_cloud_dataset: Dataset,
    ):
        # PointCloud media_type at dataset level
        with pytest.raises(DatumaroError, match="only Image media_type is allowed"):
            actual = compute_image_statistics(fxt_point_cloud_dataset)

        # Exceptional case of #873, dataset has Image media_type but DatasetItem has PointCloud.
        with patch.object(Dataset, "media_type", return_value=Image):
            with pytest.warns(UserWarning, match="only Image media_type is allowed"):
                actual = compute_image_statistics(fxt_point_cloud_dataset)
            assert actual["dataset"] == IMAGE_STATS_SCHEMA["dataset"]


class AnnStatisticsTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_stats(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Caption("hello"),
                        Caption("world"),
                        Label(
                            2,
                            attributes={
                                "x": 1,
                                "y": "2",
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            2,
                            2,
                            label=2,
                            attributes={
                                "score": 0.5,
                            },
                        ),
                        Bbox(
                            5,
                            6,
                            2,
                            2,
                            attributes={
                                "x": 1,
                                "y": "3",
                                "occluded": True,
                            },
                        ),
                        Points([1, 2, 2, 0, 1, 1], label=0),
                        Mask(
                            label=3,
                            image=np.array(
                                [
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ),
                    ],
                ),
                DatasetItem(
                    id=2,
                    media=Image.from_numpy(data=np.ones((2, 4, 3))),
                    annotations=[
                        Label(
                            2,
                            attributes={
                                "x": 2,
                                "y": "2",
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            2,
                            2,
                            label=3,
                            attributes={
                                "score": 0.5,
                            },
                        ),
                        Bbox(
                            5,
                            6,
                            2,
                            2,
                            attributes={
                                "x": 2,
                                "y": "3",
                                "occluded": False,
                            },
                        ),
                        Ellipse(
                            5,
                            6,
                            2,
                            2,
                            attributes={
                                "x": 2,
                                "y": "3",
                                "occluded": False,
                            },
                        ),
                        RotatedBbox(
                            4,
                            4,
                            2,
                            2,
                            20,
                            attributes={
                                "tiny": True,
                            },
                        ),
                        Cuboid2D(
                            [
                                (1, 1),
                                (3, 1),
                                (3, 3),
                                (1, 3),
                                (1.5, 1.5),
                                (3.5, 1.5),
                                (3.5, 3.5),
                                (1.5, 3.5),
                            ],
                            label=3,
                            id=5,
                            z_order=2,
                            attributes={
                                "x": 1,
                                "y": "2",
                            },
                        ),
                    ],
                ),
                DatasetItem(id=3),
                DatasetItem(id="2.2", media=Image.from_numpy(data=np.ones((2, 4, 3)))),
            ],
            categories=["label_%s" % i for i in range(4)],
        )

        expected = {
            "images count": 4,
            "annotations count": 13,
            "unannotated images count": 2,
            "unannotated images": ["3", "2.2"],
            "annotations by type": {
                "label": {
                    "count": 2,
                },
                "polygon": {
                    "count": 0,
                },
                "polyline": {
                    "count": 0,
                },
                "bbox": {
                    "count": 4,
                },
                "mask": {
                    "count": 1,
                },
                "points": {
                    "count": 1,
                },
                "caption": {
                    "count": 2,
                },
                "rotated_bbox": {
                    "count": 1,
                },
                "cuboid_3d": {"count": 0},
                "super_resolution_annotation": {"count": 0},
                "depth_annotation": {"count": 0},
                "ellipse": {"count": 1},
                "feature_vector": {"count": 0},
                "tabular": {"count": 0},
                "cuboid_2d": {"count": 1},
                "unknown": {"count": 0},
            },
            "annotations": {
                "labels": {
                    "count": 7,
                    "distribution": {
                        "label_0": [1, 1 / 7],
                        "label_1": [0, 0.0],
                        "label_2": [3, 3 / 7],
                        "label_3": [3, 3 / 7],
                    },
                    "attributes": {
                        "x": {
                            "count": 3,  # annotations with no label are skipped
                            "values count": 2,
                            "values present": ["1", "2"],
                            "distribution": {
                                "1": [2, 2 / 3],
                                "2": [1, 1 / 3],
                            },
                        },
                        "y": {
                            "count": 3,  # annotations with no label are skipped
                            "values count": 1,
                            "values present": ["2"],
                            "distribution": {
                                "2": [3, 3 / 3],
                            },
                        },
                        # must not include "special" attributes like "occluded"
                    },
                },
                "segments": {
                    "avg. area": (4 * 2 + 9 * 1) / 3,
                    "area distribution": [
                        {"min": 4.0, "max": 4.5, "count": 2, "percent": 2 / 3},
                        {"min": 4.5, "max": 5.0, "count": 0, "percent": 0.0},
                        {"min": 5.0, "max": 5.5, "count": 0, "percent": 0.0},
                        {"min": 5.5, "max": 6.0, "count": 0, "percent": 0.0},
                        {"min": 6.0, "max": 6.5, "count": 0, "percent": 0.0},
                        {"min": 6.5, "max": 7.0, "count": 0, "percent": 0.0},
                        {"min": 7.0, "max": 7.5, "count": 0, "percent": 0.0},
                        {"min": 7.5, "max": 8.0, "count": 0, "percent": 0.0},
                        {"min": 8.0, "max": 8.5, "count": 0, "percent": 0.0},
                        {"min": 8.5, "max": 9.0, "count": 1, "percent": 1 / 3},
                    ],
                    "pixel distribution": {
                        "label_0": [0, 0.0],
                        "label_1": [0, 0.0],
                        "label_2": [4, 4 / 17],
                        "label_3": [13, 13 / 17],
                    },
                },
            },
        }

        actual = compute_ann_statistics(dataset)

        assert actual == expected

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_stats_with_empty_dataset(self):
        label_names = ["label_%s" % i for i in range(4)]
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1),
                DatasetItem(id=3),
            ],
            categories=label_names,
        )

        expected = self._get_stats_template(label_names)
        expected["images count"] = 2
        expected["unannotated images count"] = 2
        expected["unannotated images"] = ["1", "3"]

        actual = compute_ann_statistics(dataset)
        assert actual == expected

    @mark_requirement(Requirements.DATUM_BUG_1214)
    def test_stats_with_invalid_label(self):
        label_names = ["label_%s" % i for i in range(3)]
        dataset = Dataset.from_iterable(
            iterable=[DatasetItem(id=f"item{i}", annotations=[Label(i)]) for i in range(4)],
            categories=label_names,
        )

        expected = self._get_stats_template(label_names)
        expected["images count"] = 4
        expected["annotations count"] = 4
        expected["annotations by type"]["label"]["count"] = 4
        expected["annotations"]["labels"]["count"] = 4
        expected["annotations"]["labels"]["distribution"] = {
            "label_0": [1, 0.25],
            "label_1": [1, 0.25],
            "label_2": [1, 0.25],
            3: [1, 0.25],  # label which does not exist in categories.
        }

        actual = compute_ann_statistics(dataset)

        assert actual == expected

    @staticmethod
    def _get_stats_template(label_names: list):
        return {
            "images count": 0,
            "annotations count": 0,
            "unannotated images count": 0,
            "unannotated images": [],
            "annotations by type": {
                "label": {"count": 0},
                "polygon": {"count": 0},
                "polyline": {"count": 0},
                "bbox": {"count": 0},
                "mask": {"count": 0},
                "points": {"count": 0},
                "caption": {"count": 0},
                "cuboid_3d": {"count": 0},
                "super_resolution_annotation": {"count": 0},
                "depth_annotation": {"count": 0},
                "ellipse": {"count": 0},
                "feature_vector": {"count": 0},
                "tabular": {"count": 0},
                "rotated_bbox": {"count": 0},
                "cuboid_2d": {"count": 0},
                "unknown": {"count": 0},
            },
            "annotations": {
                "labels": {
                    "count": 0,
                    "distribution": {n: [0, 0] for n in label_names},
                    "attributes": {},
                },
                "segments": {
                    "avg. area": 0.0,
                    "area distribution": [],
                    "pixel distribution": {n: [0, 0] for n in label_names},
                },
            },
        }
