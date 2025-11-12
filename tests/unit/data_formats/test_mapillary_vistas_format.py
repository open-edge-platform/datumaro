# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
from typing import Any, Dict

import numpy as np
import pytest

from datumaro.components.annotation import AnnotationType, LabelCategories, Mask, MaskCategories, Polygon
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.media import Image
from datumaro.plugins.data_formats.mapillary_vistas.format import (
    MapillaryVistasLabelMaps,
    make_mapillary_instance_categories,
)
from datumaro.plugins.data_formats.mapillary_vistas.importer import MapillaryVistasImporter
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

from .base import TestDataFormatBase

DATASET_DIR = get_test_asset_path("mapillary_vistas_dataset")
VAL_DATASET_DIR = get_test_asset_path("mapillary_vistas_dataset", "val")


@pytest.fixture
def fxt_dataset_instances_w_polygon():
    label_cat = LabelCategories.from_iterable(
        ["animal--bird", "construction--barrier--separator", "object--vehicle--bicycle"]
    )
    mask_cat = MaskCategories({0: (165, 42, 42), 1: (128, 128, 128), 2: (119, 11, 32)})

    return Dataset.from_iterable(
        [
            DatasetItem(
                id="0",
                subset="val",
                annotations=[
                    Mask(image=np.array([[1, 1, 1, 0, 0]] * 5), id=0, label=0),
                    Mask(image=np.array([[0, 0, 0, 1, 1]] * 5), id=0, label=1),
                    Polygon(points=[0, 0, 1, 0, 2, 0, 2, 4, 0, 4], label=0),
                    Polygon(points=[3, 0, 4, 0, 4, 1, 4, 4, 3, 4], label=1),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="1",
                subset="val",
                annotations=[
                    Mask(image=np.array([[0, 0, 1, 0, 0]] * 5), id=0, label=1),
                    Mask(image=np.array([[1, 1, 0, 0, 0]] * 5), id=0, label=2),
                    Mask(image=np.array([[0, 0, 0, 1, 1]] * 5), id=1, label=2),
                    Polygon(points=[2, 0, 2, 1, 2, 2, 2, 3, 2, 4], label=1),
                    Polygon(points=[0, 0, 1, 0, 1, 4, 4, 0, 0, 0], label=2),
                    Polygon(points=[3, 0, 4, 0, 4, 4, 3, 4, 3, 0], label=2),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="2",
                subset="train",
                annotations=[
                    Mask(image=np.array([[1, 0, 0, 0, 0]] * 5), id=0, label=0),
                    Mask(image=np.array([[0, 0, 1, 0, 0]] * 5), id=1, label=0),
                    Mask(image=np.array([[0, 0, 0, 0, 1]] * 5), id=2, label=0),
                    Mask(image=np.array([[0, 1, 0, 0, 0]] * 5), id=0, label=1),
                    Mask(image=np.array([[0, 0, 0, 1, 0]] * 5), id=1, label=1),
                    Polygon(points=[0, 0, 0, 1, 0, 2, 0, 3, 0, 4], label=0),
                    Polygon(points=[2, 0, 2, 1, 2, 2, 2, 3, 2, 4], label=0),
                    Polygon(points=[4, 0, 4, 1, 4, 2, 4, 3, 4, 4], label=0),
                    Polygon(points=[1, 0, 1, 1, 1, 2, 1, 3, 1, 4], label=1),
                    Polygon(points=[3, 0, 3, 1, 3, 2, 3, 3, 3, 4], label=1),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
        ],
        categories={AnnotationType.label: label_cat, AnnotationType.mask: mask_cat},
    )


@pytest.fixture
def fxt_dataset_instances_wo_polygon():
    label_cat = LabelCategories.from_iterable(
        ["animal--bird", "construction--barrier--separator", "object--vehicle--bicycle"]
    )
    mask_cat = MaskCategories({0: (165, 42, 42), 1: (128, 128, 128), 2: (119, 11, 32)})

    return Dataset.from_iterable(
        [
            DatasetItem(
                id="0",
                subset="val",
                annotations=[
                    Mask(image=np.array([[1, 1, 1, 0, 0]] * 5), id=0, label=0),
                    Mask(image=np.array([[0, 0, 0, 1, 1]] * 5), id=0, label=1),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="1",
                subset="val",
                annotations=[
                    Mask(image=np.array([[0, 0, 1, 0, 0]] * 5), id=0, label=1),
                    Mask(image=np.array([[1, 1, 0, 0, 0]] * 5), id=0, label=2),
                    Mask(image=np.array([[0, 0, 0, 1, 1]] * 5), id=1, label=2),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="2",
                subset="train",
                annotations=[
                    Mask(image=np.array([[1, 0, 0, 0, 0]] * 5), id=0, label=0),
                    Mask(image=np.array([[0, 0, 1, 0, 0]] * 5), id=1, label=0),
                    Mask(image=np.array([[0, 0, 0, 0, 1]] * 5), id=2, label=0),
                    Mask(image=np.array([[0, 1, 0, 0, 0]] * 5), id=0, label=1),
                    Mask(image=np.array([[0, 0, 0, 1, 0]] * 5), id=1, label=1),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
        ],
        categories={AnnotationType.label: label_cat, AnnotationType.mask: mask_cat},
    )


@pytest.fixture
def fxt_dataset_panoptic_w_polygon():
    label_cat = LabelCategories.from_iterable(
        [
            ("animal--bird", "animal"),
            ("construction--barrier--separator", "construction"),
            ("object--vehicle--bicycle", "object"),
        ]
    )
    mask_cat = MaskCategories({0: (165, 42, 42), 1: (128, 128, 128), 2: (119, 11, 32)})

    return Dataset.from_iterable(
        [
            DatasetItem(
                id="0",
                subset="val",
                annotations=[
                    Mask(
                        image=np.array([[1, 1, 1, 0, 0]] * 5),
                        id=1,
                        group=1,
                        label=0,
                        attributes={"is_crowd": True},
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 1, 1]] * 5),
                        id=2,
                        group=2,
                        label=1,
                        attributes={"is_crowd": False},
                    ),
                    Polygon(points=[0, 0, 1, 0, 2, 0, 2, 4, 0, 4], label=0),
                    Polygon(points=[3, 0, 4, 0, 4, 1, 4, 4, 3, 4], label=1),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="1",
                subset="val",
                annotations=[
                    Mask(
                        image=np.array([[1, 1, 0, 0, 0]] * 5),
                        id=1,
                        group=1,
                        label=2,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 0, 1, 0, 0]] * 5),
                        id=2,
                        group=2,
                        label=1,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 1, 1]] * 5),
                        id=3,
                        group=3,
                        label=2,
                        attributes={"is_crowd": True},
                    ),
                    Polygon(points=[2, 0, 2, 1, 2, 2, 2, 3, 2, 4], label=1),
                    Polygon(points=[0, 0, 1, 0, 1, 4, 4, 0, 0, 0], label=2),
                    Polygon(points=[3, 0, 4, 0, 4, 4, 3, 4, 3, 0], label=2),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="2",
                subset="train",
                annotations=[
                    Mask(
                        image=np.array([[1, 0, 0, 0, 0]] * 5),
                        id=1,
                        group=1,
                        label=0,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 1, 0, 0, 0]] * 5),
                        id=2,
                        group=2,
                        label=1,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 0, 1, 0, 0]] * 5),
                        id=3,
                        group=3,
                        label=0,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 1, 0]] * 5),
                        id=4,
                        group=4,
                        label=1,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 0, 1]] * 5),
                        id=5,
                        group=5,
                        label=0,
                        attributes={"is_crowd": False},
                    ),
                    Polygon(points=[0, 0, 0, 1, 0, 2, 0, 3, 0, 4], label=0),
                    Polygon(points=[2, 0, 2, 1, 2, 2, 2, 3, 2, 4], label=0),
                    Polygon(points=[4, 0, 4, 1, 4, 2, 4, 3, 4, 4], label=0),
                    Polygon(points=[1, 0, 1, 1, 1, 2, 1, 3, 1, 4], label=1),
                    Polygon(points=[3, 0, 3, 1, 3, 2, 3, 3, 3, 4], label=1),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
        ],
        categories={AnnotationType.label: label_cat, AnnotationType.mask: mask_cat},
    )


@pytest.fixture
def fxt_dataset_panoptic_wo_polygon():
    label_cat = LabelCategories.from_iterable(
        [
            ("animal--bird", "animal"),
            ("construction--barrier--separator", "construction"),
            ("object--vehicle--bicycle", "object"),
        ]
    )
    mask_cat = MaskCategories({0: (165, 42, 42), 1: (128, 128, 128), 2: (119, 11, 32)})

    return Dataset.from_iterable(
        [
            DatasetItem(
                id="0",
                subset="val",
                annotations=[
                    Mask(
                        image=np.array([[1, 1, 1, 0, 0]] * 5),
                        id=1,
                        group=1,
                        label=0,
                        attributes={"is_crowd": True},
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 1, 1]] * 5),
                        id=2,
                        group=2,
                        label=1,
                        attributes={"is_crowd": False},
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="1",
                subset="val",
                annotations=[
                    Mask(
                        image=np.array([[1, 1, 0, 0, 0]] * 5),
                        id=1,
                        group=1,
                        label=2,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 0, 1, 0, 0]] * 5),
                        id=2,
                        group=2,
                        label=1,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 1, 1]] * 5),
                        id=3,
                        group=3,
                        label=2,
                        attributes={"is_crowd": True},
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="2",
                subset="train",
                annotations=[
                    Mask(
                        image=np.array([[1, 0, 0, 0, 0]] * 5),
                        id=1,
                        group=1,
                        label=0,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 1, 0, 0, 0]] * 5),
                        id=2,
                        group=2,
                        label=1,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 0, 1, 0, 0]] * 5),
                        id=3,
                        group=3,
                        label=0,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 1, 0]] * 5),
                        id=4,
                        group=4,
                        label=1,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 0, 1]] * 5),
                        id=5,
                        group=5,
                        label=0,
                        attributes={"is_crowd": False},
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
        ],
        categories={AnnotationType.label: label_cat, AnnotationType.mask: mask_cat},
    )


@pytest.fixture
def fxt_dataset_original_config():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="0",
                subset="val",
                annotations=[
                    Mask(image=np.array([[1, 1, 1, 0, 0]] * 5), id=0, label=0),
                    Mask(image=np.array([[0, 0, 0, 1, 1]] * 5), id=0, label=1),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="1",
                subset="val",
                annotations=[
                    Mask(image=np.array([[0, 0, 1, 0, 0]] * 5), id=0, label=1),
                    Mask(image=np.array([[1, 1, 0, 0, 0]] * 5), id=0, label=2),
                    Mask(image=np.array([[0, 0, 0, 1, 1]] * 5), id=1, label=2),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
        ],
        categories=make_mapillary_instance_categories(MapillaryVistasLabelMaps["v1.2"]),
    )


@pytest.fixture
def fxt_dataset_keep_category():
    labels = [f"class-{i}" for i in range(101)]
    labels[1] = ("animal--bird", "animal")
    labels[10] = ("construction--barrier--separator", "construction")
    labels[100] = ("object--vehicle--bicycle", "object")

    label_cat = LabelCategories.from_iterable(labels)
    mask_cat = MaskCategories({1: (165, 42, 42), 10: (128, 128, 128), 100: (119, 11, 32)})

    return Dataset.from_iterable(
        [
            DatasetItem(
                id="0",
                subset="val",
                annotations=[
                    Mask(
                        image=np.array([[1, 1, 1, 0, 0]] * 5),
                        id=1,
                        group=1,
                        label=1,
                        attributes={"is_crowd": True},
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 1, 1]] * 5),
                        id=2,
                        group=2,
                        label=10,
                        attributes={"is_crowd": False},
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="1",
                subset="val",
                annotations=[
                    Mask(
                        image=np.array([[1, 1, 0, 0, 0]] * 5),
                        id=1,
                        group=1,
                        label=100,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 0, 1, 0, 0]] * 5),
                        id=2,
                        group=2,
                        label=10,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 1, 1]] * 5),
                        id=3,
                        group=3,
                        label=100,
                        attributes={"is_crowd": True},
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
        ],
        categories={AnnotationType.label: label_cat, AnnotationType.mask: mask_cat},
    )


class MapillaryVistasTest(TestDataFormatBase):
    IMPORTER = MapillaryVistasImporter
    USE_TEST_CAN_EXPORT_AND_IMPORT = False

    @pytest.mark.parametrize(
        "fxt_dataset_dir",
        [DATASET_DIR],
    )
    def test_can_detect(self, fxt_dataset_dir: str):
        detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(fxt_dataset_dir)

        for detected_format in detected_formats:
            assert detected_format in [
                "mapillary_vistas",
                "mapillary_vistas_instances",
                "mapillary_vistas_panoptic",
            ]

    @pytest.mark.parametrize(
        "fxt_dataset_dir,fxt_expected_dataset,fxt_importer_name,fxt_import_kwargs",
        [
            (
                DATASET_DIR,
                "fxt_dataset_instances_wo_polygon",
                "mapillary_vistas_instances",
                {"format_version": "v1.2"},
            ),
            (
                DATASET_DIR,
                "fxt_dataset_instances_wo_polygon",
                "mapillary_vistas_instances",
                {"format_version": "v2.0"},
            ),
            (
                DATASET_DIR,
                "fxt_dataset_instances_w_polygon",
                "mapillary_vistas_instances",
                {"format_version": "v2.0", "parse_polygon": True},
            ),
            (
                DATASET_DIR,
                "fxt_dataset_panoptic_wo_polygon",
                "mapillary_vistas_panoptic",
                {"format_version": "v1.2"},
            ),
            (
                DATASET_DIR,
                "fxt_dataset_panoptic_wo_polygon",
                "mapillary_vistas_panoptic",
                {"format_version": "v2.0"},
            ),
            (
                DATASET_DIR,
                "fxt_dataset_panoptic_w_polygon",
                "mapillary_vistas_panoptic",
                {"format_version": "v2.0", "parse_polygon": True},
            ),
            (
                VAL_DATASET_DIR,
                "fxt_dataset_original_config",
                "mapillary_vistas_instances",
                {"format_version": "v1.2", "use_original_config": True},
            ),
            (
                VAL_DATASET_DIR,
                "fxt_dataset_keep_category",
                "mapillary_vistas_panoptic",
                {"format_version": "v2.0", "keep_original_category_ids": True},
            ),
        ],
    )
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_importer_name: str,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        helper_tc = request.getfixturevalue("helper_tc")
        exptected_dataset = request.getfixturevalue(fxt_expected_dataset)

        dataset = Dataset.import_from(fxt_dataset_dir, fxt_importer_name, **fxt_import_kwargs)

        compare_datasets(helper_tc, exptected_dataset, dataset, require_media=True)
