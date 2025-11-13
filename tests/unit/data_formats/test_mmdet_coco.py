# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from datumaro.components.annotation import Bbox, Mask, Polygon
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.plugins.data_formats.mmdet import MmdetCocoImporter
from tests.utils.assets import get_test_asset_path

from .base import TestDataFormatBase

DUMMY_DATASET_DIR = get_test_asset_path("coco_dataset", "mmdet_coco")


@pytest.fixture
def fxt_mmdet_coco_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="a",
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                attributes={"id": 5},
                annotations=[Bbox(2, 2, 3, 1, label=1, group=1, id=1, attributes={"is_crowd": False})],
            ),
            DatasetItem(
                id="b",
                subset="val",
                media=Image.from_numpy(data=np.ones((10, 5, 3))),
                attributes={"id": 40},
                annotations=[
                    Polygon(
                        [0, 0, 1, 0, 1, 2, 0, 2],
                        label=0,
                        id=1,
                        group=1,
                        attributes={"is_crowd": False, "x": 1, "y": "hello"},
                    ),
                    Bbox(
                        0.0,
                        0.0,
                        1.0,
                        2.0,
                        id=1,
                        attributes={"x": 1, "y": "hello", "is_crowd": False},
                        group=1,
                        label=0,
                        z_order=0,
                    ),
                    Mask(
                        np.array([[1, 1, 0, 0, 0]] * 10),
                        label=1,
                        id=2,
                        group=2,
                        attributes={"is_crowd": True},
                    ),
                    Bbox(
                        0.0,
                        0.0,
                        1.0,
                        9.0,
                        id=2,
                        attributes={"is_crowd": True},
                        group=2,
                        label=1,
                        z_order=0,
                    ),
                ],
            ),
        ],
        categories=["a", "b", "c"],
    )


@pytest.mark.new
class RoboflowImporterTest(TestDataFormatBase):
    @pytest.mark.parametrize(
        ["fxt_dataset_dir", "importer"],
        [
            (DUMMY_DATASET_DIR, MmdetCocoImporter),
        ],
    )
    def test_can_detect(self, fxt_dataset_dir: str, importer: Importer):
        detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(fxt_dataset_dir)
        assert importer.NAME in detected_formats

    @pytest.mark.parametrize(
        ["fxt_dataset_dir", "fxt_expected_dataset", "importer"],
        [
            (DUMMY_DATASET_DIR, "fxt_mmdet_coco_dataset", MmdetCocoImporter),
        ],
        indirect=["fxt_expected_dataset"],
    )
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        request: pytest.FixtureRequest,
        importer: Importer,
    ):
        return super().test_can_import(
            fxt_dataset_dir=fxt_dataset_dir,
            fxt_expected_dataset=fxt_expected_dataset,
            fxt_import_kwargs={},
            request=request,
            importer=importer,
        )
