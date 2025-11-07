# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os

import pytest

from datumaro.components.annotation import Bbox, RleMask
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.data_formats.segment_anything import SegmentAnythingExporter, SegmentAnythingImporter
from tests.utils.assets import get_test_asset_path

from .base import TestDataFormatBase

DATASET_DIR = get_test_asset_path("segment_anything_dataset")


class SegmentAnythingTest(TestDataFormatBase):
    IMPORTER = SegmentAnythingImporter
    EXPORTER = SegmentAnythingExporter
    USE_TEST_CAN_EXPORT_AND_IMPORT = True

    @pytest.fixture
    def fxt_expected_dataset(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    media=Image.from_file(os.path.join(DATASET_DIR, "a.jpg")),
                    annotations=[
                        Bbox(
                            3.0,
                            1.0,
                            1.0,
                            4.0,
                            label=None,
                            id=1,
                            attributes={
                                "predicted_iou": 0.2633798018925796,
                                "stability_score": 0.47182999738062237,
                                "point_coords": [[3, 4]],
                                "crop_box": [3.0, 1.0, 2.0, 5.0],
                            },
                            group=1,
                        ),
                        RleMask(
                            rle={"size": [5, 10], "counts": "`04n0"},
                            label=None,
                            id=1,
                            attributes={
                                "predicted_iou": 0.2633798018925796,
                                "stability_score": 0.47182999738062237,
                                "point_coords": [[3, 4]],
                                "crop_box": [3.0, 1.0, 2.0, 5.0],
                            },
                            group=1,
                        ),
                    ],
                    attributes={"id": 1},
                ),
                DatasetItem(
                    id="b",
                    media=Image.from_file(os.path.join(DATASET_DIR, "b.jpg")),
                    annotations=[
                        Bbox(
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            label=None,
                            id=1,
                            attributes={
                                "predicted_iou": 0.8607975925446022,
                                "stability_score": 0.36768932422921985,
                                "point_coords": [[3, 8]],
                                "crop_box": [0.0, 0.0, 1.0, 1.0],
                            },
                            group=1,
                        ),
                        RleMask(
                            rle={"size": [5, 10], "counts": "b1"},
                            label=None,
                            id=1,
                            attributes={
                                "predicted_iou": 0.8607975925446022,
                                "stability_score": 0.36768932422921985,
                                "point_coords": [[3, 8]],
                                "crop_box": [0.0, 0.0, 1.0, 1.0],
                            },
                            group=1,
                        ),
                    ],
                    attributes={"id": 2},
                ),
            ],
        )

        return expected_dataset

    @pytest.fixture
    def fxt_dataset_dir(self) -> str:
        return get_test_asset_path("segment_anything_dataset")
