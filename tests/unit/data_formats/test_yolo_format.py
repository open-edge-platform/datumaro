# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=signature-differs

from copy import deepcopy
from typing import Any, Dict, Optional, Type

import numpy as np
import pytest

from datumaro.components.annotation import Bbox
from datumaro.components.dataset import Dataset, StreamDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.exporter import Exporter
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.plugins.data_formats.yolo.exporter import YoloExporter, YoloUltralyticsExporter
from datumaro.plugins.data_formats.yolo.importer import YoloImporter
from datumaro.util.definitions import DEFAULT_SUBSET_NAME
from tests.utils.assets import get_test_asset_path

from .base import TestDataFormatBase

STRICT_DIR = get_test_asset_path("yolo_dataset", "strict")
ANNOTATIONS_DIR = get_test_asset_path("yolo_dataset", "annotations")
LABELS_DIR = get_test_asset_path("yolo_dataset", "labels")


@pytest.fixture
def fxt_train_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id=1,
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 15, 3))),
                annotations=[
                    Bbox(0, 2, 4, 2, label=2, id=0, group=0),
                    Bbox(3, 3, 2, 3, label=4, id=1, group=1),
                ],
            ),
        ],
        categories=["label_" + str(i) for i in range(10)],
    )


@pytest.fixture
def fxt_default_dataset(fxt_train_dataset):
    return fxt_train_dataset.transform("map_subsets", mapping={"train": DEFAULT_SUBSET_NAME})


@pytest.fixture
def fxt_train_val_dataset(fxt_train_dataset):
    return Dataset.from_extractors(
        *[
            deepcopy(fxt_train_dataset).transform("map_subsets", mapping={"train": subset_name})
            for subset_name in ["train", "val"]
        ]
    )


class YoloFormatTest(TestDataFormatBase):
    IMPORTER = YoloImporter

    @pytest.mark.parametrize(
        "fxt_dataset_dir",
        [STRICT_DIR, ANNOTATIONS_DIR, LABELS_DIR],
        ids=["strict", "annotations", "labels"],
    )
    def test_can_detect(self, fxt_dataset_dir: str):
        return super().test_can_detect(fxt_dataset_dir)

    @pytest.mark.parametrize("dataset_cls", [Dataset, StreamDataset])
    @pytest.mark.parametrize(
        [
            "fxt_dataset_dir",
            "fxt_expected_dataset",
            "fxt_import_kwargs",
        ],
        [
            (STRICT_DIR, "fxt_train_dataset", {}),
            (ANNOTATIONS_DIR, "fxt_default_dataset", {}),
            (LABELS_DIR, "fxt_train_val_dataset", {}),
        ],
        indirect=["fxt_expected_dataset"],
        ids=["strict", "annotations", "labels"],
    )
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        dataset_cls: Type[Dataset],
        request: pytest.FixtureRequest,
    ):
        return super().test_can_import(
            fxt_dataset_dir,
            fxt_expected_dataset,
            fxt_import_kwargs,
            request,
            dataset_cls=dataset_cls,
        )

    @pytest.mark.parametrize("dataset_cls", [Dataset, StreamDataset])
    @pytest.mark.parametrize(
        "fxt_expected_dataset, exporter",
        [
            ("fxt_train_dataset", YoloExporter),
            ("fxt_train_val_dataset", YoloUltralyticsExporter),
        ],
        indirect=["fxt_expected_dataset"],
    )
    def test_can_export_and_import(
        self,
        fxt_expected_dataset: Dataset,
        test_dir: str,
        fxt_import_kwargs: Dict[str, Any],
        fxt_export_kwargs: Dict[str, Any],
        dataset_cls: Type[Dataset],
        request: pytest.FixtureRequest,
        exporter: Optional[Exporter],
        importer: Optional[Importer] = None,
    ):
        return super().test_can_export_and_import(
            fxt_expected_dataset,
            test_dir,
            fxt_import_kwargs,
            fxt_export_kwargs,
            request,
            exporter=exporter,
            importer=importer,
            dataset_cls=dataset_cls,
        )
