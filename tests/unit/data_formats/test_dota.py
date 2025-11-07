# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Optional

import numpy as np
import pytest

from datumaro.components.annotation import RotatedBbox
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.exporter import Exporter
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.plugins.data_formats.dota import DotaExporter, DotaImporter
from tests.utils.assets import get_test_asset_path

from .base import TestDataFormatBase

DUMMY_DATASET_DOTA_DIR = get_test_asset_path("dota_dataset")


@pytest.fixture
def fxt_dota_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="train_001",
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    RotatedBbox(
                        1,
                        1,
                        2,
                        2,
                        90,
                        label=0,
                        attributes={"difficulty": 0},
                    )
                ],
            ),
            DatasetItem(
                id="train_002",
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    RotatedBbox(
                        3,
                        3,
                        4,
                        4,
                        90,
                        label=1,
                        attributes={"difficulty": 0},
                    )
                ],
            ),
            DatasetItem(
                id="val_001",
                subset="val",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    RotatedBbox(
                        0.5,
                        0.5,
                        1,
                        1,
                        90,
                        label=0,
                        attributes={"difficulty": 0},
                    ),
                    RotatedBbox(
                        5.5,
                        3.5,
                        3,
                        9,
                        90,
                        label=1,
                        attributes={"difficulty": 0},
                    ),
                ],
            ),
        ],
        categories=["label_0", "label_1"],
    )


@pytest.mark.new
class DotaImporterTest(TestDataFormatBase):
    @pytest.mark.parametrize(
        ["fxt_dataset_dir", "importer"],
        [
            (DUMMY_DATASET_DOTA_DIR, DotaImporter),
        ],
    )
    def test_can_detect(self, fxt_dataset_dir: str, importer: Importer):
        return super().test_can_detect(fxt_dataset_dir, importer)

    @pytest.mark.parametrize(
        ["fxt_dataset_dir", "fxt_expected_dataset", "importer"],
        [
            (DUMMY_DATASET_DOTA_DIR, "fxt_dota_dataset", DotaImporter),
        ],
        indirect=["fxt_expected_dataset"],
    )
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
        importer: Importer,
    ):
        return super().test_can_import(
            fxt_dataset_dir=fxt_dataset_dir,
            fxt_expected_dataset=fxt_expected_dataset,
            fxt_import_kwargs=fxt_import_kwargs,
            request=request,
            importer=importer,
        )

    @pytest.mark.parametrize(
        ["fxt_expected_dataset", "exporter", "importer"],
        [
            ("fxt_dota_dataset", DotaExporter, DotaImporter),
        ],
        indirect=["fxt_expected_dataset"],
    )
    def test_can_export_and_import(
        self,
        fxt_expected_dataset: Dataset,
        test_dir: str,
        fxt_import_kwargs: Dict[str, Any],
        fxt_export_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
        exporter: Optional[Exporter],
        importer: Optional[Importer],
    ):
        return super().test_can_export_and_import(
            fxt_expected_dataset,
            test_dir,
            fxt_import_kwargs,
            fxt_export_kwargs,
            request,
            exporter=exporter,
            importer=importer,
        )
