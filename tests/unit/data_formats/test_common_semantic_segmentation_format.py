# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import shutil
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import pytest

from datumaro.components.annotation import Mask
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatasetImportError
from datumaro.components.media import Image
from datumaro.plugins.data_formats.common_semantic_segmentation import (
    CommonSemanticSegmentationImporter,
    CommonSemanticSegmentationWithSubsetDirsImporter,
    make_categories,
)

from .base import TestDataFormatBase

from tests.utils.assets import get_test_asset_path

DUMMY_DATASET_DIR = get_test_asset_path("common_semantic_segmentation_dataset", "dataset")

DUMMY_NON_STANDARD_DATASET_DIR = get_test_asset_path(
    "common_semantic_segmentation_dataset",
    "non_standard_dataset",
)


@pytest.fixture
def fxt_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="0001",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 1, 1]]), label=3),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), label=5),
                ],
            ),
            DatasetItem(
                id="0002",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(image=np.array([[1, 1, 1, 0, 0]]), label=1),
                    Mask(image=np.array([[0, 0, 0, 1, 1]]), label=4),
                ],
            ),
        ],
        categories=make_categories(
            OrderedDict(
                [
                    ("Void", (0, 0, 0)),
                    ("Animal", (64, 128, 64)),
                    ("Archway", (192, 0, 128)),
                    ("Bicyclist", (0, 128, 192)),
                    ("Child", (192, 128, 64)),
                    ("Road", (128, 64, 128)),
                ]
            )
        ),
    )


@pytest.fixture
def fxt_non_standard_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="0001",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 1, 1]]), label=3),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), label=5),
                ],
            ),
            DatasetItem(
                id="0002",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 0, 0]]), label=1),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), label=5),
                    Mask(image=np.array([[0, 0, 0, 1, 1]]), label=7),
                ],
            ),
        ],
        categories=make_categories(
            OrderedDict(
                [
                    ("Void", (0, 0, 0)),
                    ("Animal", (64, 128, 64)),
                    ("Archway", (192, 0, 128)),
                    ("Bicyclist", (0, 128, 192)),
                    ("Child", (192, 128, 64)),
                    ("Road", (128, 64, 128)),
                    ("Pedestrian", (64, 64, 0)),
                    ("SignSymbol", (128, 128, 128)),
                ]
            )
        ),
    )


IDS = ["DUMMY_DATASET_DIR", "DUMMY_NON_STANDARD_DATASET_DIR"]


class CommonSemanticSegmentationImporterTest(TestDataFormatBase):
    IMPORTER = CommonSemanticSegmentationImporter

    @pytest.mark.parametrize(
        "fxt_dataset_dir",
        [DUMMY_DATASET_DIR, DUMMY_NON_STANDARD_DATASET_DIR],
        ids=IDS,
    )
    def test_can_detect(self, fxt_dataset_dir: str):
        return super().test_can_detect(fxt_dataset_dir)

    @pytest.mark.parametrize(
        ["fxt_dataset_dir", "fxt_expected_dataset", "fxt_import_kwargs"],
        [
            (DUMMY_DATASET_DIR, "fxt_dataset", {}),
            (
                DUMMY_NON_STANDARD_DATASET_DIR,
                "fxt_non_standard_dataset",
                {"image_prefix": "image_", "mask_prefix": "gt_"},
            ),
        ],
        indirect=["fxt_expected_dataset"],
        ids=IDS,
    )
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        return super().test_can_import(
            fxt_dataset_dir, fxt_expected_dataset, fxt_import_kwargs, request
        )

    @pytest.mark.parametrize(
        ["fxt_dataset_dir", "fxt_expected_dataset", "fxt_import_kwargs"],
        [
            (DUMMY_DATASET_DIR, "fxt_dataset", {}),
            (
                DUMMY_NON_STANDARD_DATASET_DIR,
                "fxt_non_standard_dataset",
                {"image_prefix": "image_", "mask_prefix": "gt_"},
            ),
        ],
        indirect=["fxt_expected_dataset"],
        ids=IDS,
    )
    def test_cannot_import_nested(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
        test_dir: str,
    ):
        shutil.copytree(fxt_dataset_dir, test_dir, dirs_exist_ok=True)
        subdir_name = "subdir"
        subdir = os.path.join(test_dir, subdir_name)
        os.makedirs(subdir)
        for _file in os.listdir(test_dir):
            if _file != subdir_name:
                file_path = os.path.join(test_dir, _file)
                shutil.move(file_path, subdir)
        with pytest.raises(DatasetImportError) as exc_info:
            super().test_can_import(test_dir, fxt_expected_dataset, fxt_import_kwargs, request)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, FileNotFoundError)


class CommonSemanticSegmentationWithSubsetDirsImporterTest(TestDataFormatBase):
    IMPORTER = CommonSemanticSegmentationWithSubsetDirsImporter

    @pytest.mark.parametrize(
        "fxt_dataset_dir_with_subset_dirs",
        [DUMMY_DATASET_DIR, DUMMY_NON_STANDARD_DATASET_DIR],
        indirect=["fxt_dataset_dir_with_subset_dirs"],
        ids=IDS,
    )
    def test_can_detect(self, fxt_dataset_dir_with_subset_dirs: str):
        return super().test_can_detect(fxt_dataset_dir_with_subset_dirs)

    @pytest.mark.parametrize(
        [
            "fxt_dataset_dir_with_subset_dirs",
            "fxt_expected_dataset_with_subsets",
            "fxt_import_kwargs",
        ],
        [
            (DUMMY_DATASET_DIR, "fxt_dataset", {}),
            (
                DUMMY_NON_STANDARD_DATASET_DIR,
                "fxt_non_standard_dataset",
                {"image_prefix": "image_", "mask_prefix": "gt_"},
            ),
        ],
        indirect=["fxt_dataset_dir_with_subset_dirs", "fxt_expected_dataset_with_subsets"],
        ids=IDS,
    )
    def test_can_import(
        self,
        fxt_dataset_dir_with_subset_dirs: str,
        fxt_expected_dataset_with_subsets: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        return super().test_can_import(
            fxt_dataset_dir_with_subset_dirs,
            fxt_expected_dataset_with_subsets,
            fxt_import_kwargs,
            request,
        )
