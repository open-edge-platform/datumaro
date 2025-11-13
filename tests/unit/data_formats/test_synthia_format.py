# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=signature-differs

from typing import Any, Dict

import numpy as np
import pytest

from datumaro.components.annotation import AnnotationType, LabelCategories, Mask, MaskCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.data_formats.synthia.base import make_categories
from datumaro.plugins.data_formats.synthia.format import SynthiaAlLabelMap, SynthiaRandLabelMap, SynthiaSfLabelMap
from datumaro.plugins.data_formats.synthia.importer import SynthiaAlImporter, SynthiaRandImporter, SynthiaSfImporter
from tests.utils.assets import get_test_asset_path

from .base import TestDataFormatBase

DUMMY_RAND_DATASET_DIR = get_test_asset_path("synthia_dataset", "rand")
DUMMY_SF_DATASET_DIR = get_test_asset_path("synthia_dataset", "sf")
DUMMY_AL_DATASET_DIR = get_test_asset_path("synthia_dataset", "al")
DUMMY_DATASET_DIR_CUSTOM_LABELMAP = get_test_asset_path("synthia_dataset", "rand_custom_labelmap")
DUMMY_DATASET_DIR_META_FILE = get_test_asset_path("synthia_dataset", "rand_meta_file")


@pytest.fixture
def fxt_synthia_rand_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="000000",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(
                        np.array([1, 1, 0, 0, 0]),
                        label=1,
                    ),
                    Mask(
                        np.array([0, 0, 1, 1, 1]),
                        label=10,
                    ),
                ],
            ),
            DatasetItem(
                id="000001",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(
                        np.array([1, 0, 0, 0, 0]),
                        label=8,
                    ),
                    Mask(
                        np.array([0, 1, 1, 0, 0]),
                        label=11,
                    ),
                    Mask(
                        np.array([0, 0, 0, 1, 1]),
                        label=3,
                    ),
                ],
            ),
        ],
        categories=make_categories(label_map=SynthiaRandLabelMap),
    )


@pytest.fixture
def fxt_synthia_rand_custom_label_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="000000",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(np.array([[1, 1, 1, 0, 0]]), label=1),
                    Mask(np.array([[0, 0, 0, 1, 1]]), label=4),
                ],
            ),
            DatasetItem(
                id="000001",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(np.array([[1, 1, 0, 0, 0]]), label=2),
                    Mask(np.array([[0, 0, 1, 1, 0]]), label=3),
                    Mask(np.array([[0, 0, 0, 0, 1]]), label=4),
                ],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(["background", "sky", "building", "person", "road"]),
            AnnotationType.mask: MaskCategories(
                {
                    0: (0, 0, 0),
                    1: (0, 0, 64),
                    2: (0, 128, 128),
                    3: (128, 0, 64),
                    4: (0, 192, 128),
                }
            ),
        },
    )


@pytest.fixture
def fxt_synthia_rand_meta_file_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="000000",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(np.array([[1, 1, 1, 0, 0]]), label=1),
                    Mask(np.array([[0, 0, 0, 1, 1]]), label=4),
                ],
            ),
            DatasetItem(
                id="000001",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(np.array([[1, 1, 0, 0, 0]]), label=2),
                    Mask(np.array([[0, 0, 1, 1, 0]]), label=3),
                    Mask(np.array([[0, 0, 0, 0, 1]]), label=4),
                ],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(["background", "sky", "building", "person", "road"]),
            AnnotationType.mask: MaskCategories(
                {
                    0: (0, 0, 0),
                    1: (0, 0, 64),
                    2: (0, 128, 128),
                    3: (128, 0, 64),
                    4: (0, 192, 128),
                }
            ),
        },
    )


@pytest.fixture
def fxt_synthia_sf_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="000000",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(
                        np.array([[1, 1, 0, 0, 0]]),
                        label=1,
                    ),
                    Mask(
                        np.array([[0, 0, 1, 1, 1]]),
                        label=10,
                    ),
                ],
            ),
            DatasetItem(
                id="000001",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(
                        np.array([[1, 0, 0, 0, 0]]),
                        label=8,
                    ),
                    Mask(
                        np.array([[0, 1, 1, 0, 0]]),
                        label=11,
                    ),
                    Mask(
                        np.array([[0, 0, 0, 1, 1]]),
                        label=3,
                    ),
                ],
            ),
        ],
        categories=make_categories(label_map=SynthiaSfLabelMap),
    )


@pytest.fixture
def fxt_synthia_al_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="000000",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(
                        np.array([[1, 1, 0, 0, 0]]),
                        label=1,
                    ),
                    Mask(
                        np.array([[0, 0, 1, 1, 1]]),
                        label=10,
                    ),
                ],
            ),
            DatasetItem(
                id="000001",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(
                        np.array([[1, 0, 0, 0, 0]]),
                        label=8,
                    ),
                    Mask(
                        np.array([[0, 1, 1, 0, 0]]),
                        label=11,
                    ),
                    Mask(
                        np.array([[0, 0, 0, 1, 1]]),
                        label=3,
                    ),
                ],
            ),
        ],
        categories=make_categories(label_map=SynthiaAlLabelMap),
    )


class SynthiaRandFormatTest(TestDataFormatBase):
    IMPORTER = SynthiaRandImporter
    USE_TEST_CAN_EXPORT_AND_IMPORT = False

    @pytest.mark.parametrize(
        "fxt_dataset_dir",
        [DUMMY_RAND_DATASET_DIR],
        ids=["rand"],
    )
    def test_can_detect(self, fxt_dataset_dir: str):
        return super().test_can_detect(fxt_dataset_dir)

    @pytest.mark.parametrize(
        [
            "fxt_dataset_dir",
            "fxt_expected_dataset",
            "fxt_import_kwargs",
        ],
        [
            (DUMMY_RAND_DATASET_DIR, "fxt_synthia_rand_dataset", {}),
            (DUMMY_DATASET_DIR_CUSTOM_LABELMAP, "fxt_synthia_rand_custom_label_dataset", {}),
            (DUMMY_DATASET_DIR_META_FILE, "fxt_synthia_rand_meta_file_dataset", {}),
        ],
        indirect=["fxt_expected_dataset"],
        ids=["rand", "custom_label", "meta_file"],
    )
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        return super().test_can_import(
            fxt_dataset_dir,
            fxt_expected_dataset,
            fxt_import_kwargs,
            request,
        )


class SynthiaSfFormatTest(TestDataFormatBase):
    IMPORTER = SynthiaSfImporter
    USE_TEST_CAN_EXPORT_AND_IMPORT = False

    @pytest.mark.parametrize(
        "fxt_dataset_dir",
        [DUMMY_SF_DATASET_DIR],
        ids=["sf"],
    )
    def test_can_detect(self, fxt_dataset_dir: str):
        return super().test_can_detect(fxt_dataset_dir)

    @pytest.mark.parametrize(
        [
            "fxt_dataset_dir",
            "fxt_expected_dataset",
            "fxt_import_kwargs",
        ],
        [
            (DUMMY_SF_DATASET_DIR, "fxt_synthia_sf_dataset", {}),
        ],
        indirect=["fxt_expected_dataset"],
        ids=["sf"],
    )
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        return super().test_can_import(
            fxt_dataset_dir,
            fxt_expected_dataset,
            fxt_import_kwargs,
            request,
        )


class SynthiaAlFormatTest(TestDataFormatBase):
    IMPORTER = SynthiaAlImporter
    USE_TEST_CAN_EXPORT_AND_IMPORT = False

    @pytest.mark.parametrize(
        "fxt_dataset_dir",
        [DUMMY_AL_DATASET_DIR],
        ids=["al"],
    )
    def test_can_detect(self, fxt_dataset_dir: str):
        return super().test_can_detect(fxt_dataset_dir)

    @pytest.mark.parametrize(
        [
            "fxt_dataset_dir",
            "fxt_expected_dataset",
            "fxt_import_kwargs",
        ],
        [
            (DUMMY_AL_DATASET_DIR, "fxt_synthia_al_dataset", {}),
        ],
        indirect=["fxt_expected_dataset"],
        ids=["al"],
    )
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        return super().test_can_import(
            fxt_dataset_dir,
            fxt_expected_dataset,
            fxt_import_kwargs,
            request,
        )
