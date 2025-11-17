# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import shutil
from copy import deepcopy

import pytest

from datumaro import Dataset
from tests.utils.test_utils import TestDir


@pytest.fixture
def fxt_dummy_dataset():
    yield Dataset.from_iterable([])


@pytest.fixture
def fxt_expected_dataset(request: pytest.FixtureRequest):
    fxt_name = getattr(request, "param", "fxt_dummy_dataset")
    yield request.getfixturevalue(fxt_name)


@pytest.fixture
def fxt_import_kwargs():
    yield {}


@pytest.fixture
def fxt_export_kwargs():
    yield {}


@pytest.fixture
def fxt_dataset_dir_with_subset_dirs(test_dir: str, request: pytest.FixtureRequest):
    fxt_dataset_dir = request.param
    with TestDir(f"{test_dir}_with_subsets") as new_test_dir:
        for subset in ["train", "val", "test"]:
            dst = os.path.join(new_test_dir, subset)
            shutil.copytree(fxt_dataset_dir, dst)

        yield new_test_dir


@pytest.fixture
def fxt_expected_dataset_with_subsets(request: pytest.FixtureRequest):
    fxt_name = getattr(request, "param", "fxt_dummy_dataset")

    dataset: Dataset = request.getfixturevalue(fxt_name)

    # TODO: replace this logic with a high-level logic
    # (I don't know it's existence, if not exist, we should implement it)
    # which merges subsets into a dataset.
    items = []
    subsets = []
    for subset_name in ["train", "val", "test"]:
        subset = deepcopy(dataset).transform("map_subsets", mapping={"default": subset_name})
        subsets += [subset]
        for item in subset:
            items += [item]

    yield Dataset.from_extractors(*subsets)
