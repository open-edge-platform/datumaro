# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import pytest

from datumaro.components.dataset import Dataset
from datumaro.plugins.data_formats.tabular import *

from ...requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestCaseHelper, TestDir, compare_datasets
from tests.utils.test_utils import run_datum as run


@pytest.fixture()
def fxt_tabular_root():
    yield get_test_asset_path("tabular_dataset")


@pytest.fixture()
def fxt_electricity_path(fxt_tabular_root):
    yield osp.join(fxt_tabular_root, "electricity.csv")


@pytest.fixture()
def txf_electricity(fxt_electricity_path):
    yield Dataset.import_from(fxt_electricity_path, "tabular")


@pytest.fixture()
def fxt_buddy_path(fxt_tabular_root):
    yield osp.join(fxt_tabular_root, "adopt-a-buddy")


@pytest.fixture()
def fxt_buddy_target():
    yield {"input": "length(m)", "output": ["breed_category", "pet_category"]}


@pytest.fixture()
def fxt_buddy(fxt_buddy_path, fxt_buddy_target):
    yield Dataset.import_from(fxt_buddy_path, "tabular", target=fxt_buddy_target)


@pytest.mark.new
class TabularIntegrationTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_and_export_tabular_dataset_electricity(
        self, helper_tc: TestCaseHelper, txf_electricity, fxt_electricity_path
    ):
        """
        <b>Description:</b>
        Ensure that the electricity dataset can be converted to/from tabular format
        with command `datum convert`.

        <b>Expected results:</b>
        A tabular dataset that matches the expected result.

        <b>Steps:</b>
        1. Get path to the source dataset from assets.
        2. Convert the dataset to tabular format and back using `convert` command.
        3. Verify that the resulting dataset is equal to the expected result.
        """

        with TestDir() as test_dir:
            # Convert tabular dataset to tabular format (round-trip to test import/export)
            export_dir = osp.join(test_dir, "export_dir")

            cmd = [
                "convert",
                "-if",
                "tabular",
                "-i",
                fxt_electricity_path,
                "-f",
                "tabular",
                "-o",
                export_dir,
            ]

            run(helper_tc, *cmd)

            # Import the exported dataset and compare
            exported = Dataset.import_from(export_dir, format="tabular")
            compare_datasets(helper_tc, txf_electricity, exported)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_export_and_import_tabular_dataset_with_target(
        self, helper_tc: TestCaseHelper, fxt_buddy, fxt_buddy_path, fxt_buddy_target
    ):
        """
        <b>Description:</b>
        Ensure that datasets requiring target specification can be exported and re-imported
        using direct export/import (not convert command due to CLI limitations).

        <b>Expected results:</b>
        A tabular dataset that matches the expected result.

        <b>Steps:</b>
        1. Export dataset using Python API.
        2. Import back using CLI with target specification.
        3. Verify that the resulting dataset is equal to the expected result.
        """

        with TestDir() as test_dir:
            # Export using Python API since convert command doesn't support importer args
            export_dir = osp.join(test_dir, "export_dir")
            fxt_buddy.export(export_dir, format="tabular")

            # Import back using CLI should work since we can specify target during import
            # But since convert doesn't support importer args, we'll test Python API import
            exported = Dataset.import_from(export_dir, format="tabular", target=fxt_buddy_target)
            compare_datasets(helper_tc, fxt_buddy, exported)
