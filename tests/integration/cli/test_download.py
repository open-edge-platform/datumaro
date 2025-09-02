# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
import contextlib
import io
import os.path as osp
from unittest import TestCase
from unittest.case import skipIf

import pytest

from datumaro.cli.commands.downloaders.kaggle import KAGGLE_API_KEY_EXISTS
from datumaro.components.dataset import Dataset
from datumaro.components.extractor_tfds import AVAILABLE_TFDS_DATASETS, TFDS_EXTRACTOR_AVAILABLE
from datumaro.util import parse_json

from tests.utils.test_utils import TestCaseHelper, TestDir, compare_datasets, mock_tfds_data
from tests.utils.test_utils import run_datum as run


@pytest.mark.skipif(not TFDS_EXTRACTOR_AVAILABLE, reason="TFDS is not installed")
class DownloadGetTest:
    _helper_tc = TestCaseHelper()

    @mock_tfds_data(subsets=("train", "val"))
    def test_download(self, test_dir: str):
        expected_dataset = Dataset(AVAILABLE_TFDS_DATASETS["mnist"].make_extractor())

        run(
            self._helper_tc,
            "download",
            "tfds",
            "get",
            "--dataset-id=tfds:mnist",
            f"--output-dir={test_dir}",
            "--",
            "--save-media",
        )

        actual_dataset = Dataset.import_from(test_dir, "mnist")
        compare_datasets(self._helper_tc, expected_dataset, actual_dataset, require_media=True)

    @mock_tfds_data(subsets=("train", "val"))
    def test_download_custom_format(self, test_dir: str, helper_tc):
        expected_dataset = Dataset(AVAILABLE_TFDS_DATASETS["mnist"].make_extractor())

        run(
            self._helper_tc,
            "download",
            "tfds",
            "get",
            "--dataset-id=tfds:mnist",
            "--output-format=datumaro",
            f"--output-dir={test_dir}",
            "--",
            "--save-media",
        )

        actual_dataset = Dataset.load(test_dir)
        compare_datasets(self._helper_tc, expected_dataset, actual_dataset, require_media=True)

    def test_download_fails_on_existing_dir_without_overwrite(self, test_dir: str):
        with open(osp.join(test_dir, "text.txt"), "w"):
            pass

        run(
            self._helper_tc,
            "download",
            "tfds",
            "get",
            "--dataset-id=tfds:mnist",
            "--output-format=datumaro",
            f"--output-dir={test_dir}",
            expected_code=1,
        )

    @mock_tfds_data()
    def test_download_works_on_existing_dir_without_overwrite(self, test_dir: str):
        with open(osp.join(test_dir, "text.txt"), "w"):
            pass

        run(
            self._helper_tc,
            "download",
            "tfds",
            "get",
            "--dataset-id=tfds:mnist",
            "--output-format=datumaro",
            f"--output-dir={test_dir}",
            "--overwrite",
            "--",
            "--save-media",
        )

    @mock_tfds_data(subsets=("train", "val"))
    def test_download_subset(self, test_dir: str):
        expected_dataset = Dataset(
            AVAILABLE_TFDS_DATASETS["mnist"].make_extractor().get_subset("train")
        )

        run(
            self._helper_tc,
            "download",
            "tfds",
            "get",
            "--dataset-id=tfds:mnist",
            "--output-format=datumaro",
            f"--output-dir={test_dir}",
            "--subset=train",
            "--",
            "--save-media",
        )

        actual_dataset = Dataset.load(test_dir)
        compare_datasets(self._helper_tc, expected_dataset, actual_dataset, require_media=True)

    @mock_tfds_data(subsets=("train", "val"))
    def test_download_invalid_subset(self, test_dir: str):
        run(
            self._helper_tc,
            "download",
            "tfds",
            "get",
            "--dataset-id=tfds:mnist",
            "--output-format=datumaro",
            f"--output-dir={test_dir}",
            "--subset=test",
            expected_code=1,
        )

    @pytest.mark.skipif(not KAGGLE_API_KEY_EXISTS, reason="Kaggle API key missing")
    @pytest.mark.parametrize(
        ("dataset_id", "extra_args"),
        (
            ("andrewmvd/face-mask-detection", {}),  # kaggle_voc
            ("kmader/skin-cancer-mnist-ham10000", {}),  # kaggle_image_csv
            ("alexattia/the-simpsons-characters-dataset", {}),  # kaggle_image_txt
            ("dasmehdixtr/drone-dataset-uav", {"subset": "drone_dataset_yolo"}),  # kaggle_yolo,
            ("aalborguniversity/aau-rainsnow", {}),  # kaggle_coco,
            ("kmader/finding-lungs-in-ct-data", {}),  # kaggle_image_mask
            ("moltean/fruits", {}),  # imagenet
        ),
    )
    def test_kaggle(self, dataset_id, extra_args, test_dir: str):
        args = [
            "download",
            "kaggle",
            "get",
            f"--dataset-id={dataset_id}",
            f"--output-dir={test_dir}",
        ]
        if "subset" in extra_args:
            args += ["--subset", extra_args.pop("subset")]
        if extra_args:
            args += ["--", *[f"--{key}={value}" for key, value in extra_args.items()]]
        run(
            self._helper_tc,
            *args,
        )


@pytest.mark.skipif(not TFDS_EXTRACTOR_AVAILABLE, reason="TFDS is not installed")
class DownloadDescribeTest:
    _helper_tc = TestCaseHelper()

    @mock_tfds_data()
    def test_text(self):
        output_file = io.StringIO()

        with contextlib.redirect_stdout(output_file):
            run(self._helper_tc, "download", "tfds", "describe")

        output = output_file.getvalue()

        # Since the output is not in a structured format, it's difficult to test
        # that it looks exactly as we want it to. As a simplification, we'll
        # just check that it contains all the data that we expect.

        for name, dataset in AVAILABLE_TFDS_DATASETS.items():
            assert f"tfds:{name}" in output

            dataset_metadata = dataset.query_remote_metadata()

            for attribute in (
                "default_output_format",
                "download_size",
                "home_url",
                "human_name",
                "num_classes",
                "version",
            ):
                assert str(getattr(dataset_metadata, attribute)) in output

            expected_description = dataset_metadata.description
            # We indent the description, so it's not going to occur in the output stream
            # verbatim. Just make sure the first line is there instead.
            expected_description = expected_description.split("\n", maxsplit=1)[0]

            assert expected_description in output

            for subset_name, subset_metadata in dataset_metadata.subsets.items():
                assert subset_name in output
                assert str(subset_metadata.num_items) in output

    @mock_tfds_data()
    def test_json(self):
        output_file = io.StringIO()

        with contextlib.redirect_stdout(output_file):
            run(self._helper_tc, "download", "tfds", "describe", "--report-format=json")

        output = parse_json(output_file.getvalue())

        assert output.keys() == {f"tfds:{name}" for name in AVAILABLE_TFDS_DATASETS}

        for name, dataset in AVAILABLE_TFDS_DATASETS.items():
            dataset_metadata = dataset.query_remote_metadata()
            dataset_description = output[f"tfds:{name}"]
            for attribute in (
                "default_output_format",
                "description",
                "download_size",
                "home_url",
                "human_name",
                "num_classes",
                "version",
            ):
                assert dataset_description.pop(attribute) == getattr(dataset_metadata, attribute)

            subset_descriptions = dataset_description.pop("subsets")
            assert subset_descriptions.keys() == dataset_metadata.subsets.keys()

            for subset_name, subset_metadata in dataset_metadata.subsets.items():
                assert subset_descriptions[subset_name] == {"num_items": subset_metadata.num_items}

            # Make sure we checked all attributes
            assert not dataset_description

    @pytest.mark.parametrize("format", ["text", "json"])
    @mock_tfds_data()
    def test_report_file(self, format: str, test_dir: str):
        stdout_file = io.StringIO()

        with contextlib.redirect_stdout(stdout_file):
            run(self._helper_tc, "download", "tfds", "describe", f"--report-format={format}")

        stdout_output = stdout_file.getvalue()

        redirect_path = osp.join(test_dir, "report.txt")
        run(
            self._helper_tc,
            "download",
            "tfds",
            "describe",
            f"--report-format={format}",
            f"--report-file={redirect_path}",
        )

        with open(redirect_path, "r") as redirect_file:
            redirected_output = redirect_file.read()

        assert redirected_output == stdout_output

    @pytest.mark.skipif(not KAGGLE_API_KEY_EXISTS, reason="Kaggle API key missing")
    def test_kaggle(self):
        run(
            self._helper_tc,
            "download",
            "kaggle",
            "describe",
        )
