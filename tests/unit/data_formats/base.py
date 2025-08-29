# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
from typing import Any, Dict, Optional, Type

import pytest

from datumaro.components.dataset import Dataset, StreamDataset
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.exporter import Exporter
from datumaro.components.importer import Importer

from tests.utils.test_utils import check_is_stream, compare_datasets


class TestDataFormatBase:
    IMPORTER: Importer
    EXPORTER: Exporter

    def test_can_detect(self, fxt_dataset_dir: str, importer: Optional[Importer] = None):
        if importer is None:
            importer = getattr(self, "IMPORTER", None)

        if importer is None:
            pytest.skip(reason="importer is None.")

        detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(fxt_dataset_dir)
        assert [importer.NAME] == detected_formats, f"{[importer.NAME]} != {detected_formats}"

    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
        importer: Optional[Importer] = None,
        dataset_cls: Type[Dataset] = Dataset,
    ):
        if importer is None:
            importer = getattr(self, "IMPORTER", None)

        if importer is None:
            pytest.skip(reason="importer is None.")

        helper_tc = request.getfixturevalue("helper_tc")
        dataset = dataset_cls.import_from(fxt_dataset_dir, importer.NAME, **fxt_import_kwargs)
        stream = True if dataset_cls == StreamDataset else False
        check_is_stream(dataset, stream)

        compare_datasets(helper_tc, fxt_expected_dataset, dataset, require_media=True)

    def test_can_export_and_import(
        self,
        fxt_expected_dataset: Dataset,
        test_dir: str,
        fxt_import_kwargs: Dict[str, Any],
        fxt_export_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
        exporter: Optional[Exporter] = None,
        importer: Optional[Importer] = None,
        dataset_cls: Type[Dataset] = Dataset,
    ):
        if exporter is None:
            exporter = getattr(self, "EXPORTER", None)
        if importer is None:
            importer = getattr(self, "IMPORTER", None)

        if exporter is None or importer is None:
            pytest.skip(reason="exporter or importer is None.")

        helper_tc = request.getfixturevalue("helper_tc")

        stream = True if dataset_cls == StreamDataset else False
        exporter.convert(
            fxt_expected_dataset,
            save_dir=test_dir,
            save_media=True,
            stream=stream,
            **fxt_export_kwargs,
        )
        dataset = dataset_cls.import_from(test_dir, importer.NAME, **fxt_import_kwargs)
        check_is_stream(dataset, stream)

        compare_datasets(helper_tc, fxt_expected_dataset, dataset, require_media=True)
