# pylint: disable=arguments-differ
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT


import sys
from typing import Any

import numpy as np
import pytest

from datumaro.components.annotation import Annotation, Bbox
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.data_formats.datumaro_binary import DatumaroBinaryExporter, DatumaroBinaryImporter
from datumaro.plugins.data_formats.datumaro_binary.format import DatumaroBinaryPath
from datumaro.plugins.data_formats.datumaro_binary.mapper import *
from datumaro.plugins.data_formats.datumaro_binary.mapper.annotation import AnnotationMapper
from tests.utils.test_utils import compare_datasets_strict

from .test_datumaro_format import DatumaroFormatTest as TestBase


class DatumaroBinaryFormatTest(TestBase):
    exporter = DatumaroBinaryExporter
    importer = DatumaroBinaryImporter
    format = DatumaroBinaryImporter.NAME
    ann_ext = DatumaroBinaryPath.ANNOTATION_EXT

    @pytest.mark.parametrize(
        "fxt_dataset",
        ("fxt_test_datumaro_format_dataset", "fxt_test_datumaro_format_video_dataset"),
    )
    @pytest.mark.parametrize(
        ["compare", "require_media", "fxt_import_kwargs", "fxt_export_kwargs"],
        [
            pytest.param(
                compare_datasets_strict,
                True,
                {},
                {},
                id="test_default",
            ),
            pytest.param(
                compare_datasets_strict,
                True,
                {},
                {"max_blob_size": 1},  # 1 byte
                id="test_multi_blobs",
            ),
            pytest.param(
                compare_datasets_strict,
                True,
                {"num_workers": 2},
                {"num_workers": 2},
                id="test_multi_processing",
            ),
        ],
    )
    def test_dm_binary_own_features(
        self,
        fxt_dataset,
        compare,
        require_media,
        test_dir,
        fxt_import_kwargs,
        fxt_export_kwargs,
        helper_tc,
        request,
    ):
        return super().test_can_save_and_load(
            fxt_dataset,
            compare,
            require_media,
            test_dir,
            fxt_import_kwargs,
            fxt_export_kwargs,
            False,  # Disable stream
            helper_tc,
            request,
        )

    def test_can_import_images_exported_from_memory_without_media(self, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[Bbox(1, 2, 3, 4, label=0)],
                )
            ],
            categories=["label_0"],
        )
        self.exporter.convert(source_dataset, save_dir=test_dir, save_media=False)

        imported = Dataset.import_from(test_dir, self.format)

        item = imported.get("a", "train")
        assert item.annotations == source_dataset.get("a", "train").annotations
        assert item.media.size == (10, 6)


class MapperTest:
    @staticmethod
    def _test(mapper: Mapper, expected: Any):
        _bytes = mapper.forward(expected)
        actual, _ = mapper.backward(_bytes)
        assert expected == actual

        prefix = bytes("asdf4312", "utf-8")
        suffix = bytes("qwer5332", "utf-8")
        offset = len(prefix)
        _bytes = prefix + _bytes + suffix
        actual, offset = mapper.backward(_bytes, offset=offset)

        assert expected == actual
        assert offset == len(_bytes) - len(suffix)

    @staticmethod
    def _get_ann_mapper(ann: Annotation) -> AnnotationMapper:
        name = ann.__class__.__name__
        return getattr(sys.modules[__name__], name + "Mapper")

    @pytest.mark.parametrize(
        "mapper,expected",
        [
            (
                StringMapper,
                "9sd#&(d!d.x]+=",
            ),
            (
                IntListMapper,
                (
                    0,
                    1,
                    2,
                ),
            ),
            (
                FloatListMapper,
                (
                    0.0,
                    1.0,
                    2.0,
                ),
            ),
            (
                DictMapper,
                {
                    "string": "test",
                    "int": 0,
                    "float": 0.0,
                    "string_list": ["test0", "test1", "test2"],
                    "int_list": [0, 1, 2],
                    "float_list": [0.0, 0.1, 0.2],
                },
            ),
        ],
    )
    def test_common_mapper(self, mapper: Mapper, expected: Any):
        self._test(mapper, expected)

    @pytest.mark.parametrize(
        "fxt_dataset",
        ("fxt_test_datumaro_format_dataset", "fxt_test_datumaro_format_video_dataset"),
    )
    def test_annotations_mapper(self, fxt_dataset, request):
        """Test all annotations in fxt_dataset"""
        mapper = DatasetItemMapper
        fxt_dataset = request.getfixturevalue(fxt_dataset)
        for item in fxt_dataset:
            for ann in item.annotations:
                mapper = self._get_ann_mapper(ann)
                self._test(mapper, ann)
