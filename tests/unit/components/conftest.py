# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock

import pytest

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset_base import CategoriesInfo, DatasetInfo, DatasetItem, IDataset
from datumaro.components.media import MediaElement
from datumaro.util.definitions import DEFAULT_SUBSET_NAME


@pytest.fixture
def fxt_n_items() -> int:
    return 10


@pytest.fixture
def fxt_infos() -> DatasetInfo:
    return {"dummy": "info"}


@pytest.fixture
def fxt_categories() -> CategoriesInfo:
    return {AnnotationType.label: LabelCategories.from_iterable(["car", "cat", "dog"])}


@pytest.fixture
def fxt_stream_extractor(fxt_n_items: int, fxt_infos: DatasetInfo, fxt_categories: CategoriesInfo) -> MagicMock:
    stream_extractor = MagicMock(spec=IDataset)

    items = [
        DatasetItem(
            id=f"item_{i}",
            annotations=[Label(label=i % len(fxt_categories.get(AnnotationType.label)))],
        )
        for i in range(fxt_n_items)
    ]

    def _reset_iter():
        stream_extractor.__iter__.return_value = iter(items)

    stream_extractor.reset_iter = _reset_iter
    stream_extractor.reset_iter()

    stream_extractor.__len__.return_value = len(items)
    stream_extractor.infos.return_value = fxt_infos
    stream_extractor.categories.return_value = fxt_categories
    stream_extractor.media_type.return_value = MediaElement
    stream_extractor.ann_types.return_value = {AnnotationType.label}
    stream_extractor.is_stream = True

    stream_extractor.subsets.return_value = {DEFAULT_SUBSET_NAME: stream_extractor}

    return stream_extractor
