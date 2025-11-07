# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Set
from unittest.mock import MagicMock

from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset_base import CategoriesInfo, DatasetInfo
from datumaro.components.dataset_storage import StreamDatasetStorage
from datumaro.plugins.transforms import MapSubsets, ProjectInfos, RandomSplit, RemapLabels, Rename
from datumaro.util.definitions import DEFAULT_SUBSET_NAME


class StreamDatasetStorageTest:
    def _test_loop(
        self,
        fxt_stream_extractor: MagicMock,
        storage: StreamDatasetStorage,
        n_calls: int,
        id_pattern: str = "item_{idx}",
        subset: str = DEFAULT_SUBSET_NAME,
    ) -> None:
        for _ in range(n_calls):
            go_through_loop = False
            fxt_stream_extractor.reset_iter()
            for idx, item in enumerate(storage):
                assert item.id == id_pattern.format(idx=idx)
                assert item.subset == subset
                go_through_loop = True
            assert go_through_loop

    def test_iterator(self, fxt_stream_extractor: MagicMock, fxt_n_items: int):
        storage = StreamDatasetStorage(source=fxt_stream_extractor)
        assert len(storage) == fxt_n_items

        n_calls = 3
        self._test_loop(fxt_stream_extractor, storage, n_calls)

        # Iterator should be called 3 times (n_calls = 3)
        assert fxt_stream_extractor.__iter__.call_count == n_calls

    def test_subsets(self, fxt_stream_extractor: MagicMock):
        storage = StreamDatasetStorage(source=fxt_stream_extractor)

        self._test_subsets(fxt_stream_extractor, storage)

        n_calls = 3
        self._test_loop(fxt_stream_extractor, storage, n_calls)

        # Iterator should be called 3 times (n_calls = 3)
        assert fxt_stream_extractor.__iter__.call_count == n_calls

    def _test_subsets(self, fxt_stream_extractor, storage, expect: Set[str] = {DEFAULT_SUBSET_NAME}):
        fxt_stream_extractor.reset_iter()
        subsets = storage.subsets()
        assert set(subsets.keys()) == expect

    def test_info(self, fxt_stream_extractor: MagicMock, fxt_infos: DatasetInfo):
        storage = StreamDatasetStorage(source=fxt_stream_extractor)

        assert storage.infos() == fxt_infos
        fxt_stream_extractor.infos.assert_called_once()

    def test_categories(self, fxt_stream_extractor: MagicMock, fxt_categories: CategoriesInfo):
        storage = StreamDatasetStorage(source=fxt_stream_extractor)
        assert storage.categories() == fxt_categories
        fxt_stream_extractor.categories.assert_called_once()

    def test_item_transform(self, fxt_stream_extractor: MagicMock):
        storage = StreamDatasetStorage(source=fxt_stream_extractor)
        n_calls = 1

        self._test_loop(fxt_stream_extractor, storage, n_calls)
        assert fxt_stream_extractor.__iter__.call_count == 1

        # Stack transform 1 level
        storage.transform(Rename, regex="|item_|rename_|")
        self._test_loop(fxt_stream_extractor, storage, n_calls, id_pattern="rename_{idx}")
        assert fxt_stream_extractor.__iter__.call_count == 2

        # Stack transform 2 level
        storage.transform(Rename, regex="|rename_|renameagain_|")
        self._test_loop(fxt_stream_extractor, storage, n_calls, id_pattern="renameagain_{idx}")
        assert fxt_stream_extractor.__iter__.call_count == 3

    def test_subset_transform(self, fxt_stream_extractor: MagicMock):
        storage = StreamDatasetStorage(source=fxt_stream_extractor)

        # No need to iterate since the source already has subset info.
        self._test_subsets(fxt_stream_extractor, storage)
        assert fxt_stream_extractor.__iter__.call_count == 0

        # Stack transform 1 level, should run 1 iter to get the subset info after transform
        storage.transform(RandomSplit, splits=[("train", 0.5), ("val", 0.5)], seed=3003)
        self._test_subsets(fxt_stream_extractor, storage, expect={"train", "val"})
        assert fxt_stream_extractor.__iter__.call_count == 1

        # Stack transform 2 level, should run 1 iter more to get the subset info after transform
        storage.transform(MapSubsets, mapping={"train": DEFAULT_SUBSET_NAME, "val": DEFAULT_SUBSET_NAME})
        self._test_subsets(fxt_stream_extractor, storage)
        assert fxt_stream_extractor.__iter__.call_count == 2

    def test_info_transform(self, fxt_stream_extractor: MagicMock, fxt_infos: DatasetInfo):
        storage = StreamDatasetStorage(source=fxt_stream_extractor)

        assert storage.infos() == fxt_infos

        dst_infos = {"new": "info"}
        storage.transform(ProjectInfos, dst_infos=dst_infos)

        assert storage.infos().get("new") == "info"
        assert fxt_stream_extractor.__iter__.call_count == 0

    def test_categories_transform(self, fxt_stream_extractor: MagicMock, fxt_categories: CategoriesInfo):
        storage = StreamDatasetStorage(source=fxt_stream_extractor)

        assert storage.categories() == fxt_categories

        mapping = {"car": "apple", "cat": "banana", "dog": "cinnamon"}
        storage.transform(RemapLabels, mapping=mapping)

        actual = set(cat.name for cat in storage.categories()[AnnotationType.label])
        expect = set(mapping.values())
        assert actual == expect

        assert fxt_stream_extractor.__iter__.call_count == 0

    def test_mixed_transform(
        self,
        fxt_stream_extractor: MagicMock,
        fxt_infos: DatasetInfo,
        fxt_categories: CategoriesInfo,
    ):
        n_calls = 1
        storage = StreamDatasetStorage(source=fxt_stream_extractor)

        # Check extractor infos
        assert storage.infos() == fxt_infos

        dst_infos = {"new": "info"}
        storage.transform(ProjectInfos, dst_infos=dst_infos)
        assert fxt_stream_extractor.__iter__.call_count == 0

        # Check extractor categories
        assert storage.categories() == fxt_categories

        mapping = {"car": "apple", "cat": "banana", "dog": "cinnamon"}
        storage.transform(RemapLabels, mapping=mapping)
        assert fxt_stream_extractor.__iter__.call_count == 0

        # Stack Rename (ItemTransform) on the top
        storage.transform(Rename, regex="|item_|rename_|")
        assert fxt_stream_extractor.__iter__.call_count == 0

        # Check ProjectInfos
        assert storage.infos().get("new") == "info"

        # Check RemapLabels
        actual = set(cat.name for cat in storage.categories()[AnnotationType.label])
        expect = set(mapping.values())
        assert actual == expect

        # Check Rename
        self._test_loop(fxt_stream_extractor, storage, n_calls, id_pattern="rename_{idx}")
        assert fxt_stream_extractor.__iter__.call_count == n_calls
