# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
from typing import Sequence

from datumaro.components.contexts.importer import _ImportFail
from datumaro.components.dataset_base import SubsetBase
from datumaro.components.merge import ExactMerge
from datumaro.components.merge.extractor_merger import ExtractorMerger, check_identicalness
from datumaro.plugins.data_formats.coco.base import _CocoBase


class COCOTaskMergedBase(SubsetBase):
    __not_plugin__ = True

    def __init__(
        self,
        sources: Sequence[_CocoBase],
        subset: str,
    ):
        super().__init__(subset=subset, ctx=None)
        self._infos = check_identicalness([s.infos() for s in sources])
        self._media_type = check_identicalness([s.media_type() for s in sources])
        self._is_stream = check_identicalness([s.is_stream for s in sources])
        self._categories = ExactMerge.merge_categories([s.categories() for s in sources])

        self._sources = sources
        self._item_keys = None

    def __iter__(self):
        if len(self._sources) == 1:
            yield from self._sources[0]
        else:
            for item_key in self.item_keys:
                items = [item for s in self._sources if (item := s.get_dataset_item(item_key)) is not None]
                assert len(items) > 0

                item, remainders = items[0], items[1:]

                for remainder in remainders:
                    item = ExactMerge.merge_items(item, remainder)
                yield item

    def __len__(self):
        if len(self._sources) == 1:
            return len(self._sources[0])
        return len(self.item_keys)

    @property
    def item_keys(self):
        if self._item_keys is None:
            self._item_keys = set()
            for s in self._sources:
                self._item_keys.update(s.iter_item_ids())

        return self._item_keys

    @property
    def is_stream(self) -> bool:
        return self._is_stream


class COCOExtractorMerger(ExtractorMerger):
    __not_plugin__ = True

    def __init__(self, sources: Sequence[_CocoBase]):
        if len(sources) == 0:
            raise _ImportFail("It should not be empty.")

        self._infos = check_identicalness([s.infos() for s in sources])
        self._media_type = check_identicalness([s.media_type() for s in sources])
        self._is_stream = check_identicalness([s.is_stream for s in sources])
        self._categories = ExactMerge.merge_categories([s.categories() for s in sources])

        ann_types = set()
        for source in sources:
            ann_types.union(source.ann_types())
        self._ann_types = ann_types

        grouped_by_subset = defaultdict(list)

        for s in sources:
            grouped_by_subset[s.subset] += [s]

        self._subsets = {subset: [COCOTaskMergedBase(sources, subset)] for subset, sources in grouped_by_subset.items()}
