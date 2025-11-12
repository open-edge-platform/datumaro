# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from copy import copy
from enum import Enum, auto
from typing import Any, Iterator, Optional, Set, Tuple, Type, Union

from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset_base import CategoriesInfo, DatasetInfo, DatasetItem, IDataset
from datumaro.components.media import MediaElement
from datumaro.util.definitions import DEFAULT_SUBSET_NAME

__all__ = ["DatasetItemStorage", "DatasetItemStorageDatasetView", "ItemStatus"]


class ItemStatus(Enum):
    added = auto()
    modified = auto()
    removed = auto()


class DatasetItemStorage:
    def __init__(self):
        self.data = {}  # { subset_name: { id: DatasetItem } }
        self._traversal_order = {}  # maintain the order of elements
        self._order = []  # allow indexing

    def __iter__(self) -> Iterator[DatasetItem]:
        for item in self._traversal_order.values():
            yield item

    def __len__(self) -> int:
        return len(self._traversal_order)

    def is_empty(self) -> bool:
        # Subsets might contain removed items, so this may differ from __len__
        return all(len(s) == 0 for s in self.data.values())

    def put(self, item: DatasetItem) -> bool:
        subset = self.data.setdefault(item.subset, {})
        is_new = subset.get(item.id) is None
        self._traversal_order[(item.id, item.subset)] = item
        if is_new:
            self._order.append((item.id, item.subset))
        subset[item.id] = item
        return is_new

    def get(
        self, id: Union[str, DatasetItem], subset: Optional[str] = None, dummy: Any = None
    ) -> Optional[DatasetItem]:
        if isinstance(id, DatasetItem):
            id, subset = id.id, id.subset
        else:
            id = str(id)
            subset = subset or DEFAULT_SUBSET_NAME

        return self.data.get(subset, {}).get(id, dummy)

    def remove(self, id: Union[str, DatasetItem], subset: Optional[str] = None) -> bool:
        if isinstance(id, DatasetItem):
            id, subset = id.id, id.subset
        else:
            id = str(id)
            subset = subset or DEFAULT_SUBSET_NAME

        subset_data = self.data.setdefault(subset, {})
        is_removed = subset_data.get(id) is not None
        subset_data[id] = None
        if is_removed:
            # TODO : investigate why "del subset_data[id]" cannot replace "subset_data[id] = None".
            self._traversal_order.pop((id, subset))
            self._order.remove((id, subset))
        return is_removed

    def __contains__(self, x: Union[DatasetItem, Tuple[str, str]]) -> bool:
        if not isinstance(x, tuple):
            x = [x]
        dummy = 0
        return self.get(*x, dummy=dummy) is not dummy

    def get_subset(self, name):
        return self.data.get(name, {})

    def subsets(self):
        return self.data

    def get_annotated_items(self):
        return sum(bool(s.annotations) for s in self._traversal_order.values())

    def get_datasetitem_by_path(self, path):
        for s in self._traversal_order.values():
            if getattr(s.media, "path", None) == path:
                return s

    def get_annotations(self):
        annotations_by_type = {t.name: {"count": 0} for t in AnnotationType}
        for item in self._traversal_order.values():
            for ann in item.annotations:
                annotations_by_type[ann.type.name]["count"] += 1
        return sum(t["count"] for t in annotations_by_type.values())

    def __copy__(self):
        copied = DatasetItemStorage()
        copied._traversal_order = copy(self._traversal_order)
        copied._order = copy(self._order)
        copied.data = copy(self.data)
        return copied

    def __getitem__(self, idx: int) -> DatasetItem:
        _id, subset = self._order[idx]
        return self.data[subset][_id]


class DatasetItemStorageDatasetView(IDataset):
    class Subset(IDataset):
        def __init__(self, parent: "DatasetItemStorageDatasetView", name: str):
            super().__init__()
            self.parent = parent
            self.name = name
            self._length = None

        @property
        def _data(self):
            return self.parent._get_subset_data(self.name)

        def __iter__(self):
            for item in self._data.values():
                if item:
                    yield item

        def __len__(self):
            if self._length is not None:
                return self._length

            self._length = 0
            for item in self._data.values():
                if item is not None:
                    self._length += 1
            return self._length

        def put(self, item):
            self._length = None
            return self._data.put(item)

        def get(self, id, subset=None):
            assert (subset or DEFAULT_SUBSET_NAME) == (self.name or DEFAULT_SUBSET_NAME)
            return self._data.get(id, subset)

        def remove(self, id, subset=None):
            assert (subset or DEFAULT_SUBSET_NAME) == (self.name or DEFAULT_SUBSET_NAME)
            self._length = None
            return self._data.remove(id, subset)

        def get_subset(self, name):
            assert (name or DEFAULT_SUBSET_NAME) == (self.name or DEFAULT_SUBSET_NAME)
            return self

        def subsets(self):
            return {self.name or DEFAULT_SUBSET_NAME: self}

        def infos(self):
            return self.parent.infos()

        def categories(self):
            return self.parent.categories()

        def media_type(self):
            return self.parent.media_type()

        def ann_types(self):
            return self.parent.ann_types()

    def __init__(
        self,
        parent: DatasetItemStorage,
        infos: DatasetInfo,
        categories: CategoriesInfo,
        media_type: Optional[Type[MediaElement]],
        ann_types: Optional[Set[AnnotationType]],
    ):
        self._parent = parent
        self._infos = infos
        self._categories = categories
        self._media_type = media_type
        self._ann_types = ann_types

    def __iter__(self):
        yield from self._parent

    def __len__(self):
        return len(self._parent)

    def infos(self):
        return self._infos

    def categories(self):
        return self._categories

    def get_subset(self, name):
        return self.Subset(self, name)

    def _get_subset_data(self, name):
        return self._parent.get_subset(name)

    def subsets(self):
        return {k: self.get_subset(k) for k in self._parent.subsets()}

    def get(self, id, subset=None):
        return self._parent.get(id, subset=subset)

    def media_type(self):
        return self._media_type

    def ann_types(self):
        return self._ann_types
