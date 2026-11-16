# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset_base import DatasetItem, IDataset
from datumaro.components.dataset_item_storage import DatasetItemStorage
from datumaro.components.merge import Merger

__all__ = ["UnionMerge"]


class UnionMerge(Merger):
    """
    Merge several datasets with "union" policy:

    - Label categories are merged according to the union of their label names.
    For example, if Dataset-A has {"car", "cat", "dog"} and Dataset-B has
    {"car", "bus", "truck"} labels, the merged dataset will have
    {"bust", "car", "cat", "dog", "truck"} labels.

    - If there are two or more dataset items whose (id, subset) pairs match each other,
    both are included in the merged dataset. At this time, since the same (id, subset)
    pair cannot be duplicated in the dataset, we add a suffix to the id of each source item.
    For example, if Dataset-A has DatasetItem(id="magic", subset="train") and Dataset-B has
    also DatasetItem(id="magic", subset="train"), the merged dataset will have
    DatasetItem(id="magic-0", subset="train") and DatasetItem(id="magic-1", subset="train").
    """

    def __init__(self, **options):
        super().__init__(**options)
        self._matching_table = {}

    def merge(self, sources: Sequence[IDataset]) -> DatasetItemStorage:
        dict_items: Dict[Tuple[str, str], List[DatasetItem]] = defaultdict(list)

        for source_idx, source in enumerate(sources):
            matching_table = self._matching_table.get(source_idx, None)
            for item in source:
                if matching_table:
                    # Remap the labels on copies, so the source dataset keeps its own label ids
                    item = item.wrap(
                        annotations=[ann.wrap(label=matching_table[ann.label]) for ann in item.annotations]
                    )
                dict_items[item.id, item.subset].append(item)

        item_storage = DatasetItemStorage()

        for items in dict_items.values():
            if len(items) == 1:
                item_storage.put(items[0])
            else:
                for idx, item in enumerate(items):
                    # Add prefix
                    item_storage.put(item.wrap(id=f"{item.id}-{idx}"))

        return item_storage

    def merge_categories(self, sources: Sequence[IDataset]) -> Dict:
        dst_categories = {}

        label_cat = self._merge_label_categories(sources)
        if label_cat is None:
            label_cat = LabelCategories()
        dst_categories[AnnotationType.label] = label_cat

        return dst_categories

    def _merge_label_categories(self, sources: Sequence[IDataset]) -> LabelCategories:
        dst_cat = LabelCategories()
        for src_id, src_categories in enumerate(sources):
            src_cat = src_categories.get(AnnotationType.label)
            if src_cat is None:
                continue

            for src_label in src_cat.items:
                src_idx = src_cat.find(src_label.name)[0]
                dst_idx = dst_cat.find(src_label.name)[0]
                if dst_idx is None:
                    dst_cat.add(src_label.name, src_label.parent, src_label.attributes)
                    dst_idx = dst_cat.find(src_label.name)[0]

                if self._matching_table.get(src_id, None):
                    self._matching_table[src_id].update({src_idx: dst_idx})
                else:
                    self._matching_table[src_id] = {src_idx: dst_idx}

        return dst_cat
