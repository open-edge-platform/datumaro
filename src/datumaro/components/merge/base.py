# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
import logging as log
import os
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Set, Type

from datumaro.components.abstracts.merger import IMergerContext
from datumaro.components.annotation import AnnotationType
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import CategoriesInfo, DatasetInfo, IDataset
from datumaro.components.dataset_item_storage import DatasetItemStorageDatasetView
from datumaro.components.errors import (
    ConflictingCategoriesError,
    DatasetMergeError,
    DatasetQualityError,
    MediaTypeError,
)
from datumaro.components.media import MediaElement
from datumaro.util import dump_json_file


class Merger(IMergerContext, CliPlugin):
    """Merge multiple datasets into one dataset"""

    def __init__(self, **options):
        super().__init__(**options)
        self.__dict__["_sources"] = None
        self.errors = []

    @staticmethod
    def merge_infos(sources: Sequence[DatasetInfo]) -> Dict:
        """Merge several :class:`IDataset` into one :class:`IDataset`"""
        infos = {}
        for source in sources:
            for k, v in source.items():
                if k in infos:
                    log.warning(
                        "Duplicated infos field %s: overwrite from %s to %s", k, infos[k], v
                    )
                infos[k] = v
        return infos

    @staticmethod
    def merge_categories(sources: Sequence[CategoriesInfo]) -> Dict:
        categories = {}
        for source_idx, source in enumerate(sources):
            for cat_type, source_cat in source.items():
                existing_cat = categories.setdefault(cat_type, source_cat)
                if existing_cat != source_cat and len(source_cat) != 0:
                    if len(existing_cat) == 0:
                        categories[cat_type] = source_cat
                    else:
                        raise ConflictingCategoriesError(
                            "Merging of datasets with different categories is "
                            "only allowed in 'merge' command.",
                            sources=list(range(source_idx)),
                        )
        return categories

    @staticmethod
    def merge_media_types(sources: Sequence[IDataset]) -> Optional[Type[MediaElement]]:
        if sources:
            media_type = sources[0].media_type()
            for s in sources:
                if not issubclass(s.media_type(), media_type) or not issubclass(
                    media_type, s.media_type()
                ):
                    # Symmetric comparison is needed in the case of subclasses:
                    # eg. Image and RoIImage
                    raise MediaTypeError("Datasets have different media types")
            return media_type

        return None

    @staticmethod
    def merge_ann_types(sources: Sequence[IDataset]) -> Optional[Set[AnnotationType]]:
        ann_types = set()
        for source in sources:
            ann_types.union(source.ann_types())
        return ann_types

    def __call__(self, *datasets: IDataset) -> DatasetItemStorageDatasetView:
        infos = self.merge_infos(d.infos() for d in datasets)
        categories = self.merge_categories(d.categories() for d in datasets)
        media_type = self.merge_media_types(datasets)
        ann_types = self.merge_ann_types(datasets)
        return DatasetItemStorageDatasetView(
            parent=self.merge(datasets),
            infos=infos,
            categories=categories,
            media_type=media_type,
            ann_types=ann_types,
        )

    def save_merge_report(self, path: str) -> None:
        item_errors = OrderedDict()
        source_errors = OrderedDict()
        all_errors = []

        for e in self.errors:
            if isinstance(e, DatasetQualityError):
                item_errors[str(e.item_id)] = item_errors.get(str(e.item_id), 0) + 1
            elif isinstance(e, DatasetMergeError):
                for s in e.sources:
                    source_errors[str(s)] = source_errors.get(s, 0) + 1
                item_errors[str(e.item_id)] = item_errors.get(str(e.item_id), 0) + 1

            all_errors.append(str(e))

        errors = OrderedDict(
            [
                ("Item errors", item_errors),
                ("Source errors", source_errors),
                ("All errors", all_errors),
            ]
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)

        dump_json_file(path, errors, indent=True)

    def get_any_label_name(self, ann, label_id):
        raise NotImplementedError
