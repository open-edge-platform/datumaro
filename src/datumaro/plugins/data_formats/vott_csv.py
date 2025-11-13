# Copyright (C) 2021-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import csv
import errno
import os.path as osp
from typing import List, Optional

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class VottCsvPath:
    ANNO_FILE_SUFFIX = "-export.csv"


class VottCsvBase(SubsetBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isfile(path):
            raise FileNotFoundError(errno.ENOENT, "Can't find annotations file", path)

        if not subset:
            subset = osp.splitext(osp.basename(path))[0].rsplit("-", maxsplit=1)[0]

        super().__init__(subset=subset, ctx=ctx)

        if has_meta_file(path):
            self._categories = {AnnotationType.label: LabelCategories.from_iterable(parse_meta_file(path).keys())}
        else:
            self._categories = {AnnotationType.label: LabelCategories()}

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        label_categories = self._categories[AnnotationType.label]

        with open(path, encoding="utf-8") as content:
            for row in csv.DictReader(content):
                item_id = osp.splitext(row["image"])[0]

                if item_id not in items:
                    items[item_id] = DatasetItem(
                        id=item_id,
                        subset=self._subset,
                        media=Image.from_file(path=osp.join(osp.dirname(path), row["image"])),
                    )

                annotations = items[item_id].annotations

                label_name = row.get("label")
                x_min = row.get("xmin")
                y_min = row.get("ymin")
                x_max = row.get("xmax")
                y_max = row.get("ymax")

                if label_name and x_min and y_min and x_max and y_max:
                    label_idx = label_categories.find(label_name)[0]
                    if label_idx is None:
                        label_idx = label_categories.add(label_name)

                    x_min = float(x_min)
                    y_min = float(y_min)
                    w = float(x_max) - x_min
                    h = float(y_max) - y_min

                    annotations.append(Bbox(x_min, y_min, w, h, label=label_idx))
                    self._ann_types.add(AnnotationType.bbox)

        return items


class VottCsvImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, ".csv", "vott_csv")

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("*" + VottCsvPath.ANNO_FILE_SUFFIX)

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [osp.splitext(VottCsvPath.ANNO_FILE_SUFFIX)[1]]
