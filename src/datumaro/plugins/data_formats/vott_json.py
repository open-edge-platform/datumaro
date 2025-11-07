# Copyright (C) 2022-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import os.path as osp
from typing import List, Optional

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util import parse_json_file
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class VottJsonPath:
    ANNO_FILE_SUFFIX = "-export.json"


class VottJsonBase(SubsetBase):
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
        anno_dict = parse_json_file(path)

        label_categories = self._categories[AnnotationType.label]
        tags = anno_dict.get("tags", [])
        for label in tags:
            label_name = label.get("name")
            label_idx = label_categories.find(label_name)[0]
            if label_idx is None:
                label_idx = label_categories.add(label_name)

        items = {}
        for id, asset in anno_dict.get("assets", {}).items():
            item_id = osp.splitext(asset.get("asset", {}).get("name"))[0]
            annotations = []
            for region in asset.get("regions", []):
                tags = region.get("tags", [])
                if not tags:
                    bbox = region.get("boundingBox", {})
                    if bbox:
                        annotations.append(
                            Bbox(
                                float(bbox["left"]),
                                float(bbox["top"]),
                                float(bbox["width"]),
                                float(bbox["height"]),
                                attributes={"id": region.get("id")},
                            )
                        )

                for tag in region.get("tags", []):
                    label_idx = label_categories.find(tag)[0]
                    if label_idx is None:
                        label_idx = label_categories.add(tag)

                    bbox = region.get("boundingBox", {})
                    if bbox:
                        annotations.append(
                            Bbox(
                                float(bbox["left"]),
                                float(bbox["top"]),
                                float(bbox["width"]),
                                float(bbox["height"]),
                                label=label_idx,
                                attributes={"id": region.get("id")},
                            )
                        )
                        self._ann_types.add(AnnotationType.bbox)

            items[item_id] = DatasetItem(
                id=item_id,
                subset=self._subset,
                attributes={"id": id},
                media=Image.from_file(path=osp.join(osp.dirname(path), asset.get("asset", {}).get("path"))),
                annotations=annotations,
            )

        return items


class VottJsonImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, ".json", "vott_json")

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("*" + VottJsonPath.ANNO_FILE_SUFFIX)

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [osp.splitext(VottJsonPath.ANNO_FILE_SUFFIX)[1]]
