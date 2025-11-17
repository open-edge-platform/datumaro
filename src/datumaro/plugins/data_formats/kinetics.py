# Copyright (C) 2022-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import csv
import errno
import os
import os.path as osp
from typing import List, Optional

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Video
from datumaro.plugins.data_formats.video import VIDEO_EXTENSIONS
from datumaro.rust_api import JsonSectionPageMapper
from datumaro.util import parse_json, parse_json_file
from datumaro.util.os_util import find_files


class KineticsBase(DatasetBase):
    def __init__(self, path: str, *, ctx: Optional[ImportContext] = None):
        if not osp.isdir(path):
            raise NotADirectoryError(errno.ENOTDIR, "Can't find dataset directory", path)
        self._path = path

        super().__init__(media_type=Video, ctx=ctx)

        self._annotation_files = {}
        for ann_file in find_files(path, ["csv", "json"]):
            filename = osp.splitext(osp.basename(ann_file))[0]
            if (filename not in self._annotation_files) or (self._annotation_files.get(filename, "").endswith("json")):
                self._annotation_files[filename] = ann_file

        self._subsets = {}
        self._subset_media_files = {}
        self._categories = {AnnotationType.label: LabelCategories()}
        self._items = []

        for ann_file in sorted(self._annotation_files.values()):
            if ann_file.endswith("csv"):
                self._load_items_from_csv(ann_file)
            else:
                self._load_items_from_json(ann_file)

    def __iter__(self):
        return iter(self._items)

    def categories(self):
        return self._categories

    def _subset_path(self, subset):
        if subset in self._subsets:
            return self._subsets[subset]

        subset_path = osp.join(self._path, subset)
        self._subsets[subset] = subset_path if osp.isdir(subset_path) else self._path

        return self._subsets[subset]

    def _media_files(self, subset):
        if subset in self._subset_media_files:
            return self._subset_media_files[subset]

        subset_path = self._subset_path(subset)

        self._subset_media_files[subset] = {}
        for root, _, files in os.walk(subset_path):
            for f in files:
                key, file_extension = osp.splitext(f)
                if file_extension.lstrip(".") in VIDEO_EXTENSIONS:
                    self._subset_media_files[subset][key] = osp.join(root, f)

        return self._subset_media_files[subset]

    def _search_media_by_id(self, media_files, item_id):
        if item_id in media_files:
            return media_files[item_id]

        for media_file, file_path in media_files.items():
            if media_file.rfind(item_id) > -1:
                return file_path

        return None

    def _create_item(self, item_id, label, time_start, time_end, subset):
        label_id, _ = self._categories[AnnotationType.label].find(label)
        if label_id is None:
            label_id = self._categories[AnnotationType.label].add(label)

        media_path = self._search_media_by_id(self._media_files(subset), item_id)
        media = Video(media_path) if media_path else None

        self._ann_types.add(AnnotationType.label)

        return DatasetItem(
            id=item_id,
            annotations=[
                Label(
                    label=label_id,
                    attributes={"time_start": int(time_start), "time_end": int(time_end)},
                )
            ],
            subset=subset,
            media=media,
        )

    def _load_items_from_csv(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for item_desc in csv.DictReader(f):
                self._items.append(
                    self._create_item(
                        item_desc["youtube_id"],
                        item_desc["label"],
                        item_desc["time_start"],
                        item_desc["time_end"],
                        item_desc["split"],
                    )
                )

    def _load_items_from_json(self, path):
        subset_desc = parse_json_file(path)
        for item_id, item_desc in subset_desc.items():
            self._items.append(
                self._create_item(
                    item_id,
                    item_desc["annotations"]["label"],
                    item_desc["annotations"]["segment"][0],
                    item_desc["annotations"]["segment"][1],
                    item_desc["subset"],
                )
            )


class KineticsImporter(Importer):
    _ANNO_EXTS = [".json", ".csv"]

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        with context.require_any():
            with context.alternative():
                ann_file = context.require_file(f"*{cls._ANNO_EXTS[0]}")

                with context.probe_text_file(
                    ann_file,
                    "JSON file must contain an youtube 'url' key",
                ) as f:
                    fpath = osp.join(context.root_path, ann_file)
                    page_mapper = JsonSectionPageMapper(fpath)
                    sections = page_mapper.sections()

                    page_map = next(iter(sections.values()))
                    offset, size = page_map["offset"], page_map["size"]

                    f.seek(offset, 0)
                    contents = parse_json(f.read(size))
                    if not isinstance(contents, dict):
                        raise Exception
                    if "youtube" not in contents.get("url", ""):
                        raise Exception

            with context.alternative():
                ann_file = context.require_file(f"*{cls._ANNO_EXTS[1]}")

                with context.probe_text_file(
                    ann_file,
                    "CSV file must contain 'youtube_id' in header",
                ) as f:
                    header = f.readline()
                    if "youtube_id" not in header.split(","):
                        raise Exception

    @classmethod
    def find_sources(cls, path):
        if find_files(path, ["csv", "json"]):
            return [{"url": path, "format": KineticsBase.NAME}]
        return []

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return cls._ANNO_EXTS
