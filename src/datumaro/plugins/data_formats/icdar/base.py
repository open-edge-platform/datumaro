# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import csv
import errno
import glob
import logging as log
import os.path as osp
from typing import List, Optional

import numpy as np

from datumaro.components.annotation import Bbox, Caption, ExtractedMask, MaskCategories, Polygon
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import InvalidAnnotationError
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util.image import IMAGE_EXTENSIONS, find_images
from datumaro.util.mask_tools import lazy_mask

from .format import IcdarPath, IcdarTask


class _IcdarBase(SubsetBase):
    def __init__(
        self,
        path: str,
        task: IcdarTask,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        self._path = path
        self._task = task

        if task is IcdarTask.word_recognition:
            if not osp.isfile(path):
                raise FileNotFoundError(errno.ENOENT, "Can't find annotations file", path)

            if not subset:
                subset = osp.basename(osp.dirname(path))
            super().__init__(subset=subset, ctx=ctx)

            self._dataset_dir = osp.dirname(osp.dirname(path))

            self._items = list(self._load_recognition_items().values())
        elif task in {IcdarTask.text_localization, IcdarTask.text_segmentation}:
            if not osp.isdir(path):
                raise NotADirectoryError(errno.ENOTDIR, "Can't read dataset directory with annotation files", path)

            if not subset:
                subset = osp.basename(path)
            super().__init__(subset=subset, ctx=ctx)

            self._dataset_dir = osp.dirname(path)

            if task is IcdarTask.text_localization:
                self._items = list(self._load_localization_items().values())
            else:
                self._items = list(self._load_segmentation_items().values())

    def _load_recognition_items(self):
        items = {}

        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                objects = line.split(", ")
                if len(objects) == 2:
                    image = objects[0]
                    captions = []
                    for caption in objects[1:]:
                        if caption[0] != '"' or caption[-1] != '"':
                            log.warning("Line %s: unexpected number of quotes" % line)
                        else:
                            captions.append(caption.replace("\\", "")[1:-1])
                else:
                    image = objects[0][:-1]
                    captions = []

                item_id = osp.splitext(image)[0]
                image_path = osp.join(osp.dirname(self._path), IcdarPath.IMAGES_DIR, image)
                if item_id not in items:
                    items[item_id] = DatasetItem(item_id, subset=self._subset, media=Image.from_file(path=image_path))

                annotations = items[item_id].annotations
                for caption in captions:
                    annotations.append(Caption(caption))

                for ann in annotations:
                    self._ann_types.add(ann.type)

        return items

    def _load_localization_items(self):
        items = {}

        image_dir = osp.join(self._path, IcdarPath.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace("\\", "/"): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        for path in glob.iglob(osp.join(self._path, "**", "*.txt"), recursive=True):
            item_id = osp.splitext(osp.relpath(path, self._path))[0]
            if osp.basename(item_id).startswith("gt_"):
                item_id = osp.join(osp.dirname(item_id), osp.basename(item_id)[3:])
            item_id = item_id.replace("\\", "/")

            if item_id not in items:
                image = None
                image_path = images.get(item_id)
                if image_path:
                    image = Image.from_file(path=image_path)

                items[item_id] = DatasetItem(item_id, subset=self._subset, media=image)
            annotations = items[item_id].annotations

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    objects = line.split('"')
                    if len(objects) > 1:
                        if len(objects) == 3:
                            text = objects[1]
                        else:
                            raise InvalidAnnotationError("Line %s: unexpected number of quotes in filename" % line)
                    else:
                        text = ""
                    objects = objects[0].split()
                    if len(objects) == 1:
                        objects = objects[0].split(",")

                    if len(objects) >= 8:
                        points = [float(p) for p in objects[:8]]

                        attributes = {}
                        if len(text) > 0:
                            attributes["text"] = text
                        elif len(objects) == 9:
                            text = objects[8]
                            attributes["text"] = text

                        annotations.append(Polygon(points, attributes=attributes))
                    elif len(objects) >= 4:
                        x = float(objects[0])
                        y = float(objects[1])
                        w = float(objects[2]) - x
                        h = float(objects[3]) - y

                        attributes = {}
                        if len(text) > 0:
                            attributes["text"] = text
                        elif len(objects) == 5:
                            text = objects[4]
                            attributes["text"] = text

                        annotations.append(Bbox(x, y, w, h, attributes=attributes))

            for ann in annotations:
                self._ann_types.add(ann.type)

        return items

    def _load_segmentation_items(self):
        items = {}

        image_dir = osp.join(self._path, IcdarPath.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace("\\", "/"): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        for path in glob.iglob(osp.join(self._path, "**", "*.txt"), recursive=True):
            item_id = osp.splitext(osp.relpath(path, self._path))[0]
            item_id = item_id.replace("\\", "/")
            if item_id.endswith("_GT"):
                item_id = item_id[:-3]

            if item_id not in items:
                image = None
                image_path = images.get(item_id)
                if image_path:
                    image = Image.from_file(path=image_path)

                items[item_id] = DatasetItem(item_id, subset=self._subset, media=image)
            annotations = items[item_id].annotations

            colors = [(255, 255, 255)]
            chars = [""]
            centers = [0]
            groups = [0]
            group = 1
            number_in_group = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        if number_in_group == 1:
                            groups[len(groups) - 1] = 0
                        else:
                            group += 1
                        number_in_group = 0
                        continue

                    objects = line.split()
                    if objects[0][0] == "#":
                        objects[0] = objects[0][1:]
                        objects[9] = '" "'
                        objects.pop()
                    if len(objects) != 10:
                        raise InvalidAnnotationError(
                            'Line %s contains the wrong number of arguments, e.g. \'241 73 144 1 4 0 3 1 4 "h"' % line
                        )

                    centers.append(objects[3] + " " + objects[4])
                    groups.append(group)
                    colors.append(tuple(int(o) for o in objects[:3]))
                    char = objects[9]
                    if char[0] == '"' and char[-1] == '"':
                        char = char[1:-1]
                    chars.append(char)
                    number_in_group += 1
            if number_in_group == 1:
                groups[len(groups) - 1] = 0

            mask_categories = MaskCategories({i: colors[i] for i in range(len(colors))})
            inverse_cls_colormap = mask_categories.inverse_colormap

            gt_path = osp.join(self._path, item_id + "_GT" + IcdarPath.GT_EXT)
            if osp.isfile(gt_path):
                # load mask through cache
                index_mask = lazy_mask(gt_path, inverse_cls_colormap)
                np_index_mask = index_mask()

                classes = np.unique(np_index_mask)
                for label_id in classes:
                    if label_id == 0:
                        continue
                    i = int(label_id)
                    annotations.append(
                        ExtractedMask(
                            index_mask=index_mask,
                            index=label_id,
                            group=groups[i],
                            attributes={
                                "index": i - 1,
                                "color": " ".join(str(p) for p in colors[i]),
                                "text": chars[i],
                                "center": centers[i],
                            },
                        )
                    )

            for ann in annotations:
                self._ann_types.add(ann.type)

        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c


class IcdarWordRecognitionBase(_IcdarBase):
    def __init__(self, path, **kwargs):
        kwargs["task"] = IcdarTask.word_recognition
        super().__init__(path, **kwargs)


class IcdarTextLocalizationBase(_IcdarBase):
    def __init__(self, path, **kwargs):
        kwargs["task"] = IcdarTask.text_localization
        super().__init__(path, **kwargs)


class IcdarTextSegmentationBase(_IcdarBase):
    def __init__(self, path, **kwargs):
        kwargs["task"] = IcdarTask.text_segmentation
        super().__init__(path, **kwargs)


class IcdarWordRecognitionImporter(Importer):
    _ANNO_EXT = ".txt"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        annot_path = context.require_file(f"*/gt{cls._ANNO_EXT}")

        with context.probe_text_file(
            annot_path,
            "must be a ICDAR-like annotation file",
        ) as f:
            reader = csv.reader(f, doublequote=False, escapechar="\\", skipinitialspace=True)
            fields = next(reader)
            if len(fields) != 2:
                raise Exception
            if osp.splitext(fields[0])[1] not in IMAGE_EXTENSIONS:
                raise Exception

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, ".txt", "icdar_word_recognition")

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]


class IcdarTextLocalizationImporter(Importer):
    _ANNO_EXT = ".txt"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(f"**/gt_*{cls._ANNO_EXT}")

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, "", "icdar_text_localization")

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]


class IcdarTextSegmentationImporter(Importer):
    _ANNO_EXT = ".txt"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        gt_txt_path = context.require_file(f"**/*_GT{cls._ANNO_EXT}")
        gt_bmp_path = osp.splitext(gt_txt_path)[0] + ".bmp"
        context.require_file(glob.escape(gt_bmp_path))

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, "", "icdar_text_segmentation")

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]
