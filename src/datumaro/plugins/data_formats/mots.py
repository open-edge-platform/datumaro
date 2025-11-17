# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

# Implements MOTS format https://www.vision.rwth-aachen.de/page/mots

import logging as log
import os
import os.path as osp
from enum import Enum
from glob import iglob
from typing import List, Optional

import numpy as np

from datumaro.components.annotation import AnnotationType, ExtractedMask, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util.image import find_images, lazy_image, save_image
from datumaro.util.mask_tools import merge_masks
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class MotsPath:
    MASKS_DIR = "instances"
    IMAGE_DIR = "images"
    IMAGE_EXT = ".jpg"
    LABELS_FILE = "labels.txt"
    MAX_INSTANCES = 1000


class MotsLabels(Enum):
    background = 0
    car = 1
    pedestrian = 2
    ignored = 10


class MotsPngExtractor(SubsetBase):
    @staticmethod
    def detect_dataset(path):
        if osp.isdir(osp.join(path, MotsPath.MASKS_DIR)):
            return [{"url": path, "format": MotsPngExtractor.NAME}]
        return []

    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        assert osp.isdir(path), path
        super().__init__(subset=subset, ctx=ctx)
        self._images_dir = osp.join(path, "images")
        self._anno_dir = osp.join(path, MotsPath.MASKS_DIR)
        if has_meta_file(path):
            self._categories = {AnnotationType.label: LabelCategories.from_iterable(parse_meta_file(path).keys())}
        else:
            self._categories = self._parse_categories(osp.join(self._anno_dir, MotsPath.LABELS_FILE))
        self._items = self._parse_items()

    def _parse_categories(self, path):
        if osp.isfile(path):
            labels = []
            with open(path, encoding="utf-8") as f:
                for label in f:
                    label = label.strip()
                    if label:
                        labels.append(label)
        else:
            labels = [l.name for l in MotsLabels]
        return {AnnotationType.label: LabelCategories.from_iterable(labels)}

    def _parse_items(self):
        items = []

        image_dir = self._images_dir
        if osp.isdir(image_dir):
            images = {osp.splitext(osp.relpath(p, image_dir))[0]: p for p in find_images(image_dir, recursive=True)}
        else:
            images = {}

        for p in sorted(iglob(self._anno_dir + "/**/*.png", recursive=True)):
            item_id = osp.splitext(osp.relpath(p, self._anno_dir))[0]
            image = images.get(item_id)
            if image:
                image = Image.from_file(path=image)
            items.append(
                DatasetItem(
                    id=item_id,
                    subset=self._subset,
                    media=image,
                    annotations=self._parse_annotations(p),
                )
            )
        return items

    @staticmethod
    def _lazy_extract_mask(mask, v):
        return lambda: mask == v

    def _parse_annotations(self, path):
        index_mask = lazy_image(path, dtype=np.uint16)
        np_index_mask = index_mask()

        masks = []
        for obj_id in np.unique(np_index_mask):
            class_id, instance_id = divmod(obj_id, MotsPath.MAX_INSTANCES)
            z_order = 0
            if class_id == 0:
                continue  # background
            if class_id == 10 and len(self._categories[AnnotationType.label]) < 10:
                z_order = 1
                class_id = self._categories[AnnotationType.label].find(MotsLabels.ignored.name)[0]
            else:
                class_id -= 1
            masks.append(
                ExtractedMask(
                    index_mask=index_mask,
                    index=obj_id,
                    label=class_id,
                    z_order=z_order,
                    attributes={"track_id": instance_id},
                )
            )
            self._ann_types.add(AnnotationType.mask)
        return masks


class MotsImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            return []

        subsets = MotsPngExtractor.detect_dataset(path)
        if not subsets:
            for p in os.listdir(path):
                detected = MotsPngExtractor.detect_dataset(osp.join(path, p))
                for s in detected:
                    s.setdefault("options", {})["subset"] = p
                subsets.extend(detected)
        return subsets

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [".png", ".txt"]


class MotsPngExporter(Exporter):
    DEFAULT_IMAGE_EXT = MotsPath.IMAGE_EXT

    def _apply_impl(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(self._save_dir, exist_ok=True)

        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)

        for subset_name, subset in self._extractor.subsets().items():
            subset_dir = osp.join(self._save_dir, subset_name)
            image_dir = osp.join(subset_dir, MotsPath.IMAGE_DIR)
            anno_dir = osp.join(subset_dir, MotsPath.MASKS_DIR)
            os.makedirs(anno_dir, exist_ok=True)

            for item in subset:
                log.debug("Converting item '%s'", item.id)

                if self._save_media:
                    if item.media and item.media.has_data:
                        self._save_image(item, subdir=image_dir)
                    else:
                        log.debug("Item '%s' has no image", item.id)

                self._save_annotations(item, anno_dir)

            with open(osp.join(anno_dir, MotsPath.LABELS_FILE), "w", encoding="utf-8") as f:
                f.write("\n".join(l.name for l in subset.categories()[AnnotationType.label].items))

    def _save_annotations(self, item, anno_dir):
        masks = [a for a in item.annotations if a.type == AnnotationType.mask]
        if not masks:
            return

        instance_ids = [int(a.attributes["track_id"]) for a in masks]
        masks = sorted(zip(masks, instance_ids), key=lambda e: e[0].z_order)
        mask = merge_masks((m.image, MotsPath.MAX_INSTANCES * (1 + m.label) + id) for m, id in masks)
        save_image(osp.join(anno_dir, item.id + ".png"), mask, create_dir=True, dtype=np.uint16)
