# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import glob
import logging as log
import os
import os.path as osp
import re
from functools import partial
from typing import List, Optional

import numpy as np

from datumaro.components.annotation import AnnotationType, ExtractedMask, LabelCategories, Mask, Polygon
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.rust_api import JsonSectionPageMapper
from datumaro.util import parse_json
from datumaro.util.image import IMAGE_EXTENSIONS, find_images, lazy_image, load_image
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class Ade20k2020Path:
    MASK_PATTERN = re.compile(
        r""".+_seg
        | .+_parts_\d+
        | instance_.+
    """,
        re.VERBOSE,
    )


class Ade20k2020Base(DatasetBase):
    def __init__(self, path: str, *, ctx: Optional[ImportContext] = None):
        if not osp.isdir(path):
            raise NotADirectoryError(errno.ENOTDIR, "Can't find dataset directory", path)

        # exclude dataset meta file
        subsets = [subset for subset in os.listdir(path) if osp.splitext(subset)[-1] != ".json"]
        if len(subsets) < 1:
            raise FileNotFoundError(errno.ENOENT, "Can't find subsets in directory", path)

        super().__init__(subsets=sorted(subsets), ctx=ctx)
        self._path = path

        self._items = []
        self._categories = {}

        if has_meta_file(self._path):
            self._categories = {AnnotationType.label: LabelCategories.from_iterable(parse_meta_file(self._path).keys())}

        for subset in self._subsets:
            self._load_items(subset)

    def __iter__(self):
        return iter(self._items)

    def categories(self):
        return self._categories

    def _load_items(self, subset):
        labels = self._categories.setdefault(AnnotationType.label, LabelCategories())
        path = osp.join(self._path, subset)

        images = [i for i in find_images(path, recursive=True)]
        for image_path in sorted(images):
            item_id = osp.splitext(osp.relpath(image_path, path))[0]

            if Ade20k2020Path.MASK_PATTERN.fullmatch(osp.basename(item_id)):
                continue

            item_annotations = []
            item_info = self._load_item_info(image_path)
            for item in item_info:
                label_idx = labels.find(item["label_name"])[0]
                if label_idx is None:
                    labels.add(item["label_name"])

            mask_path = osp.splitext(image_path)[0] + "_seg.png"
            max_part_level = max([p["part_level"] for p in item_info])
            for part_level in range(max_part_level + 1):
                if not osp.exists(mask_path):
                    log.warning("Can`t find part level %s mask for %s" % (part_level, image_path))
                    continue

                mask = lazy_image(mask_path, loader=self._load_class_mask)

                classes = {(v["class_idx"], v["label_name"]) for v in item_info if v["part_level"] == part_level}

                for class_idx, label_name in classes:
                    label_id = labels.find(label_name)[0]
                    item_annotations.append(
                        ExtractedMask(
                            index_mask=mask,
                            index=class_idx,
                            label=label_id,
                            id=class_idx,
                            group=class_idx,
                            z_order=part_level,
                        )
                    )

                mask_path = osp.splitext(image_path)[0] + "_parts_%s.png" % (part_level + 1)

            for item in item_info:
                instance_path = osp.join(osp.dirname(image_path), item["instance_mask"])
                if not osp.isfile(instance_path):
                    log.warning("Can`t find instance mask: %s" % instance_path)
                    continue

                mask = lazy_image(instance_path, loader=self._load_instance_mask)

                label_id = labels.find(item["label_name"])[0]
                instance_id = item["id"]
                attributes = {k: True for k in item["attributes"]}
                polygon_points = item["polygon_points"]

                item_annotations.append(
                    Mask(
                        label=label_id,
                        image=partial(self._get_instance_mask, mask),
                        id=instance_id,
                        attributes=attributes,
                        z_order=item["part_level"],
                        group=instance_id,
                    )
                )

                if len(item["polygon_points"]) % 2 == 0 and len(item["polygon_points"]) // 2 >= 3:
                    item_annotations.append(
                        Polygon(
                            polygon_points,
                            label=label_id,
                            attributes=attributes,
                            id=instance_id,
                            z_order=item["part_level"],
                            group=instance_id,
                        )
                    )

            self._items.append(
                DatasetItem(
                    item_id,
                    subset=subset,
                    media=Image.from_file(path=image_path),
                    annotations=item_annotations,
                )
            )
            for ann in item_annotations:
                self._ann_types.add(ann.type)

    def _load_item_info(self, path):
        json_path = osp.splitext(path)[0] + ".json"
        item_info = []
        if not osp.isfile(json_path):
            raise FileNotFoundError(errno.ENOENT, "Can't find annotation file for image %s" % path, json_path)

        with open(json_path, "r", encoding="latin-1") as f:
            item_objects = parse_json(f.read())["annotation"]["object"]
            for obj in item_objects:
                polygon_points = []
                for x, y in zip(obj["polygon"]["x"], obj["polygon"]["y"]):
                    polygon_points.append(x)
                    polygon_points.append(y)

                attributes = obj["attributes"]
                if isinstance(attributes, str):
                    attributes = [attributes]

                item_info.append(
                    {
                        "id": obj["id"],
                        "class_idx": obj["name_ndx"],
                        "part_level": obj["parts"]["part_level"],
                        "occluded": int(obj["occluded"] == "yes"),
                        "crop": obj["crop"],
                        "label_name": obj["raw_name"],
                        "attributes": attributes,
                        "instance_mask": obj["instance_mask"],
                        "polygon_points": polygon_points,
                    }
                )

        return item_info

    @staticmethod
    def _load_instance_mask(path):
        mask = load_image(path)
        _, instance_mask = np.unique(mask, return_inverse=True)
        return instance_mask.reshape(mask.shape)

    @staticmethod
    def _load_class_mask(path):
        mask = load_image(path)
        return ((mask[:, :, 2] / 10).astype(np.int32) << 8) + mask[:, :, 1].astype(np.int32)

    @staticmethod
    def _get_instance_mask(mask: lazy_image) -> np.ndarray:
        return mask() == 1


class Ade20k2020Importer(Importer):
    _ANNO_EXT = ".json"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        annot_path = context.require_file(f"*/**/*{cls._ANNO_EXT}")

        with context.probe_text_file(
            annot_path,
            'must be a JSON object with an "annotation" key',
        ):
            fpath = osp.join(context.root_path, annot_path)
            page_mapper = JsonSectionPageMapper(fpath)
            sections = page_mapper.sections()

            if "annotation" not in sections:
                raise Exception

    @classmethod
    def find_sources(cls, path):
        for i in range(5):
            for i in glob.iglob(osp.join(path, *("*" * i))):
                if osp.splitext(i)[1].lower() in IMAGE_EXTENSIONS:
                    return [
                        {
                            "url": path,
                            "format": Ade20k2020Base.NAME,
                        }
                    ]
        return []

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]
