# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import fnmatch
import glob
import logging as log
import os
import os.path as osp
from typing import List, Optional

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset import DatasetItem
from datumaro.components.dataset_base import DatasetBase
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util.image import find_images


class MarsPath:
    SUBSET_DIR_PATTERN = "bbox_*"
    IMAGE_DIR_PATTERNS = ["[0-9]" * 4, "00-1"]
    IMAGE_NAME_POSTFIX = "C[0-9]" + "T" + "[0-9]" * 4 + "F" + "[0-9]" * 3 + ".jpg"


class MarsBase(DatasetBase):
    def __init__(self, path: str, *, ctx: Optional[ImportContext] = None):
        assert osp.isdir(path), path
        super().__init__(ctx=ctx)

        self._dataset_dir = path
        self._subsets = {
            subset_dir.split("_", maxsplit=1)[1]: osp.join(path, subset_dir)
            for subset_dir in os.listdir(path)
            if (osp.isdir(osp.join(path, subset_dir)) and fnmatch.fnmatch(subset_dir, MarsPath.SUBSET_DIR_PATTERN))
        }

        self._categories = self._load_categories()
        self._items = []
        for subset, subset_path in self._subsets.items():
            self._items.extend(self._load_items(subset, subset_path))

    def __iter__(self):
        yield from self._items

    def categories(self):
        return self._categories

    def _load_categories(self):
        dirs = sorted(
            [
                dir_name
                for subset_path in self._subsets.values()
                for dir_name in os.listdir(subset_path)
                if (
                    osp.isdir(osp.join(self._dataset_dir, subset_path, dir_name))
                    and any(fnmatch.fnmatch(dir_name, image_dir) for image_dir in MarsPath.IMAGE_DIR_PATTERNS)
                )
            ]
        )
        return {AnnotationType.label: LabelCategories.from_iterable(dirs)}

    def _load_items(self, subset, path):
        items = []
        for label_cat in self._categories[AnnotationType.label]:
            label = label_cat.name
            label_id = self._categories[AnnotationType.label].find(label)[0]
            for image_path in find_images(osp.join(path, label)):
                image_name = osp.basename(image_path)
                item_id = osp.splitext(image_name)[0]
                pedestrian_id = image_name[0:4]

                if not fnmatch.fnmatch(image_name, label + MarsPath.IMAGE_NAME_POSTFIX):
                    items.append(DatasetItem(id=item_id, image=image_path))
                    continue

                if pedestrian_id != label:
                    log.warning(
                        f"The image {image_path} will be skip because"
                        "pedestrian id for it does not match with"
                        f"the directory name: {label}"
                    )
                    continue

                items.append(
                    DatasetItem(
                        id=item_id,
                        media=Image.from_file(path=image_path),
                        subset=subset,
                        annotations=[Label(label=label_id)],
                        attributes={
                            "person_id": pedestrian_id,
                            "camera_id": int(image_name[5]),
                            "track_id": int(image_name[7:11]),
                            "frame_id": int(image_name[12:15]),
                        },
                    )
                )
                self._ann_types.add(AnnotationType.label)

        return items


class MarsImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext):
        with context.require_any():
            for image_dir in MarsPath.IMAGE_DIR_PATTERNS:
                with context.alternative():
                    context.require_file(
                        "/".join(
                            [
                                MarsPath.SUBSET_DIR_PATTERN,
                                image_dir,
                                image_dir + MarsPath.IMAGE_NAME_POSTFIX,
                            ]
                        )
                    )

    @classmethod
    def find_sources(cls, path):
        patterns = [
            "/".join((path, subset_dir, image_dir, image_dir + MarsPath.IMAGE_NAME_POSTFIX))
            for image_dir in MarsPath.IMAGE_DIR_PATTERNS
            for subset_dir in os.listdir(path)
            if (osp.isdir(osp.join(path, subset_dir)) and fnmatch.fnmatch(subset_dir, MarsPath.SUBSET_DIR_PATTERN))
        ]

        for pattern in patterns:
            try:
                next(glob.iglob(pattern))
                return [{"url": path, "format": "mars"}]
            except StopIteration:
                continue

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [osp.splitext(MarsPath.IMAGE_NAME_POSTFIX)[1]]
