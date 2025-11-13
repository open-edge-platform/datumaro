# Copyright (C) 2022-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import glob
import os.path as osp
from typing import List, Optional

import nibabel as nib
import numpy as np

from datumaro.components.annotation import AnnotationType, ExtractedMask, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import MultiframeImage


class BratsPath:
    IMAGES_DIR = "images"
    LABELS = "labels"
    DATA_EXT = ".nii.gz"


class BratsBase(SubsetBase):
    def __init__(self, path: str, *, ctx: Optional[ImportContext] = None):
        if not osp.isdir(path):
            raise NotADirectoryError(errno.ENOTDIR, "Can't find dataset directory", path)

        self._subset_suffix = osp.basename(path)[len(BratsPath.IMAGES_DIR) :]
        subset = None
        if self._subset_suffix == "Tr":
            subset = "train"
        elif self._subset_suffix == "Ts":
            subset = "test"
        super().__init__(subset=subset, media_type=MultiframeImage, ctx=ctx)

        self._root_dir = osp.dirname(path)
        self._categories = self._load_categories()
        self._items = list(self._load_items(path).values())

    def _load_categories(self):
        label_cat = LabelCategories()

        labels_path = osp.join(self._root_dir, BratsPath.LABELS)
        if osp.isfile(labels_path):
            with open(labels_path, encoding="utf-8") as f:
                for line in f:
                    label_cat.add(line.strip())

        return {AnnotationType.label: label_cat}

    def _load_items(self, path):
        items = {}

        for image_path in glob.glob(osp.join(path, f"*{BratsPath.DATA_EXT}")):
            data = nib.load(image_path).get_fdata()

            item_id = osp.basename(image_path)[: -len(BratsPath.DATA_EXT)]

            images = [0] * data.shape[2]
            for i in range(data.shape[2]):
                images[i] = data[:, :, i]

            items[item_id] = DatasetItem(
                id=item_id, subset=self._subset, media=MultiframeImage(images, path=image_path)
            )

        masks_dir = osp.join(self._root_dir, BratsPath.LABELS + self._subset_suffix)
        for mask in glob.glob(osp.join(masks_dir, f"*{BratsPath.DATA_EXT}")):
            data = nib.load(mask).get_fdata()

            item_id = osp.basename(image_path)[: -len(BratsPath.DATA_EXT)]

            if item_id not in items:
                items[item_id] = DatasetItem(id=item_id)

            anno = []
            for i in range(data.shape[2]):
                np_mask = data[:, :, i]
                classes = np.unique(np_mask)
                for class_id in classes:
                    anno.append(
                        ExtractedMask(
                            index_mask=np_mask,
                            index=class_id,
                            label=class_id,
                            attributes={"image_id": i},
                        )
                    )
                    self._ann_types.add(AnnotationType.mask)

            items[item_id].annotations = anno

        return items


class BratsImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        with context.require_any(), context.alternative():
            context.require_file(f"*/*{BratsPath.DATA_EXT}")

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, "", "brats", filename=f"{BratsPath.IMAGES_DIR}*")

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [osp.splitext(BratsPath.DATA_EXT)[-1]]
