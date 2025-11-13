# Copyright (C) 2021-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import os.path as osp
from typing import Optional

import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, ExtractedMask, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image
from datumaro.util.image import find_images, lazy_image
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file

from .format import KittiLabelMap, KittiPath, KittiTask, make_kitti_categories, parse_label_map


class _KittiBase(SubsetBase):
    def __init__(
        self,
        path: str,
        task: KittiTask,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        assert osp.isdir(path), path
        self._path = path
        self._task = task

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
        super().__init__(subset=subset, ctx=ctx)

        self._categories = self._load_categories(osp.dirname(self._path))
        self._items = list(self._load_items().values())

    def _load_categories(self, path):
        if self._task == KittiTask.segmentation:
            return self._load_categories_segmentation(path)
        if self._task == KittiTask.detection:
            if has_meta_file(path):
                return {AnnotationType.label: LabelCategories.from_iterable(parse_meta_file(path).keys())}

            return {AnnotationType.label: LabelCategories()}

    def _load_categories_segmentation(self, path):
        label_map = None
        if has_meta_file(path):
            label_map = parse_meta_file(path)
        else:
            label_map_path = osp.join(path, KittiPath.LABELMAP_FILE)
            if osp.isfile(label_map_path):
                label_map = parse_label_map(label_map_path)
            else:
                label_map = KittiLabelMap

        self._labels = [label for label in label_map]
        return make_kitti_categories(label_map)

    def _load_items(self):
        items = {}
        image_dir = osp.join(self._path, KittiPath.IMAGES_DIR)
        image_path_by_id = {
            osp.splitext(osp.relpath(p, image_dir))[0]: p for p in find_images(image_dir, recursive=True)
        }

        segm_dir = osp.join(self._path, KittiPath.INSTANCES_DIR)
        if self._task == KittiTask.segmentation:
            for instances_path in find_images(segm_dir, exts=KittiPath.MASK_EXT, recursive=True):
                item_id = osp.splitext(osp.relpath(instances_path, segm_dir))[0]
                anns = []

                instances_mask = lazy_image(instances_path, dtype=np.int32)
                np_instances_mask = instances_mask()
                segm_ids = np.unique(np_instances_mask)
                for segm_id in segm_ids:
                    semantic_id = segm_id >> 8
                    ann_id = int(segm_id % 256)
                    isCrowd = ann_id == 0
                    anns.append(
                        ExtractedMask(
                            index_mask=instances_mask,
                            index=segm_id,
                            label=semantic_id,
                            id=ann_id,
                            attributes={"is_crowd": isCrowd},
                        )
                    )
                    self._ann_types.add(AnnotationType.mask)

                image = image_path_by_id.pop(item_id, None)
                if image:
                    image = Image.from_file(path=image)

                items[item_id] = DatasetItem(id=item_id, annotations=anns, media=image, subset=self._subset)

        det_dir = osp.join(self._path, KittiPath.LABELS_DIR)
        if self._task == KittiTask.detection:
            for labels_path in sorted(glob.glob(osp.join(det_dir, "**", "*.txt"), recursive=True)):
                item_id = osp.splitext(osp.relpath(labels_path, det_dir))[0]
                anns = []

                with open(labels_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line_idx, line in enumerate(lines):
                    line = line.split()
                    assert len(line) == 15 or len(line) == 16

                    x1, y1 = float(line[4]), float(line[5])
                    x2, y2 = float(line[6]), float(line[7])

                    attributes = {}
                    attributes["truncated"] = float(line[1]) != 0
                    attributes["occluded"] = int(line[2]) != 0

                    if len(line) == 16:
                        attributes["score"] = float(line[15])

                    label_id = self.categories()[AnnotationType.label].find(line[0])[0]
                    if label_id is None:
                        label_id = self.categories()[AnnotationType.label].add(line[0])

                    anns.append(
                        Bbox(
                            x=x1,
                            y=y1,
                            w=x2 - x1,
                            h=y2 - y1,
                            id=line_idx,
                            attributes=attributes,
                            label=label_id,
                        )
                    )
                    self._ann_types.add(AnnotationType.bbox)

                image = image_path_by_id.pop(item_id, None)
                if image:
                    image = Image.from_file(path=image)

                items[item_id] = DatasetItem(id=item_id, annotations=anns, media=image, subset=self._subset)

        for item_id, image_path in image_path_by_id.items():
            if item_id in items:
                continue

            items[item_id] = DatasetItem(id=item_id, subset=self._subset, media=Image.from_file(path=image_path))

        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c


class KittiSegmentationBase(_KittiBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, task=KittiTask.segmentation, **kwargs)


class KittiDetectionBase(_KittiBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, task=KittiTask.detection, **kwargs)
