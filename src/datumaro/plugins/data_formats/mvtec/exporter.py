# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from enum import Enum, auto

import cv2
import numpy as np

from datumaro.components.annotation import AnnotationType
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.media import Image
from datumaro.util import parse_str_enum_value

from .format import MvtecPath, MvtecTask


class LabelmapType(Enum):
    kitti = auto()
    source = auto()


class MvtecExporter(Exporter):
    DEFAULT_IMAGE_EXT = MvtecPath.IMAGE_EXT

    @staticmethod
    def _split_tasks_string(s):
        return [MvtecTask[i.strip().lower()] for i in s.split(",")]

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument(
            "--tasks",
            type=cls._split_tasks_string,
            help="MVTec task filter, comma-separated list of {%s} "
            "(default: all)" % ", ".join(t.name for t in MvtecTask),
        )
        return parser

    def __init__(
        self,
        extractor,
        save_dir,
        tasks=None,
        **kwargs,
    ):
        super().__init__(extractor, save_dir, **kwargs)

        assert tasks is None or isinstance(tasks, (MvtecTask, list, set))
        if tasks is None:
            tasks = set(MvtecTask)
        elif isinstance(tasks, MvtecTask):
            tasks = {tasks}
        else:
            tasks = set(parse_str_enum_value(t, MvtecTask) for t in tasks)
        self._tasks = tasks

    def _apply_impl(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(self._save_dir, exist_ok=True)

        for subset_name, subset in self._extractor.subsets().items():
            if MvtecTask.segmentation in self._tasks or MvtecTask.detection in self._tasks:
                os.makedirs(osp.join(self._save_dir, MvtecPath.MASK_DIR), exist_ok=True)

            for item in subset:
                labels = []
                for ann in item.annotations:
                    if ann.type in [AnnotationType.label, AnnotationType.mask, AnnotationType.bbox]:
                        labels.append(self.get_label(ann.label))

                if self._save_media:
                    self._save_image(item, subdir=subset_name)

                bboxes = [a for a in item.annotations if a.type == AnnotationType.bbox]
                if bboxes and MvtecTask.detection in self._tasks:
                    mask_path = osp.join(MvtecPath.MASK_DIR, item.id + MvtecPath.MASK_POSTFIX + MvtecPath.MASK_EXT)

                    mask = np.zeros((*item.media.size,), dtype=np.uint8)
                    for bbox in bboxes:
                        x, y, h, w = bbox.get_bbox()
                        mask = cv2.rectangle(
                            mask,
                            np.array((x, y), dtype=np.int32),
                            np.array((x + w, y + h), dtype=np.int32),
                            (1),
                            -1,
                        )

                    if not osp.exists(osp.join(self._save_dir, osp.dirname(mask_path))):
                        os.mkdir(osp.join(self._save_dir, osp.dirname(mask_path)))
                    cv2.imwrite(osp.join(self._save_dir, mask_path), mask * 255)

                masks = [a for a in item.annotations if a.type == AnnotationType.mask]
                if masks and MvtecTask.segmentation in self._tasks:
                    mask_path = osp.join(MvtecPath.MASK_DIR, item.id + MvtecPath.MASK_POSTFIX + MvtecPath.MASK_EXT)

                    if not osp.exists(osp.join(self._save_dir, osp.dirname(mask_path))):
                        os.mkdir(osp.join(self._save_dir, osp.dirname(mask_path)))
                    cv2.imwrite(osp.join(self._save_dir, mask_path), masks[0].image.astype(np.uint8) * 255)

    def get_label(self, label_id):
        return self._extractor.categories()[AnnotationType.label].items[label_id].name


class MvtecClassificationExporter(MvtecExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = MvtecTask.classification
        super().__init__(*args, **kwargs)


class MvtecSegmentationExporter(MvtecExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = MvtecTask.segmentation
        super().__init__(*args, **kwargs)


class MvtecDetectionExporter(MvtecExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = MvtecTask.detection
        super().__init__(*args, **kwargs)
