# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os.path as osp
from glob import glob
from typing import List

from datumaro.components.errors import DatasetNotFoundError
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer

from .format import KittiPath, KittiTask


class KittiImporter(Importer):
    DETECT_CONFIDENCE = FormatDetectionConfidence.MEDIUM

    _TASKS = {
        KittiTask.segmentation: ("kitti_segmentation", KittiPath.INSTANCES_DIR),
        KittiTask.detection: ("kitti_detection", KittiPath.LABELS_DIR),
    }

    def __call__(self, path, **extra_params):
        subsets = self.find_sources(path)

        if len(subsets) == 0:
            raise DatasetNotFoundError(path, self.NAME)

        # TODO: should be removed when proper label merging is implemented
        conflicting_types = {"kitti_segmentation", "kitti_detection"}
        ann_types = set(t for s in subsets.values() for t in s) & conflicting_types
        if len(ann_types) >= 1:
            selected_ann_type = sorted(ann_types)[0]
        if len(ann_types) > 1:
            log.warning(
                "Not implemented: "
                "Found potentially conflicting source types with labels: %s. "
                "Only one type will be used: %s" % (", ".join(ann_types), selected_ann_type)
            )

        sources = []
        for ann_files in subsets.values():
            for ann_type, ann_file in ann_files.items():
                if ann_type in conflicting_types and ann_type is not selected_ann_type:
                    log.warning("Not implemented: conflicting source '%s' is skipped." % ann_file)
                    continue
                log.info("Found a dataset at '%s'" % ann_file)

                sources.append(
                    {
                        "url": ann_file,
                        "format": ann_type,
                        "options": dict(extra_params),
                    }
                )

        return sources

    @classmethod
    def find_sources(cls, path):
        subsets = {}

        for extractor_type, task_dir in cls._TASKS.values():
            subset_paths = glob(osp.join(path, "**", task_dir), recursive=True)
            for subset_path in subset_paths:
                path = osp.normpath(osp.join(subset_path, ".."))
                subset_name = osp.splitext(osp.basename(path))[0]
                subsets.setdefault(subset_name, {})[extractor_type] = path

        return subsets

    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> FormatDetectionConfidence:
        sub_importers = [KittiDetectionImporter, KittiSegmentationImporter]
        with context.require_any():
            for importer_cls in sub_importers:
                with context.alternative():
                    importer_cls.detect(context)

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        sub_importers = [KittiDetectionImporter, KittiSegmentationImporter]
        return list({ext for importer in sub_importers for ext in importer.get_file_extensions()})


class KittiDetectionImporter(KittiImporter):
    _TASK = KittiTask.detection
    _TASKS = {_TASK: KittiImporter._TASKS[_TASK]}
    _ANNO_EXT = ".txt"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        # left color camera label files
        context.require_file(f"**/label_2/*_*{cls._ANNO_EXT}")
        return cls.DETECT_CONFIDENCE

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]


class KittiSegmentationImporter(KittiImporter):
    _TASK = KittiTask.segmentation
    _TASKS = {_TASK: KittiImporter._TASKS[_TASK]}
    _FORMAT_EXT = ".png"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        # instance segmentation masks
        context.require_file(f"**/instance/*{cls._FORMAT_EXT}")
        return cls.DETECT_CONFIDENCE

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._FORMAT_EXT]
