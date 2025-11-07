# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from typing import List, Optional, Type

from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.merge.extractor_merger import ExtractorMerger

from .format import VocPath, VocTask


class _VocImporter(Importer):
    _TASKS = {
        VocTask.voc: ("voc", "Main"),
        VocTask.voc_classification: ("voc_classification", "Main"),
        VocTask.voc_detection: ("voc_detection", "Main"),
        VocTask.voc_segmentation: ("voc_segmentation", "Segmentation"),
        VocTask.voc_instance_segmentation: ("voc_instance_segmentation", "Segmentation"),
        VocTask.voc_layout: ("voc_layout", "Layout"),
        VocTask.voc_action: ("voc_action", "Action"),
    }
    ANNO_EXT = ".txt"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        # The `voc` format is inherently ambiguous with `voc_classification`,
        # `voc_detection`, etc. To remove the ambiguity (and thus make it
        # possible to use autodetection with the VOC datasets), disable
        # autodetection for the single-task formats.

        with context.require_any():
            task_dirs = {task_dir for _, task_dir in cls._TASKS.values()}
            for task_dir in sorted(task_dirs):
                with context.alternative():
                    context.require_file(osp.join(VocPath.SUBSETS_DIR, task_dir, f"*{cls.ANNO_EXT}"))

    @classmethod
    def find_sources(cls, path):
        subsets = []

        # find root path for the dataset and use it for all tasks
        root_path = None
        for extractor_type, task_dir in cls._TASKS.values():
            if osp.isfile(path) and not osp.basename(osp.dirname(path)) == task_dir:
                continue

            task_subsets = cls._find_sources_recursive(
                root_path or path,
                "txt",
                extractor_type,
                dirname=osp.join(VocPath.SUBSETS_DIR, task_dir),
                file_filter=lambda p: "_" not in osp.basename(p),
                max_depth=0 if root_path else 3,
            )

            if not task_subsets:
                continue

            subsets.extend(task_subsets)

            if not root_path:
                root_path = osp.dirname(osp.dirname(osp.dirname(task_subsets[0]["url"])))

        return subsets

    @property
    def can_stream(self) -> bool:
        return True

    def get_extractor_merger(self) -> Optional[Type[ExtractorMerger]]:
        return ExtractorMerger

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls.ANNO_EXT]


class VocImporter(_VocImporter):
    _TASK = VocTask.voc
    _TASKS = {_TASK: _VocImporter._TASKS[_TASK]}


class VocClassificationImporter(_VocImporter):
    _TASK = VocTask.voc_classification
    _TASKS = {_TASK: _VocImporter._TASKS[_TASK]}


class VocDetectionImporter(_VocImporter):
    _TASK = VocTask.voc_detection
    _TASKS = {_TASK: _VocImporter._TASKS[_TASK]}


class VocSegmentationImporter(_VocImporter):
    _TASK = VocTask.voc_segmentation
    _TASKS = {_TASK: _VocImporter._TASKS[_TASK]}


class VocInstanceSegmentationImporter(_VocImporter):
    _TASK = VocTask.voc_instance_segmentation
    _TASKS = {_TASK: _VocImporter._TASKS[_TASK]}


class VocLayoutImporter(_VocImporter):
    _TASK = VocTask.voc_layout
    _TASKS = {_TASK: _VocImporter._TASKS[_TASK]}


class VocActionImporter(_VocImporter):
    _TASK = VocTask.voc_action
    _TASKS = {_TASK: _VocImporter._TASKS[_TASK]}
