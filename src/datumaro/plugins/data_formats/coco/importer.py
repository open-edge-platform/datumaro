# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os.path as osp
from glob import glob
from typing import List, Optional

from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME
from datumaro.components.errors import DatasetNotFoundError
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.merge.extractor_merger import ExtractorMerger
from datumaro.plugins.data_formats.coco.base import (
    CocoCaptionsBase,
    CocoImageInfoBase,
    CocoInstancesBase,
    CocoLabelsBase,
    CocoPanopticBase,
    CocoPersonKeypointsBase,
    CocoStuffBase,
)
from datumaro.plugins.data_formats.coco.extractor_merger import COCOExtractorMerger

from .format import CocoImporterType, CocoTask


class CocoImporter(Importer):
    _TASKS = {
        CocoTask.instances: CocoInstancesBase,
        CocoTask.person_keypoints: CocoPersonKeypointsBase,
        CocoTask.captions: CocoCaptionsBase,
        CocoTask.labels: CocoLabelsBase,
        CocoTask.image_info: CocoImageInfoBase,
        CocoTask.panoptic: CocoPanopticBase,
        CocoTask.stuff: CocoStuffBase,
    }
    _IMPORTER_TYPE = CocoImporterType.default
    _ANNO_EXT = ".json"

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--keep-original-category-ids",
            action="store_true",
            help="Add dummy label categories so that category indices "
            "correspond to the category IDs in the original annotation "
            "file",
        )
        return parser

    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> FormatDetectionConfidence:
        num_tasks = 0
        for task in cls._TASKS:
            try:
                context.require_files(f"annotations/{task.name}_*{cls._ANNO_EXT}")
                num_tasks += 1
            except Exception:
                pass
        if num_tasks > 1:
            log.warning("Multiple COCO tasks are detected. The detected format will be `coco` instead.")
            return FormatDetectionConfidence.MEDIUM
        context.raise_unsupported()

    def __call__(self, path, stream: bool = False, **extra_params):
        subsets = self.find_sources(path)

        if len(subsets) == 0:
            raise DatasetNotFoundError(path, self.NAME)

        # TODO: should be removed when proper label merging is implemented
        conflicting_types = {
            CocoTask.instances,
            CocoTask.person_keypoints,
            CocoTask.labels,
            CocoTask.panoptic,
            CocoTask.stuff,
        }
        ann_types = set(t for s in subsets.values() for t in s) & conflicting_types
        if len(ann_types) >= 1:
            selected_ann_type = sorted(ann_types, key=lambda x: x.name)[0]
        if len(ann_types) > 1:
            log.warning(
                "Not implemented: "
                "Found potentially conflicting source types with labels: %s. "
                "Only one type will be used: %s" % (", ".join(t.name for t in ann_types), selected_ann_type.name)
            )

        sources = []
        for subset, ann_files in subsets.items():
            for ann_type, ann_file in ann_files.items():
                if ann_type in conflicting_types and ann_type is not selected_ann_type:
                    log.warning("Not implemented: conflicting source '%s' is skipped." % ann_file)
                    continue
                log.info("Found a dataset at '%s'" % ann_file)

                options = dict(extra_params)
                options["coco_importer_type"] = self._IMPORTER_TYPE
                options["subset"] = subset

                if stream:
                    options["stream"] = True

                sources.append(
                    {
                        "url": ann_file,
                        "format": self._TASKS[ann_type].NAME,
                        "options": options,
                    }
                )

        return sources

    @classmethod
    def find_sources(cls, path):
        def detect_coco_task(filename):
            for task in CocoTask:
                if filename.startswith(task.name + "_"):
                    return task
            return None

        if osp.isfile(path):
            if len(cls._TASKS) == 1:
                return {"": {next(iter(cls._TASKS)): path}}

            subset_paths = [path] if path.endswith(".json") else []
        else:
            subset_paths = glob(osp.join(path, "**", "*_*.json"), recursive=True)

        subsets = {}
        for subset_path in subset_paths:
            ann_type = detect_coco_task(osp.basename(subset_path))

            if ann_type not in cls._TASKS:
                log.warning(
                    "File '%s' was skipped, could't match this file "
                    "with any of these tasks: %s" % (subset_path, ",".join(e.NAME for e in cls._TASKS.values()))
                )
                continue

            parts = osp.splitext(osp.basename(subset_path))[0].split(ann_type.name + "_", maxsplit=1)
            subset_name = parts[1] if len(parts) == 2 else DEFAULT_SUBSET_NAME
            subsets.setdefault(subset_name, {})[ann_type] = subset_path

        return subsets

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]

    @property
    def can_stream(self) -> bool:
        return True

    def get_extractor_merger(self) -> Optional[ExtractorMerger]:
        return COCOExtractorMerger


class CocoImageInfoImporter(CocoImporter):
    _TASK = CocoTask.image_info
    _TASKS = {_TASK: CocoImporter._TASKS[_TASK]}

    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> FormatDetectionConfidence:
        context.require_file(f"annotations/{cls._TASK.name}_*{cls._ANNO_EXT}")
        return FormatDetectionConfidence.LOW


class CocoCaptionsImporter(CocoImageInfoImporter):
    _TASK = CocoTask.captions
    _TASKS = {_TASK: CocoImporter._TASKS[_TASK]}


class CocoInstancesImporter(CocoImageInfoImporter):
    _TASK = CocoTask.instances
    _TASKS = {_TASK: CocoImporter._TASKS[_TASK]}


class CocoPersonKeypointsImporter(CocoImageInfoImporter):
    _TASK = CocoTask.person_keypoints
    _TASKS = {_TASK: CocoImporter._TASKS[_TASK]}


class CocoLabelsImporter(CocoImageInfoImporter):
    _TASK = CocoTask.labels
    _TASKS = {_TASK: CocoImporter._TASKS[_TASK]}


class CocoPanopticImporter(CocoImageInfoImporter):
    _TASK = CocoTask.panoptic
    _TASKS = {_TASK: CocoImporter._TASKS[_TASK]}


class CocoStuffImporter(CocoImageInfoImporter):
    _TASK = CocoTask.stuff
    _TASKS = {_TASK: CocoImporter._TASKS[_TASK]}
