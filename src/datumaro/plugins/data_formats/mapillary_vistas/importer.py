# Copyright (C) 2022-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
import glob
import logging as log
import os.path as osp
from typing import List

from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME
from datumaro.components.errors import DatasetNotFoundError
from datumaro.components.importer import Importer
from datumaro.util import str_to_bool

from .base import MapillaryVistasInstancesBase, MapillaryVistasPanopticBase
from .format import MapillaryVistasPath, MapillaryVistasTask


class MapillaryVistasImporter(Importer):
    _TASKS = {
        MapillaryVistasTask.instances: MapillaryVistasInstancesBase,
        MapillaryVistasTask.panoptic: MapillaryVistasPanopticBase,
    }

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--format-version",
            default="v2.0",
            type=str,
            help="Use original config*.json file for your version of dataset",
        )
        parser.add_argument(
            "--parse-polygon",
            type=str_to_bool,
            default=False,
            help="Use original config*.json file for your version of dataset",
        )
        parser.add_argument(
            "--use-original-config",
            action="store_true",
            help="Use original config*.json file for your version of dataset",
        )
        parser.add_argument(
            "--keep-original-category-ids",
            action="store_true",
            help="Add dummy label categories so that category indices "
            "correspond to the category IDs in the original annotation "
            "file",
        )
        return parser

    def __call__(self, path, **extra_params):
        subsets = self.find_sources(path)

        if len(subsets) == 0:
            raise DatasetNotFoundError(path, self.NAME)

        tasks = list(set(task for subset in subsets.values() for task in subset))
        selected_task = tasks[0]
        if len(tasks) > 1:
            task_types = ",".join(task.name for task in tasks)
            log.warning(
                f"Found potentially conflicting source types: {task_types}"
                f"Only one one type will be used: {selected_task.name}"
            )

        if selected_task == MapillaryVistasTask.instances:
            has_config = any(
                [osp.isfile(osp.join(path, config)) for config in MapillaryVistasPath.CONFIG_FILES.values()]
            )

            if not has_config and not extra_params.get("use_original_config"):
                raise DatasetNotFoundError(
                    path,
                    self.NAME,
                    "Failed to find config*.json at '{path}'. See extra args for using original configs.",
                )

        return [
            {"url": url, "format": self._TASKS[task].NAME, "options": dict(extra_params)}
            for _, subset_info in subsets.items()
            for task, url in subset_info.items()
            if task == selected_task
        ]

    @classmethod
    def find_sources(cls, path):
        subsets = {}

        suffixes = [
            osp.join(ann_dir, subdir)
            for ann_dir, subdirs in MapillaryVistasPath.ANNOTATION_DIRS.items()
            for subdir in subdirs
        ]

        for suffix in suffixes:
            task = MapillaryVistasPath.CLASS_BY_DIR[osp.basename(suffix)]
            if task not in cls._TASKS:
                continue

            if osp.isdir(osp.join(path, suffix)):
                return {DEFAULT_SUBSET_NAME: {task: path}}

            for ann_path in glob.glob(osp.join(path, "*", suffix)):
                subset = osp.dirname(osp.dirname(osp.relpath(ann_path, path)))
                subsets.setdefault(subset, {})[task] = osp.join(path, subset)

        return subsets

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [".jpg", ".png", ".json"]


class MapillaryVistasInstancesImporter(MapillaryVistasImporter):
    _TASK = MapillaryVistasTask.instances
    _TASKS = {_TASK: MapillaryVistasImporter._TASKS[_TASK]}


class MapillaryVistasPanopticImporter(MapillaryVistasImporter):
    _TASK = MapillaryVistasTask.panoptic
    _TASKS = {_TASK: MapillaryVistasImporter._TASKS[_TASK]}
