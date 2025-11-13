# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from typing import List

from datumaro.components.errors import DatasetImportError
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer

from .format import Kitti3dPath


class Kitti3dImporter(Importer):
    _ANNO_EXT = ".txt"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        context.require_file(f"{Kitti3dPath.CALIB_DIR}/*.txt")
        cls._check_ann_file(context.require_file(f"{Kitti3dPath.LABEL_DIR}/*.txt"), context)
        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def _check_ann_file(cls, fpath: str, context: FormatDetectionContext) -> bool:
        with context.probe_text_file(fpath, "Requirements for the annotation file of Kitti 3D format") as fp:
            for line in fp:
                fields = line.rstrip("\n").split(" ")
                if len(fields) == 15 or len(fields) == 16:
                    return True
                raise DatasetImportError(
                    f"Kitti 3D format txt file should have 15 or 16 fields for "
                    f"each line, but the read line has {len(fields)} fields: "
                    f"fields={fields}."
                )
            raise DatasetImportError("Empty file is not allowed.")

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]

    @classmethod
    def find_sources(cls, path):
        # return [{"url": path, "format": "kitti3d"}]
        sources = cls._find_sources_recursive(
            path, "", "kitti3d", dirname=Kitti3dPath.LABEL_DIR, file_filter=lambda p: osp.isdir(p)
        )
        if len(sources) == 0:
            return [{"url": path, "format": "kitti3d"}]
        return sources
