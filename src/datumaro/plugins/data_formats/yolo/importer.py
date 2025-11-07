# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from collections import defaultdict
from io import TextIOWrapper
from typing import Any, Dict, List, Optional, Type

from datumaro.components.errors import DatasetImportError
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.merge.extractor_merger import ExtractorMerger
from datumaro.util.os_util import extract_subset_name_from_parent

from .format import YoloFormatType, YoloLoosePath, YoloPath, YoloUltralyticsPath


class _YoloStrictImporter(Importer):
    _FORMAT_EXT = ".data"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(f"obj{cls._FORMAT_EXT}")

    @classmethod
    def find_sources(cls, path: str) -> List[Dict[str, Any]]:
        sources = cls._find_sources_recursive(path, ".data", YoloFormatType.yolo_strict.name)

        def _extract_subset_wise_sources(source) -> List[Dict[str, Any]]:
            config_path = source["url"]
            config = YoloPath._parse_config(config_path)
            subsets = [k for k in config if k not in YoloPath.RESERVED_CONFIG_KEYS]
            return [
                {
                    "url": config_path,
                    "format": YoloFormatType.yolo_strict.name,
                    "options": {"subset": subset},
                }
                for subset in subsets
            ]

        return sum([_extract_subset_wise_sources(source) for source in sources], [])

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._FORMAT_EXT]


class _YoloLooseImporter(Importer):
    META_FILE = YoloLoosePath.NAMES_FILE
    FORMAT = YoloFormatType.yolo_loose.name

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        context.require_file(cls.META_FILE)

        with context.require_any():
            with context.alternative():
                cls._check_ann_file(context.require_file("[Aa]nnotations/**/*.txt"), context)
            with context.alternative():
                cls._check_ann_file(context.require_file("[Ll]abels/**/*.txt"), context)

        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def _check_ann_file(cls, fpath: str, context: FormatDetectionContext) -> None:
        with context.probe_text_file(fpath, "Requirements for the annotation file of yolo format") as fp:
            cls._check_ann_file_impl(fp)

    @classmethod
    def _check_ann_file_impl(cls, fp: TextIOWrapper) -> bool:
        for line in fp:
            fields = line.rstrip("\n").split(" ")
            if len(fields) != 5:
                raise DatasetImportError(
                    f"Yolo format txt file should have 5 fields for each line, "
                    f"but the read line has {len(fields)} fields: fields={fields}."
                )

            for field in fields:
                if not field.replace(".", "").isdigit():
                    raise DatasetImportError(f"Each field should be a number but fields={fields}.")

            # Check the first line only
            return True

        raise DatasetImportError("Empty file is not allowed.")

    @classmethod
    def _find_loose(cls, path: str, dirname: str) -> List[Dict[str, Any]]:
        def _filter_ann_file(fpath: str):
            try:
                with open(fpath, "r") as fp:
                    return cls._check_ann_file_impl(fp)
            except DatasetImportError:
                return False

        sources = cls._find_sources_recursive(
            path,
            ext=".txt",
            extractor_name="",
            dirname=dirname,
            file_filter=_filter_ann_file,
            filename="**/*",
            max_depth=1,
            recursive=True,
        )
        if len(sources) == 0:
            return []

        subsets = defaultdict(list)

        for source in sources:
            subsets[extract_subset_name_from_parent(source["url"], path)].append(source["url"])

        sources = [
            {
                "url": osp.join(path),
                "format": cls.FORMAT,
                "options": {
                    "subset": subset,
                    "urls": urls,
                },
            }
            for subset, urls in subsets.items()
        ]
        return sources

    @classmethod
    def find_sources(cls, path: str) -> List[Dict[str, Any]]:
        # Check obj.names first
        filename, ext = osp.splitext(cls.META_FILE)
        obj_names_files = cls._find_sources_recursive(
            path,
            ext=ext,
            extractor_name="",
            dirname="",
            filename=filename,
            max_depth=1,
            recursive=False,
        )
        if len(obj_names_files) == 0:
            return []

        sources = []

        for obj_names_file in obj_names_files:
            base_path = osp.dirname(obj_names_file["url"])
            if found := cls._find_loose(base_path, "[Aa]nnotations"):
                sources += found

            if found := cls._find_loose(path, "[Ll]abels"):
                sources += found

        return sources

    @property
    def can_stream(self) -> bool:
        return True

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [".txt"]


class _YoloUltralyticsImporter(_YoloLooseImporter):
    META_FILE = YoloUltralyticsPath.META_FILE
    FORMAT = YoloFormatType.yolo_ultralytics.name

    @classmethod
    def _check_ann_file_impl(cls, fp: TextIOWrapper) -> bool:
        try:
            return _YoloLooseImporter._check_ann_file_impl(fp)
        except DatasetImportError as e:
            if e.args[0] == "Empty file is not allowed.":
                return True
            raise


class YoloImporter(Importer):
    SUB_IMPORTERS: Dict[YoloFormatType, Importer] = {
        YoloFormatType.yolo_strict: _YoloStrictImporter,
        YoloFormatType.yolo_loose: _YoloLooseImporter,
        YoloFormatType.yolo_ultralytics: _YoloUltralyticsImporter,
    }

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        with context.require_any():
            for importer_cls in cls.SUB_IMPORTERS.values():
                with context.alternative():
                    return importer_cls.detect(context)

        context.fail("Any yolo format is not detected.")

    @classmethod
    def find_sources(cls, path: str) -> List[Dict[str, Any]]:
        for importer_cls in cls.SUB_IMPORTERS.values():
            if sources := importer_cls.find_sources(path):
                return sources

        return []

    def get_extractor_merger(self) -> Optional[Type[ExtractorMerger]]:
        return ExtractorMerger

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return list({ext for importer in cls.SUB_IMPORTERS.values() for ext in importer.get_file_extensions()})
