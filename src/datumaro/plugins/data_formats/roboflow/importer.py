# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import warnings
from collections import defaultdict
from glob import glob
from io import TextIOWrapper
from typing import Any, Dict, List, Type

from defusedxml import ElementTree

from datumaro.components.errors import DatasetImportError
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.merge.extractor_merger import ExtractorMerger


class RoboflowCocoImporter(Importer):
    FORMAT = "roboflow_coco"
    ANN_FILE_NAME = "_annotations.coco.json"

    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> FormatDetectionConfidence:
        context.require_file(osp.join("*", cls.ANN_FILE_NAME))
        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def find_sources(cls, path):
        subset_paths = glob(osp.join(path, "**", cls.ANN_FILE_NAME), recursive=True)

        sources = []
        for subset_path in subset_paths:
            subset_name = osp.basename(osp.dirname(subset_path))
            sources.append({"url": subset_path, "format": cls.FORMAT, "options": {"subset": subset_name}})

        return sources

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [osp.splitext(cls.ANN_FILE_NAME)[1]]

    @property
    def can_stream(self) -> bool:
        return True

    def get_extractor_merger(self) -> Type[ExtractorMerger]:
        return ExtractorMerger


class RoboflowVocImporter(Importer):
    FORMAT = "roboflow_voc"
    FORMAT_EXT = ".xml"
    ANN_DIR_NAME = ""

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        with context.require_any(), context.alternative():
            cls._check_ann_file(context.require_file("**/" + cls.ANN_DIR_NAME + "*" + cls.FORMAT_EXT), context)

        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def _check_ann_file(cls, fpath: str, context: FormatDetectionContext) -> None:
        with context.probe_text_file(fpath, "Requirements for the annotation file of voc format") as fp:
            cls._check_ann_file_impl(fp)

    @classmethod
    def _check_ann_file_impl(cls, fp: TextIOWrapper) -> bool:
        root = ElementTree.parse(fp).getroot()

        if root.tag != "annotation":
            raise DatasetImportError("Roboflow VOC format xml file should have the annotation tag.")

        if not root.find("source/database").text == "roboflow.ai":
            raise DatasetImportError("Roboflow VOC format xml file should have the source/database with `roboflow.ai`.")

        return True

    @classmethod
    def _get_sources(cls, path: str) -> Dict[Any, List[Any]]:
        def _filter_ann_file(fpath: str):
            try:
                with open(fpath, "r") as fp:
                    return cls._check_ann_file_impl(fp)
            except DatasetImportError:
                return False

        sources = cls._find_sources_recursive(
            path,
            ext=cls.FORMAT_EXT,
            extractor_name="",
            dirname=cls.ANN_DIR_NAME,
            file_filter=_filter_ann_file,
            filename="**/*",
            max_depth=2,
            recursive=True,
        )
        if len(sources) == 0:
            return []

        return sources

    @classmethod
    def find_sources(cls, path: str) -> List[Dict[str, Any]]:
        sources = cls._get_sources(path)

        subsets = {}
        for source in sources:
            subset_name = osp.dirname(source["url"]).split(os.sep)[-1]
            subsets[subset_name] = osp.dirname(source["url"])

        return [
            {
                "url": url,
                "format": cls.FORMAT,
                "options": {
                    "subset": subset,
                },
            }
            for subset, url in subsets.items()
        ]

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls.FORMAT_EXT]


class RoboflowYoloImporter(RoboflowVocImporter):
    FORMAT = "roboflow_yolo"
    FORMAT_EXT = ".txt"
    ANN_DIR_NAME = "labels/"

    @classmethod
    def _check_ann_file_impl(cls, fp: TextIOWrapper) -> bool:
        for line in fp:
            fields = line.rstrip("\n").split(" ")
            if len(fields) != 5:
                raise DatasetImportError(
                    f"Roboflow Yolo format txt file should have 5 fields for each line, "
                    f"but the read line has {len(fields)} fields: fields={fields}."
                )

            for field in fields:
                if not field.replace(".", "").isdigit():
                    raise DatasetImportError(f"Each field should be a number but fields={fields}.")

            # Check the first line only
            return True

        raise DatasetImportError("Empty file is not allowed.")

    @classmethod
    def find_sources(cls, path: str) -> List[Dict[str, Any]]:
        sources = cls._get_sources(path)

        subsets = defaultdict(list)
        for source in sources:
            subset_name = osp.dirname(source["url"]).split(os.sep)[-2]
            subsets[subset_name].append(source["url"])

        return [
            {
                "url": osp.dirname(osp.dirname(urls[0])),
                "format": cls.FORMAT,
                "options": {
                    "subset": subset,
                    "urls": urls,
                },
            }
            for subset, urls in subsets.items()
        ]


class RoboflowYoloObbImporter(RoboflowYoloImporter):
    FORMAT = "roboflow_yolo_obb"
    FORMAT_EXT = ".txt"
    ANN_DIR_NAME = "labelTxt/"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        warnings.warn(
            f"FormatDetectionConfidence of '{cls.FORMAT}' is lowered because of 'dota' format support. "
            f"It will be deprecated in datumaro==1.8.0.",
            DeprecationWarning,
        )
        with context.require_any(), context.alternative():
            cls._check_ann_file(context.require_file("**/" + cls.ANN_DIR_NAME + "*" + cls.FORMAT_EXT), context)

        return FormatDetectionConfidence.LOW

    @classmethod
    def _check_ann_file_impl(cls, fp: TextIOWrapper) -> bool:
        for line in fp:
            fields = line.rstrip("\n").split(" ")
            if len(fields) != 10:
                raise DatasetImportError(
                    f"Roboflow Yolo OBB format txt file should have 10 fields for each line, "
                    f"but the read line has {len(fields)} fields: fields={fields}."
                )

            # Check the first line only
            return True

        raise DatasetImportError("Empty file is not allowed.")


class RoboflowCreateMlImporter(RoboflowCocoImporter):
    FORMAT = "roboflow_create_ml"
    ANN_FILE_NAME = "_annotations.createml.json"


class RoboflowMulticlassImporter(RoboflowCocoImporter):
    FORMAT = "roboflow_multiclass"
    ANN_FILE_NAME = "_classes.csv"
