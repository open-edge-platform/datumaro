# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from typing import Dict, List, Optional

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.errors import DatasetImportError
from datumaro.rust_api import JsonSectionPageMapper
from datumaro.util import parse_json


class SegmentAnythingImporter(Importer):
    _N_JSON_TO_TEST = 9
    _MAX_ANNOTATION_SECTION_BYTES = 100 * 1024 * 1024  # 100 MiB
    _ANNO_EXT = ".json"

    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> Optional[FormatDetectionConfidence]:
        # test maximum 10 annotation files only
        for ctr, file in enumerate(context.require_files_iter(f"*{cls._ANNO_EXT}")):
            with context.probe_text_file(
                file, "Annotation format is not Segmentat-Anything format", is_binary_file=True
            ) as f:
                fpath = os.path.join(context.root_path, file)
                page_mapper = JsonSectionPageMapper(fpath)
                sections = page_mapper.sections()

                if set(sections.keys()) != {"annotations", "image"}:
                    raise DatasetImportError

                offset, size = sections["image"]["offset"], sections["image"]["size"]
                f.seek(offset, 0)
                img_contents = parse_json(f.read(size))

                if set(img_contents.keys()) != {
                    "image_id",
                    "width",
                    "height",
                    "file_name",
                }:
                    raise DatasetImportError

                offset, size = sections["annotations"]["offset"], sections["annotations"]["size"]

                if size > cls._MAX_ANNOTATION_SECTION_BYTES:
                    msg = f"Annotation section is too huge. It exceeded {cls._MAX_ANNOTATION_SECTION_BYTES} bytes."
                    raise DatasetImportError(msg)

                f.seek(offset, 0)
                ann_contents = parse_json(f.read(size))

                if not {"id", "segmentation", "bbox"}.issubset(set(ann_contents[0])):
                    raise DatasetImportError

            if ctr > cls._N_JSON_TO_TEST:
                break

    @classmethod
    def find_sources(cls, path) -> List[Dict]:
        if not os.path.isdir(path):
            return []
        return [{"url": path, "format": cls.NAME}]

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]
