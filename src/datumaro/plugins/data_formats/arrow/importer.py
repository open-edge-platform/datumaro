# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from typing import Dict, List, Optional

import pyarrow as pa

from datumaro.components.errors import DatasetImportError
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer

from .format import DatumaroArrow

__all__ = ["ArrowImporter"]


class ArrowImporter(Importer):
    _FORMAT_EXT = ".arrow"

    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> Optional[FormatDetectionConfidence]:
        if context.root_path.endswith(".arrow"):
            cls._verify_datumaro_arrow_format(context.root_path)
        else:
            for arrow_file in context.require_files("*.arrow"):
                with context.probe_text_file(
                    arrow_file,
                    f"{arrow_file} is not Datumaro arrow format.",
                    is_binary_file=True,
                ) as f:
                    f.close()
                    cls._verify_datumaro_arrow_format(os.path.join(context.root_path, arrow_file))

    @classmethod
    def find_sources(cls, path: str) -> List[Dict]:
        def _filter(path: str) -> bool:
            try:
                cls._verify_datumaro_arrow_format(path)
                return True
            except DatasetImportError:
                return False

        return cls._find_sources_recursive(
            path=path,
            ext=cls._FORMAT_EXT,
            extractor_name=cls.NAME,
            file_filter=_filter,
            max_depth=0,
        )

    @classmethod
    def find_sources_with_params(cls, path: str, **extra_params) -> List[Dict]:
        sources = cls.find_sources(path)
        # Merge sources into one config but multiple file_paths
        return [
            {
                "url": path,
                "format": cls.NAME,
                "options": {"file_paths": [source["url"] for source in sources]},
            }
        ]

    @staticmethod
    def _verify_datumaro_arrow_format(file: str) -> None:
        with pa.memory_map(file, "r") as mm_file, pa.ipc.open_file(mm_file) as reader:
            schema = reader.schema
        DatumaroArrow.check_signature(schema.metadata.get(b"signature", b"").decode())
        DatumaroArrow.check_version(schema.metadata.get(b"version", b"").decode())
        DatumaroArrow.check_schema(schema)

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._FORMAT_EXT]
