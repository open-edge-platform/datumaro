# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
from pathlib import Path
from typing import List, Optional

from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.exporter import Exporter
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util.image import IMAGE_EXTENSIONS, find_images


class ImageDirImporter(Importer):
    """
    Reads images from a directory as a dataset.
    """

    DETECT_CONFIDENCE = FormatDetectionConfidence.EXTREME_LOW

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--subset",
            help="The name of the subset for the produced dataset items (default: none)",
        )
        return parser

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        path = Path(context.root_path)
        for item in path.iterdir():
            if item.is_dir():
                context.fail("Only flat image directories are supported")
            elif item.suffix.lower() not in IMAGE_EXTENSIONS:
                context.fail(f"File {item} is not an image.")
        return super().detect(context)

    @classmethod
    def find_sources(cls, path):
        path = Path(path)
        if not path.is_dir():
            return []

        return [{"url": str(path), "format": ImageDirBase.NAME}]

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return list(IMAGE_EXTENSIONS)


class ImageDirBase(SubsetBase):
    def __init__(
        self,
        url: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(subset=subset, ctx=ctx)
        url = Path(url)
        assert url.is_dir(), url

        for path in find_images(str(url)):
            item_id = Path(path).stem
            self._items.append(DatasetItem(id=item_id, subset=self._subset, media=Image.from_file(path=path)))
        self._ann_types = set()

    @property
    def is_stream(self) -> bool:
        return True


class ImageDirExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".jpg"

    def _apply_impl(self):
        os.makedirs(self._save_dir, exist_ok=True)

        for item in self._extractor:
            if item.media:
                self._save_image(item)
            else:
                log.debug("Item '%s' has no image info", item.id)
