# Copyright (C) 2021-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from enum import Enum
from typing import List, Optional
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, ZIP_STORED, ZipFile

from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.exporter import Exporter
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util import parse_str_enum_value
from datumaro.util.image import IMAGE_EXTENSIONS, encode_image


class Compression(Enum):
    ZIP_STORED = ZIP_STORED
    ZIP_DEFLATED = ZIP_DEFLATED
    ZIP_BZIP2 = ZIP_BZIP2
    ZIP_LZMA = ZIP_LZMA


class ImageZipPath:
    DEFAULT_ARCHIVE_NAME = "default.zip"
    DEFAULT_COMPRESSION = Compression.ZIP_STORED


class ImageZipBase(SubsetBase):
    def __init__(
        self,
        url: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(subset=subset, media_type=Image, ctx=ctx)

        assert url.endswith(".zip"), url

        with ZipFile(url, "r") as zf:
            for path in zf.filelist:
                item_id, extension = osp.splitext(path.filename)
                if extension.lower() not in IMAGE_EXTENSIONS:
                    continue
                image = Image.from_bytes(data=zf.read(path.filename))
                self._items.append(DatasetItem(id=item_id, media=image, subset=self._subset))

        self._ann_types = set()


class ImageZipImporter(Importer):
    _FORMAT_EXT = ".zip"

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, cls._FORMAT_EXT, "image_zip")

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._FORMAT_EXT]


class ImageZipExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".jpg"

    @staticmethod
    def _get_compression_method(s):
        try:
            return Compression[s.upper()]
        except KeyError:
            import argparse

            raise argparse.ArgumentTypeError()

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument(
            "--name",
            type=str,
            default=ImageZipPath.DEFAULT_ARCHIVE_NAME,
            help="Name of output zipfile (default: %(default)s)",
        )

        parser.add_argument(
            "--compression",
            type=cls._get_compression_method,
            default=ImageZipPath.DEFAULT_COMPRESSION.name,
            help="Archive compression method.\nAvailable methods: {} (default: %(default)s)".format(
                ", ".join(e.name for e in Compression)
            ),
        )

        return parser

    def __init__(self, extractor, save_dir, name=None, compression=None, **kwargs):
        super().__init__(extractor, save_dir, **kwargs)

        if name is None:
            name = ImageZipPath.DEFAULT_ARCHIVE_NAME

        compression = parse_str_enum_value(compression, Compression, default=ImageZipPath.DEFAULT_COMPRESSION)

        self._archive_name = name
        self._compression = compression.value

    def _apply_impl(self):
        os.makedirs(self._save_dir, exist_ok=True)

        archive_path = osp.join(self._save_dir, self._archive_name)

        if osp.exists(archive_path):
            raise FileExistsError(
                "Zip file: %s, already exist, specify archive name with --name extra argument" % archive_path
            )

        with ZipFile(archive_path, "w", self._compression) as zf:
            for item in self._extractor:
                if item.media:
                    self._archive_image(zf, item)
                else:
                    log.debug("Item '%s' has no image info", item.id)

    def _archive_image(self, zipfile, item):
        image_name = self._make_image_filename(item)
        path = getattr(item.media, "path", None)
        if path is not None and osp.isfile(path):
            zipfile.write(item.media.path, arcname=image_name)
        elif item.media.has_data:
            zipfile.writestr(image_name, encode_image(item.media.data, osp.splitext(image_name)[1]))
