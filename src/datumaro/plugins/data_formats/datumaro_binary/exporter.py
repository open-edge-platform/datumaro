# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=no-self-use

import os.path as osp
import struct
import warnings
from io import BufferedWriter
from multiprocessing.pool import ApplyResult, Pool
from typing import Any, List, Optional, Union

from datumaro.components.dataset_base import DatasetItem, IDataset
from datumaro.components.errors import DatumaroError, PathSeparatorInSubsetNameError
from datumaro.components.exporter import ExportContext, ExportContextComponent, Exporter
from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter
from datumaro.plugins.data_formats.datumaro.exporter import _SubsetWriter as __SubsetWriter
from datumaro.plugins.data_formats.datumaro.format import DATUMARO_FORMAT_VERSION

from .format import DatumaroBinaryPath
from .mapper import DictMapper
from .mapper.common import IntListMapper
from .mapper.dataset_item import DatasetItemMapper


class _SubsetWriter(__SubsetWriter):
    """"""

    def __init__(
        self,
        context: Exporter,
        subset: str,
        ann_file: str,
        export_context: ExportContextComponent,
        max_blob_size: int = DatumaroBinaryPath.MAX_BLOB_SIZE,
    ):
        super().__init__(context, subset, ann_file, export_context)

        self._fp: Optional[BufferedWriter] = None
        self._data["items"]: List[Union[bytes, ApplyResult]] = []
        self._bytes: List[Union[bytes, ApplyResult]] = self._data["items"]
        self._item_cnt = 0
        media_type = context._extractor.media_type()
        self._media_type = {"media_type": media_type._type}

        if max_blob_size != DatumaroBinaryPath.MAX_BLOB_SIZE:
            warnings.warn(
                f"You provide max_blob_size={max_blob_size}, "
                "but it is not recommended to provide an arbitrary max_blob_size."
            )

        self._max_blob_size = max_blob_size

    def _sign(self):
        self._fp.write(DatumaroBinaryPath.SIGNATURE.encode())

    def _dump_encryption_field(self) -> int:
        # Encrypted files are no longer supported in this version of Datumaro.
        msg = b""

        return self._fp.write(struct.pack(f"I{len(msg)}s", len(msg), msg))

    def _dump_header(self, header: Any):
        msg = DictMapper.forward(header)

        length = struct.pack("I", len(msg))
        return self._fp.write(length + msg)

    def _dump_version(self):
        self._dump_header(
            {
                "dm_format_version": DATUMARO_FORMAT_VERSION,
                "media_encryption": False,
            },
        )

    def _dump_info(self):
        self._dump_header(self.infos)

    def _dump_categories(self):
        self._dump_header(self.categories)

    def _dump_media_type(self):
        self._dump_header(self._media_type)

    def add_item(self, item: DatasetItem, pool: Optional[Pool] = None, *args, **kwargs):
        if pool is not None:
            self._bytes.append(
                pool.apply_async(
                    self.add_item_impl,
                    (
                        item,
                        self.export_context,
                    ),
                )
            )
        else:
            self._bytes.append(self.add_item_impl(item, self.export_context))

        self._item_cnt += 1

    @staticmethod
    def add_item_impl(
        item: DatasetItem,
        context: ExportContextComponent,
    ) -> bytes:
        with _SubsetWriter.context_save_media(item, context=context):
            return DatasetItemMapper.forward(item)

    def _dump_items(self, pool: Optional[Pool] = None):
        # Await async results
        if pool is not None:
            self._bytes = [
                result.get(timeout=DatumaroBinaryPath.MP_TIMEOUT)
                for result in self._bytes
                if isinstance(result, ApplyResult)
            ]

        # Divide items to blobs
        blobs = [bytearray()]
        cur_blob = blobs[-1]
        for _bytes in self._bytes:
            cur_blob += _bytes

            if len(cur_blob) > self._max_blob_size:
                blobs += [bytearray()]
                cur_blob = blobs[-1]

        # Encrypt blobs
        blobs = [bytes(blob) for blob in blobs if len(blob) > 0]

        # Dump blob sizes first
        blob_sizes = IntListMapper.forward([len(blob) for blob in blobs])
        n_blob_sizes = len(blob_sizes)
        self._fp.write(struct.pack(f"<I{n_blob_sizes}s", n_blob_sizes, blob_sizes))

        # Dump blobs
        for blob in blobs:
            items_bytes = blob
            n_items_bytes = len(items_bytes)
            self._fp.write(struct.pack(f"<{n_items_bytes}s", items_bytes))

    def write(self, pool: Optional[Pool] = None, *args, **kwargs):
        try:
            with open(self.ann_file, "wb") as fp:
                self._fp = fp
                self._sign()
                self._dump_version()
                self._dump_encryption_field()
                self._dump_info()
                self._dump_categories()
                self._dump_media_type()
                self._dump_items(pool)
        finally:
            self._fp = None


class DatumaroBinaryExporter(DatumaroExporter):
    DEFAULT_IMAGE_EXT = DatumaroBinaryPath.IMAGE_EXT
    PATH_CLS = DatumaroBinaryPath

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="The number of multi-processing workers for export. "
            "If num_workers = 0, do not use multiprocessing (default: %(default)s).",
        )

        return parser

    def __init__(
        self,
        extractor: IDataset,
        save_dir: str,
        *,
        save_media: Optional[bool] = None,
        image_ext: Optional[str] = None,
        default_image_ext: Optional[str] = None,
        save_dataset_meta: bool = False,
        ctx: Optional[ExportContext] = None,
        num_workers: int = 0,
        max_blob_size: int = DatumaroBinaryPath.MAX_BLOB_SIZE,
        **kwargs,
    ):
        """
        Parameters
        ----------
        num_workers
            The number of multi-processing workers for export. If num_workers = 0, do not use multiprocessing.
        max_blob_size
            The maximum size of DatasetItem serialization blob. Changing from the default is not recommended.
        """

        if num_workers < 0:
            raise DatumaroError(
                f"num_workers should be non-negative but num_workers={num_workers}."
            )
        self._num_workers = num_workers

        self._max_blob_size = max_blob_size

        super().__init__(
            extractor,
            save_dir,
            save_media=save_media,
            image_ext=image_ext,
            default_image_ext=default_image_ext,
            save_dataset_meta=save_dataset_meta,
            ctx=ctx,
        )

    def create_writer(
        self, subset: str, images_dir: str, pcd_dir: str, video_dir: str
    ) -> _SubsetWriter:
        export_context = ExportContextComponent(
            save_dir=self._save_dir,
            save_media=self._save_media,
            images_dir=images_dir,
            pcd_dir=pcd_dir,
            video_dir=video_dir,
            image_ext=self._image_ext,
            default_image_ext=self._default_image_ext,
        )

        if osp.sep in subset:
            raise PathSeparatorInSubsetNameError(subset)

        return _SubsetWriter(
            context=self,
            subset=subset,
            ann_file=osp.join(self._annotations_dir, subset + self.PATH_CLS.ANNOTATION_EXT),
            export_context=export_context,
            max_blob_size=self._max_blob_size,
        )

    def _apply_impl(self, *args, **kwargs):
        if self._num_workers == 0:
            return super()._apply_impl()

        with Pool(processes=self._num_workers) as pool:
            return super()._apply_impl(pool)
