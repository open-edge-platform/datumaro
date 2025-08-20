# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import struct
from io import BufferedReader
from multiprocessing.pool import AsyncResult, Pool
from typing import Any, Dict, List, Optional

from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatasetImportError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image, MediaElement, MediaType, PointCloud, Video, VideoFrame
from datumaro.plugins.data_formats.datumaro_binary.format import DatumaroBinaryPath
from datumaro.plugins.data_formats.datumaro_binary.mapper import DictMapper
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import IntListMapper
from datumaro.plugins.data_formats.datumaro_binary.mapper.dataset_item import DatasetItemMapper

from ..datumaro.base import DatumaroBase, JsonReader


class DatumaroBinaryBase(DatumaroBase):
    """"""

    def __init__(
        self,
        path: str,
        *,
        num_workers: int = 0,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        """
        Parameters
        ----------
        path
            Directory path to import DatumaroBinary format dataset
        num_workers
            The number of multi-processing workers for import. If num_workers = 0, do not use multiprocessing.
        """
        self._fp: Optional[BufferedReader] = None
        self._num_workers = num_workers
        super().__init__(path, subset=subset, ctx=ctx)

    def _get_dm_format_version(self, path: str) -> str:
        with open(path, "rb") as fp:
            self._fp = fp
            self._check_signature()
            dm_format_version = self._read_version()
        return dm_format_version

    def _load_impl(self, path: str) -> None:
        """Actual implementation of loading Datumaro binary format."""
        try:
            with open(path, "rb") as fp:
                self._fp = fp
                self._check_signature()
                self._read_version()
                self._check_encryption_field()
                self._read_info()
                self._read_categories()
                self._read_media_type()
                self._read_items()
        finally:
            self._fp = None

    def _check_signature(self):
        signature = self._fp.read(DatumaroBinaryPath.SIGNATURE_LEN).decode()
        DatumaroBinaryPath.check_signature(signature)

    def _check_encryption_field(self):
        len_byte = self._fp.read(4)
        _bytes = self._fp.read(struct.unpack("I", len_byte)[0])

        if _bytes != b"":
            raise DatasetImportError(
                "Encrypted files are no longer supported since Datumaro 1.12.0."
            )

    def _read_header(self):
        len_byte = self._fp.read(4)
        _bytes = self._fp.read(struct.unpack("I", len_byte)[0])
        header, _ = DictMapper.backward(_bytes)
        return header

    def _read_version(self) -> Dict[str, Any]:
        version_header = self._read_header()
        return version_header["dm_format_version"]

    def _read_info(self):
        self._infos = self._read_header()

    def _read_categories(self):
        categories = self._read_header()
        self._categories = JsonReader._load_categories({"categories": categories})

    def _read_media_type(self):
        media_type = self._read_header()["media_type"]
        if media_type == MediaType.IMAGE:
            self._media_type = Image
        elif media_type == MediaType.POINT_CLOUD:
            self._media_type = PointCloud
        elif media_type == MediaType.VIDEO:
            self._media_type = Video
        elif media_type == MediaType.VIDEO_FRAME:
            self._media_type = VideoFrame
        elif media_type == MediaType.MEDIA_ELEMENT:
            self._media_type = MediaElement
        else:
            raise NotImplementedError(f"media_type={media_type} is currently not supported.")

    def _read_items(self) -> None:
        (n_blob_sizes_bytes,) = struct.unpack("<I", self._fp.read(4))
        blob_sizes_bytes = self._fp.read(n_blob_sizes_bytes)
        blob_sizes, _ = IntListMapper.backward(blob_sizes_bytes, 0)

        media_path_prefix = {
            MediaType.IMAGE: osp.join(self._images_dir, self._subset),
            MediaType.POINT_CLOUD: osp.join(self._pcd_dir, self._subset),
            MediaType.VIDEO: osp.join(self._video_dir, self._subset),
            MediaType.VIDEO_FRAME: osp.join(self._video_dir, self._subset),
        }

        if self._num_workers > 0:
            self._items = self._read_items_mp(blob_sizes, media_path_prefix)
        else:
            self._items = self._read_items_sp(blob_sizes, media_path_prefix)

        for item in self._items:
            for ann in item.annotations:
                self._ann_types.add(ann.type)

    def _read_items_mp(
        self, blob_sizes: List[int], media_path_prefix: Dict[MediaType, str]
    ) -> List[DatasetItem]:
        async_results: List[AsyncResult] = []

        with Pool(processes=self._num_workers) as pool:
            for blob_size in blob_sizes:
                blob_bytes = self._fp.read(blob_size)
                async_results += [
                    pool.apply_async(
                        self._read_blob,
                        (
                            blob_bytes,
                            media_path_prefix,
                        ),
                    )
                ]

            return [
                item
                for async_result in async_results
                for item in async_result.get(timeout=DatumaroBinaryPath.MP_TIMEOUT)
            ]

    def _read_items_sp(
        self, blob_sizes: List[int], media_path_prefix: Dict[MediaType, str]
    ) -> List[DatasetItem]:
        items_list = [
            self._read_blob(self._fp.read(blob_size), media_path_prefix) for blob_size in blob_sizes
        ]

        return [item for items in items_list for item in items]

    @staticmethod
    def _read_blob(blob_bytes: bytes, media_path_prefix: Dict[MediaType, str]) -> List[DatasetItem]:
        items = []
        offset = 0

        # Extract items
        while offset < len(blob_bytes):
            item, offset = DatasetItemMapper.backward(blob_bytes, offset, media_path_prefix)
            items.append(item)

        assert offset == len(blob_bytes)

        return items

    @property
    def is_stream(self) -> bool:
        return False

    def infos(self):
        return self._infos

    def categories(self):
        return self._categories

    def media_type(self):
        return self._media_type

    def ann_types(self):
        return self._ann_types

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> DatasetItem:
        yield from self._items
