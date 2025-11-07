# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import struct
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Type

import pyarrow as pa

from datumaro.components.annotation import AnnotationType, Categories
from datumaro.components.dataset_base import CategoriesInfo, DatasetBase, DatasetInfo, DatasetItem, IDataset, SubsetBase
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image, MediaElement, MediaType
from datumaro.components.merge.extractor_merger import check_identicalness
from datumaro.plugins.data_formats.arrow.format import DatumaroArrow
from datumaro.plugins.data_formats.datumaro.base import JsonReader
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper
from datumaro.util.definitions import DEFAULT_SUBSET_NAME

from .mapper.dataset_item import DatasetItemMapper


class ArrowSubsetBase(SubsetBase):
    __not_plugin__ = True

    def __init__(
        self,
        lookup: Dict[str, DatasetItem],
        infos: Dict[str, Any],
        categories: Dict[AnnotationType, Categories],
        subset: str,
        media_type: Type[MediaElement] = Image,
        ann_types: Set[AnnotationType] = None,
    ):
        super().__init__(length=len(lookup), subset=subset, media_type=media_type, ann_types=ann_types, ctx=None)

        self._lookup = lookup
        self._infos = infos
        self._categories = categories

    def __iter__(self) -> Iterator[DatasetItem]:
        for item in self._lookup.values():
            yield item

    def __len__(self) -> int:
        return len(self._lookup)

    def get(self, item_id: str, subset: Optional[str] = None) -> Optional[DatasetItem]:
        if subset != self._subset:
            return None

        try:
            return self._lookup[item_id]
        except KeyError:
            return None


@dataclass(frozen=True)
class Metadata:
    infos: Dict
    categories: Dict
    media_type: Type[MediaElement]


class ArrowBase(DatasetBase):
    def __init__(
        self,
        root_path: str,
        *,
        file_paths: List[str],
        ctx: Optional[ImportContext] = None,
    ):
        self._root_path = root_path
        tables = [pa.ipc.open_file(pa.memory_map(path, "r")).read_all() for path in file_paths]
        metadatas = [self._load_schema_metadata(table) for table in tables]

        table = pa.concat_tables(tables)
        subsets = table.column(DatumaroArrow.SUBSET_FIELD).unique().to_pylist()
        media_type = check_identicalness([metadata.media_type for metadata in metadatas])

        super().__init__(length=len(table), subsets=subsets, media_type=media_type, ctx=ctx)

        self._infos = check_identicalness([metadata.infos for metadata in metadatas])
        self._categories = check_identicalness([metadata.categories for metadata in metadatas])

        self._init_cache(file_paths, subsets)

    @staticmethod
    def _load_schema_metadata(table: pa.Table) -> Metadata:
        schema = table.schema

        _infos, _ = DictMapper.backward(schema.metadata.get(b"infos", b"\x00\x00\x00\x00"))
        infos = JsonReader._load_infos({"infos": _infos})

        _categories, _ = DictMapper.backward(schema.metadata.get(b"categories", b"\x00\x00\x00\x00"))
        categories = JsonReader._load_categories({"categories": _categories})

        (media_type,) = struct.unpack("<I", schema.metadata.get(b"media_type", b"\x00\x00\x00\x00"))
        media_type = MediaType(media_type).media

        return Metadata(infos=infos, categories=categories, media_type=media_type)

    def infos(self) -> DatasetInfo:
        return self._infos

    def categories(self) -> CategoriesInfo:
        return self._categories

    def __iter__(self) -> Iterator[DatasetItem]:
        for lookup in self._lookup.values():
            for item in lookup.values():
                yield item

    def _init_cache(self, file_paths: List[str], subsets: List[str]):
        self._lookup: Dict[str, Dict[str, DatasetItem]] = {subset: {} for subset in subsets}

        total = len(self)
        cnt = 0
        pbar = self._ctx.progress_reporter
        pbar.start(total=total, desc="Importing")

        ann_types = set()
        for table_path in file_paths:
            with pa.OSFile(table_path, "r") as source, pa.ipc.open_file(source) as reader:
                table = reader.read_all()
                for idx in range(len(table)):
                    item = DatasetItemMapper.backward(idx, table, table_path)
                    self._lookup[item.subset][item.id] = item
                    for ann in item.annotations:
                        ann_types.add(ann.type)
                    pbar.report_status(cnt)
                    cnt += 1
        self._ann_types = ann_types

        self._subsets = {
            subset: ArrowSubsetBase(
                lookup=lookup,
                infos=self._infos,
                categories=self._categories,
                subset=self._subsets,
                media_type=self._media_type,
                ann_types=self._ann_types,
            )
            for subset, lookup in self._lookup.items()
        }

        pbar.finish()

    def get(self, item_id: str, subset: Optional[str] = None) -> Optional[DatasetItem]:
        subset = subset or DEFAULT_SUBSET_NAME

        try:
            return self._lookup[subset][item_id]
        except KeyError:
            return None

    @property
    def lookup(self) -> Dict[str, Dict[str, int]]:
        return self._lookup

    def subsets(self) -> Dict[str, IDataset]:
        return self._subsets

    def get_subset(self, name: str) -> IDataset:
        return self._subsets[name]
