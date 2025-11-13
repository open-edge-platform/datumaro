# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Dict, Optional, Tuple

from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import MediaType
from datumaro.plugins.data_formats.datumaro_binary.mapper.annotation import AnnotationListMapper

from .common import DictMapper, Mapper, StringMapper
from .media import MediaMapper


class DatasetItemMapper(Mapper):
    @staticmethod
    def forward(obj: DatasetItem) -> bytes:
        bytes_arr = bytearray()
        bytes_arr.extend(StringMapper.forward(obj.id))
        bytes_arr.extend(StringMapper.forward(obj.subset))
        bytes_arr.extend(MediaMapper.forward(obj.media))
        bytes_arr.extend(DictMapper.forward(obj.attributes))
        bytes_arr.extend(AnnotationListMapper.forward(obj.annotations))
        return bytes(bytes_arr)

    @staticmethod
    def backward(
        _bytes: bytes, offset: int = 0, media_path_prefix: Optional[Dict[MediaType, str]] = None
    ) -> Tuple[DatasetItem, int]:
        id, offset = StringMapper.backward(_bytes, offset)
        subset, offset = StringMapper.backward(_bytes, offset)
        media, offset = MediaMapper.backward(_bytes, offset, media_path_prefix)
        attributes, offset = DictMapper.backward(_bytes, offset)
        annotations, offset = AnnotationListMapper.backward(_bytes, offset)
        return (
            DatasetItem(id=id, subset=subset, media=media, attributes=attributes, annotations=annotations),
            offset,
        )
