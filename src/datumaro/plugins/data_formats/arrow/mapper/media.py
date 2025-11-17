# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pyarrow as pa

from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image, MediaElement, MediaType, PointCloud
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import Mapper
from datumaro.util.image import decode_image, encode_image, load_image


class MediaMapper(Mapper):
    @classmethod
    def forward(cls, obj: Optional[MediaElement], **options) -> Dict[str, Any]:
        if obj is None:
            return {"type": int(MediaType.NONE)}
        if obj._type == MediaType.IMAGE:
            return ImageMapper.forward(obj, **options)
        if obj._type == MediaType.POINT_CLOUD:
            return PointCloudMapper.forward(obj, **options)
        if obj._type == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.forward(obj)
        raise DatumaroError(f"{obj._type} is not allowed for MediaMapper.")

    @classmethod
    def backward(
        cls,
        media_struct: pa.StructScalar,
        idx: int,
        table: pa.Table,
        table_path: str,
    ) -> Optional[MediaElement]:
        media_type = MediaType(media_struct.get("type").as_py())

        if media_type == MediaType.NONE:
            return None
        if media_type == MediaType.IMAGE:
            return ImageMapper.backward(media_struct, idx, table, table_path)
        if media_type == MediaType.POINT_CLOUD:
            return PointCloudMapper.backward(media_struct, idx, table, table_path)
        if media_type == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.backward(media_struct, idx, table, table_path)
        raise DatumaroError(f"{media_type} is not allowed for MediaMapper.")


class MediaElementMapper(Mapper):
    MAGIC_PATH = "/NOT/A/REAL/PATH"
    MEDIA_TYPE = MediaType.MEDIA_ELEMENT

    @classmethod
    def forward(cls, obj: MediaElement) -> Dict[str, Any]:
        return {"type": int(obj.type)}

    @classmethod
    def backward(
        cls,
        media_dict: Dict[str, Any],
        idx: int,
        table: pa.Table,
        table_path: str,
    ) -> MediaElement:
        return MediaElement()


class ImageMapper(MediaElementMapper):
    MEDIA_TYPE = MediaType.IMAGE
    AVAILABLE_SCHEMES = ("AS-IS", "PNG", "TIFF", "JPEG/95", "JPEG/75", "NONE")

    @classmethod
    def encode(cls, obj: Image, scheme: str = "PNG") -> Optional[bytes]:
        if scheme is None or scheme == "NONE":
            return None
        if scheme == "AS-IS":
            _bytes = obj.bytes
            if _bytes is not None:
                return _bytes
            # try to encode in PNG
            scheme = "PNG"

        options = {}
        if scheme.startswith("JPEG"):
            quality = int(scheme.split("/")[-1])
            options["ext"] = "JPEG"
            options["jpeg_quality"] = quality
        elif scheme == "PNG":
            options["ext"] = "PNG"
        elif scheme == "TIFF":
            options["ext"] = "TIFF"
        else:
            raise NotImplementedError

        data = obj.data
        if data is not None:
            return encode_image(obj.data, **options)
        return None

    @classmethod
    def decode(cls, path: Optional[str] = None, data: Optional[bytes] = None) -> Optional[np.ndarray]:
        if path is None and data is None:
            return None
        if data is not None:
            return decode_image(data, np.uint8)
        if path is not None:
            return load_image(path, np.uint8)

    @classmethod
    def forward(cls, obj: Image, encoder: Union[str, Callable[[Image], bytes]] = "PNG") -> Dict[str, Any]:
        out = super().forward(obj)

        _bytes = encoder(obj) if isinstance(encoder, Callable) else cls.encode(obj, scheme=encoder)

        path = None if _bytes is not None else getattr(obj, "path", None)

        out["image"] = {
            "has_bytes": _bytes is not None,
            "bytes": _bytes,
            "path": path,
            "size": obj.size,
        }
        return out

    @classmethod
    def backward(
        cls,
        media_struct: pa.StructScalar,
        idx: int,
        table: pa.Table,
        table_path: str,
    ) -> Image:
        image_struct = media_struct.get("image")

        if path := image_struct.get("path").as_py():
            return Image.from_file(
                path=path,
                size=image_struct.get("size").as_py(),
            )

        return Image.from_bytes(
            data=lambda: pa.ipc.open_file(pa.memory_map(table_path, "r"))
            .read_all()
            .column("media")[idx]
            .get("image")
            .get("bytes")
            .as_py(),
            size=image_struct.get("size").as_py(),
        )

    @classmethod
    def backward_extra_image(
        cls, image_struct: pa.StructScalar, idx: int, table: pa.Table, extra_image_idx: int
    ) -> Image:
        if path := image_struct.get("path").as_py():
            return Image.from_file(
                path=path,
                size=image_struct.get("size").as_py(),
            )

        return Image.from_bytes(
            data=lambda: table.column("media")[idx]
            .get("point_cloud")
            .get("extra_images")[extra_image_idx]
            .get("bytes")
            .as_py(),
            size=image_struct.get("size").as_py(),
        )


# TODO: share binary for extra images
class PointCloudMapper(MediaElementMapper):
    MEDIA_TYPE = MediaType.POINT_CLOUD
    B64_PREFIX = "//B64_ENCODED//"

    @classmethod
    def forward(cls, obj: PointCloud, encoder: Union[str, Callable[[PointCloud], bytes]] = "PNG") -> Dict[str, Any]:
        out = super().forward(obj)

        if isinstance(encoder, Callable):
            _bytes = encoder(obj)
        elif encoder != "NONE":
            _bytes = obj.data
        else:
            _bytes = None

        path = None if _bytes is not None else getattr(obj, "path", None)

        out["point_cloud"] = {
            "has_bytes": _bytes is not None,
            "bytes": _bytes,
            "path": path,
            "extra_images": [ImageMapper.forward(img, encoder=encoder)["image"] for img in obj.extra_images],
        }

        return out

    @classmethod
    def backward(
        cls,
        media_struct: pa.StructScalar,
        idx: int,
        table: pa.Table,
        table_path: str,
    ) -> PointCloud:
        point_cloud_struct = media_struct.get("point_cloud")

        extra_images = [
            ImageMapper.backward_extra_image(image_struct, idx, table, extra_image_idx)
            for extra_image_idx, image_struct in enumerate(point_cloud_struct.get("extra_images"))
        ]

        if path := point_cloud_struct.get("path").as_py():
            return PointCloud.from_file(path=path, extra_images=extra_images)

        return PointCloud.from_bytes(
            data=pa.ipc.open_file(pa.memory_map(table_path, "r"))
            .read_all()
            .column("media")[idx]
            .get("point_cloud")
            .get("bytes")
            .as_py(),
            extra_images=extra_images,
        )
