# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import struct
from typing import Dict, Optional, Tuple

from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image, MediaElement, MediaType, PointCloud, Video, VideoFrame

from .common import Mapper, StringMapper


class MediaMapper(Mapper):
    @classmethod
    def forward(cls, obj: Optional[MediaElement]) -> bytes:
        if obj is None:
            return struct.pack("<I", MediaType.NONE)
        if obj._type == MediaType.IMAGE:
            return ImageMapper.forward(obj)
        if obj._type == MediaType.POINT_CLOUD:
            return PointCloudMapper.forward(obj)
        if obj._type == MediaType.VIDEO:
            return VideoMapper.forward(obj)
        if obj._type == MediaType.VIDEO_FRAME:
            return VideoFrameMapper.forward(obj)
        if obj._type == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.forward(obj)
        raise DatumaroError(f"{obj._type} is not allowed for MediaMapper.")

    @classmethod
    def backward(
        cls,
        _bytes: bytes,
        offset: int = 0,
        media_path_prefix: Optional[Dict[MediaType, str]] = None,
    ) -> Tuple[Optional[MediaElement], int]:
        (media_type,) = struct.unpack_from("<I", _bytes, offset)

        if media_type == MediaType.NONE:
            return None, offset + 4
        if media_type == MediaType.IMAGE:
            return ImageMapper.backward(_bytes, offset, media_path_prefix)
        if media_type == MediaType.POINT_CLOUD:
            return PointCloudMapper.backward(_bytes, offset, media_path_prefix)
        if media_type == MediaType.VIDEO:
            return VideoMapper.backward(_bytes, offset, media_path_prefix)
        if media_type == MediaType.VIDEO_FRAME:
            return VideoFrameMapper.backward(_bytes, offset, media_path_prefix)
        if media_type == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.backward(_bytes, offset, media_path_prefix)
        raise DatumaroError(f"{media_type} is not allowed for MediaMapper.")


class MediaElementMapper(Mapper):
    MAGIC_PATH = "/NOT/A/REAL/PATH"
    MEDIA_TYPE = MediaType.MEDIA_ELEMENT

    @classmethod
    def forward(cls, obj: MediaElement) -> bytes:
        bytes_arr = bytearray()
        bytes_arr.extend(struct.pack("<I", obj.type))
        path = getattr(obj, "path", cls.MAGIC_PATH)
        bytes_arr.extend(StringMapper.forward(path))

        return bytes(bytes_arr)

    @classmethod
    def backward_dict(
        cls,
        _bytes: bytes,
        offset: int = 0,
        media_path_prefix: Optional[Dict[MediaType, str]] = None,
    ) -> Tuple[Dict, int]:
        (media_type,) = struct.unpack_from("<I", _bytes, offset)
        assert media_type == cls.MEDIA_TYPE, f"Expect {cls.MEDIA_TYPE} but actual is {media_type}."
        offset += 4
        path, offset = StringMapper.backward(_bytes, offset)
        if path == cls.MAGIC_PATH:
            path = None
        return {
            "type": media_type,
            "path": path
            if path == cls.MAGIC_PATH or media_path_prefix is None
            else osp.join(media_path_prefix[cls.MEDIA_TYPE], path),
        }, offset

    @classmethod
    def backward(
        cls,
        _bytes: bytes,
        offset: int = 0,
        media_path_prefix: Optional[Dict[MediaType, str]] = None,
    ) -> Tuple[MediaElement, int]:
        _, offset = cls.backward_dict(_bytes, offset, media_path_prefix)
        return MediaElement(), offset


class ImageMapper(MediaElementMapper):
    MAGIC_SIZE_FOR_NONE = (-1583, -1597)
    MEDIA_TYPE = MediaType.IMAGE

    @classmethod
    def forward(cls, obj: Image) -> bytes:
        size = obj.size if obj.has_size else cls.MAGIC_SIZE_FOR_NONE

        bytes_arr = bytearray()
        bytes_arr.extend(super().forward(obj))
        bytes_arr.extend(struct.pack("<ii", size[0], size[1]))

        return bytes(bytes_arr)

    @classmethod
    def backward(
        cls,
        _bytes: bytes,
        offset: int = 0,
        media_path_prefix: Optional[Dict[MediaType, str]] = None,
    ) -> Tuple[Image, int]:
        media_dict, offset = cls.backward_dict(_bytes, offset, media_path_prefix)
        height, width = struct.unpack_from("<ii", _bytes, offset)
        size = (height, width)
        offset += 8
        return (
            Image.from_file(path=media_dict["path"], size=size if size != cls.MAGIC_SIZE_FOR_NONE else None),
            offset,
        )


class VideoMapper(MediaElementMapper):
    MAGIC_END_FRAME_FOR_NONE = 4294967295  # max value of unsigned int32
    MEDIA_TYPE = MediaType.VIDEO

    @classmethod
    def forward(cls, obj: Video) -> bytes:
        end_frame = obj._end_frame if obj._end_frame else cls.MAGIC_END_FRAME_FOR_NONE

        bytes_arr = bytearray()
        bytes_arr.extend(super().forward(obj))
        bytes_arr.extend(struct.pack("<III", obj._step, obj._start_frame, end_frame))

        return bytes(bytes_arr)

    @classmethod
    def backward(
        cls,
        _bytes: bytes,
        offset: int = 0,
        media_path_prefix: Optional[Dict[MediaType, str]] = None,
    ) -> Tuple[Video, int]:
        media_dict, offset = cls.backward_dict(_bytes, offset, media_path_prefix)
        step, start_frame, end_frame = struct.unpack_from("<III", _bytes, offset)
        offset += 12
        video = Video(
            path=media_dict["path"],
            step=step,
            start_frame=start_frame,
            end_frame=end_frame if end_frame != cls.MAGIC_END_FRAME_FOR_NONE else None,
        )
        return (video, offset)


class VideoFrameMapper(MediaElementMapper):
    MEDIA_TYPE = MediaType.VIDEO_FRAME

    @classmethod
    def forward(cls, obj: VideoFrame) -> bytes:
        bytes_arr = bytearray()
        bytes_arr.extend(super().forward(obj))
        bytes_arr.extend(struct.pack("<I", obj.index))

        return bytes(bytes_arr)

    @classmethod
    def backward(
        cls,
        _bytes: bytes,
        offset: int = 0,
        media_path_prefix: Optional[Dict[MediaType, str]] = None,
    ) -> Tuple[VideoFrame, int]:
        media_dict, offset = cls.backward_dict(_bytes, offset, media_path_prefix)
        (frame_index,) = struct.unpack_from("<I", _bytes, offset)
        video = Video(media_dict["path"])
        offset += 4
        return (
            VideoFrame(video, frame_index),
            offset,
        )


class PointCloudMapper(MediaElementMapper):
    MEDIA_TYPE = MediaType.POINT_CLOUD

    @classmethod
    def forward(cls, obj: PointCloud) -> bytes:
        bytes_arr = bytearray()
        bytes_arr.extend(super().forward(obj))
        bytes_arr.extend(struct.pack("<I", len(obj.extra_images)))
        for img in obj.extra_images:
            bytes_arr.extend(ImageMapper.forward(img))

        return bytes(bytes_arr)

    @classmethod
    def backward(
        cls,
        _bytes: bytes,
        offset: int = 0,
        media_path_prefix: Optional[Dict[MediaType, str]] = None,
    ) -> Tuple[PointCloud, int]:
        media_dict, offset = cls.backward_dict(_bytes, offset, media_path_prefix)
        (len_extra_images,) = struct.unpack_from("<I", _bytes, offset)
        offset += 4

        extra_images = []
        for _ in range(len_extra_images):
            img, offset = ImageMapper.backward(_bytes, offset, media_path_prefix)
            extra_images.append(img)

        return PointCloud.from_file(path=media_dict["path"], extra_images=extra_images), offset
