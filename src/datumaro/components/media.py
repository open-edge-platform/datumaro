# Copyright (C) 2021-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import errno
import io
import os
import os.path as osp
import shutil
from copy import deepcopy
from enum import IntEnum
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import cv2
import imagesize
import numpy as np

from datumaro.components.errors import DatumaroError, MediaShapeError
from datumaro.util.definitions import BboxIntCoords
from datumaro.util.image import (
    _image_loading_errors,
    copyto_image,
    decode_image,
    lazy_image,
    load_image,
    save_image,
)

if TYPE_CHECKING:
    import pandas as pd
else:
    from datumaro.util.import_util import lazy_import

    pd = lazy_import("pandas")


AnyData = TypeVar("AnyData", bytes, np.ndarray)


class MediaType(IntEnum):
    NONE = 0
    MEDIA_ELEMENT = 1
    IMAGE = 2
    BYTE_IMAGE = 3
    VIDEO_FRAME = 4
    VIDEO = 5
    POINT_CLOUD = 6
    MULTIFRAME_IMAGE = 7
    ROI_IMAGE = 8
    MOSAIC_IMAGE = 9
    TABLE_ROW = 10

    @property
    def media(self) -> Optional[Type[MediaElement]]:
        if self == self.NONE:
            return None
        if self == self.MEDIA_ELEMENT:
            return MediaElement
        if self == self.IMAGE:
            return Image
        if self == self.VIDEO_FRAME:
            return VideoFrame
        if self == self.VIDEO:
            return Video
        if self == self.POINT_CLOUD:
            return PointCloud
        if self == self.MULTIFRAME_IMAGE:
            return MultiframeImage
        if self == self.ROI_IMAGE:
            return RoIImage
        if self == self.MOSAIC_IMAGE:
            return MosaicImage
        if self == self.TABLE_ROW:
            return TableRow
        raise NotImplementedError


class MediaElement(Generic[AnyData]):
    _type = MediaType.MEDIA_ELEMENT

    def __init__(self, *args, **kwargs) -> None:
        pass

    def as_dict(self) -> Dict[str, Any]:
        # NOTE:
        # attributes starting with a single underscore are assumed
        # to be arguments of __init__ method and
        # attributes starting with double underscores are assuemd
        # to be not directly related to __init__ method.
        return {
            key[1:]: value
            for key, value in self.__dict__.items()
            if key.startswith("_") and not key.startswith(f"_{self.__class__.__name__}")
        }

    def from_self(self, **kwargs):
        attrs = deepcopy(self.as_dict())
        attrs.update(kwargs)
        return self.__class__(**attrs)

    @property
    def type(self) -> MediaType:
        return self._type

    @property
    def data(self) -> Optional[AnyData]:
        return None

    @property
    def has_data(self) -> bool:
        return False

    @property
    def bytes(self) -> Optional[bytes]:
        return None

    def __eq__(self, other: object) -> bool:
        other_type = getattr(other, "type", None)
        if self.type != other_type:
            return False
        return True

    def save(
        self,
        fp: Union[str, io.IOBase],
    ):
        raise NotImplementedError


class FromFileMixin:
    def __init__(self, path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert path, "Path can't be empty"
        self._path = path

    @property
    def path(self) -> str:
        """Path to the media file"""
        # TODO: do we need this replace?
        return self._path.replace("\\", "/")

    @property
    def bytes(self) -> Optional[bytes]:
        if self.has_data:
            with open(self._path, "rb") as f:
                _bytes = f.read()
            return _bytes
        return None

    @property
    def has_data(self) -> bool:
        return os.path.exists(self.path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={repr(self._path)})"


class FromDataMixin(Generic[AnyData]):
    def __init__(self, data: Union[Callable[[], AnyData], AnyData], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = data

    @property
    def data(self) -> Optional[AnyData]:
        if callable(self._data):
            return self._data()
        return self._data

    @property
    def bytes(self) -> Optional[bytes]:
        if self.has_data:
            _bytes = self._data() if callable(self._data) else self._data
            if isinstance(_bytes, bytes):
                return _bytes
        return None

    @property
    def has_data(self) -> bool:
        return self._data is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data=" + repr(self._data)[:20].replace("\n", "") + "...)"


class Image(MediaElement[np.ndarray]):
    _type = MediaType.IMAGE

    _DEFAULT_EXT = ".png"

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        ext: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        assert self.__class__ != Image, (
            f"Directly initalizing {self.__class__.__name__} is not supported. "
            f"Please use one of fractory functions ({self.__class__.__name__}.from_file(), "
            f"{self.__class__.__name__}.from_numpy(), {self.__class__.__name__}.from_bytes())."
        )
        super().__init__(*args, **kwargs)
        self._dtype = np.uint8

        if ext is not None:
            if not ext.startswith("."):
                ext = "." + ext
            ext = ext.lower()
        self._ext = ext

        if size is not None:
            assert (
                len(size) == 2 and 0 < size[0] and 0 < size[1]
            ), f"Invalid image size info '{size}'"
            size = tuple(map(int, size))
        self._size = size  # (H, W)

    @classmethod
    def from_file(cls, path: str, *args, **kwargs):
        return ImageFromFile(path, *args, **kwargs)

    @classmethod
    def from_numpy(
        cls,
        data: Union[np.ndarray, Callable[[], np.ndarray]],
        *args,
        **kwargs,
    ):
        return ImageFromNumpy(data, *args, **kwargs)

    @classmethod
    def from_bytes(
        cls,
        data: Union[bytes, Callable[[], bytes]],
        *args,
        **kwargs,
    ):
        return ImageFromBytes(data, *args, **kwargs)

    @property
    def has_size(self) -> bool:
        """Indicates that size info is cached and won't require image loading"""
        return self._size is not None

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        """Returns (H, W)"""

        if self._size is None:
            try:
                data = self.data
            except _image_loading_errors:
                return None
            if data is not None:
                self._size = tuple(map(int, data.shape[:2]))
        return self._size

    @property
    def ext(self) -> Optional[str]:
        """Media file extension (with the leading dot)"""
        return self._ext

    def _get_ext_to_save(self, fp: Union[str, io.IOBase], ext: Optional[str] = None):
        if isinstance(fp, str):
            assert ext is None, "'ext' must be empty if string is given."
            ext = osp.splitext(osp.basename(fp))[1].lower()
        else:
            ext = ext if ext else self._DEFAULT_EXT
        return ext

    def __eq__(self, other):
        # Do not compare `_type`
        # sicne Image is subclass of RoIImage and MosaicImage
        if not isinstance(other, __class__):
            return False
        return (np.array_equal(self.size, other.size)) and (np.array_equal(self.data, other.data))


class ImageFromFile(FromFileMixin, Image):
    def __init__(
        self,
        path: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(path, *args, **kwargs)
        self.__data = lazy_image(self.path)

        # extension from file name and real extension can be differ
        self._ext = self._ext if self._ext else osp.splitext(osp.basename(path))[1]

    @property
    def data(self) -> Optional[np.ndarray]:
        """Image data in BGRA HWC [0; 255] (uint8) format"""

        if not self.has_data:
            return None

        if self.__data._dtype != self._dtype:
            self.__data._loader = partial(load_image, dtype=self._dtype)
        data = self.__data()

        if self._size is None and data is not None:
            if not 2 <= data.ndim <= 3:
                raise MediaShapeError("An image should have 2 (gray) or 3 (bgra) dims.")
            self._size = tuple(map(int, data.shape[:2]))
        return data

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        """Returns (H, W)"""

        if self._size is None:
            try:
                width, height = imagesize.get(self.path)
                assert width != -1 and height != -1
                self._size = (height, width)
            except Exception:
                _ = super().size
        return self._size

    def save(
        self,
        fp: Union[str, io.IOBase],
        ext: Optional[str] = None,
    ):
        cur_path = osp.abspath(self.path) if self.path else None
        cur_ext = self.ext
        new_ext = self._get_ext_to_save(fp, ext)
        if isinstance(fp, str):
            os.makedirs(osp.dirname(fp), exist_ok=True)

        if cur_path is not None and osp.isfile(cur_path):
            if cur_ext == new_ext:
                copyto_image(src=cur_path, dst=fp)
            else:
                save_image(fp, self.data, ext=new_ext)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cur_path)

    def get_data_as_dtype(self, dtype: Optional[np.dtype] = np.uint8) -> Optional[np.ndarray]:
        """Get image data with a specific data type"""
        self._dtype = dtype
        return self.data


class ImageFromData(FromDataMixin, Image):
    def save(
        self,
        fp: Union[str, io.IOBase],
        ext: Optional[str] = None,
    ):
        data = self.data
        if data is None:
            raise ValueError(f"{self.__class__.__name__} is empty.")
        new_ext = self._get_ext_to_save(fp, ext)
        if isinstance(fp, str):
            os.makedirs(osp.dirname(fp), exist_ok=True)
        save_image(fp, data, ext=new_ext)


class ImageFromNumpy(ImageFromData):
    def __init__(
        self,
        data: Union[Callable[[], bytes], bytes],
        *args,
        **kwargs,
    ):
        super().__init__(data=data, *args, **kwargs)

    @property
    def data(self) -> Optional[np.ndarray]:
        """Image data in BGRA HWC [0; 255] (uint8) format"""

        data = super().data

        if isinstance(data, np.ndarray) and data.dtype != self._dtype:
            data = np.clip(data, 0.0, 255.0).astype(self._dtype)
        if self._size is None and data is not None:
            if not 2 <= data.ndim <= 3:
                raise MediaShapeError("An image should have 2 (gray) or 3 (bgra) dims.")
            self._size = tuple(map(int, data.shape[:2]))
        return data

    @property
    def has_size(self) -> bool:
        """Indicates that size info is cached and won't require image loading"""
        return self._size is not None or isinstance(self._data, np.ndarray)

    def get_data_as_dtype(self, dtype: Optional[np.dtype] = np.uint8) -> Optional[np.ndarray]:
        """Get image data with a specific data type"""
        self._dtype = dtype
        return self.data


class ImageFromBytes(ImageFromData):
    _FORMAT_MAGICS = (
        (b"\x89PNG\r\n\x1a\n", ".png"),
        (b"\xff\xd8\xff", ".jpg"),
        (b"BM", ".bmp"),
    )

    def __init__(
        self,
        data: Union[Callable[[], bytes], bytes],
        *args,
        **kwargs,
    ):
        super().__init__(data=data, *args, **kwargs)

        if self._ext is None and isinstance(data, bytes):
            self._ext = self._guess_ext(data)

    @classmethod
    def _guess_ext(cls, data: bytes) -> Optional[str]:
        return next(
            (ext for magic, ext in cls._FORMAT_MAGICS if data.startswith(magic)),
            None,
        )

    @property
    def data(self) -> Optional[np.ndarray]:
        """Image data in BGRA HWC [0; 255] (uint8) format"""

        data = super().data

        if isinstance(data, bytes):
            data = decode_image(data, dtype=self._dtype)
        if self._size is None and data is not None:
            if not 2 <= data.ndim <= 3:
                raise MediaShapeError("An image should have 2 (gray) or 3 (bgra) dims.")
            self._size = tuple(map(int, data.shape[:2]))
        return data

    def get_data_as_dtype(self, dtype: Optional[np.dtype] = np.uint8) -> Optional[np.ndarray]:
        """Get image data with a specific data type"""

        if dtype != np.uint8:
            raise ValueError("ImageFromBytes only support `dtype=np.uint8`.")
        self._dtype = dtype
        return self.data


class VideoFrame(ImageFromNumpy):
    _type = MediaType.VIDEO_FRAME

    _DEFAULT_EXT = None

    def __init__(self, video: Video, index: int):
        self._video = video
        self._index = index

        super().__init__(data=lambda: self._video.get_frame_data(self._index))

    def as_dict(self) -> Dict[str, Any]:
        attrs = super().as_dict()
        return {
            "video": attrs["video"],
            "index": attrs["index"],
        }

    @property
    def size(self) -> Tuple[int, int]:
        return self._video.frame_size

    @property
    def index(self) -> int:
        return self._index

    @property
    def video(self) -> Video:
        return self._video

    @property
    def path(self) -> str:
        return self._video.path

    def from_self(self, **kwargs):
        attrs = deepcopy(self.as_dict())
        if "path" in kwargs:
            attrs.update({"video": self.video.from_self(**kwargs)})
            kwargs.pop("path")
        attrs.update(kwargs)
        return self.__class__(**attrs)

    def __getstate__(self):
        # Return only the picklable parts of the state.
        state = self.__dict__.copy()
        del state["_data"]
        return state

    def __setstate__(self, state):
        # Restore the objects' state.
        self.__dict__.update(state)
        # Reinitialize unpichlable attributes
        self._data = lambda: self._video.get_frame_data(self._index)


class _VideoFrameIterator(Iterator[VideoFrame]):
    """
    Provides sequential access to the video frames.
    """

    _video: Video
    _iterator: Iterator[VideoFrame]
    _pos: int
    _current_frame_data: Optional[np.ndarray]

    def __init__(self, video: Video):
        self._video = video
        self._reset()

    def _reset(self):
        self._video._reset_reader()
        self._iterator = self._decode(self._video._get_reader())
        self._pos = -1
        self._current_frame_data = None

    def _decode(self, cap) -> Iterator[VideoFrame]:
        """
        Decodes video frames using opencv
        """

        self._pos = -1

        success, frame = cap.read()
        while success:
            self._pos += 1
            if self._video._includes_frame(self._pos):
                self._current_frame_data = frame.astype(float)
                yield self._make_frame(index=self._pos)

            success, frame = cap.read()

        if self._video._frame_count is None:
            self._video._frame_count = self._pos + 1
            if self._video._end_frame and self._video._end_frame >= self._video._frame_count:
                raise ValueError(
                    f"The end_frame value({self._video._end_frame}) of the video "
                    f"must be less than the frame count({self._video._frame_count})."
                )

    def _make_frame(self, index) -> VideoFrame:
        return VideoFrame(self._video, index=index)

    def __next__(self):
        return next(self._iterator)

    def __getitem__(self, idx: int) -> VideoFrame:
        if not self._video._includes_frame(idx):
            raise IndexError(f"Video doesn't contain frame #{idx}.")

        return self._navigate_to(idx)

    def get_frame_data(self, idx: int) -> np.ndarray:
        self._navigate_to(idx)
        return self._current_frame_data

    def _navigate_to(self, idx: int) -> VideoFrame:
        """
        Iterates over frames to the required position.
        """

        if idx < 0:
            raise IndexError()

        if idx < self._pos:
            self._reset()

        if self._pos < idx:
            try:
                while self._pos < idx:
                    v = self.__next__()
            except StopIteration as e:
                raise IndexError() from e
        else:
            v = self._make_frame(index=self._pos)

        return v


class Video(MediaElement, Iterable[VideoFrame]):
    _type = MediaType.VIDEO

    """
    Provides random access to the video frames.
    """

    def __init__(
        self,
        path: str,
        step: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._path = path

        assert 0 <= start_frame
        if end_frame is not None:
            assert start_frame <= end_frame
            # we can't know the video length here,
            # so we cannot validate if the end_frame is valid.
        assert 0 < step
        self._step = step
        self._start_frame = start_frame
        self._end_frame = end_frame

        self._reader = None
        self._iterator: Optional[_VideoFrameIterator] = None
        self._frame_size: Optional[Tuple[int, int]] = None

        # We don't provide frame count unless we have a reliable source of
        # this information.
        # - Not all videos provide length / duration metainfo
        # - We can get an estimation based on metadata, but it
        #   can be invalid or inaccurate due to variable frame rate
        #   or fractional values rounded up. Relying on the value will give
        #   errors during requesting frames.
        # https://stackoverflow.com/a/47796468
        self._frame_count = None
        self._length = None

    def close(self):
        self._iterator = None

        if self._reader is not None:
            self._reader.release()
            self._reader = None

    def __getitem__(self, idx: int) -> VideoFrame:
        if not self._includes_frame(idx):
            raise IndexError(f"Video doesn't contain frame #{idx}.")

        return self._get_iterator()[idx]

    def get_frame_data(self, idx: int) -> VideoFrame:
        if not self._includes_frame(idx):
            raise IndexError(f"Video doesn't contain frame #{idx}.")

        return self._get_iterator().get_frame_data(idx)

    def __iter__(self) -> Iterator[VideoFrame]:
        """
        Iterates over frames lazily, if possible.
        """

        if self._frame_count is not None:
            # Decoding is not necessary to get frame pointers
            # However, it can be inacurrate
            end_frame = self._get_end_frame()
            for index in range(self._start_frame, end_frame + 1, self._step):
                yield VideoFrame(video=self, index=index)
        else:
            # Need to decode to iterate over frames
            yield from self._get_iterator()

    @property
    def length(self) -> Optional[int]:
        """
        Returns frame count of the closed interval [start_frame, end_frame],
        if video provides such information.

        Note that not all videos provide length / duration metainfo, so the
        result may be undefined.

        Also note, that information may be inaccurate because of variable
        FPS in video or incorrect metainfo. The count is only guaranteed to
        be valid after video is completely read once.

        The count is affected by the frame filtering options of the object,
        i.e. start frame, end frame and frame step.
        """

        if self._length is None:
            end_frame = self._get_end_frame()

            if end_frame is not None:
                length = (end_frame + 1 - self._start_frame) // self._step
                if 0 >= length:
                    raise ValueError(
                        "There is no valid frame for the closed interval"
                        f"[start_frame({self._start_frame}),"
                        f" end_frame({end_frame})] with step({self._step})."
                    )
                self._length = length

        return self._length

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Returns (H, W)"""

        if self._frame_size is None:
            self._frame_size = self._get_frame_size()
        return self._frame_size

    def _get_frame_size(self) -> Tuple[int, int]:
        cap = self._get_reader()
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if h and w:
            frame_size = (int(h), int(w))
        else:
            image = next(self._get_iterator()).data
            frame_size = image.shape[0:2]

        return frame_size

    def _get_end_frame(self):
        # Note that end_frame could less than the last frame of the video
        if self._end_frame is not None and self._frame_count is not None:
            end_frame = min(self._end_frame, self._frame_count)
        elif self._end_frame is not None:
            end_frame = self._end_frame
        elif self._frame_count is not None:
            end_frame = self._frame_count - 1
        else:
            end_frame = None

        return end_frame

    def _includes_frame(self, i):
        if self._start_frame <= i:
            if (i - self._start_frame) % self._step == 0:
                end_frame = self._get_end_frame()
                if end_frame is None or i <= end_frame:
                    return True

        return False

    def _get_iterator(self):
        if self._iterator is None:
            self._iterator = _VideoFrameIterator(self)
        return self._iterator

    def _get_reader(self):
        if self._reader is None:
            self._reset_reader()
        return self._reader

    def _reset_reader(self):
        if self._reader is not None:
            self._reader.release()
        self._reader = cv2.VideoCapture(self._path)
        assert self._reader.isOpened()

    def __eq__(self, other: object) -> bool:
        def _get_frame(obj: Video, idx: int):
            try:
                return obj[idx]
            except IndexError:
                return None

        if not isinstance(other, __class__):
            return False
        if self._start_frame != other._start_frame or self._step != other._step:
            return False

        # The video path can vary if a dataset is copied.
        # So, we need to check if the video data is the same instead of checking paths.
        if self._end_frame is not None and self._end_frame == other._end_frame:
            for idx in range(self._start_frame, self._end_frame + 1, self._step):
                if self[idx] != other[idx]:
                    return False
            return True

        end_frame = self._end_frame or other._end_frame
        if end_frame is None:
            last_frame = None
            for idx, frame in enumerate(self):
                if frame != _get_frame(other, frame.index):
                    return False
                last_frame = frame
            # check if the actual last frames are same
            try:
                other[last_frame.index + self._step if last_frame else self._start_frame]
            except IndexError:
                return True
            return False

        # _end_frame values, only one of the two is valid
        for idx in range(self._start_frame, end_frame + 1, self._step):
            frame = _get_frame(self, idx)
            if frame is None:
                return False
            if frame != _get_frame(other, idx):
                return False
        # check if the actual last frames are same
        idx_next = end_frame + self._step
        return None is (_get_frame(self, idx_next) or _get_frame(other, idx_next))

    def __hash__(self):
        # Required for caching
        return hash((self._path, self._step, self._start_frame, self._end_frame))

    def save(
        self,
        fp: Union[str, io.IOBase],
    ):
        if isinstance(fp, str):
            os.makedirs(osp.dirname(fp), exist_ok=True)
        if isinstance(fp, str):
            if fp != self.path:
                shutil.copyfile(self.path, fp)
        elif isinstance(fp, io.IOBase):
            with open(self.path, "rb") as f_video:
                fp.write(f_video.read())

    @property
    def path(self) -> str:
        """Path to the media file"""
        return self._path

    @property
    def ext(self) -> str:
        """Media file extension (with the leading dot)"""
        return osp.splitext(osp.basename(self.path))[1]


class PointCloud(MediaElement[bytes]):
    _type = MediaType.POINT_CLOUD

    def __init__(
        self,
        extra_images: Optional[Union[List[Image], Callable[[], List[Image]]]] = None,
        *args,
        **kwargs,
    ):
        assert self.__class__ != PointCloud, (
            f"Directly initalizing {self.__class__.__name__} is not supported. "
            f"Please use one of fractory function ({self.__class__.__name__}.from_file(), "
            f"{self.__class__.__name__}.from_bytes())."
        )
        super().__init__(*args, **kwargs)
        self._extra_images = extra_images or []

    @classmethod
    def from_file(cls, path: str, *args, **kwargs):
        return PointCloudFromFile(path, *args, **kwargs)

    @classmethod
    def from_bytes(cls, data: Union[bytes, Callable[[], bytes]], *args, **kwargs):
        return PointCloudFromBytes(data, *args, **kwargs)

    @property
    def extra_images(self) -> List[Image]:
        if callable(self._extra_images):
            extra_images = self._extra_images()
            assert isinstance(extra_images, list) and all(
                [isinstance(image, Image) for image in extra_images]
            )
            return extra_images
        return self._extra_images

    def _save_extra_images(
        self,
        fn: Callable[[int, Image], Dict[str, Any]],
    ):
        for i, img in enumerate(self.extra_images):
            if img.has_data:
                kwargs: Dict[str, Any] = {}
                kwargs.update(fn(i, img))
                img.save(**kwargs)

    def __eq__(self, other: object) -> bool:
        return (
            super().__eq__(other)
            and (self.data == other.data)
            and self.extra_images == other.extra_images
        )


class PointCloudFromFile(FromFileMixin, PointCloud):
    @property
    def data(self) -> Optional[bytes]:
        if self.has_data:
            with open(self.path, "rb") as f:
                bytes_data = f.read()
            return bytes_data
        return None

    def save(
        self,
        fp: Union[str, io.IOBase],
        extra_images_fn: Optional[Callable[[int, Image], Dict[str, Any]]] = None,
    ):
        cur_path = osp.abspath(self.path) if self.path else None

        if cur_path is not None and osp.isfile(cur_path):
            with open(cur_path, "rb") as reader:
                _bytes = reader.read()
            if isinstance(fp, str):
                os.makedirs(osp.dirname(fp), exist_ok=True)
                with open(fp, "wb") as f:
                    f.write(_bytes)
            else:
                fp.write(_bytes)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cur_path)

        if extra_images_fn is not None:
            self._save_extra_images(extra_images_fn)


class PointCloudFromData(FromDataMixin, PointCloud):
    def save(
        self,
        fp: Union[str, io.IOBase],
        extra_images_fn: Optional[Callable[[int, Image], Dict[str, Any]]] = None,
    ):
        _bytes = self.data
        if _bytes is None:
            raise ValueError(f"{self.__class__.__name__} is empty.")
        if isinstance(fp, str):
            os.makedirs(osp.dirname(fp), exist_ok=True)
            with open(fp, "wb") as f:
                f.write(_bytes)
        else:
            fp.write(_bytes)

        if extra_images_fn is not None:
            self._save_extra_images(extra_images_fn)


class PointCloudFromBytes(PointCloudFromData):
    @property
    def data(self) -> Optional[bytes]:
        return super().data


class MultiframeImage(MediaElement):
    _type = MediaType.MULTIFRAME_IMAGE

    def __init__(
        self,
        images: Optional[Iterable[Union[str, Image, np.ndarray, Callable[[str], np.ndarray]]]],
        *,
        path: Optional[str] = None,
    ):
        self._path = path

        if images is None:
            images = []

        self._images = [None] * len(images)
        for i, image in enumerate(images):
            assert isinstance(image, (str, Image, np.ndarray)) or callable(image)

            if isinstance(image, str):
                image = Image.from_file(path=image)
            elif isinstance(image, np.ndarray) or callable(image):
                image = Image.from_numpy(data=image)

            self._images[i] = image

        assert self._path or self._images

    @property
    def data(self) -> List[Image]:
        return self._images

    @property
    def path(self) -> str:
        """Path to the media file"""
        return self._path

    @property
    def ext(self) -> str:
        """Media file extension (with the leading dot)"""
        return osp.splitext(osp.basename(self.path))[1]


class RoIImage(Image):
    _type = MediaType.ROI_IMAGE

    def __init__(
        self,
        roi: BboxIntCoords,
        *args,
        **kwargs,
    ):
        assert self.__class__ != RoIImage, (
            f"Directly initalizing {self.__class__.__name__} is not supported. "
            f"Please use a fractory function '{self.__class__.__name__}.from_image()'. "
        )

        assert len(roi) == 4 and all(isinstance(v, int) for v in roi)
        self._roi = roi
        _, _, w, h = self._roi
        super().__init__(size=(h, w), *args, **kwargs)

    def as_dict(self) -> Dict[str, Any]:
        attrs = super().as_dict()
        attrs.pop("size", None)
        return attrs

    @classmethod
    def from_file(cls, *args, **kwargs):
        raise DatumaroError(f"Please use a factory function '{cls.__name__}.from_image'.")

    @classmethod
    def from_image(cls, data: Image, roi: BboxIntCoords, *args, **kwargs):
        if not isinstance(data, Image):
            raise TypeError(f"type(image)={type(data)} should be Image.")

        if isinstance(data, ImageFromFile):
            return RoIImageFromFile(path=data.path, roi=roi, ext=data._ext, *args, **kwargs)
        if isinstance(data, ImageFromNumpy):
            return RoIImageFromNumpy(data=data._data, roi=roi, ext=data._ext, *args, **kwargs)
        if isinstance(data, ImageFromBytes):
            return RoIImageFromBytes(data=data._data, roi=roi, ext=data._ext, *args, **kwargs)
        raise NotImplementedError

    @classmethod
    def from_numpy(cls, *args, **kwargs):
        raise DatumaroError(f"Please use a factory function '{cls.__name__}.from_image'.")

    @classmethod
    def from_bytes(cls, *args, **kwargs):
        raise DatumaroError(f"Please use a factory function '{cls.__name__}.from_image'.")

    @property
    def roi(self) -> BboxIntCoords:
        return self._roi

    def _get_roi_data(self, data: np.ndarray) -> np.ndarray:
        x, y, w, h = self._roi
        return data[y : y + h, x : x + w]

    def save(
        self,
        fp: Union[str, io.IOBase],
        ext: Optional[str] = None,
    ):
        data = self.data
        if data is None:
            raise ValueError(f"{self.__class__.__name__} is empty.")
        new_ext = self._get_ext_to_save(fp, ext)
        if isinstance(fp, str):
            os.makedirs(osp.dirname(fp), exist_ok=True)
        save_image(fp, data, ext=new_ext)


class RoIImageFromFile(FromFileMixin, RoIImage):
    def __init__(
        self,
        path: str,
        roi: BboxIntCoords,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(path, roi, *args, **kwargs)
        self.__data = lazy_image(self.path)

    @property
    def data(self) -> Optional[np.ndarray]:
        """Image data in BGRA HWC [0; 255] (uint8) format"""
        if not self.has_data:
            return None
        data = self.__data()
        return self._get_roi_data(data)


class RoIImageFromData(FromDataMixin, RoIImage):
    pass


class RoIImageFromBytes(RoIImageFromData):
    def __init__(
        self,
        data: Union[bytes, Callable[[], bytes]],
        roi: BboxIntCoords,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(data, roi, *args, **kwargs)

    @property
    def data(self) -> Optional[np.ndarray]:
        """Image data in BGRA HWC [0; 255] (uint8) format"""
        data = super().data
        if data is None:
            return None
        if isinstance(data, bytes):
            data = decode_image(data)
        return self._get_roi_data(data)


class RoIImageFromNumpy(RoIImageFromData):
    def __init__(
        self,
        data: Union[np.ndarray, Callable[[], np.ndarray]],
        roi: BboxIntCoords,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(data, roi, *args, **kwargs)

    @property
    def data(self) -> Optional[np.ndarray]:
        """Image data in BGRA HWC [0; 255] (uint8) format"""
        data = super().data
        if data is None:
            return None
        return self._get_roi_data(data)


ImageWithRoI = Tuple[Image, BboxIntCoords]


class MosaicImage(Image):
    _type = MediaType.MOSAIC_IMAGE

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        assert self.__class__ != MosaicImage, (
            f"Directly initalizing {self.__class__.__name__} is not supported. "
            f"Please use a fractory function '{self.__class__.__name__}.from_image_roi_pairs()'."
        )
        super().__init__(*args, **kwargs)

    @classmethod
    def from_file(cls, *args, **kwargs):
        raise DatumaroError(f"Please use a factory function '{cls.__name__}.from_image_roi_pairs'.")

    @classmethod
    def from_image_roi_pairs(cls, data: List[ImageWithRoI], size: Tuple[int, int], *args, **kwargs):
        return MosaicImageFromImageRoIPairs(data, size)

    @classmethod
    def from_numpy(cls, *args, **kwargs):
        raise DatumaroError(f"Please use a factory function '{cls.__name__}.from_image_roi_pairs'.")

    @classmethod
    def from_bytes(cls, *args, **kwargs):
        raise DatumaroError(f"Please use a factory function '{cls.__name__}.from_image_roi_pairs'.")


class MosaicImageFromData(FromDataMixin, MosaicImage):
    def save(
        self,
        fp: Union[str, io.IOBase],
        ext: Optional[str] = None,
    ):
        data = self.data
        if data is None:
            raise ValueError(f"{self.__class__.__name__} is empty.")
        new_ext = self._get_ext_to_save(fp, ext)
        if isinstance(fp, str):
            os.makedirs(osp.dirname(fp), exist_ok=True)
        save_image(fp, data, ext=new_ext)


class MosaicImageFromImageRoIPairs(MosaicImageFromData):
    def __init__(self, data: List[ImageWithRoI], size: Tuple[int, int]) -> None:
        def _get_mosaic_img() -> np.ndarray:
            h, w = self.size
            mosaic_img = np.zeros(shape=(h, w, 3), dtype=np.uint8)
            for img, roi in data:
                assert isinstance(img, Image), "MosaicImage can only take a list of Images."
                x, y, w, h = roi
                mosaic_img[y : y + h, x : x + w] = img.data
            return mosaic_img

        super().__init__(data=_get_mosaic_img, size=size)
        self._data_in = data

    def as_dict(self) -> Dict[str, Any]:
        attrs = super().as_dict()
        return {
            "data": attrs["data_in"],
            "size": attrs["size"],
        }


TableDtype = TypeVar("TableDtype", str, int, float)


class Table:
    def __init__(
        self,
    ) -> None:
        """
        Table data with multiple rows and columns.
        This provides random access to the table row.

        Initialization must be done in the child class.
        """
        assert self.__class__ != Table, (
            f"Directly initalizing {self.__class__.__name__} is not supported. "
            f"Please use one of fractory functions ({self.__class__.__name__}.from_csv(), "
            f"{self.__class__.__name__}.from_dataframe(), or ({self.__class__.__name__}.from_list())."
        )
        self._shape: Tuple[int, int] = (0, 0)

    @classmethod
    def from_csv(cls, path: str, *args, **kwargs) -> Type[Table]:
        """
        Returns Table instance creating from a csv file.

        Args:
            path (str) : Path to csv file.
        """
        return TableFromCSV(path, *args, **kwargs)

    @classmethod
    def from_dataframe(
        cls,
        data: Union[pd.DataFrame, Callable[[], pd.DataFrame]],
        *args,
        **kwargs,
    ) -> Type[Table]:
        """
        Returns Table instance creating from a pandas DataFrame.

        Args:
            data (DataFrame) : Data in pandas DataFrame format.
        """
        return TableFromDataFrame(data, *args, **kwargs)

    @classmethod
    def from_list(
        cls,
        data: List[Dict[str, TableDtype]],
        *args,
        **kwargs,
    ) -> Type[Table]:
        """
        Returns Table instance creating from a list of dicts.

        Args:
            data (list(dict(str,str|int|float))) : A list of table row data.
        """
        return TableFromListOfDict(data, *args, **kwargs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.data.equals(other)

    def __getitem__(self, idx: int) -> TableRow:
        """
        Random access to a specific row by index.
        """
        if idx >= self.shape[0]:
            raise IndexError(f"Table doesn't contain row #{idx}.")
        return TableRow(table=self, index=idx)

    def __iter__(self) -> Iterator[TableRow]:
        """
        Iterates over rows.
        """
        for index in range(self.shape[0]):
            yield TableRow(table=self, index=index)

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns table size as (#rows, #cols)"""
        return self._shape

    @property
    def columns(self) -> List[str]:
        """Returns column names"""
        return self.data.columns.to_list()

    def dtype(self, column: str) -> Optional[Type[TableDtype]]:
        """Returns native python type for a given column"""
        numpy_type = self.data.dtypes[column]
        if self.data[column].nunique() / self.shape[0] < 0.1:  # TODO
            # Convert to CategoricalDtype for efficient storage and categorical analysis
            return pd.api.types.CategoricalDtype()
        if isinstance(numpy_type, np.dtypes.ObjectDType):
            return str
        else:
            return type(np.zeros(1, numpy_type).tolist()[0])

    def features(self, column: str, unique: Optional[bool] = False) -> List[TableDtype]:
        """Get features for a given column name."""
        if unique:
            return list(self.data[column].unique())
        else:
            return self.data[column].to_list()

    def save(
        self,
        path: str,
    ):
        """
        Save table instance to a '.csv' file.

        Args:
            path (str) : Path to the output csv file.
        """
        data: pd.DataFrame = self.data
        os.makedirs(osp.dirname(path), exist_ok=True)
        data.to_csv(path, index=False)


class TableFromCSV(FromFileMixin, Table):
    def __init__(
        self,
        path: str,
        dtype: Optional[Dict] = None,
        sep: Optional[str] = None,
        encoding: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Read a '.csv' file and compose a Table instance.

        Args:
            path (str) : Path to csv file.
            dtype (optional, dict(str,str)) : Dictionay of column name -> type str ('str', 'int', or 'float').
            sep (optional, str) : Delimiter to use.
            encoding (optional, str) : Encoding to use for UTF when reading/writing (ex. 'utf-8').
        """
        super().__init__(path, *args, **kwargs)

        # assumes that the 1st row is a header.
        data: pd.DataFrame = pd.read_csv(
            path, dtype=dtype, sep=sep, engine="python", encoding=encoding, index_col=False
        )
        if data is None:
            raise ValueError(f"Can't read csv File from {path}")
        if data.shape[1] == 0:
            raise MediaShapeError("A table should have 1 or more columns.")

        self.__data = data
        self._shape = data.shape

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Table data in pandas DataFrame format"""
        return self.__data

    def select(self, columns: List[str]):
        self.__data = self.__data[columns]
        self._shape = self.__data.shape


class TableFromDataFrame(FromDataMixin, Table):
    def __init__(
        self,
        data: Union[Callable[[], pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs,
    ):
        """
        Read a pandas DataFrame and compose a Table instance.

        Args:
            data (DataFrame) : Data in pandas DataFrame format.
        """
        super().__init__(data=data, *args, **kwargs)

        if data is None:
            raise ValueError("'data' can't be None")
        if data.shape[1] == 0:
            raise MediaShapeError("A table should have 1 or more columns.")
        for col in data.columns:
            if not isinstance(col, str):
                raise TypeError("A table should have column names as a list of str values")

        self._shape = data.shape

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Table data in pandas DataFrame format"""
        return super().data


class TableFromListOfDict(TableFromDataFrame):
    def __init__(
        self,
        data: List[Dict[str, TableDtype]],
        *args,
        **kwargs,
    ):
        """
        Read a list of table row data and compose a Table instance.
        The table row data is in dictionary format.

        Args:
            data (list(dict(str,str|int|float))) : A list of table row data.
        """
        super().__init__(data=pd.DataFrame(data), *args, **kwargs)


class TableRow(MediaElement):
    _type = MediaType.TABLE_ROW

    def __init__(self, table: Table, index: int):
        """
        TableRow media refers to a Table instance and its row index.

        Args:
            table (Table) : Table instance.
            index (int) : Row index.
        """
        if table is None:
            raise ValueError("'table' can't be None")
        if index < 0 or index >= table.shape[0]:
            raise IndexError(f"'index({index})' is out of range.")
        self._table = table
        self._index = index

    @property
    def table(self) -> Table:
        """Table instance"""
        return self._table

    @property
    def index(self) -> int:
        """Row index"""
        return self._index

    @property
    def path(self) -> str:
        return self._table.data.path

    @property
    def has_data(self) -> bool:
        return self.data() is not None

    def data(self, targets: Optional[List[str]] = None) -> Dict:
        """
        Row data in dict format.

        Args:
            targets (optional, list(str)) : If this is specified,
                the values corresponding to target colums will be returned.
                Otherwise, whole row data will be returned.
        """
        row = self.table.data.iloc[self.index]
        if targets:
            row = row[targets]
        return row.to_dict()

    def __repr__(self):
        return f"TableRow(row_idx:{self.index}, data:{self.data()})"

    @classmethod
    def from_data(cls, data: Dict, *args, **kwargs):
        return TableRowFromData(data, *args, **kwargs)


class TableRowFromData(FromDataMixin, TableRow):
    def __init__(self, data: Dict, *args, **kwargs):
        super().__init__(data=data, *args, **kwargs)

    @property
    def data(self) -> Dict:
        data = super().data
        return data
