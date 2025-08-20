# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import os.path as osp
import shlex
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum, auto
from functools import partial
from io import BytesIO, IOBase
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    try:
        # Introduced in 1.20
        from numpy.typing import DTypeLike
    except ImportError:
        DTypeLike = Any


class ImageBackend(Enum):
    cv2 = auto()
    PIL = auto()


IMAGE_BACKEND: ContextVar[ImageBackend] = ContextVar("IMAGE_BACKEND")
_image_loading_errors = (FileNotFoundError,)
try:
    import cv2

    IMAGE_BACKEND.set(ImageBackend.cv2)
except ModuleNotFoundError:
    import PIL

    IMAGE_BACKEND.set(ImageBackend.PIL)
    _image_loading_errors = (*_image_loading_errors, PIL.UnidentifiedImageError)

from datumaro.util.image_cache import ImageCache
from datumaro.util.os_util import find_files


class ImageColorChannel(Enum):
    """Image color channel

    - UNCHANGED: Use the original image's channel (default)
    - COLOR_BGR: Use BGR 3 channels (it can ignore the alpha channel or convert the gray scale image)
    - COLOR_RGB: Use RGB 3 channels (it can ignore the alpha channel or convert the gray scale image)
    """

    UNCHANGED = 0
    COLOR_BGR = 1
    COLOR_RGB = 2

    def decode_by_cv2(self, image_bytes: bytes, dtype: DTypeLike = np.uint8) -> np.ndarray:
        """Convert image color channel for OpenCV image (np.ndarray)."""
        image_buffer = np.frombuffer(image_bytes, dtype=dtype)

        if self == ImageColorChannel.UNCHANGED:
            return cv2.imdecode(image_buffer, cv2.IMREAD_UNCHANGED)

        img = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

        if self == ImageColorChannel.COLOR_BGR:
            return img

        if self == ImageColorChannel.COLOR_RGB:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        raise ValueError

    def decode_by_pil(self, image_bytes: bytes) -> np.ndarray:
        """Convert image color channel for PIL Image."""
        from PIL import Image

        img = Image.open(BytesIO(image_bytes))

        if self == ImageColorChannel.UNCHANGED:
            return np.asarray(img)

        if self == ImageColorChannel.COLOR_BGR:
            img = np.asarray(img.convert("RGB"))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self == ImageColorChannel.COLOR_RGB:
            return np.asarray(img.convert("RGB"))

        raise ValueError


IMAGE_COLOR_CHANNEL: ContextVar[ImageColorChannel] = ContextVar(
    "IMAGE_COLOR_CHANNEL", default=ImageColorChannel.UNCHANGED
)


@contextmanager
def decode_image_context(image_backend: ImageBackend, image_color_channel: ImageColorChannel):
    """Change Datumaro image color channel while decoding.

    For model training, it is recommended to use this context manager
    to load images in the BGR 3-channel format. For example,

    .. code-block:: python

        import datumaro as dm
        with decode_image_context(image_backend=ImageBackend.cv2, image_color_channel=ImageColorScale.COLOR):
            item: dm.DatasetItem
            img_data = item.media_as(dm.Image).data
            assert img_data.shape[-1] == 3  # It should be a 3-channel image
    """

    curr_ctx = (IMAGE_BACKEND.get(), IMAGE_COLOR_CHANNEL.get())

    IMAGE_BACKEND.set(image_backend)
    IMAGE_COLOR_CHANNEL.set(image_color_channel)

    yield

    IMAGE_BACKEND.set(curr_ctx[0])
    IMAGE_COLOR_CHANNEL.set(curr_ctx[1])


def load_image(path: str, dtype: DTypeLike = np.uint8):
    """
    Reads an image in the HWC Grayscale/BGR(A) [0; 255] format (default dtype is uint8).
    """

    if IMAGE_BACKEND.get() == ImageBackend.cv2:
        # cv2.imread does not support paths that are not representable
        # in the locale encoding on Windows, so we read the image bytes
        # ourselves.

        with open(path, "rb") as f:
            image_bytes = f.read()

        return decode_image(image_bytes, dtype=dtype)
    elif IMAGE_BACKEND.get() == ImageBackend.PIL:
        with open(path, "rb") as f:
            image_bytes = f.read()

        return decode_image(image_bytes, dtype=dtype)

    raise NotImplementedError(IMAGE_BACKEND)


def copyto_image(src: Union[str, IOBase], dst: Union[str, IOBase]) -> None:
    if src == dst:
        return

    @contextmanager
    def _open(fp, mode):
        was_file = False
        if not isinstance(fp, IOBase):
            was_file = True
            fp = open(fp, mode)
        yield fp
        if was_file:
            fp.close()

    with _open(src, "rb") as src_fp:
        _bytes = src_fp.read()

    with _open(dst, "wb") as dst_fp:
        dst_fp.write(_bytes)


def save_image(
    dst: Union[str, IOBase],
    image: np.ndarray,
    ext: Optional[str] = None,
    create_dir: bool = False,
    dtype: DTypeLike = np.uint8,
    **kwargs,
) -> None:
    # NOTE: Check destination path for existence
    # OpenCV silently fails if target directory does not exist
    if isinstance(dst, IOBase):
        ext = ext if ext else ".png"
    else:
        dst_dir = osp.dirname(dst)
        if dst_dir:
            if create_dir:
                os.makedirs(dst_dir, exist_ok=True)
            elif not osp.isdir(dst_dir):
                raise FileNotFoundError("Directory does not exist: '%s'" % dst_dir)
        # file extension and actual encoding can be differ
        ext = ext if ext else osp.splitext(dst)[1]

    if not kwargs:
        kwargs = {}

    # NOTE: OpenCV documentation says "If the image format is not supported,
    # the image will be converted to 8-bit unsigned and saved that way".
    # Conversion from np.int32 to np.uint8 is not working properly
    backend = IMAGE_BACKEND.get()
    if dtype == np.int32:
        backend = ImageBackend.PIL
    if backend == ImageBackend.cv2:
        # cv2.imwrite does not support paths that are not representable
        # in the locale encoding on Windows, so we write the image bytes
        # ourselves.

        image_bytes = encode_image(image, ext, dtype=dtype, **kwargs)

        if isinstance(dst, str):
            with open(dst, "wb") as f:
                f.write(image_bytes)
        else:
            dst.write(image_bytes)
    elif backend == ImageBackend.PIL:
        from PIL import Image

        if ext.startswith("."):
            ext = ext[1:]

        params = {}
        params["quality"] = kwargs.get("jpeg_quality")
        if kwargs.get("jpeg_quality") == 100:
            params["subsampling"] = 0

        image = image.astype(dtype)
        if len(image.shape) == 3 and image.shape[2] in {3, 4}:
            image[:, :, :3] = image[:, :, 2::-1]  # BGR to RGB
        image = Image.fromarray(image)
        image.save(dst, format=ext, **params)
    else:
        raise NotImplementedError()


def encode_image(image: np.ndarray, ext: str, dtype: DTypeLike = np.uint8, **kwargs) -> bytes:
    if not kwargs:
        kwargs = {}

    if IMAGE_BACKEND.get() == ImageBackend.cv2:
        import cv2

        params = []

        if not ext.startswith("."):
            ext = "." + ext

        if ext.upper() in (".JPG", ".JPEG"):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), kwargs.get("jpeg_quality", 75)]

        image = image.astype(dtype)
        success, result = cv2.imencode(ext, image, params=params)
        if not success:
            raise Exception("Failed to encode image to '%s' format" % (ext))
        return result.tobytes()
    elif IMAGE_BACKEND.get() == ImageBackend.PIL:
        from PIL import Image

        if ext.startswith("."):
            ext = ext[1:]

        params = {}
        params["quality"] = kwargs.get("jpeg_quality")
        if kwargs.get("jpeg_quality") == 100:
            params["subsampling"] = 0

        image = image.astype(dtype)
        if len(image.shape) == 3 and image.shape[2] in {3, 4}:
            image[:, :, :3] = image[:, :, 2::-1]  # BGR to RGB
        image = Image.fromarray(image)
        with BytesIO() as buffer:
            image.save(buffer, format=ext, **params)
            return buffer.getvalue()
    else:
        raise NotImplementedError()


def decode_image(image_bytes: bytes, dtype: np.dtype = np.uint8) -> np.ndarray:
    ctx_color_scale = IMAGE_COLOR_CHANNEL.get()

    if np.issubdtype(dtype, np.floating):
        # PIL doesn't support floating point image loading
        # CV doesn't support floating point image with color channel setting (IMREAD_COLOR)
        with decode_image_context(
            image_backend=ImageBackend.cv2, image_color_channel=ImageColorChannel.UNCHANGED
        ):
            image = ctx_color_scale.decode_by_cv2(image_bytes, dtype=dtype)
            image = image[..., ::-1]
        if ctx_color_scale == ImageColorChannel.COLOR_BGR:
            image = image[..., ::-1]
    else:
        if IMAGE_BACKEND.get() == ImageBackend.cv2:
            image = ctx_color_scale.decode_by_cv2(image_bytes)
        elif IMAGE_BACKEND.get() == ImageBackend.PIL:
            image = ctx_color_scale.decode_by_pil(image_bytes)
        else:
            raise NotImplementedError()

    image = image.astype(dtype)

    assert len(image.shape) in {2, 3}
    if len(image.shape) == 3:
        assert image.shape[2] in {3, 4}
    return image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jp2",
    ".png",
    ".bmp",
    ".dib",
    ".tif",
    ".tiff",
    ".tga",
    ".webp",
    ".pfm",
    ".sr",
    ".ras",
    ".exr",
    ".hdr",
    ".pic",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pxm",
    ".pnm",
}


def find_images(
    dirpath: str,
    exts: Union[str, Iterable[str]] = None,
    recursive: bool = False,
    max_depth: Optional[int] = None,
    min_depth: Optional[int] = None,
) -> Iterator[str]:
    yield from find_files(
        dirpath,
        exts=exts or IMAGE_EXTENSIONS,
        recursive=recursive,
        max_depth=max_depth,
        min_depth=min_depth,
    )


def is_image(path: str) -> bool:
    trunk, ext = osp.splitext(osp.basename(path))
    return trunk and ext.lower() in IMAGE_EXTENSIONS and osp.isfile(path)


class lazy_image:
    def __init__(
        self,
        path: str,
        loader: Callable[[str], np.ndarray] = None,
        cache: Union[bool, ImageCache] = True,
        dtype: Optional[DTypeLike] = None,
    ) -> None:
        """
        Cache:
            - False: do not cache
            - True: use the global cache
            - ImageCache instance: an object to be used as cache
        """

        self._custom_loader = True

        if loader is None:
            loader = partial(load_image, dtype=dtype) if dtype else load_image
            self._custom_loader = False

        self._path = path
        self._loader = loader

        assert isinstance(cache, (ImageCache, bool))
        self._cache = cache
        self._dtype = dtype

    def __call__(self) -> np.ndarray:
        image = None
        cache_key = weakref.ref(self)

        cache = self._get_cache()
        if cache is not None:
            image = cache.get(cache_key)

        if image is None:
            image = self._loader(self._path)
            if cache is not None:
                cache.push(cache_key, image)
        return image

    def _get_cache(self) -> Optional[ImageCache]:
        if self._cache is True:
            cache = ImageCache.get_instance()
        elif self._cache is False:
            cache = None
        else:
            cache = self._cache
        return cache


ImageMeta = Dict[str, Tuple[int, int]]
"""filename -> height, width"""

DEFAULT_IMAGE_META_FILE_NAME = "images.meta"


def load_image_meta_file(image_meta_path: str) -> ImageMeta:
    """
    Loads image metadata from a file with the following format:

        <image name 1> <height 1> <width 1>

        <image name 2> <height 2> <width 2>

        ...

    Shell-like comments and quoted fields are allowed.

    This can be useful to support datasets in which image dimensions are
    required to interpret annotations.
    """
    assert isinstance(image_meta_path, str)

    if not osp.isfile(image_meta_path):
        raise FileNotFoundError("Can't read image meta file '%s'" % image_meta_path)

    image_meta = {}

    with open(image_meta_path, encoding="utf-8") as f:
        for line in f:
            fields = shlex.split(line, comments=True)
            if not fields:
                continue

            # ignore extra fields, so that the format can be extended later
            image_name, h, w = fields[:3]
            image_meta[image_name] = (int(h), int(w))

    return image_meta


def save_image_meta_file(image_meta: ImageMeta, image_meta_path: str) -> None:
    """
    Saves image_meta to the path specified by image_meta_path in the format
    defined in load_image_meta_file's documentation.
    """

    assert isinstance(image_meta_path, str)

    with open(image_meta_path, "w", encoding="utf-8") as f:
        # Add a comment about file syntax
        print("# <image name> <height> <width>", file=f)
        print("", file=f)

        for image_name, (height, width) in image_meta.items():
            print(shlex.quote(image_name), height, width, file=f)
