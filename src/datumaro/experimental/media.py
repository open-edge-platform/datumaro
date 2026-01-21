# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Media types for lazy loading and caching of image data.

This module provides the LazyImage class for lazy loading images from disk,
along with a global LRU cache to manage memory usage when working with many images.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from threading import RLock

import numpy as np
from cachetools import LRUCache, cachedmethod
from PIL import Image

# Default cache size: 256 MB
_DEFAULT_CACHE_SIZE_BYTES = 256 * 1024 * 1024


def _get_image_size(img: np.ndarray) -> int:
    """Get the size of an image in bytes."""
    return img.nbytes


class ImageCache:
    """
    Global LRU cache for lazy-loaded images.

    The size of the cache can be updated dynamically, allowing fine-grained
    control over the memory usage. Images are automatically evicted when
    the cache exceeds its maximum size, with the least recently used images
    being removed first.

    The cache is keyed by (path, format, channels_first) tuples, so the same
    image loaded with different settings will be cached separately.

    Examples:
        >>> from datumaro.experimental import ImageCache
        >>> ImageCache.set_size(512 * 1024 * 1024)  # Set cache to 512 MB
        >>> ImageCache.get_size()  # Get current cache size in bytes
        >>> ImageCache.length()  # Get number of cached images
        >>> ImageCache.clear()  # Clear all cached images
    """

    _cache: LRUCache[tuple[str, str, bool], np.ndarray] = LRUCache(
        maxsize=_DEFAULT_CACHE_SIZE_BYTES, getsizeof=_get_image_size
    )
    _lock = RLock()

    @classmethod
    def set_size(cls, maxsize_bytes: int) -> None:
        """
        Set the maximum size in bytes for the image cache.

        When the cache exceeds this size, the least recently used images are evicted.

        Args:
            maxsize_bytes: Maximum cache size in bytes. Set to 0 to disable caching.

        Example:
            >>> ImageCache.set_size(512 * 1024 * 1024)  # Set cache to 512 MB
        """
        with cls._lock:
            # Create a new cache with the new size, preserving existing items if they fit
            new_cache: LRUCache[tuple[str, str, bool], np.ndarray] = LRUCache(
                maxsize=maxsize_bytes, getsizeof=_get_image_size
            )
            if maxsize_bytes > 0:
                for key, value in cls._cache.items():
                    item_size = _get_image_size(value)
                    if new_cache.currsize + item_size <= maxsize_bytes:
                        new_cache[key] = value
            cls._cache = new_cache

    @classmethod
    def get_size(cls) -> int:
        """
        Get the current cache size in bytes.

        Returns:
            Current cache size in bytes.
        """
        with cls._lock:
            return int(cls._cache.currsize)

    @classmethod
    def get_max_size(cls) -> int:
        """
        Get the maximum cache size in bytes.

        Returns:
            Maximum cache size in bytes.
        """
        with cls._lock:
            return int(cls._cache.maxsize)

    @classmethod
    def clear(cls) -> None:
        """
        Clear all cached images from memory.

        Example:
            >>> ImageCache.clear()  # Free all cached image memory
        """
        with cls._lock:
            cls._cache.clear()

    @classmethod
    def length(cls) -> int:
        """
        Get the number of images currently cached.

        Returns:
            Number of cached images.
        """
        with cls._lock:
            return len(cls._cache)

    @classmethod
    def info(cls) -> dict[str, int]:
        """
        Get information about the current cache state.

        Returns:
            A dictionary with:
            - 'count': Number of images currently cached
            - 'current_size': Current cache size in bytes
            - 'max_size': Maximum cache size in bytes
        """
        with cls._lock:
            return {
                "count": len(cls._cache),
                "current_size": cls._cache.currsize,
                "max_size": cls._cache.maxsize,
            }


@dataclass
class LazyImage:
    """
    A wrapper class that holds an image path and provides lazy loading of image data.

    This class enables lazy loading patterns where the image is only loaded from
    disk when the `data` property is accessed. Loaded images are stored in a
    global LRU cache that limits total memory usage.

    The cache keeps the most recently accessed images in memory and automatically
    evicts the least recently used images when the cache is full. By default,
    the cache is limited to 256 MB. Use `ImageCache.set_size()` to adjust this limit.

    Attributes:
        path: The file path to the image (can be a string, Path object, or another LazyImage)
        format: The color format to use when loading ("RGB", "BGR", etc.)
        channels_first: Whether to return data in channels-first format (C, H, W)

    Examples:
        >>> lazy_img = LazyImage("/path/to/image.jpg")
        >>> print(lazy_img.path)  # Access path without loading
        /path/to/image.jpg
        >>> img_array = lazy_img.data  # Image loaded here on first access
        >>> print(img_array.shape)
        (480, 640, 3)

    Cache management:
        >>> from datumaro.experimental import ImageCache
        >>> ImageCache.set_size(512 * 1024 * 1024)  # Limit to 512 MB
        >>> ImageCache.clear()  # Clear all cached images

    Using with Sample:
        >>> from datumaro.experimental import Sample
        >>> from datumaro.experimental.fields import image_path_field
        >>> class MySample(Sample):
        ...     image: LazyImage = image_path_field()
        ...
        >>> sample = MySample(image="/path/to/image.jpg")
        >>> sample.image.path  # Returns the path string
        >>> sample.image.data  # Returns the numpy array
    """

    path: str | Path | LazyImage
    format: str = "RGB"
    channels_first: bool = False

    def __post_init__(self) -> None:
        # Ensure path is stored as a string for consistency
        # Handle LazyImage input by extracting its path
        if isinstance(self.path, LazyImage):
            object.__setattr__(self, "path", self.path.path)
        elif isinstance(self.path, Path):
            object.__setattr__(self, "path", str(self.path))

    def _cache_key(self) -> tuple[str, str, bool]:
        """Generate a cache key for this image configuration."""
        return str(self.path), self.format.upper(), self.channels_first

    def _convert_to_format(self, img: Image.Image, target_format: str) -> np.ndarray:
        """Convert PIL image to numpy array in the target format.

        Supports 8-bit and 16-bit images. The bit depth is preserved from the
        original image file - 16-bit images will return uint16 arrays.
        """
        is_16bit = img.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N")

        if target_format in ("RGB", "BGR"):
            return self._convert_to_rgb(img, is_16bit)
        if target_format == "RGBA":
            return self._convert_to_rgba(img, is_16bit)
        if target_format == "L":
            return self._convert_to_grayscale(img, is_16bit)
        return np.array(img)

    def _convert_to_rgb(self, img: Image.Image, is_16bit: bool) -> np.ndarray:
        """Convert image to RGB format."""
        if is_16bit:
            img_array = np.array(img)
            if img_array.ndim == 2:
                return np.stack([img_array] * 3, axis=-1)
            return img_array
        return np.array(img.convert("RGB"))

    def _convert_to_rgba(self, img: Image.Image, is_16bit: bool) -> np.ndarray:
        """Convert image to RGBA format."""
        if is_16bit:
            img_array = np.array(img)
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            alpha = np.full(img_array.shape[:2], np.iinfo(img_array.dtype).max, dtype=img_array.dtype)
            return np.concatenate([img_array, alpha[..., np.newaxis]], axis=-1)
        return np.array(img.convert("RGBA"))

    def _convert_to_grayscale(self, img: Image.Image, is_16bit: bool) -> np.ndarray:
        """Convert image to grayscale format."""
        if is_16bit:
            return np.array(img)
        return np.array(img.convert("L"))

    @cachedmethod(
        cache=lambda _: ImageCache._cache,
        key=lambda self: self._cache_key(),
        lock=lambda _: ImageCache._lock,
    )
    def _load_data(self) -> np.ndarray:
        """Load the image from disk (cached via @cachedmethod)."""
        with Image.open(self.path) as img:
            target_format = self.format.upper()
            img_array = self._convert_to_format(img, target_format)

            # Handle BGR format by swapping R and B channels
            if target_format == "BGR" and img_array.ndim == 3:
                img_array = img_array[..., ::-1].copy()

            # Handle channels-first format
            if self.channels_first and img_array.ndim == 3:
                img_array = img_array.transpose(2, 0, 1)

        return img_array

    @property
    def data(self) -> np.ndarray:
        """
        Load and return the image data as a numpy array.

        The image is loaded from disk on first access and cached in a global
        LRU cache. Subsequent accesses return the cached data if available.
        When the cache is full, least recently used images are automatically
        evicted based on total memory usage.

        Returns:
            numpy.ndarray: The image data as a numpy array with shape:
                - (H, W, C) if channels_first is False
                - (C, H, W) if channels_first is True

        Raises:
            FileNotFoundError: If the image file does not exist
            PIL.UnidentifiedImageError: If the file cannot be read as an image
        """
        return self._load_data()

    def clear_cache(self) -> None:
        """Remove this image from the cache to free memory."""
        with ImageCache._lock:
            ImageCache._cache.pop(self._cache_key(), None)

    @cached_property
    def width(self) -> int:
        """Get the image width without fully loading the image data."""
        with Image.open(self.path) as img:
            return img.width

    @cached_property
    def height(self) -> int:
        """Get the image height without fully loading the image data."""
        with Image.open(self.path) as img:
            return img.height

    @cached_property
    def size(self) -> tuple[int, int]:
        """Get the image size (width, height) without fully loading the image data."""
        with Image.open(self.path) as img:
            return img.size

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the image data.

        Returns:
            tuple: Shape of the image array. If channels_first is False: (H, W, C).
                   If channels_first is True: (C, H, W). For grayscale: (H, W).
        """
        return self.data.shape

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"

    def __fspath__(self) -> str:
        """Allow LazyImage to be used in os.path operations."""
        return str(self.path)
