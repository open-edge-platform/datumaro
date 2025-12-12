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


# Global image cache with byte-size limiting (default 256 MB)
_image_cache: LRUCache[tuple[str, str, bool], np.ndarray] = LRUCache(
    maxsize=_DEFAULT_CACHE_SIZE_BYTES, getsizeof=_get_image_size
)
_cache_lock = RLock()


def set_image_cache_size(maxsize_bytes: int) -> None:
    """
    Set the maximum size in bytes for the global LRU image cache.

    When the cache exceeds this size, the least recently used images are evicted.

    Args:
        maxsize_bytes: Maximum cache size in bytes. Set to 0 to disable caching.

    Example:
        >>> from datumaro.experimental import set_image_cache_size
        >>> set_image_cache_size(512 * 1024 * 1024)  # Set cache to 512 MB
    """
    global _image_cache  # noqa: PLW0603
    with _cache_lock:
        # Create a new cache with the new size, preserving existing items if they fit
        new_cache: LRUCache[tuple[str, str, bool], np.ndarray] = LRUCache(
            maxsize=maxsize_bytes, getsizeof=_get_image_size
        )
        if maxsize_bytes > 0:
            for key, value in _image_cache.items():
                try:
                    new_cache[key] = value
                except ValueError:
                    # Cache is full, stop adding items
                    break
        _image_cache = new_cache


def clear_image_cache() -> None:
    """
    Clear all cached images from memory.

    Example:
        >>> from datumaro.experimental import clear_image_cache
        >>> clear_image_cache()  # Free all cached image memory
    """
    with _cache_lock:
        _image_cache.clear()


def get_image_cache_info() -> dict[str, int]:
    """
    Get information about the current cache state.

    Returns:
        A dictionary with:
        - 'count': Number of images currently cached
        - 'current_size': Current cache size in bytes
        - 'max_size': Maximum cache size in bytes
    """
    with _cache_lock:
        return {
            "count": len(_image_cache),
            "current_size": _image_cache.currsize,
            "max_size": _image_cache.maxsize,
        }


def get_image_cache_size() -> int:
    """
    Get the current number of images in the cache.

    Returns:
        Number of images currently cached.

    Note:
        For more detailed cache information including byte sizes,
        use `get_image_cache_info()` instead.
    """
    with _cache_lock:
        return len(_image_cache)


@dataclass
class LazyImage:
    """
    A wrapper class that holds an image path and provides lazy loading of image data.

    This class enables lazy loading patterns where the image is only loaded from
    disk when the `data` property is accessed. Loaded images are stored in a
    global LRU cache that limits total memory usage.

    The cache keeps the most recently accessed images in memory and automatically
    evicts the least recently used images when the cache is full. By default,
    the cache is limited to 256 MB. Use `set_image_cache_size()` to adjust this limit.

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
        >>> from datumaro.experimental import set_image_cache_size, clear_image_cache
        >>> set_image_cache_size(512 * 1024 * 1024)  # Limit to 512 MB
        >>> clear_image_cache()  # Clear all cached images

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

    @cachedmethod(cache=lambda _: _image_cache, key=lambda self: self._cache_key(), lock=lambda _: _cache_lock)
    def _load_data(self) -> np.ndarray:
        """Load the image from disk (cached via @cachedmethod)."""
        with Image.open(self.path) as img:
            # Convert to target format
            if self.format.upper() in ("RGB", "BGR"):
                converted = img.convert("RGB")
            elif self.format.upper() == "RGBA":
                converted = img.convert("RGBA")
            elif self.format.upper() == "L":
                converted = img.convert("L")
            else:
                converted = img

            img_array = np.array(converted, dtype=np.uint8)

            # Handle BGR format by swapping R and B channels
            if self.format.upper() == "BGR" and img_array.ndim == 3:
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
        with _cache_lock:
            _image_cache.pop(self._cache_key(), None)

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

    def __str__(self) -> str:
        return str(self.path)

    def __fspath__(self) -> str:
        """Allow LazyImage to be used in os.path operations."""
        return str(self.path)
