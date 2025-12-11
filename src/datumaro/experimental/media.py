# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Media types for lazy loading and caching of image data.

This module provides the LazyImage class for lazy loading images from disk,
along with a global LRU cache to manage memory usage when working with many images.
"""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TypeAlias

import numpy as np
from PIL import Image


class _ImageCache:
    """
    Thread-safe LRU cache for image data.

    This cache limits the total number of images kept in memory.
    When the cache is full, the least recently used image is evicted.
    """

    def __init__(self, maxsize: int = 100):
        self._cache: OrderedDict[tuple[str, str, bool], np.ndarray] = OrderedDict()
        self._maxsize = maxsize
        self._lock = Lock()

    @property
    def maxsize(self) -> int:
        """Get the maximum cache size."""
        return self._maxsize

    @maxsize.setter
    def maxsize(self, value: int) -> None:
        """Set the maximum cache size and evict if necessary."""
        with self._lock:
            self._maxsize = value
            self._evict_if_needed()

    def get(self, key: tuple[str, str, bool]) -> np.ndarray | None:
        """Get an item from the cache, moving it to the end (most recently used)."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: tuple[str, str, bool], value: np.ndarray) -> None:
        """Add an item to the cache, evicting LRU items if necessary."""
        with self._lock:
            if self._maxsize <= 0:
                return
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                self._cache[key] = value
                self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Evict least recently used items if cache exceeds maxsize."""
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached images."""
        with self._lock:
            self._cache.clear()

    def remove(self, key: tuple[str, str, bool]) -> None:
        """Remove a specific item from the cache."""
        with self._lock:
            self._cache.pop(key, None)

    def __len__(self) -> int:
        """Return the number of cached images."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: tuple[str, str, bool]) -> bool:
        """Check if an item is in the cache."""
        with self._lock:
            return key in self._cache


# Global image cache instance
_image_cache = _ImageCache(maxsize=100)


def set_image_cache_size(maxsize: int) -> None:
    """
    Set the maximum number of images to keep in the global LRU cache.

    Args:
        maxsize: Maximum number of images to cache. Set to 0 to disable caching.

    Example:
        >>> from datumaro.experimental import set_image_cache_size
        >>> set_image_cache_size(50)  # Keep at most 50 images in memory
    """
    _image_cache.maxsize = maxsize


def clear_image_cache() -> None:
    """
    Clear all cached images from memory.

    Example:
        >>> from datumaro.experimental import clear_image_cache
        >>> clear_image_cache()  # Free all cached image memory
    """
    _image_cache.clear()


def get_image_cache_size() -> int:
    """
    Get the current number of images in the cache.

    Returns:
        Number of images currently cached.
    """
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
    up to 100 images are cached. Use `set_image_cache_size()` to adjust this limit.

    Attributes:
        path: The file path to the image (can be a string or Path object)
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
        >>> set_image_cache_size(50)  # Limit to 50 images
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

    path: str | Path
    format: str = "RGB"
    channels_first: bool = False

    def __post_init__(self) -> None:
        # Ensure path is stored as a string for consistency
        if isinstance(self.path, Path):
            object.__setattr__(self, "path", str(self.path))

    def _cache_key(self) -> tuple[str, str, bool]:
        """Generate a cache key for this image configuration."""
        return (str(self.path), self.format.upper(), self.channels_first)

    def _load_image(self) -> np.ndarray:
        """Load the image from disk."""
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
        evicted.

        Returns:
            numpy.ndarray: The image data as a numpy array with shape:
                - (H, W, C) if channels_first is False
                - (C, H, W) if channels_first is True

        Raises:
            FileNotFoundError: If the image file does not exist
            PIL.UnidentifiedImageError: If the file cannot be read as an image
        """
        key = self._cache_key()

        # Try to get from cache
        cached = _image_cache.get(key)
        if cached is not None:
            return cached

        # Load and cache
        img_array = self._load_image()
        _image_cache.put(key, img_array)

        return img_array

    def clear_cache(self) -> None:
        """Remove this image from the cache to free memory."""
        _image_cache.remove(self._cache_key())

    @property
    def width(self) -> int:
        """Get the image width without fully loading the image data."""
        with Image.open(self.path) as img:
            return img.width

    @property
    def height(self) -> int:
        """Get the image height without fully loading the image data."""
        with Image.open(self.path) as img:
            return img.height

    @property
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
        return f"LazyImage({self.path})"

    def __fspath__(self) -> str:
        """Allow LazyImage to be used in os.path operations."""
        return str(self.path)


# Type alias for image path fields that accept strings, Paths, or LazyImage objects.
# Use this type annotation to avoid type checker warnings when passing strings for LazyImage.
ImagePathLike: TypeAlias = str | Path | LazyImage
