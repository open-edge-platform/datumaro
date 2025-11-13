# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from datumaro.v2 import Field, Semantic
from datumaro.v2.fields.base import PolarsDataType, T
from datumaro.v2.type_registry import from_polars_data, to_numpy


@dataclass(frozen=True)
class TensorField(Field):
    """
    Represents a tensor field with semantic tags and data type information.

    This field handles n-dimensional tensor data by flattening it for storage
    and preserving shape information separately for reconstruction.

    Attributes:
        semantic: Semantic tags describing the tensor's purpose
        dtype: Polars data type for tensor elements
        channels_first: Whether the tensor uses channels-first format (C, H, W) vs channels-last (H, W, C)
    """

    semantic: Semantic
    dtype: PolarsDataType = field(default_factory=pl.UInt8)
    channels_first: bool = False

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate Polars schema with separate columns for data and shape."""
        return {name: pl.List(self.dtype), name + "_shape": pl.List(pl.Int32())}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert tensor to flattened data and shape information."""
        numpy_value = to_numpy(value, self.dtype)

        if self.channels_first and numpy_value is not None:
            numpy_value = numpy_value.swapaxes(0, -1)

        schema = self.to_polars_schema("tensor")

        numpy_value_shape: Any | None = None
        if numpy_value is not None:
            numpy_value_shape = numpy_value.shape
            numpy_value = numpy_value.reshape(-1)

        return {
            name: pl.Series(name, [numpy_value], dtype=schema["tensor"]),
            name + "_shape": pl.Series(name + "_shape", [numpy_value_shape], dtype=schema["tensor_shape"]),
        }

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct tensor from flattened data using stored shape."""
        flat_data = df[name][row_index]
        shape = df[name + "_shape"][row_index]
        numpy_data = np.array(flat_data).reshape(shape) if flat_data is not None else None

        if self.channels_first and numpy_data is not None:
            numpy_data = numpy_data.swapaxes(0, -1)

        return from_polars_data(numpy_data, target_type)  # type: ignore


def tensor_field(dtype: Any, semantic: Semantic = Semantic.Default) -> Any:
    """
    Create a TensorField instance with the specified semantic tags and data type.

    Args:
        dtype: Polars data type for tensor elements
        semantic: Semantic tags describing the tensor's purpose (optional)

    Returns:
        TensorField instance configured with the given parameters
    """
    return TensorField(semantic=semantic, dtype=dtype)


@dataclass(frozen=True)
class ImageField(TensorField):
    """
    Represents an image tensor field with format information.

    Extends TensorField to include image-specific metadata such as
    color format (RGB, BGR, etc.).

    Attributes:
        format: Image color format (e.g., "RGB", "BGR", "RGBA")
    """

    format: str = "RGB"


def image_field(
    dtype: Any,
    format: str = "RGB",
    channels_first: bool = False,
    semantic: Semantic = Semantic.Default,
) -> Any:
    """
    Create an ImageField instance with the specified parameters.

    Args:
        dtype: Polars data type for pixel values
        format: Image color format (defaults to "RGB")
        semantic: Semantic tags describing the image's purpose (optional)

    Returns:
        ImageField instance configured with the given parameters
    """
    return ImageField(semantic=semantic, dtype=dtype, format=format, channels_first=channels_first)


@dataclass(frozen=True)
class ImageBytesField(Field):
    """
    Represents an image field stored as bytes data.

    This field stores image data as encoded bytes (PNG, JPEG, BMP, etc.) and
    provides conversion capabilities to decode them into numpy arrays or encode
    numpy arrays into bytes format.

    Attributes:
        semantic: Semantic tags describing the image's purpose
        format: Image encoding format (e.g., "PNG", "JPEG", "BMP"). If None,
                auto-detects format when decoding or defaults to PNG when encoding.
    """

    semantic: Semantic

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for image bytes as binary data."""
        return {name: pl.Binary()}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert image data to bytes and store in Polars series."""
        numpy_value = to_numpy(value, pl.Binary)
        bytes_value = bytes(numpy_value) if numpy_value is not None else None
        return {name: pl.Series(name, [bytes_value], dtype=pl.Binary())}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct image from bytes data."""
        bytes_data = df[name][row_index]
        return from_polars_data(bytes_data, target_type)  # type: ignore


def image_bytes_field(semantic: Semantic = Semantic.Default) -> Any:
    """
    Create an ImageBytesField instance with the specified parameters.

    Args:
        semantic: Semantic tags describing the image's purpose (optional)

    Returns:
        ImageBytesField instance configured with the given parameters
    """
    return ImageBytesField(semantic=semantic)


@dataclass
class ImageInfo:
    """Container for image metadata (width and height)."""

    width: int
    height: int


@dataclass(frozen=True)
class ImageInfoField(Field):
    """
    Represents image metadata (width, height) as a Polars struct.
    """

    semantic: Semantic

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        return {
            name: pl.Struct(
                [
                    Field("width", pl.Int32()),
                    Field("height", pl.Int32()),
                ]
            )
        }

    def to_polars(self, name: str, value: ImageInfo | None) -> dict[str, pl.Series]:
        schema = self.to_polars_schema("info")
        if value is not None:
            data = [{"width": value.width, "height": value.height}]
        else:
            data = [None]
        return {name: pl.Series(name, data, dtype=schema["info"])}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type) -> ImageInfo | None:
        if not issubclass(target_type, ImageInfo):
            raise TypeError(f"Expected target_type to be ImageInfo, got {target_type}")
        struct_val = df[name][row_index]
        if struct_val is None:
            return None
        return ImageInfo(width=struct_val["width"], height=struct_val["height"])


def image_info_field(semantic: Semantic = Semantic.Default) -> Any:
    """
    Create an ImageInfoField instance for storing image width and height.

    Args:
        semantic: Optional semantic tags for disambiguation (e.g., Semantic.Left)

    Returns:
        ImageInfoField instance configured with the given semantic tags
    """
    return ImageInfoField(semantic=semantic)


@dataclass(frozen=True)
class ImagePathField(Field):
    """
    Represents a field containing the file path to an image on disk.

    This field stores image file paths as strings and is typically used
    as input for lazy loading operations where images are loaded on-demand.

    Attributes:
        semantic: Semantic tags describing the image path's purpose
    """

    semantic: Semantic

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for string path column."""
        return {name: pl.String()}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert path string to Polars series."""
        return {name: pl.Series(name, [str(value) if value is not None else None])}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type) -> Any:
        """Extract path string from Polars data."""
        data = df[name][row_index]
        return target_type(data) if data is not None else None


def image_path_field(semantic: Semantic = Semantic.Default) -> Any:
    """
    Create an ImagePathField instance with the specified semantic tags.

    Args:
        semantic: Semantic tags describing the image path's purpose (optional)

    Returns:
        ImagePathField instance configured with the given semantic tags
    """
    return ImagePathField(semantic=semantic)


@dataclass(frozen=True)
class ImageCallableField(Field):
    """
    Represents a field that stores a callable which returns an image as a numpy array.

    This field is useful for lazy loading scenarios where images are generated
    or loaded on-demand. The callable should return a numpy array representing
    the image data when invoked.

    Attributes:
        semantic: Semantic tags describing the callable's purpose
        format: Expected image color format (e.g., "RGB", "BGR", "RGBA")
    """

    semantic: Semantic
    format: str = "RGB"

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Return schema with Object type to store callable."""
        return {name: pl.Object}

    def to_polars(self, name: str, value: callable) -> dict[str, pl.Series]:
        """Store callable as Object in Polars series."""
        if not callable(value) and value is not None:
            raise TypeError(f"Expected callable, got {type(value)}")
        return {name: pl.Series(name, [value])}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type) -> callable:
        """Extract callable from Polars dataframe."""
        value = df[name][row_index]
        if not callable(value) and value is not None:
            raise TypeError(f"Expected callable in column {name}, got {type(value)}")
        return value


def image_callable_field(format: str = "RGB", semantic: Semantic = Semantic.Default) -> Any:
    """
    Create an ImageCallableField instance for storing image-generating callables.

    Args:
        format: Expected image color format (defaults to "RGB")
        semantic: Semantic tags describing the callable's purpose (optional)

    Returns:
        ImageCallableField instance configured with the given parameters
    """
    return ImageCallableField(semantic=semantic, format=format)
