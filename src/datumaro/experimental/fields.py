# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Field implementations for various data types including tensors, images, and bounding boxes.

This module provides concrete field implementations that handle serialization
to/from Polars DataFrames for different data types commonly used in machine
learning and computer vision applications.
"""

from dataclasses import dataclass
from typing import Any, TypeVar, Union

import numpy as np
import polars as pl
from typing_extensions import TypeAlias

from .schema import Field, Semantic
from .type_registry import from_polars_data, to_numpy

T = TypeVar("T")

PolarsDataType: TypeAlias = Union[type[pl.DataType], pl.DataType]


@dataclass(frozen=True)
class TensorField(Field):
    """
    Represents a tensor field with semantic tags and data type information.

    This field handles n-dimensional tensor data by flattening it for storage
    and preserving shape information separately for reconstruction.

    Attributes:
        semantic: Semantic tags describing the tensor's purpose
        dtype: Polars data type for tensor elements
    """

    semantic: Semantic
    dtype: PolarsDataType = pl.UInt8()

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate Polars schema with separate columns for data and shape."""
        return {name: pl.List(self.dtype), name + "_shape": pl.List(pl.Int32())}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert tensor to flattened data and shape information."""
        numpy_value = to_numpy(value, self.dtype)
        return {
            name: pl.Series(name, [numpy_value.reshape(-1)]),
            name + "_shape": pl.Series(name + "_shape", [numpy_value.shape]),
        }

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct tensor from flattened data using stored shape."""
        flat_data = df[name][row_index]
        shape = df[name + "_shape"][row_index]
        numpy_data = np.array(flat_data).reshape(shape)
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


def image_field(dtype: Any, format: str = "RGB", semantic: Semantic = Semantic.Default) -> Any:
    """
    Create an ImageField instance with the specified parameters.

    Args:
        dtype: Polars data type for pixel values
        format: Image color format (defaults to "RGB")
        semantic: Semantic tags describing the image's purpose (optional)

    Returns:
        ImageField instance configured with the given parameters
    """
    return ImageField(semantic=semantic, dtype=dtype, format=format)


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
        return {name: pl.Series(name, [bytes(numpy_value)])}

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


@dataclass(frozen=True)
class BBoxField(Field):
    """
    Represents a bounding box field with format and normalization options.

    Handles bounding box data with support for different coordinate formats
    and optional normalization to [0,1] range.

    Attributes:
        semantic: Semantic tags describing the bounding box purpose
        dtype: Polars data type for coordinate values
        format: Coordinate format (e.g., "x1y1x2y2", "xywh")
        normalize: Whether coordinates are normalized to [0,1] range
    """

    semantic: Semantic
    dtype: PolarsDataType = pl.Float32()
    format: str = "x1y1x2y2"
    normalize: bool = False

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for bounding box as list of 4-element arrays."""
        return {name: pl.List(pl.Array(self.dtype, 4))}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert bounding box tensor to Polars list format."""
        numpy_value = to_numpy(value, self.dtype)

        return {
            name: pl.Series(
                name,
                numpy_value.reshape(1, -1, 4),
                dtype=pl.List(pl.Array(self.dtype, 4)),
            )
        }

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct bounding box tensor from Polars data."""
        polars_data = df[name][row_index]
        return from_polars_data(polars_data, target_type)  # type: ignore


def bbox_field(
    dtype: Any,
    format: str = "x1y1x2y2",
    normalize: bool = False,
    semantic: Semantic = Semantic.Default,
) -> Any:
    """
    Create a BBoxField instance with the specified parameters.

    Args:
        dtype: Polars data type for coordinate values
        format: Coordinate format (defaults to "x1y1x2y2")
        normalize: Whether coordinates are normalized (defaults to False)
        semantic: Semantic tags describing the bounding box purpose (optional)

    Returns:
        BBoxField instance configured with the given parameters
    """
    return BBoxField(semantic=semantic, dtype=dtype, format=format, normalize=normalize)


@dataclass(frozen=True)
class RotatedBBoxField(Field):
    """
    Represents a rotated bounding box field with format and normalization options.

    Handles rotated bounding box data with support for different coordinate formats
    and optional normalization to [0,1] range. Stores all attributes (cx, cy, w, h, r)
    in one tensor similar to BBoxField.

    Attributes:
        semantic: Semantic tags describing the rotated bounding box purpose
        dtype: Polars data type for coordinate values
        format: Coordinate format (e.g., "cxcywhr", "cxcywha" for angle in degrees)
        normalize: Whether coordinates are normalized to [0,1] range
    """

    semantic: Semantic
    dtype: PolarsDataType = pl.Float32()
    format: str = "cxcywhr"
    normalize: bool = False

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for rotated bounding box as list of 5-element arrays."""
        return {name: pl.List(pl.Array(self.dtype, 5))}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert rotated bounding box tensor to Polars list format."""
        numpy_value = to_numpy(value, self.dtype)

        return {
            name: pl.Series(
                name,
                numpy_value.reshape(1, -1, 5),
                dtype=pl.List(pl.Array(self.dtype, 5)),
            )
        }

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct rotated bounding box tensor from Polars data."""
        polars_data = df[name][row_index]
        return from_polars_data(polars_data, target_type)  # type: ignore


def rotated_bbox_field(
    dtype: Any,
    format: str = "cxcywhr",
    normalize: bool = False,
    semantic: Semantic = Semantic.Default,
) -> Any:
    """
    Create a RotatedBBoxField instance with the specified parameters.

    Args:
        dtype: Polars data type for coordinate values
        format: Coordinate format (defaults to "cxcywhr" for cx,cy,w,h,rotation_radians)
        normalize: Whether coordinates are normalized (defaults to False)
        semantic: Semantic tags describing the rotated bounding box purpose (optional)

    Returns:
        RotatedBBoxField instance configured with the given parameters
    """
    return RotatedBBoxField(semantic=semantic, dtype=dtype, format=format, normalize=normalize)


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
        return {name: pl.Struct([pl.Field("width", pl.Int32()), pl.Field("height", pl.Int32())])}

    def to_polars(self, name: str, value: ImageInfo) -> dict[str, pl.Series]:
        return {name: pl.Series(name, [{"width": value.width, "height": value.height}])}

    def from_polars(
        self, name: str, row_index: int, df: pl.DataFrame, target_type: type
    ) -> ImageInfo:
        if not issubclass(target_type, ImageInfo):
            raise TypeError(f"Expected target_type to be ImageInfo, got {target_type}")
        struct_val = df[name][row_index]
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
        return {name: pl.Series(name, [str(value)])}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type):
        """Extract path string from Polars data."""
        return target_type(df[name][row_index])


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
class LabelField(Field):
    """
    Represents a unified label annotation field that supports both single and multi-label scenarios.

    This field automatically detects whether the input is a single label or multiple labels
    and handles the conversion accordingly:
    - Single labels: stored as Int32
    - Multi-labels: stored as List(Int32)
    """

    semantic: Semantic
    dtype: PolarsDataType = pl.UInt8()
    multi_label: bool = False  # Flag to indicate if this field should handle multi-labels
    is_list: bool = False

    @property
    def _pl_type(self) -> pl.DataType:
        pl_type = self.dtype
        if self.multi_label:
            pl_type = pl.List(pl_type)
        if self.is_list:
            pl_type = pl.List(pl_type)
        return pl_type

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema based on whether this is single or multi-label."""
        return {name: self._pl_type}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert label(s) to Polars format for single or multi-label cases."""
        pl_type = self._pl_type

        if value is None:
            return {name: pl.Series(name, [None], dtype=pl_type)}

        if self.multi_label:
            return {name: pl.Series(name, [to_numpy(value)], dtype=pl.List(self.dtype))}

        return {name: pl.Series(name, [value], dtype=pl_type)}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct label(s) from Polars data."""
        data = df[name][row_index]
        return from_polars_data(data, target_type)


def label_field(
    dtype: Any = pl.Int32(),
    semantic: Semantic = Semantic.Default,
    multi_label: bool = False,
    is_list: bool = False,
) -> Any:
    """
    Create a LabelField instance with the specified parameters.

    Args:
        dtype: Polars data type for label values (defaults to pl.Int32())
        semantic: Semantic tags describing the label purpose (optional)
        multi_label: Whether this field should handle multiple labels (defaults to False)
        is_list: Whether this field should be treated as a list type (defaults to False)

    Returns:
        LabelField instance configured with the given parameters
    """
    return LabelField(semantic=semantic, dtype=dtype, multi_label=multi_label, is_list=is_list)


def convert_numpy_object_array_to_series(data: np.ndarray) -> pl.Series:
    """
    Convert ragged numpy object arrays to Polars Series recursively.

    Handles nested object arrays containing variable-length lists.

    Example:
        >>> import numpy as np
        >>> ragged = np.array([
        ...     np.array([1, 2, 3]),
        ...     np.array([4, 5]),
        ...     np.array([6, 7, 8, 9])
        ... ], dtype=object)
        >>> series = convert_numpy_object_array_to_series(ragged)
        >>> print(series)
        shape: (3,)
        Series: '' [list[i64]]
        [
                [1, 2, 3]
                [4, 5]
                [6, 7, … 9]
        ]

        # Compare with direct conversion which results
        # into an object Series instead of a list Series:
        >>> direct = pl.Series(ragged)
        >>> print(direct)
        shape: (3,)
        Series: '' [o][object]
        [
                [1 2 3]
                [4 5]
                [6 7 8 9]
        ]
    """
    if data.dtype == object:
        return pl.Series([convert_numpy_object_array_to_series(elem) for elem in data])
    return pl.Series(data)


@dataclass(frozen=True)
class PolygonField(Field):
    """
    Represents a polygon field with format and normalization options.

    Handles polygon data with support for different coordinate formats
    and optional normalization to [0,1] range. Polygons are stored as
    variable-length lists of coordinate pairs.

    Attributes:
        semantic: Semantic tags describing the polygon purpose
        dtype: Polars data type for coordinate values
        format: Coordinate format (e.g., "xy", "yx")
        normalize: Whether coordinates are normalized to [0,1] range
    """

    semantic: Semantic
    dtype: PolarsDataType = pl.Float32()
    format: str = "xy"
    normalize: bool = False

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for polygon as list of coordinate values."""
        return {name: pl.List(pl.List(pl.Array(self.dtype, 2)))}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert polygon tensor to Polars list format."""
        numpy_value = to_numpy(value, self.dtype)

        series = convert_numpy_object_array_to_series(numpy_value)

        return {name: pl.Series(name, [series], dtype=pl.List(pl.List(pl.Array(self.dtype, 2))))}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct polygon tensor from Polars data."""
        polars_data = df[name][row_index]
        return from_polars_data(polars_data, target_type)  # type: ignore


def polygon_field(
    dtype: Any,
    format: str = "xy",
    normalize: bool = False,
    semantic: Semantic = Semantic.Default,
) -> Any:
    """
    Create a PolygonField instance with the specified parameters.

    Args:
        dtype: Polars data type for coordinate values
        format: Coordinate format (defaults to "xy")
        normalize: Whether coordinates are normalized (defaults to False)
        semantic: Semantic tags describing the polygon purpose (optional)

    Returns:
        PolygonField instance configured with the given parameters
    """
    return PolygonField(semantic=semantic, dtype=dtype, format=format, normalize=normalize)


@dataclass(frozen=True)
class MaskField(Field):
    """
    Represents a mask tensor field for binary or indexed segmentation masks.

    Similar to TensorField but specialized for masks: handles single-channel
    2D arrays with no color format specification. Uses uint8 as the default
    data type suitable for binary masks, class masks, or instance masks.

    Attributes:
        semantic: Semantic tags describing the mask purpose
        dtype: Polars data type for mask values (defaults to uint8)
    """

    semantic: Semantic
    dtype: PolarsDataType = pl.UInt8()

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate Polars schema with separate columns for data and shape."""
        return {name: pl.List(self.dtype), name + "_shape": pl.List(pl.Int32())}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert mask tensor to flattened data and shape information."""
        numpy_value = to_numpy(value, self.dtype)
        return {
            name: pl.Series(name, [numpy_value.reshape(-1)]),
            name + "_shape": pl.Series(name + "_shape", [numpy_value.shape]),
        }

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct mask tensor from flattened data using stored shape."""
        flat_data = df[name][row_index]
        shape = df[name + "_shape"][row_index]
        numpy_data = np.array(flat_data).reshape(shape)
        return from_polars_data(numpy_data, target_type)  # type: ignore


def mask_field(dtype: Any = pl.UInt8(), semantic: Semantic = Semantic.Default) -> Any:
    """
    Create a MaskField instance with the specified parameters.

    Args:
        dtype: Polars data type for mask values (defaults to pl.UInt8())
        semantic: Semantic tags describing the mask purpose (optional)

    Returns:
        MaskField instance configured with the given parameters
    """
    return MaskField(semantic=semantic, dtype=dtype)


@dataclass(frozen=True)
class InstanceMaskField(Field):
    """
    Represents an instance mask tensor field for instance segmentation masks.

    Handles 3D tensor data of shape (N, H, W) where N is the number of instances,
    H and W are the mask height and width. Each mask is a binary mask representing
    a single instance. Unlike MaskField, this does not contain category information.

    Attributes:
        semantic: Semantic tags describing the instance mask purpose
        dtype: Polars data type for mask values (defaults to bool for binary masks)
    """

    semantic: Semantic
    dtype: PolarsDataType = pl.Boolean()

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate Polars schema with separate columns for data and shape."""
        return {name: pl.List(self.dtype), name + "_shape": pl.List(pl.Int32())}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert instance mask tensor to flattened data and shape information."""
        numpy_value = to_numpy(value, self.dtype)
        return {
            name: pl.Series(name, [numpy_value.reshape(-1)]),
            name + "_shape": pl.Series(name + "_shape", [numpy_value.shape]),
        }

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct instance mask tensor from flattened data using stored shape."""
        flat_data = df[name][row_index]
        shape = df[name + "_shape"][row_index]
        numpy_data = np.array(flat_data).reshape(shape)
        return from_polars_data(numpy_data, target_type)  # type: ignore


def instance_mask_field(dtype: Any = pl.Boolean(), semantic: Semantic = Semantic.Default) -> Any:
    """
    Create an InstanceMaskField instance with the specified parameters.

    Args:
        dtype: Polars data type for mask values (defaults to pl.Boolean())
        semantic: Semantic tags describing the instance mask purpose (optional)

    Returns:
        InstanceMaskField instance configured with the given parameters
    """
    return InstanceMaskField(semantic=semantic, dtype=dtype)


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
        if not callable(value):
            raise TypeError(f"Expected callable, got {type(value)}")
        return {name: pl.Series(name, [value])}

    def from_polars(
        self, name: str, row_index: int, df: pl.DataFrame, target_type: type
    ) -> callable:
        """Extract callable from Polars dataframe."""
        value = df[name][row_index]
        if not callable(value):
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
