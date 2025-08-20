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
from enum import IntEnum
from typing import Any, TypeVar

import numpy as np
import polars as pl

from .schema import Field, Semantic
from .type_registry import from_polars_data, to_numpy

T = TypeVar("T")


@dataclass
class Label:
    """Minimal Label annotation class."""

    label: Any  # Can be int, str, or other types for flexibility


class PointsVisibility(IntEnum):
    """Point visibility enumeration."""

    HIDDEN = 0
    VISIBLE = 1
    ABSENT = 2


@dataclass
class Points:
    """Minimal Points annotation class."""

    points: list
    visibility: list
    label: int | None = None
    group: int = 0
    object_id: int = 0
    z_order: int = 0
    attributes: dict | None = None

    # Add Visibility as a class attribute for compatibility
    Visibility = PointsVisibility

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class Polygon:
    """Minimal Polygon annotation class."""

    points: list
    label: int | None = None
    group: int = 0
    object_id: int = 0
    z_order: int = 0
    attributes: dict | None = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class Ellipse:
    """Minimal Ellipse annotation class."""

    points: list
    label: int | None = None
    group: int = 0
    object_id: int = 0
    z_order: int = 0
    attributes: dict | None = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class RotatedBbox:
    """Minimal RotatedBbox annotation class."""

    points: list
    label: int | None = None
    group: int = 0
    object_id: int = 0
    z_order: int = 0
    attributes: dict | None = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


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
    dtype: Any

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

    format: str


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
    dtype: Any
    format: str
    normalize: bool

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
    dtype: Any
    multi_label: bool = False  # Flag to indicate if this field should handle multi-labels

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema based on whether this is single or multi-label."""
        if self.multi_label:
            return {name: pl.List(self.dtype)}
        else:
            return {name: self.dtype}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert label(s) to Polars format, handling both single and multi-label cases."""
        if value is None:
            return {name: pl.Series(name, [None], dtype=self.dtype)}

        # Handle single Label object
        if hasattr(value, "label"):
            return {name: pl.Series(name, [value.label], dtype=self.dtype)}

        # Handle numpy array or list (multi-label)
        elif isinstance(value, (np.ndarray, list)):
            if isinstance(value, np.ndarray):
                value_list = value.tolist()
            else:
                value_list = value
            return {name: pl.Series(name, [value_list], dtype=pl.List(self.dtype))}

        # Handle single integer value
        elif isinstance(value, (int, np.integer)):
            return {name: pl.Series(name, [value], dtype=self.dtype)}

        else:
            raise ValueError(f"Unsupported label value type: {type(value)}")

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct label(s) from Polars data."""
        data = df[name][row_index]

        # If data is a list, it's multi-label
        if isinstance(data, list):
            if target_type == np.ndarray or target_type is np.ndarray:
                return np.array(data, dtype=np.int64)
            elif target_type is list:
                return data
            else:
                # Try to create target_type with the array/list
                return target_type(data)

        # Single label case
        else:
            if hasattr(target_type, "__annotations__") and "label" in target_type.__annotations__:
                return target_type(label=data)
            else:
                return target_type(data)


def label_field(
    dtype: Any = pl.Int32(), semantic: Semantic = Semantic.Default, multi_label: bool = False
) -> Any:
    """
    Create a LabelField instance with the specified parameters.

    Args:
        dtype: Polars data type for label values (defaults to pl.Int32())
        semantic: Semantic tags describing the label purpose (optional)
        multi_label: Whether this field should handle multiple labels (defaults to False)

    Returns:
        LabelField instance configured with the given parameters
    """
    return LabelField(semantic=semantic, dtype=dtype, multi_label=multi_label)


@dataclass(frozen=True)
class SubsetField(Field):
    """
    Represents a dataset subset field for train/validation/test splits.
    """

    semantic: Semantic
    dtype: Any

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for subset as string."""
        return {name: pl.String()}

    def to_polars(self, name: str, value: str) -> dict[str, pl.Series]:
        """Convert subset to Polars format."""
        return {name: pl.Series(name, [value], dtype=pl.String())}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct subset from Polars data."""
        subset_value = df[name][row_index]
        if target_type is str:
            return subset_value
        return target_type(subset_value)


def subset_field(dtype: Any = pl.String(), semantic: Semantic = Semantic.Default) -> Any:
    """
    Create a SubsetField instance with the specified parameters.

    Args:
        dtype: Polars data type for subset values (defaults to pl.String())
        semantic: Semantic tags describing the subset purpose (optional)

    Returns:
        SubsetField instance configured with the given parameters
    """
    return SubsetField(semantic=semantic, dtype=dtype)


@dataclass(frozen=True)
class PolygonField(Field):
    """
    Represents a polygon annotation field.
    """

    semantic: Semantic
    dtype: Any

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for polygon as list of coordinates."""
        return {
            f"{name}_points": pl.List(pl.Float32()),
            f"{name}_label": pl.Int32(),
            f"{name}_group": pl.Int32(),
            f"{name}_object_id": pl.Int32(),
            f"{name}_z_order": pl.Int32(),
            f"{name}_attributes": pl.Struct([]),
        }

    def to_polars(self, name: str, value: Polygon) -> dict[str, pl.Series]:
        """Convert polygon to Polars format."""
        return {
            f"{name}_points": pl.Series(
                f"{name}_points", [value.points], dtype=pl.List(pl.Float32())
            ),
            f"{name}_label": pl.Series(
                f"{name}_label", [value.label if value.label is not None else -1], dtype=pl.Int32()
            ),
            f"{name}_group": pl.Series(f"{name}_group", [value.group], dtype=pl.Int32()),
            f"{name}_object_id": pl.Series(
                f"{name}_object_id", [value.object_id], dtype=pl.Int32()
            ),
            f"{name}_z_order": pl.Series(f"{name}_z_order", [value.z_order], dtype=pl.Int32()),
            f"{name}_attributes": pl.Series(
                f"{name}_attributes", [value.attributes], dtype=pl.Struct([])
            ),
        }

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct polygon from Polars data."""
        points = df[f"{name}_points"][row_index]
        label = df[f"{name}_label"][row_index]
        group = df[f"{name}_group"][row_index]
        object_id = df[f"{name}_object_id"][row_index]
        z_order = df[f"{name}_z_order"][row_index]
        attributes = df[f"{name}_attributes"][row_index]

        return target_type(
            points=points,
            label=None if label == -1 else label,
            group=group,
            object_id=object_id,
            z_order=z_order,
            attributes=attributes,
        )


def polygon_field(dtype: Any = pl.Float32(), semantic: Semantic = Semantic.Default) -> Any:
    """
    Create a PolygonField instance with the specified parameters.

    Args:
        dtype: Polars data type for polygon coordinate values (defaults to pl.Float32())
        semantic: Semantic tags describing the polygon purpose (optional)

    Returns:
        PolygonField instance configured with the given parameters
    """
    return PolygonField(semantic=semantic, dtype=dtype)


@dataclass(frozen=True)
class EllipseField(Field):
    """
    Represents an ellipse annotation field.
    """

    semantic: Semantic
    dtype: Any

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for ellipse as list of coordinates."""
        return {
            f"{name}_points": pl.List(pl.Float32()),
            f"{name}_label": pl.Int32(),
            f"{name}_group": pl.Int32(),
            f"{name}_object_id": pl.Int32(),
            f"{name}_z_order": pl.Int32(),
            f"{name}_attributes": pl.Struct([]),
        }

    def to_polars(self, name: str, value: Ellipse) -> dict[str, pl.Series]:
        """Convert ellipse to Polars format."""
        return {
            f"{name}_points": pl.Series(
                f"{name}_points", [value.points], dtype=pl.List(pl.Float32())
            ),
            f"{name}_label": pl.Series(
                f"{name}_label", [value.label if value.label is not None else -1], dtype=pl.Int32()
            ),
            f"{name}_group": pl.Series(f"{name}_group", [value.group], dtype=pl.Int32()),
            f"{name}_object_id": pl.Series(
                f"{name}_object_id", [value.object_id], dtype=pl.Int32()
            ),
            f"{name}_z_order": pl.Series(f"{name}_z_order", [value.z_order], dtype=pl.Int32()),
            f"{name}_attributes": pl.Series(
                f"{name}_attributes", [value.attributes], dtype=pl.Struct([])
            ),
        }

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct ellipse from Polars data."""
        points = df[f"{name}_points"][row_index]
        label = df[f"{name}_label"][row_index]
        group = df[f"{name}_group"][row_index]
        object_id = df[f"{name}_object_id"][row_index]
        z_order = df[f"{name}_z_order"][row_index]
        attributes = df[f"{name}_attributes"][row_index]

        # Extract ellipse parameters from points
        x1, y1, x2, y2 = points

        return target_type(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            label=None if label == -1 else label,
            group=group,
            object_id=object_id,
            z_order=z_order,
            attributes=attributes,
        )


def ellipse_field(dtype: Any = pl.Float32(), semantic: Semantic = Semantic.Default) -> Any:
    """
    Create an EllipseField instance with the specified parameters.

    Args:
        dtype: Polars data type for ellipse coordinate values (defaults to pl.Float32())
        semantic: Semantic tags describing the ellipse purpose (optional)

    Returns:
        EllipseField instance configured with the given parameters
    """
    return EllipseField(semantic=semantic, dtype=dtype)


@dataclass(frozen=True)
class RotatedBboxField(Field):
    """
    Represents a rotated bounding box annotation field.
    """

    semantic: Semantic
    dtype: Any

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for rotated bounding box as list of parameters."""
        return {
            f"{name}_points": pl.List(pl.Float32()),
            f"{name}_label": pl.Int32(),
            f"{name}_group": pl.Int32(),
            f"{name}_object_id": pl.Int32(),
            f"{name}_z_order": pl.Int32(),
            f"{name}_attributes": pl.Struct([]),
        }

    def to_polars(self, name: str, value: RotatedBbox) -> dict[str, pl.Series]:
        """Convert rotated bounding box to Polars format."""
        return {
            f"{name}_points": pl.Series(
                f"{name}_points", [value.points], dtype=pl.List(pl.Float32())
            ),
            f"{name}_label": pl.Series(
                f"{name}_label", [value.label if value.label is not None else -1], dtype=pl.Int32()
            ),
            f"{name}_group": pl.Series(f"{name}_group", [value.group], dtype=pl.Int32()),
            f"{name}_object_id": pl.Series(
                f"{name}_object_id", [value.object_id], dtype=pl.Int32()
            ),
            f"{name}_z_order": pl.Series(f"{name}_z_order", [value.z_order], dtype=pl.Int32()),
            f"{name}_attributes": pl.Series(
                f"{name}_attributes", [value.attributes], dtype=pl.Struct([])
            ),
        }

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct rotated bounding box from Polars data."""
        points = df[f"{name}_points"][row_index]
        label = df[f"{name}_label"][row_index]
        group = df[f"{name}_group"][row_index]
        object_id = df[f"{name}_object_id"][row_index]
        z_order = df[f"{name}_z_order"][row_index]
        attributes = df[f"{name}_attributes"][row_index]

        # Extract rotated bbox parameters from points
        cx, cy, w, h, r = points

        return target_type(
            cx=cx,
            cy=cy,
            w=w,
            h=h,
            r=r,
            label=None if label == -1 else label,
            group=group,
            object_id=object_id,
            z_order=z_order,
            attributes=attributes,
        )


def rotated_bbox_field(dtype: Any = pl.Float32(), semantic: Semantic = Semantic.Default) -> Any:
    """
    Create a RotatedBboxField instance with the specified parameters.

    Args:
        dtype: Polars data type for rotated bbox coordinate values (defaults to pl.Float32())
        semantic: Semantic tags describing the rotated bbox purpose (optional)

    Returns:
        RotatedBboxField instance configured with the given parameters
    """
    return RotatedBboxField(semantic=semantic, dtype=dtype)


# Generic object-carrying fields for media and annotations
@dataclass(frozen=True)
class MediaField(Field):
    """
    Field for storing an arbitrary Datumaro media object (Image, Video, PointCloud, ...).
    Stored as a Python object using Polars Object dtype.
    """

    semantic: Semantic

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        return {name: pl.Object()}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        return {name: pl.Series(name, [value], dtype=pl.Object())}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> Any:
        # Return the stored object as-is to preserve original media type
        return df[name][row_index]


def media_field(semantic: Semantic = Semantic.Default) -> Any:
    return MediaField(semantic=semantic)


@dataclass(frozen=True)
class AnnotationField(Field):
    """
    Field for storing a Datumaro Annotation object.
    Stored as a Python object using Polars Object dtype.
    """

    semantic: Semantic

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        return {name: pl.Object()}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        return {name: pl.Series(name, [value], dtype=pl.Object())}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> Any:
        # Return the stored object as-is to preserve original annotation type
        return df[name][row_index]


def annotation_field(semantic: Semantic = Semantic.Default) -> Any:
    return AnnotationField(semantic=semantic)
