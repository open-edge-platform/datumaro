# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from datumaro.experimental.fields.base import Field, PolarsDataType, Semantic, T, convert_numpy_object_array_to_series
from datumaro.experimental.type_registry import from_polars_data, to_numpy


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
    dtype: PolarsDataType = field(default_factory=pl.Float32)
    format: str = "x1y1x2y2"
    normalize: bool = False

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for bounding box as list of 4-element arrays."""
        return {name: pl.List(pl.Array(self.dtype, 4))}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert bounding box tensor to Polars list format."""
        numpy_value = to_numpy(value, self.dtype)

        if numpy_value is not None:
            data: Any = numpy_value.reshape(1, -1, 4)
        else:
            data = [None]

        return {
            name: pl.Series(
                name,
                data,
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
    dtype: PolarsDataType = field(default_factory=pl.Float32)
    format: str = "cxcywhr"
    normalize: bool = False

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for rotated bounding box as list of 5-element arrays."""
        return {name: pl.List(pl.Array(self.dtype, 5))}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert rotated bounding box tensor to Polars list format."""
        numpy_value = to_numpy(value, self.dtype)

        if numpy_value is not None:
            data: Any = numpy_value.reshape(1, -1, 5)
        else:
            data = [None]

        return {
            name: pl.Series(
                name,
                data,
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
    dtype: PolarsDataType = field(default_factory=pl.UInt8)
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
        return {name: pl.Series(name, [value], dtype=self._pl_type)}

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


@dataclass(frozen=True)
class ScoreField(Field):
    """
    Represents a prediction score.

    By default stores a single float value in a Float32 column. If is_list=True,
    stores a list of float values, matching multi-prediction scenarios.
    """

    semantic: Semantic
    dtype: PolarsDataType = field(default_factory=pl.Float32)
    is_list: bool = False

    @property
    def _pl_type(self) -> pl.DataType:
        pl_type = self.dtype
        if self.is_list:
            pl_type = pl.List(pl_type)
        return pl_type

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        return {name: self._pl_type}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        return {name: pl.Series(name, [value], dtype=self._pl_type)}

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        data = df[name][row_index]
        if target_type is list:
            return list(data) if data is not None else None  # type: ignore[return-value]
        return from_polars_data(data, target_type)


def score_field(
    dtype: Any = pl.Float32(),
    semantic: Semantic = Semantic.Default,
    is_list: bool = False,
) -> Any:
    """
    Create a ScoreField instance.

    Args:
        dtype: Polars data type for score values (defaults to pl.Float32())
        semantic: Semantic tags describing the score purpose (optional)
        is_list: Whether this field should be treated as a list type (defaults to False)

    Returns:
        ScoreField instance configured with the given parameters
    """
    return ScoreField(semantic=semantic, dtype=dtype, is_list=is_list)


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
    dtype: PolarsDataType = field(default_factory=pl.Float32)
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
class KeypointsField(Field):
    """
    Represents a keypoints field with coordinate and visibility information.

    Handles keypoint data where each keypoint has (x, y) coordinates and a (v) visibility state.
    The keypoints are stored as triplets [[x1, y1, v1], [x2, y2, v2], ...] where each triplet
    contains x coordinate, y coordinate, and visibility (0=absent, 1=hidden, 2=visible).

    Attributes:
        semantic: Semantic tags describing the keypoints purpose
        dtype: Polars data type for coordinate values
        normalize: Whether coordinates are normalized to [0,1] range
    """

    semantic: Semantic
    dtype: PolarsDataType = field(default_factory=pl.Float32)
    normalize: bool = False

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for keypoints as list of 3-element arrays (x, y, visibility)."""
        return {name: pl.List(pl.Array(self.dtype, 3))}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert keypoints tensor to Polars list format."""
        numpy_value = to_numpy(value, self.dtype)

        if numpy_value is not None:
            data: Any = numpy_value.reshape(1, -1, 3)
        else:
            data = [None]

        return {
            name: pl.Series(
                name,
                data,
                dtype=pl.List(pl.Array(self.dtype, 3)),
            )
        }

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct keypoints tensor from Polars data."""
        polars_data = df[name][row_index]
        return from_polars_data(polars_data, target_type)  # type: ignore


@dataclass(frozen=True)
class EllipseField(Field):
    """
    Represents an ellipse field with format and normalization options.

    Handles ellipse data with support for different coordinate formats
    and optional normalization to [0,1] range.

    Attributes:
        semantic: Semantic tags describing the ellipses purpose
        dtype: Polars data type for coordinate values
        format: Coordinate format (e.g., "x1y1x2y2", "xywh")
        normalize: Whether coordinates are normalized to [0,1] range
    """

    semantic: Semantic
    dtype: PolarsDataType = field(default_factory=pl.Float32)
    format: str = "x1y1x2y2"
    normalize: bool = False

    def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
        """Generate schema for ellipses as list of 4-element arrays."""
        return {name: pl.List(pl.Array(self.dtype, 4))}

    def to_polars(self, name: str, value: Any) -> dict[str, pl.Series]:
        """Convert ellipse tensor to Polars list format."""
        numpy_value = to_numpy(value, self.dtype)

        if numpy_value is not None:
            data: Any = numpy_value.reshape(1, -1, 4)
        else:
            data = [None]

        return {
            name: pl.Series(
                name,
                data,
                dtype=pl.List(pl.Array(self.dtype, 4)),
            )
        }

    def from_polars(self, name: str, row_index: int, df: pl.DataFrame, target_type: type[T]) -> T:
        """Reconstruct an ellipse tensor from Polars data."""
        polars_data = df[name][row_index]
        return from_polars_data(polars_data, target_type)  # type: ignore
