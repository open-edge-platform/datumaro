# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import polars as pl
import pytest

from datumaro.experimental.fields import BBoxField, PolygonField, bbox_field, polygon_field
from datumaro.experimental.filtering.filter_registry import (
    FilterRegistry,
    _compute_filter_mask,
    create_filtering_plan,
    create_filtering_transform,
)
from datumaro.experimental.schema import AttributeInfo, Schema
from datumaro.experimental.transform import IdentityTransform


@pytest.fixture
def empty_annotations_df():
    """Create a test DataFrame with empty and non-empty annotations."""
    return pl.DataFrame(
        {
            "bboxes": [
                [],  # Empty bboxes
                [[0, 0, 10, 10]],  # One bbox
                [[0, 0, 10, 10], [20, 20, 30, 30]],  # Two bboxes
            ],
            "polygons": [
                [[0, 0, 10, 0, 10, 10, 0, 10]],  # One polygon
                [],  # Empty polygons
                [[0, 0, 10, 0, 10, 10, 0, 10], [20, 20, 30, 20, 30, 30, 20, 30]],  # Two polygons
            ],
        }
    )


def test_filters_are_registered():
    """Test that filters are registered for bbox and polygon fields."""
    bbox_filter = FilterRegistry.get_filter(BBoxField)
    polygon_filter = FilterRegistry.get_filter(PolygonField)

    assert bbox_filter is not None
    assert polygon_filter is not None


def test_empty_bbox_filter(empty_annotations_df):
    """Test filtering empty bboxes."""
    schema = Schema(
        {
            "bboxes": AttributeInfo(type=np.ndarray, field=bbox_field(dtype=pl.Float64())),
        }
    )

    plan = create_filtering_plan(schema)
    filtered_df = empty_annotations_df.filter(_compute_filter_mask(empty_annotations_df, plan))

    assert len(filtered_df) == 2  # Should keep rows with non-empty bboxes
    assert all(len(bboxes) > 0 for bboxes in filtered_df["bboxes"])


def test_empty_polygon_filter(empty_annotations_df):
    """Test filtering empty polygons."""
    schema = Schema(
        {
            "polygons": AttributeInfo(type=np.ndarray, field=polygon_field(dtype=pl.Float64())),
        }
    )

    plan = create_filtering_plan(schema)
    filtered_df = empty_annotations_df.filter(_compute_filter_mask(empty_annotations_df, plan))

    assert len(filtered_df) == 2  # Should keep rows with non-empty polygons
    assert all(len(polygons) > 0 for polygons in filtered_df["polygons"])


def test_combined_filters(empty_annotations_df):
    """Test filtering both empty bboxes and polygons."""
    schema = Schema(
        {
            "bboxes": AttributeInfo(type=np.ndarray, field=bbox_field(dtype=pl.Float64())),
            "polygons": AttributeInfo(type=np.ndarray, field=polygon_field(dtype=pl.Float64())),
        }
    )

    plan = create_filtering_plan(schema)
    filtered_df = empty_annotations_df.filter(_compute_filter_mask(empty_annotations_df, plan))

    assert len(filtered_df) == 1  # Only one row has both non-empty
    assert all(len(bboxes) > 0 for bboxes in filtered_df["bboxes"])
    assert all(len(polygons) > 0 for polygons in filtered_df["polygons"])


def test_transform_application():
    """Test applying the filter transform."""
    df = pl.DataFrame({"bboxes": [[], [[0, 0, 10, 10]], [[20, 20, 30, 30]]]})
    schema = Schema(
        {
            "bboxes": AttributeInfo(type=list, field=bbox_field(dtype=pl.Float64())),
        }
    )

    transform = create_filtering_transform()(IdentityTransform(df, schema))
    filtered_df = transform.apply(["bboxes"])

    assert len(filtered_df) == 2  # Should keep only non-empty bboxes
    assert all(len(bboxes) > 0 for bboxes in filtered_df["bboxes"])
