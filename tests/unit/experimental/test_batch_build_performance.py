"""Tests for batch-build performance optimizations.

These tests cover the PyArrow-accelerated batch column construction used by
``Dataset.append_batch`` and ``convert_from_legacy`` to avoid the slow
per-sample Series creation + concat pattern.
"""

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from datumaro.components.annotation import AnnotationType, Bbox, Polygon
from datumaro.components.annotation import LabelCategories as LegacyLabelCategories
from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.experimental.arrow_utils import build_series_bulk, numpy_to_nested_lists, polars_dtype_to_pyarrow
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import (
    Subset,
    bbox_field,
    image_path_field,
    label_field,
    numeric_field,
    polygon_field,
    subset_field,
)
from datumaro.experimental.fields.annotations import BBoxField, PolygonField
from datumaro.experimental.fields.types import NumericField, StringField
from datumaro.experimental.legacy import convert_from_legacy

# ---------------------------------------------------------------------------
# Helper sample types
# ---------------------------------------------------------------------------


class DetectionSample(Sample):
    image: str = image_path_field()
    bboxes: np.ndarray | None = bbox_field(dtype=pl.Float32())
    labels: np.ndarray | None = label_field(dtype=pl.UInt32(), is_list=True)
    subset: Subset = subset_field()


class PolygonSample(Sample):
    image: str = image_path_field()
    polygons: np.ndarray | None = polygon_field(dtype=pl.Float32())
    labels: np.ndarray | None = label_field(dtype=pl.UInt32(), is_list=True)


class SimpleSample(Sample):
    name: str | None = numeric_field(dtype=pl.Float32(), semantic="score")


# ---------------------------------------------------------------------------
# numpy_to_nested_lists
# ---------------------------------------------------------------------------


class NumpyToNestedListsTest:
    """Tests for numpy_to_nested_lists helper."""

    def test_none_returns_none(self):
        assert numpy_to_nested_lists(None) is None

    def test_regular_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = numpy_to_nested_lists(arr)
        assert result == [[1.0, 2.0], [3.0, 4.0]]
        assert isinstance(result, list)

    def test_object_array_ragged(self):
        """Ragged numpy object array (variable-length sub-arrays)."""
        inner1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        inner2 = np.array([[5.0, 6.0]], dtype=np.float32)
        outer = np.empty(2, dtype=object)
        outer[0] = inner1
        outer[1] = inner2
        result = numpy_to_nested_lists(outer)
        assert result == [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]]

    def test_plain_python_value_passthrough(self):
        """Non-numpy values are returned as-is."""
        assert numpy_to_nested_lists(42) == 42
        assert numpy_to_nested_lists("hello") == "hello"

    def test_1d_array(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        assert numpy_to_nested_lists(arr) == [1, 2, 3]


# ---------------------------------------------------------------------------
# polars_dtype_to_pyarrow
# ---------------------------------------------------------------------------


class PolarsDtypeToPyarrowTest:
    """Tests for Polars-to-PyArrow type conversion."""

    def test_scalar_types(self):
        assert polars_dtype_to_pyarrow(pl.Float32()) == pa.float32()
        assert polars_dtype_to_pyarrow(pl.Float64()) == pa.float64()
        assert polars_dtype_to_pyarrow(pl.Int32()) == pa.int32()
        assert polars_dtype_to_pyarrow(pl.UInt8()) == pa.uint8()
        assert polars_dtype_to_pyarrow(pl.Boolean()) == pa.bool_()
        assert polars_dtype_to_pyarrow(pl.String()) == pa.string()

    def test_list_type(self):
        pa_type = polars_dtype_to_pyarrow(pl.List(pl.Float32()))
        assert pa_type == pa.list_(pa.float32())

    def test_array_type(self):
        pa_type = polars_dtype_to_pyarrow(pl.Array(pl.Float32(), 4))
        assert pa_type == pa.list_(pa.float32(), 4)

    def test_nested_list_of_array(self):
        """List(List(Array(Float32, 2))) — polygon type."""
        dtype = pl.List(pl.List(pl.Array(pl.Float32(), 2)))
        pa_type = polars_dtype_to_pyarrow(dtype)
        # Should be list<list<fixed_size_list[2]<float>>>
        assert isinstance(pa_type, pa.ListType)

    def test_struct_type(self):
        dtype = pl.Struct([pl.Field("width", pl.Int32()), pl.Field("height", pl.Int32())])
        pa_type = polars_dtype_to_pyarrow(dtype)
        assert isinstance(pa_type, pa.StructType)
        assert pa_type.num_fields == 2

    def test_dataset_converters_version_matches(self):
        """polars_dtype_to_pyarrow handles a representative set of nested dtypes."""
        for dtype in (
            pl.List(pl.Array(pl.Float32(), 4)),
            pl.List(pl.List(pl.Array(pl.Float32(), 2))),
            pl.Array(pl.UInt32(), 3),
            pl.Struct([pl.Field("a", pl.Int32()), pl.Field("b", pl.Float64())]),
        ):
            pa_type = polars_dtype_to_pyarrow(dtype)
            # Should succeed round-trip through a PyArrow array constructor with an empty list.
            arr = pa.array([], type=pa_type)
            assert arr.type == pa_type


# ---------------------------------------------------------------------------
# build_series_bulk
# ---------------------------------------------------------------------------


class BuildSeriesBulkTest:
    """Tests for build_series_bulk (dataset.py version)."""

    def test_simple_int_series(self):
        s = build_series_bulk("x", [1, 2, 3], pl.Int32())
        assert s.name == "x"
        assert len(s) == 3
        assert s.to_list() == [1, 2, 3]

    def test_simple_float_series(self):
        s = build_series_bulk("f", [1.0, None, 3.0], pl.Float32())
        assert len(s) == 3
        assert s[1] is None

    def test_string_series(self):
        s = build_series_bulk("s", ["a", "b", "c"], pl.String())
        assert s.to_list() == ["a", "b", "c"]

    def test_list_of_arrays_via_pyarrow(self):
        """Bbox-like type: List(Array(Float32, 4))."""
        dtype = pl.List(pl.Array(pl.Float32(), 4))
        values = [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0]],
        ]
        s = build_series_bulk("bboxes", values, dtype)
        assert len(s) == 2

    def test_nested_polygon_type_via_pyarrow(self):
        """Polygon type: List(List(Array(Float32, 2)))."""
        dtype = pl.List(pl.List(pl.Array(pl.Float32(), 2)))
        values = [
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],  # 1 polygon with 3 points
            [[[7.0, 8.0], [9.0, 10.0]]],  # 1 polygon with 2 points
        ]
        s = build_series_bulk("polygons", values, dtype)
        assert len(s) == 2

    def test_struct_type_via_pyarrow(self):
        """Struct type for image_info-like data."""
        dtype = pl.Struct([pl.Field("width", pl.Int32()), pl.Field("height", pl.Int32())])
        values = [{"width": 640, "height": 480}, {"width": 1920, "height": 1080}]
        s = build_series_bulk("info", values, dtype)
        assert len(s) == 2

    def test_none_values_in_nested_type(self):
        """None values should be handled gracefully."""
        dtype = pl.List(pl.Array(pl.Float32(), 4))
        values = [None, [[1.0, 2.0, 3.0, 4.0]], None]
        s = build_series_bulk("bboxes", values, dtype)
        assert len(s) == 3
        assert s[0] is None
        assert s[2] is None

    def test_variable_width_lists(self):
        """Variable-width lists of ints go through the PyArrow fast path cleanly."""
        values = [[1, 2, 3], [4, 5], [6]]
        s = build_series_bulk("labels", values, pl.List(pl.Int32()))
        assert len(s) == 3
        assert s[0].to_list() == [1, 2, 3]
        assert s[1].to_list() == [4, 5]


# ---------------------------------------------------------------------------
# BBoxField.to_python_scalars
# ---------------------------------------------------------------------------


class BBoxFieldToPythonScalarsTest:
    """Tests for BBoxField.to_python_scalars fast path."""

    def test_returns_nested_lists_and_canvas(self):
        bf = BBoxField(dtype=pl.Float32())
        arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        result = bf.to_python_scalars("bboxes", arr)
        assert set(result.keys()) == {"bboxes", "bboxes_canvas_size"}
        assert result["bboxes"] == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        assert isinstance(result["bboxes"], list)
        # Canvas size is only populated for tv_tensors.BoundingBoxes, None otherwise.
        assert result["bboxes_canvas_size"] is None

    def test_none_value(self):
        bf = BBoxField(dtype=pl.Float32())
        result = bf.to_python_scalars("bboxes", None)
        assert result == {"bboxes": None, "bboxes_canvas_size": None}


# ---------------------------------------------------------------------------
# Field.to_python_scalars
# ---------------------------------------------------------------------------


class FieldToPythonScalarsTest:
    """Tests for Field.to_python_scalars (base + overrides)."""

    def test_base_field_default_delegates_to_to_polars(self):
        """Default to_python_scalars extracts s[0] from to_polars result."""
        field = NumericField(semantic="test", dtype=pl.Float32())
        result = field.to_python_scalars("score", 3.14)
        assert result == {"score": pytest.approx(3.14, abs=1e-5)}

    def test_string_field_scalars(self):
        field = StringField(semantic="id")
        result = field.to_python_scalars("name", "hello")
        assert result == {"name": "hello"}

    def test_string_field_none(self):
        field = StringField(semantic="id")
        result = field.to_python_scalars("name", None)
        assert result == {"name": None}

    def test_polygon_field_scalars_returns_nested_lists(self):
        """PolygonField.to_python_scalars should return nested Python lists, not Series."""
        pf = PolygonField(dtype=pl.Float32())

        # Build a proper ragged numpy polygon
        inner = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        outer = np.empty(1, dtype=object)
        outer[0] = inner

        result = pf.to_python_scalars("polygons", outer)
        assert "polygons" in result
        # Should be a nested list, not a Series
        assert isinstance(result["polygons"], list)
        assert result["polygons"] == [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]

    def test_polygon_field_scalars_none(self):
        pf = PolygonField(dtype=pl.Float32())
        result = pf.to_python_scalars("polygons", None)
        assert result == {"polygons": None}


# ---------------------------------------------------------------------------
# Dataset.append_batch (integration)
# ---------------------------------------------------------------------------


class AppendBatchBulkTest:
    """Tests that append_batch uses the bulk path and produces correct results."""

    def test_append_batch_simple(self):
        dataset = Dataset(
            DetectionSample,
            categories={"labels": LabelCategories(labels=("cat", "dog"))},
        )
        samples = [
            DetectionSample(
                image="/img1.jpg",
                bboxes=np.array([[10, 20, 50, 60]], dtype=np.float32),
                labels=np.array([0], dtype=np.uint32),
                subset=Subset.TRAINING,
            ),
            DetectionSample(
                image="/img2.jpg",
                bboxes=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
                labels=np.array([1, 0], dtype=np.uint32),
                subset=Subset.VALIDATION,
            ),
        ]
        dataset.append_batch(samples)

        assert len(dataset) == 2
        s0 = dataset[0]
        s1 = dataset[1]
        # image_path_field returns a LazyImage; access its .path
        image0 = s0.image
        assert str(getattr(image0, "path", image0)) == "/img1.jpg"
        np.testing.assert_array_equal(s0.bboxes, np.array([[10, 20, 50, 60]], dtype=np.float32))
        np.testing.assert_array_equal(s0.labels, np.array([0], dtype=np.uint32))
        assert len(s1.bboxes) == 2

    def test_append_batch_with_polygons(self):
        """Test that polygon fields go through the fast to_python_scalars path."""
        dataset = Dataset(PolygonSample, categories={"labels": LabelCategories(labels=("a", "b"))})

        # Build polygon data: 1 polygon with 3 points
        poly1_inner = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        poly1 = np.empty(1, dtype=object)
        poly1[0] = poly1_inner

        # Build polygon data: 2 polygons
        poly2_a = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        poly2_b = np.array([[50.0, 60.0], [70.0, 80.0], [90.0, 100.0]], dtype=np.float32)
        poly2 = np.empty(2, dtype=object)
        poly2[0] = poly2_a
        poly2[1] = poly2_b

        samples = [
            PolygonSample(image="/img1.jpg", polygons=poly1, labels=np.array([0], dtype=np.uint32)),
            PolygonSample(image="/img2.jpg", polygons=poly2, labels=np.array([1, 0], dtype=np.uint32)),
        ]
        dataset.append_batch(samples)

        assert len(dataset) == 2
        s0 = dataset[0]
        assert s0.polygons is not None
        # Check shape: 1 polygon, 3 points, 2 coords
        assert len(s0.polygons) == 1
        np.testing.assert_array_almost_equal(s0.polygons[0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_append_batch_with_none_polygons(self):
        """Test that None polygon values are handled correctly in batch mode."""
        dataset = Dataset(PolygonSample, categories={"labels": LabelCategories(labels=("a",))})

        samples = [
            PolygonSample(image="/img1.jpg", polygons=None, labels=None),
            PolygonSample(image="/img2.jpg", polygons=None, labels=None),
        ]
        dataset.append_batch(samples)

        assert len(dataset) == 2
        assert dataset[0].polygons is None
        assert dataset[1].polygons is None

    def test_append_batch_variable_length_labels(self):
        """Labels with different counts per sample should work (type relaxation)."""
        dataset = Dataset(
            DetectionSample,
            categories={"labels": LabelCategories(labels=("a", "b", "c"))},
        )
        samples = [
            DetectionSample(
                image="/img1.jpg",
                bboxes=np.array([[1, 2, 3, 4]], dtype=np.float32),
                labels=np.array([0], dtype=np.uint32),
                subset=Subset.TRAINING,
            ),
            DetectionSample(
                image="/img2.jpg",
                bboxes=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32),
                labels=np.array([0, 1, 2], dtype=np.uint32),
                subset=Subset.TRAINING,
            ),
        ]
        dataset.append_batch(samples)

        assert len(dataset) == 2
        assert len(dataset[0].labels) == 1
        assert len(dataset[1].labels) == 3

    def test_append_batch_empty_list(self):
        """Appending an empty list should be a no-op."""
        dataset = Dataset(DetectionSample)
        dataset.append_batch([])
        assert len(dataset) == 0

    def test_append_batch_single_item(self):
        """Single-item batch should work correctly."""
        dataset = Dataset(
            DetectionSample,
            categories={"labels": LabelCategories(labels=("x",))},
        )
        samples = [
            DetectionSample(
                image="/img.jpg",
                bboxes=np.array([[1, 2, 3, 4]], dtype=np.float32),
                labels=np.array([0], dtype=np.uint32),
                subset=Subset.TRAINING,
            ),
        ]
        dataset.append_batch(samples)
        assert len(dataset) == 1


# ---------------------------------------------------------------------------
# convert_from_legacy end-to-end (correctness after batch optimization)
# ---------------------------------------------------------------------------


class ConvertFromLegacyBatchTest:
    """End-to-end tests verifying convert_from_legacy correctness after batch optimization."""

    def test_bbox_dataset_correctness(self):
        """Verify that bbox values are correctly preserved through batch build."""
        label_categories = LegacyLabelCategories()
        label_categories.add("cat")
        label_categories.add("dog")

        items = [
            DatasetItem(
                id="item1",
                media=Image.from_file("/img1.jpg", size=(480, 640)),
                annotations=[Bbox(10, 20, 30, 40, label=0), Bbox(50, 60, 70, 80, label=1)],
            ),
            DatasetItem(
                id="item2",
                media=Image.from_file("/img2.jpg", size=(100, 200)),
                annotations=[Bbox(1, 2, 3, 4, label=0)],
            ),
        ]
        dataset = LegacyDataset.from_iterable(
            items,
            ann_types={AnnotationType.bbox},
            categories={AnnotationType.label: label_categories},
        )

        result = convert_from_legacy(dataset)

        assert len(result) == 2
        s0 = result[0]
        s1 = result[1]

        # Two bboxes for first sample
        assert s0.bboxes.shape == (2, 4)
        # One bbox for second sample
        assert s1.bboxes.shape == (1, 4)
        # Labels
        np.testing.assert_array_equal(s0.labels, [0, 1])
        np.testing.assert_array_equal(s1.labels, [0])

    def test_polygon_dataset_correctness(self):
        """Verify that polygon values survive the batch build path."""
        label_categories = LegacyLabelCategories()
        label_categories.add("road")

        poly1 = Polygon(points=[10, 20, 30, 40, 50, 60], label=0)
        poly2 = Polygon(points=[1, 2, 3, 4, 5, 6, 7, 8], label=0)

        items = [
            DatasetItem(
                id="item1",
                media=Image.from_file("/img1.jpg", size=(100, 200)),
                annotations=[poly1],
            ),
            DatasetItem(
                id="item2",
                media=Image.from_file("/img2.jpg", size=(100, 200)),
                annotations=[poly2],
            ),
        ]
        dataset = LegacyDataset.from_iterable(
            items,
            ann_types={AnnotationType.polygon},
            categories={AnnotationType.label: label_categories},
        )

        result = convert_from_legacy(dataset)
        assert len(result) == 2

        # Polygon 1: 3 points
        s0 = result[0]
        assert s0.polygons is not None
        assert len(s0.polygons) == 1  # 1 polygon
        assert len(s0.polygons[0]) == 3  # 3 points

        # Polygon 2: 4 points
        s1 = result[1]
        assert s1.polygons is not None
        assert len(s1.polygons) == 1
        assert len(s1.polygons[0]) == 4

    def test_empty_dataset_roundtrip(self):
        """Empty dataset should survive batch build."""
        dataset = LegacyDataset.from_iterable([])
        result = convert_from_legacy(dataset)
        assert len(result) == 0

    def test_multiple_annotation_types(self):
        """Dataset with both bboxes and polygons should be handled correctly."""
        label_categories = LegacyLabelCategories()
        label_categories.add("car")

        bbox = Bbox(10, 20, 30, 40, label=0)
        poly = Polygon(points=[1, 2, 3, 4, 5, 6], label=0)

        items = [
            DatasetItem(
                id="item1",
                media=Image.from_file("/img1.jpg", size=(100, 200)),
                annotations=[bbox, poly],
            ),
        ]
        dataset = LegacyDataset.from_iterable(
            items,
            ann_types={AnnotationType.bbox, AnnotationType.polygon},
            categories={AnnotationType.label: label_categories},
        )

        result = convert_from_legacy(dataset)
        assert len(result) == 1
        s0 = result[0]
        # Should have both polygons and bboxes (polygon converter generates bboxes)
        assert s0.polygons is not None

    def test_large_batch_variable_annotations(self):
        """Batch with many items with varying annotation counts."""
        label_categories = LegacyLabelCategories()
        label_categories.add("obj")

        items = []
        for i in range(50):
            # Each item has i+1 bboxes
            bboxes = [Bbox(j * 10, j * 20, 30, 40, label=0) for j in range(i + 1)]
            items.append(
                DatasetItem(
                    id=f"item_{i}",
                    media=Image.from_file(f"/img_{i}.jpg", size=(100, 200)),
                    annotations=bboxes,
                )
            )
        dataset = LegacyDataset.from_iterable(
            items,
            ann_types={AnnotationType.bbox},
            categories={AnnotationType.label: label_categories},
        )

        result = convert_from_legacy(dataset)
        assert len(result) == 50

        # First item should have 1 bbox, last item should have 50
        assert result[0].bboxes.shape[0] == 1
        assert result[49].bboxes.shape[0] == 50

    def test_none_annotations_handled(self):
        """Items with no annotations should have None for annotation fields."""
        label_categories = LegacyLabelCategories()
        label_categories.add("obj")

        items = [
            DatasetItem(
                id="annotated",
                media=Image.from_file("/img1.jpg", size=(100, 200)),
                annotations=[Bbox(10, 20, 30, 40, label=0)],
            ),
            DatasetItem(
                id="unannotated",
                media=Image.from_file("/img2.jpg", size=(100, 200)),
                annotations=[],
            ),
        ]
        dataset = LegacyDataset.from_iterable(
            items,
            ann_types={AnnotationType.bbox},
            categories={AnnotationType.label: label_categories},
        )

        result = convert_from_legacy(dataset)
        assert len(result) == 2

        # First sample should have bboxes
        assert result[0].bboxes is not None
        assert result[0].bboxes.shape[0] > 0
        # Second sample should have empty bboxes (no annotations)
        assert result[1].bboxes is not None
        assert result[1].bboxes.shape[0] == 0
