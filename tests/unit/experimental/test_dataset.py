"""
Unit tests for Dataset class.
"""

from typing import Any

import numpy as np
import polars as pl

from datumaro.experimental.dataset import (
    AttributeInfo,
    Dataset,
    Sample,
    Schema,
    convert_sample_to_schema,
)
from datumaro.experimental.fields import ImageInfo, bbox_field, image_field, image_info_field
from datumaro.experimental.schema import Semantic


def test_dataset_creation_from_sample_class():
    """Test Dataset creation from Sample class."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8, format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32, normalize=False)
        image_info: ImageInfo = image_info_field()

    dataset = Dataset(TestSample)

    assert dataset.schema is not None
    assert len(dataset.schema.attributes) == 3
    assert "image" in dataset.schema.attributes
    assert "bbox" in dataset.schema.attributes
    assert "image_info" in dataset.schema.attributes
    assert len(dataset.df) == 0


def test_dataset_creation_from_schema():
    """Test Dataset creation from explicit Schema."""
    schema = Schema(
        attributes={
            "image": AttributeInfo(
                type=np.ndarray, annotation=image_field(dtype=pl.UInt8, format="RGB")
            ),
            "bbox": AttributeInfo(
                type=np.ndarray,
                annotation=bbox_field(dtype=pl.Float32, normalize=False),
            ),
        }
    )

    dataset = Dataset(schema)

    assert dataset.schema == schema
    assert len(dataset.df) == 0


def test_dataset_append_sample():
    """Test adding samples to dataset."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8, format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32, normalize=False)
        image_info: ImageInfo = image_info_field()

    dataset = Dataset(TestSample)

    sample = TestSample(
        image=np.array([[[255, 0, 0]], [[0, 255, 0]]], dtype=np.uint8),
        bbox=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        image_info=ImageInfo(width=1, height=2),
    )

    dataset.append(sample)

    assert len(dataset.df) == 1
    retrieved_sample = dataset[0]
    assert isinstance(retrieved_sample, TestSample)
    assert np.allclose(retrieved_sample.bbox, sample.bbox)
    assert retrieved_sample.image_info.width == 1
    assert retrieved_sample.image_info.height == 2


def test_dataset_multiple_samples():
    """Test adding multiple samples to dataset."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8, format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32, normalize=False)
        image_info: ImageInfo = image_info_field()

    dataset = Dataset(TestSample)

    # Add first sample
    sample1 = TestSample(
        image=np.array([[[255, 0, 0]], [[0, 255, 0]]], dtype=np.uint8),
        bbox=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        image_info=ImageInfo(width=1, height=2),
    )
    dataset.append(sample1)

    # Add second sample
    sample2 = TestSample(
        image=np.array([[[0, 0, 255]], [[255, 255, 0]]], dtype=np.uint8),
        bbox=np.array([[0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
        image_info=ImageInfo(width=1, height=2),
    )
    dataset.append(sample2)

    assert len(dataset.df) == 2

    retrieved_sample1 = dataset[0]
    retrieved_sample2 = dataset[1]

    assert np.allclose(retrieved_sample1.bbox, sample1.bbox)
    assert np.allclose(retrieved_sample2.bbox, sample2.bbox)


def test_dataset_item_modification():
    """Test modifying dataset items."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8, format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32, normalize=False)
        image_info: ImageInfo = image_info_field()

    dataset = Dataset(TestSample)

    original_sample = TestSample(
        image=np.array([[[255, 0, 0]], [[0, 255, 0]]], dtype=np.uint8),
        bbox=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        image_info=ImageInfo(width=1, height=2),
    )
    dataset.append(original_sample)

    # Modify the sample
    modified_sample = TestSample(
        image=np.array([[[0, 0, 255]], [[255, 255, 0]]], dtype=np.uint8),
        bbox=np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32),
        image_info=ImageInfo(width=1, height=2),
    )
    dataset[0] = modified_sample

    retrieved_sample = dataset[0]
    assert np.allclose(retrieved_sample.bbox, modified_sample.bbox)


def test_dataset_from_dataframe():
    """Test creating dataset from existing DataFrame."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8, format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32, normalize=False)
        image_info: ImageInfo = image_info_field()

    # Create a dataset and add a sample
    original_dataset = Dataset(TestSample)
    sample = TestSample(
        image=np.array([[[255, 0, 0]], [[0, 255, 0]]], dtype=np.uint8),
        bbox=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        image_info=ImageInfo(width=1, height=2),
    )
    original_dataset.append(sample)

    # Create new dataset from DataFrame
    new_dataset = Dataset.from_dataframe(original_dataset.df, TestSample)

    assert len(new_dataset.df) == 1
    retrieved_sample = new_dataset[0]
    assert isinstance(retrieved_sample, TestSample)
    assert np.allclose(retrieved_sample.bbox, sample.bbox)


def test_stereo_sample_with_semantics():
    """Test dataset with stereo samples using semantic tags."""

    class StereoSample(Sample):
        left_image: np.ndarray[Any, Any] = image_field(
            dtype=pl.UInt8, format="RGB", semantic=Semantic.Left
        )
        right_image: np.ndarray[Any, Any] = image_field(
            dtype=pl.UInt8, format="BGR", semantic=Semantic.Right
        )
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32, normalize=True)
        left_image_info: ImageInfo = image_info_field(Semantic.Left)
        right_image_info: ImageInfo = image_info_field(Semantic.Right)

    dataset = Dataset(StereoSample)

    sample = StereoSample(
        left_image=np.array([[[255, 0, 0]], [[0, 255, 0]]], dtype=np.uint8),
        right_image=np.array([[[0, 0, 255]], [[255, 255, 0]]], dtype=np.uint8),
        bbox=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        left_image_info=ImageInfo(width=1, height=2),
        right_image_info=ImageInfo(width=1, height=2),
    )

    dataset.append(sample)

    assert len(dataset.df) == 1
    retrieved_sample = dataset[0]
    assert isinstance(retrieved_sample, StereoSample)
    assert np.allclose(retrieved_sample.bbox, sample.bbox)
    assert retrieved_sample.left_image_info.width == 1
    assert retrieved_sample.right_image_info.width == 1


def test_dynamic_schema_definition():
    """Test dataset creation with dynamic schema without explicit Sample class."""
    schema = Schema(
        attributes={
            "image": AttributeInfo(
                type=np.ndarray, annotation=image_field(dtype=pl.UInt8, format="RGB")
            ),
            "bbox": AttributeInfo(
                type=np.ndarray,
                annotation=bbox_field(dtype=pl.Float32, normalize=False),
            ),
            "image_info": AttributeInfo(type=ImageInfo, annotation=image_info_field()),
        }
    )

    dataset = Dataset(schema)

    # Create a generic sample
    sample = Sample(
        image=np.array([[[255, 0, 0]], [[0, 255, 0]]], dtype=np.uint8),
        bbox=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        image_info=ImageInfo(width=1, height=2),
    )

    dataset.append(sample)

    assert len(dataset.df) == 1
    retrieved_sample = dataset[0]
    assert isinstance(retrieved_sample, Sample)
    assert np.allclose(getattr(retrieved_sample, "bbox"), getattr(sample, "bbox"))


def test_convert_sample_to_same_schema():
    """Test individual sample conversion function."""

    class SourceSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8, format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32, normalize=False)

    class TargetSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8, format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32, normalize=False)

    source_sample = SourceSample(
        image=np.array([[[255, 0, 0]], [[0, 255, 0]]], dtype=np.uint8),
        bbox=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
    )

    source_schema = SourceSample.infer_schema()

    converted_sample = convert_sample_to_schema(source_sample, source_schema, TargetSample)
    assert isinstance(converted_sample, TargetSample)
    assert np.allclose(converted_sample.bbox, source_sample.bbox)


def test_convert_sample_to_different_schema():
    """Test individual sample conversion function between different schemas."""

    class SourceSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8, format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32, normalize=False)
        image_info: ImageInfo = image_info_field()

    class TargetSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8, format="BGR")  # Different format
        bbox: np.ndarray[Any, Any] = bbox_field(
            dtype=pl.Float32, normalize=True
        )  # Different normalization
        image_info: ImageInfo = image_info_field()

    source_sample = SourceSample(
        image=np.array([[[255, 0, 0]], [[0, 255, 0]]], dtype=np.uint8),
        bbox=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        image_info=ImageInfo(width=1, height=2),
    )

    source_schema = SourceSample.infer_schema()

    converted_sample = convert_sample_to_schema(source_sample, source_schema, TargetSample)
    assert isinstance(converted_sample, TargetSample)


def test_dataset_polars_schema_generation():
    """Test Polars schema generation from dataset schema."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8, format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32, normalize=False)
        image_info: ImageInfo = image_info_field()

    dataset = Dataset(TestSample)
    polars_schema = dataset.df.schema

    # Should contain all expected columns
    expected_columns = ["image", "image_shape", "bbox", "image_info"]
    for col in expected_columns:
        assert col in polars_schema

    # Check specific types
    assert polars_schema["image"] == pl.List(pl.UInt8())
    assert polars_schema["bbox"] == pl.List(pl.Array(pl.Float32, 4))


def test_dataset_lazy_converters_property():
    """Test lazy converters property."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8, format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32, normalize=False)

    dataset = Dataset(TestSample)
    lazy_converters = dataset.lazy_converters

    assert isinstance(lazy_converters, list)
    # Initially should be empty
    assert len(lazy_converters) == 0
