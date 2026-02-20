"""
Unit tests for Dataset class.
"""

import sys
from typing import Any

import numpy as np
import polars as pl
import pytest

from datumaro.experimental.categories import LabelCategories, MaskCategories
from datumaro.experimental.dataset import AttributeInfo, Dataset, Sample, Schema, convert_sample_to_schema
from datumaro.experimental.fields import (
    ImageInfo,
    Subset,
    TileInfo,
    bbox_field,
    image_field,
    image_info_field,
    label_field,
    mask_field,
    subset_field,
    tile_field,
)


def test_sample_validation_pass():
    class MySample(Sample):
        bbox: np.ndarray = bbox_field(dtype=pl.Float32())
        image: np.ndarray = image_field(dtype=pl.UInt8(), format="RGB")
        tile: TileInfo = tile_field()
        mask: np.ndarray = mask_field(dtype=pl.UInt8())

    # Valid sample
    valid_sample = MySample(
        bbox=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        image=np.array([[[255, 0, 0]]], dtype=np.uint8),
        tile=TileInfo(source_sample_idx=0, x=0, y=0, width=1, height=1),
        mask=np.array([[1, 0], [0, 1]], dtype=np.uint8),
    )
    # Test that validation raises an error
    with pytest.raises(Exception):
        valid_sample.validate()


def test_validate_fields_with_categories_on_append():
    categories = LabelCategories(labels=("cat", "dog"))
    schema = Schema(attributes={"label": AttributeInfo(type=int, field=label_field(), categories=categories)})
    ds = Dataset(schema)

    # Append valid sample (label=1)
    ds.append(Sample(label=1))

    # Append invalid sample (label=2, out of range)
    with pytest.raises(ValueError, match="exceed"):
        ds.append(Sample(label=2))


def test_validate_fields_with_categories_on_validate():
    categories = LabelCategories(labels=("cat", "dog"))
    schema = Schema(attributes={"label": AttributeInfo(type=int, field=label_field(), categories=categories)})
    ds = Dataset(schema)
    ds.append(Sample(label=0))
    ds.append(Sample(label=1))

    # Test that validation raises an error
    with pytest.raises(Exception):
        ds.validate_fields_with_categories()

    # Manually tamper with df to add an invalid label
    ds.df = ds.df.with_columns(pl.lit(3).alias("label"))
    with pytest.raises(ValueError, match="exceed"):
        ds.validate_fields_with_categories()


def test_sample_validation_fail():
    class MySample(Sample):
        bbox: np.ndarray = bbox_field(dtype=pl.Float32())
        image: np.ndarray = image_field(dtype=pl.UInt8(), format="RGB")
        tile: TileInfo = tile_field()
        mask: np.ndarray = mask_field(dtype=pl.UInt8())

    # Invalid sample: wrong dtype for image
    with pytest.raises(TypeError):
        MySample(
            bbox=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            image="invalid_image_data",
            tile=TileInfo(source_sample_idx=0, x=0, y=0, width=1, height=1),
            mask=np.array([[1, 0], [0, 1]], dtype=np.uint8),
        ).validate()


def test_append_dataset():
    class MySample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)

    # Define sample instances
    sample1 = MySample(
        image=np.array([1, 2, 3], dtype=np.uint8),
        bbox=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    )
    sample2 = MySample(
        image=np.array([4, 5, 6], dtype=np.uint8),
        bbox=np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
    )
    sample3 = MySample(
        image=np.array([7, 8, 9], dtype=np.uint8),
        bbox=np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float32),
    )
    sample4 = MySample(
        image=np.array([10, 11, 12], dtype=np.uint8),
        bbox=np.array([1.3, 1.4, 1.5, 1.6], dtype=np.float32),
    )

    # Create two datasets with the same schema
    ds1 = Dataset(MySample)
    ds2 = Dataset(MySample)

    # Append samples to each dataset
    ds1.append(sample1)
    ds1.append(sample2)
    ds2.append(sample3)
    ds2.append(sample4)

    # Append ds2 into ds1
    ds1.append_dataset(ds2)

    # Check that ds1 now contains all samples
    assert len(ds1) == 4

    # Extract all samples and check their values
    samples = [ds1[i] for i in range(len(ds1))]
    expected = [sample1, sample2, sample3, sample4]
    for s, e in zip(samples, expected):
        np.testing.assert_array_equal(s.image, e.image)
        np.testing.assert_array_equal(s.bbox, e.bbox.reshape(1, -1))


def test_filter_by_subset_raises_without_subset_field():
    class NoSubsetSample(Sample):
        image: np.ndarray = image_field(dtype=pl.UInt8(), format="RGB")

    dataset = Dataset(NoSubsetSample)
    dataset.append(NoSubsetSample(image=np.array([1])))
    with pytest.raises(RuntimeError, match="Dataset does not have an attribute for 'SubsetField'"):
        dataset.filter_by_subset(Subset.TRAINING)


def test_filter_by_subset_filters_correctly():
    class SubsetSample(Sample):
        image: np.ndarray = image_field(dtype=pl.UInt8(), format="RGB")
        subset: Subset = subset_field()

    dataset = Dataset(SubsetSample)
    dataset.append(SubsetSample(image=np.array([1]), subset=Subset.TRAINING))
    dataset.append(SubsetSample(image=np.array([2]), subset=Subset.VALIDATION))
    dataset.append(SubsetSample(image=np.array([3]), subset=Subset.TRAINING))

    filtered = dataset.filter_by_subset(Subset.TRAINING)
    assert len(filtered) == 2
    for sample in filtered:
        assert sample.subset == Subset.TRAINING


def test_filter_by_multiple_subsets():
    class SubsetSample(Sample):
        image: np.ndarray = image_field(dtype=pl.UInt8(), format="RGB")
        subset: Subset = subset_field()

    dataset = Dataset(SubsetSample)
    dataset.append(SubsetSample(image=np.array([1]), subset=Subset.TRAINING))
    dataset.append(SubsetSample(image=np.array([2]), subset=Subset.VALIDATION))
    dataset.append(SubsetSample(image=np.array([3]), subset=Subset.TESTING))
    dataset.append(SubsetSample(image=np.array([4]), subset=Subset.TRAINING))

    # Test with list of subsets
    filtered = dataset.filter_by_subset([Subset.TRAINING, Subset.VALIDATION])
    assert len(filtered) == 3
    for sample in filtered:
        assert sample.subset in (Subset.TRAINING, Subset.VALIDATION)

    # Test with tuple of subsets
    filtered = dataset.filter_by_subset((Subset.VALIDATION, Subset.TESTING))
    assert len(filtered) == 2
    for sample in filtered:
        assert sample.subset in (Subset.VALIDATION, Subset.TESTING)


def test_dataset_creation_from_sample_class():
    """Test Dataset creation from Sample class."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)
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
            "image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.UInt8(), format="RGB")),
            "bbox": AttributeInfo(
                type=np.ndarray,
                field=bbox_field(dtype=pl.Float32(), normalize=False),
            ),
        }
    )

    dataset = Dataset(schema)

    assert dataset.schema == schema
    assert len(dataset.df) == 0


def test_dataset_append_sample():
    """Test adding samples to dataset."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)
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
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)
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
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)
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
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)
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
        left_image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB", semantic="left")
        right_image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="BGR", semantic="right")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=True)
        left_image_info: ImageInfo = image_info_field("left")
        right_image_info: ImageInfo = image_info_field("right")

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
            "image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.UInt8(), format="RGB")),
            "bbox": AttributeInfo(
                type=np.ndarray,
                field=bbox_field(dtype=pl.Float32(), normalize=False),
            ),
            "image_info": AttributeInfo(type=ImageInfo, field=image_info_field()),
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
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)

    class TargetSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)

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
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)
        image_info: ImageInfo = image_info_field()

    class TargetSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="BGR")  # Different format
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=True)  # Different normalization
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
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)
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


def test_dataset_len():
    """Test __len__ method returns correct dataset size."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)
        image_info: ImageInfo = image_info_field()

    dataset = Dataset(TestSample)

    # Initially empty
    assert len(dataset) == 0

    # Add one sample
    sample1 = TestSample(
        image=np.array([[[255, 0, 0]], [[0, 255, 0]]], dtype=np.uint8),
        bbox=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        image_info=ImageInfo(width=1, height=2),
    )
    dataset.append(sample1)
    assert len(dataset) == 1

    # Add second sample
    sample2 = TestSample(
        image=np.array([[[0, 0, 255]], [[255, 255, 0]]], dtype=np.uint8),
        bbox=np.array([[0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
        image_info=ImageInfo(width=1, height=2),
    )
    dataset.append(sample2)
    assert len(dataset) == 2

    # Remove a sample
    del dataset[0]

    assert len(dataset) == 1


def test_dataset_iter():
    """Test __iter__ method allows iteration over dataset samples."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)
        image_info: ImageInfo = image_info_field()

    dataset = Dataset(TestSample)

    # Create sample data for testing
    samples = [
        TestSample(
            image=np.array([[[255, 0, 0]], [[0, 255, 0]]], dtype=np.uint8),
            bbox=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            image_info=ImageInfo(width=1, height=2),
        ),
        TestSample(
            image=np.array([[[0, 0, 255]], [[255, 255, 0]]], dtype=np.uint8),
            bbox=np.array([[0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
            image_info=ImageInfo(width=2, height=3),
        ),
        TestSample(
            image=np.array([[[128, 64, 192]], [[96, 160, 32]]], dtype=np.uint8),
            bbox=np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32),
            image_info=ImageInfo(width=3, height=4),
        ),
    ]

    # Add samples to dataset
    for sample in samples:
        dataset.append(sample)

    # Test iteration with for loop
    iterated_samples: list[TestSample] = []
    for sample in dataset:
        iterated_samples.append(sample)

    # Verify we got all samples
    assert len(iterated_samples) == len(samples)

    # Verify each sample matches what we expect
    for original, iterated in zip(samples, iterated_samples):
        assert isinstance(iterated, TestSample)
        np.testing.assert_array_equal(iterated.bbox, original.bbox)
        assert iterated.image_info.width == original.image_info.width
        assert iterated.image_info.height == original.image_info.height

    # Test iteration with list comprehension
    list_samples = [sample for sample in dataset]
    assert len(list_samples) == 3

    # Test iteration is consistent (calling iterator multiple times)
    first_iteration = list(dataset)
    second_iteration = list(dataset)
    assert len(first_iteration) == len(second_iteration)

    # Test empty dataset iteration
    empty_dataset = Dataset(TestSample)
    empty_list = list(empty_dataset)
    assert len(empty_list) == 0


def test_dataset_delitem():
    """Test __delitem__ method allows deletion of dataset samples."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)
        image_info: ImageInfo = image_info_field()

    dataset = Dataset(TestSample)

    # Create and add sample data
    samples = [
        TestSample(
            image=np.array([[[255, 0, 0]], [[0, 255, 0]]], dtype=np.uint8),
            bbox=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            image_info=ImageInfo(width=1, height=2),
        ),
        TestSample(
            image=np.array([[[0, 0, 255]], [[255, 255, 0]]], dtype=np.uint8),
            bbox=np.array([[0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
            image_info=ImageInfo(width=2, height=3),
        ),
        TestSample(
            image=np.array([[[128, 64, 192]], [[96, 160, 32]]], dtype=np.uint8),
            bbox=np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32),
            image_info=ImageInfo(width=3, height=4),
        ),
    ]

    # Add samples to dataset
    for sample in samples:
        dataset.append(sample)

    # Initially should have 3 samples
    assert len(dataset) == 3

    # Delete middle sample (index 1)
    del dataset[1]
    assert len(dataset) == 2

    # Verify remaining samples are correct (original indices 0 and 2)
    remaining_sample_0 = dataset[0]
    remaining_sample_1 = dataset[1]  # This was originally index 2

    np.testing.assert_array_equal(remaining_sample_0.bbox, samples[0].bbox)
    np.testing.assert_array_equal(remaining_sample_1.bbox, samples[2].bbox)
    assert remaining_sample_0.image_info.width == 1
    assert remaining_sample_1.image_info.width == 3

    # Delete first sample (index 0)
    del dataset[0]
    assert len(dataset) == 1

    # Only the original third sample should remain
    last_sample = dataset[0]
    np.testing.assert_array_equal(last_sample.bbox, samples[2].bbox)
    assert last_sample.image_info.width == 3

    # Delete last sample
    del dataset[0]
    assert len(dataset) == 0

    # Test deleting from empty dataset should raise IndexError
    with pytest.raises(IndexError, match="Row index out of bounds"):
        del dataset[0]

    # Test deleting with out-of-bounds indices
    dataset.append(samples[0])  # Add one sample back
    assert len(dataset) == 1

    with pytest.raises(IndexError, match="Row index out of bounds"):
        del dataset[1]  # Index 1 is out of bounds

    with pytest.raises(IndexError, match="Row index out of bounds"):
        del dataset[-1]  # Negative indices not supported


def test_dataset_with_categories():
    """Test Dataset creation and usage with categories."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)
        image_info: ImageInfo = image_info_field()

    # Create label categories
    label_categories = LabelCategories(labels=("person", "car"))

    # Create mask categories
    mask_categories = MaskCategories(colormap={0: (0, 0, 0), 1: (255, 0, 0)})

    # Create dataset with categories dictionary mapping attributes to categories
    categories = {
        "bbox": label_categories,  # bbox field gets label categories
        "image": mask_categories,  # image field gets mask categories
    }
    dataset = Dataset(TestSample, categories=categories)

    # Test categories are stored correctly in schema AttributeInfo
    schema = dataset.schema
    assert schema.attributes["bbox"].categories is not None
    assert schema.attributes["image"].categories is not None
    assert isinstance(schema.attributes["bbox"].categories, LabelCategories)
    assert isinstance(schema.attributes["image"].categories, MaskCategories)

    # Test label categories
    labels = schema.attributes["bbox"].categories
    assert len(labels) == 2
    assert "person" in labels
    assert "car" in labels

    # Test mask categories
    masks = schema.attributes["image"].categories
    assert len(masks.colormap) == 2
    assert masks.colormap[0] == (0, 0, 0)
    assert masks.colormap[1] == (255, 0, 0)


def test_schema_copy_independence():
    """Test that schema modifications don't affect the original cached schema."""

    class TestSample(Sample):
        image: np.ndarray[Any, Any] = image_field(dtype=pl.UInt8(), format="RGB")
        bbox: np.ndarray[Any, Any] = bbox_field(dtype=pl.Float32(), normalize=False)

    # Create categories
    label_categories = LabelCategories(labels=("person",))

    # Create first dataset with categories
    dataset1 = Dataset(TestSample, categories={"bbox": label_categories})

    # Create second dataset without categories
    dataset2 = Dataset(TestSample)

    # Verify that the second dataset doesn't have categories from the first
    schema1 = dataset1.schema
    schema2 = dataset2.schema
    assert schema1.attributes["bbox"].categories is not None
    assert schema2.attributes["bbox"].categories is None

    # Verify that the cached schema from TestSample.infer_schema() is not modified
    original_schema = TestSample.infer_schema()
    assert original_schema.attributes["bbox"].categories is None
    assert original_schema.attributes["image"].categories is None

    # Create third dataset with different categories
    label_categories2 = LabelCategories(labels=("car", "truck"))

    dataset3 = Dataset(TestSample, categories={"image": label_categories2})

    # Verify independence through schema AttributeInfo
    schema1 = dataset1.schema
    schema2 = dataset2.schema
    schema3 = dataset3.schema

    assert schema1.attributes["bbox"].categories is not None
    assert schema2.attributes["bbox"].categories is None
    assert schema2.attributes["image"].categories is None
    assert schema3.attributes["bbox"].categories is None
    assert schema3.attributes["image"].categories is not None

    assert isinstance(schema1.attributes["bbox"].categories, LabelCategories)
    assert isinstance(schema3.attributes["image"].categories, LabelCategories)

    assert len(schema3.attributes["image"].categories) == 2  # car, truck
    assert len(schema1.attributes["bbox"].categories) == 1  # person


def test_union_type_handling():
    """Test Union type handling with both modern (A | B) and typing.Union syntax."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    from typing import Union

    from datumaro.experimental.type_registry import from_polars_data

    polars_data = [1.0, 2.0, 3.0]

    # Modern syntax
    if sys.version_info >= (3, 10):
        union_type_modern = torch.Tensor | np.ndarray
        result = from_polars_data(polars_data, union_type_modern)
        assert isinstance(result, torch.Tensor)
        assert result.tolist() == [1.0, 2.0, 3.0]

    # typing.Union syntax
    union_type_typing = Union[torch.Tensor, np.ndarray]
    result2 = from_polars_data(polars_data, union_type_typing)
    assert isinstance(result2, torch.Tensor)
    assert result2.tolist() == [1.0, 2.0, 3.0]


def test_dataset_with_optional_field():
    """Test dataset with np.ndarray | None field using mask_field, mixing None and array values."""

    class TestSample(Sample):
        mask: np.ndarray | None = mask_field(dtype=pl.UInt8())

    dataset = Dataset(TestSample, categories={"mask": MaskCategories.generate(size=256)})

    # Add samples with both None and ndarray values
    samples = [
        TestSample(mask=None),
        TestSample(mask=np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)),
        TestSample(mask=None),
        TestSample(mask=np.array([[255, 128, 64]], dtype=np.uint8)),
    ]
    for s in samples:
        dataset.append(s)

    # Check that values are preserved correctly
    assert len(dataset) == 4

    # First sample should have None mask
    sample_0 = dataset[0]
    assert sample_0.mask is None

    # Second sample should have the array
    sample_1 = dataset[1]
    assert sample_1.mask is not None
    np.testing.assert_array_equal(sample_1.mask, np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8))

    # Third sample should have None mask
    sample_2 = dataset[2]
    assert sample_2.mask is None

    # Fourth sample should have the array
    sample_3 = dataset[3]
    assert sample_3.mask is not None
    np.testing.assert_array_equal(sample_3.mask, np.array([[255, 128, 64]], dtype=np.uint8))


def test_is_type_optional_with_union_none():
    """Test is_type_optional correctly identifies Union types with None."""
    from typing import Union

    from datumaro.experimental.type_registry import is_type_optional

    # Test modern syntax (Python 3.10+)
    assert is_type_optional(int | None) is True
    assert is_type_optional(str | None) is True
    assert is_type_optional(np.ndarray | None) is True

    # Test typing.Union syntax
    assert is_type_optional(Union[int, None]) is True
    assert is_type_optional(Union[str, None]) is True
    assert is_type_optional(Union[np.ndarray, None]) is True

    # Test complex unions with None
    assert is_type_optional(int | str | None) is True
    assert is_type_optional(Union[int, str, None]) is True


def test_is_type_optional_with_non_optional_types():
    """Test is_type_optional correctly identifies non-optional types."""
    from typing import Union

    from datumaro.experimental.type_registry import is_type_optional

    # Non-optional simple types
    assert is_type_optional(int) is False
    assert is_type_optional(str) is False
    assert is_type_optional(np.ndarray) is False

    # Non-optional unions (without None)
    assert is_type_optional(int | str) is False
    assert is_type_optional(Union[int, str]) is False
    assert is_type_optional(Union[int, str, float]) is False


def test_getitem_missing_column_for_optional_field():
    """Test __getitem__ returns None for optional fields when columns are missing."""
    from datumaro.experimental.fields import numeric_field

    class SourceSample(Sample):
        value: int = numeric_field(dtype=pl.Int32(), semantic="main")

    class TargetSample(Sample):
        value: int = numeric_field(dtype=pl.Int32(), semantic="main")
        optional_value: int | None = numeric_field(dtype=pl.Int32(), semantic="optional")

    # Create source dataset
    source_dataset = Dataset(SourceSample)
    source_dataset.append(SourceSample(value=42))

    # Create target dataset by manually setting up the schema mismatch
    # This simulates a conversion where the optional field's column is not present
    target_schema = TargetSample.infer_schema()
    target_dataset = Dataset.from_dataframe(
        df=source_dataset.df,  # DataFrame without 'optional_value' column
        dtype_or_schema=TargetSample,
        schema=target_schema,
    )

    # Accessing the sample should return None for the missing optional field
    sample = target_dataset[0]
    assert sample.value == 42
    assert sample.optional_value is None


def test_getitem_missing_column_for_required_field_raises():
    """Test __getitem__ raises KeyError for required fields when columns are missing."""
    from datumaro.experimental.fields import numeric_field

    class SourceSample(Sample):
        value: int = numeric_field(dtype=pl.Int32(), semantic="value")

    class TargetSample(Sample):
        value: int = numeric_field(dtype=pl.Int32(), semantic="value")
        required_value: int = numeric_field(dtype=pl.Int32(), semantic="required")  # Required, not optional

    # Create source dataset
    source_dataset = Dataset(SourceSample)
    source_dataset.append(SourceSample(value=42))

    # Create target dataset with schema that expects a column that doesn't exist
    target_schema = TargetSample.infer_schema()
    target_dataset = Dataset.from_dataframe(
        df=source_dataset.df,  # DataFrame without 'required_value' column
        dtype_or_schema=TargetSample,
        schema=target_schema,
    )

    # Accessing the sample should raise KeyError for the missing required field
    with pytest.raises(KeyError, match=r"Required columns.*'required_value'.*not found"):
        target_dataset[0]


def test_getitem_with_multiple_optional_missing_columns():
    """Test __getitem__ handles multiple missing optional columns correctly."""
    from datumaro.experimental.fields import numeric_field, string_field

    class SourceSample(Sample):
        name: str = string_field(semantic="name")

    class TargetSample(Sample):
        name: str = string_field(semantic="name")
        optional_a: int | None = numeric_field(dtype=pl.Int32(), semantic="a")
        optional_b: str | None = string_field(semantic="b")
        optional_c: float | None = numeric_field(dtype=pl.Float32(), semantic="c")

    # Create source dataset with only 'name'
    source_df = pl.DataFrame({"name": ["test"]})

    # Create target dataset
    target_schema = TargetSample.infer_schema()
    target_dataset = Dataset.from_dataframe(
        df=source_df,
        dtype_or_schema=TargetSample,
        schema=target_schema,
    )

    # Access sample - all optional fields should be None
    sample = target_dataset[0]
    assert sample.name == "test"
    assert sample.optional_a is None
    assert sample.optional_b is None
    assert sample.optional_c is None


# ── filter_by_labels tests ──────────────────────────────────────────────────


def test_filter_by_labels_basic_filtering():
    """Test basic filtering with single-value LabelField: strings, integers, single/multiple, matches."""
    categories = LabelCategories(labels=("cat", "dog", "bird"))
    schema = Schema(attributes={"label": AttributeInfo(type=int, field=label_field(), categories=categories)})
    ds = Dataset(schema)
    ds.append(Sample(label=0))  # cat
    ds.append(Sample(label=1))  # dog
    ds.append(Sample(label=2))  # bird
    ds.append(Sample(label=0))  # cat

    # Filter by single label name (list)
    filtered = ds.filter_by_labels(["cat"])
    assert len(filtered) == 2
    assert all(s.label == 0 for s in filtered)

    # Filter by single label name (string - not list)
    filtered = ds.filter_by_labels("dog")
    assert len(filtered) == 1
    assert filtered[0].label == 1

    # Filter by multiple label names
    filtered = ds.filter_by_labels(["dog", "bird"])
    assert len(filtered) == 2
    assert {filtered[0].label, filtered[1].label} == {1, 2}

    # Filter by single integer index
    filtered = ds.filter_by_labels(0)
    assert len(filtered) == 2

    # Filter by multiple integer indices
    filtered = ds.filter_by_labels([1, 2])
    assert len(filtered) == 2

    # Mix strings and integers
    filtered = ds.filter_by_labels(["cat", 2])
    assert len(filtered) == 3

    # All match
    filtered = ds.filter_by_labels(["cat", "dog", "bird"])
    assert len(filtered) == 4

    # No match
    ds2 = Dataset(schema)
    ds2.append(Sample(label=0))
    filtered = ds2.filter_by_labels(["bird"])
    assert len(filtered) == 0


def test_filter_by_labels_list_field():
    """Filter a dataset with is_list=True LabelField using strings and integers."""
    categories = LabelCategories(labels=("cat", "dog", "bird"))
    schema = Schema(
        attributes={
            "labels": AttributeInfo(
                type=list,
                field=label_field(dtype=pl.UInt8(), is_list=True),
                categories=categories,
            )
        }
    )
    ds = Dataset(schema)
    ds.append(Sample(labels=[0, 1]))  # cat, dog
    ds.append(Sample(labels=[2]))  # bird
    ds.append(Sample(labels=[1, 2]))  # dog, bird
    ds.append(Sample(labels=[0]))  # cat

    # Filter by label name
    filtered = ds.filter_by_labels(["dog"])
    assert len(filtered) == 2  # rows 0, 2

    # Filter by integer index
    filtered = ds.filter_by_labels([1])
    assert len(filtered) == 2

    # Filter by multiple labels
    filtered = ds.filter_by_labels(["cat", "bird"])
    assert len(filtered) == 4  # every row has at least one

    # Filter by mixed string and integer
    filtered = ds.filter_by_labels(["bird", 1])
    assert len(filtered) == 3


def test_filter_by_labels_multi_label_field():
    """Filter a dataset with multi_label=True LabelField using strings and integers."""
    categories = LabelCategories(labels=("sunny", "rainy", "cloudy"))
    schema = Schema(
        attributes={
            "weather": AttributeInfo(
                type=list,
                field=label_field(dtype=pl.UInt8(), multi_label=True),
                categories=categories,
            )
        }
    )
    ds = Dataset(schema)
    ds.append(Sample(weather=[0, 2]))  # sunny + cloudy
    ds.append(Sample(weather=[1]))  # rainy
    ds.append(Sample(weather=[0, 1]))  # sunny + rainy

    # Filter by label name
    filtered = ds.filter_by_labels(["sunny"], label_field_name="weather")
    assert len(filtered) == 2  # rows 0, 2

    # Filter by integer index
    filtered = ds.filter_by_labels([0], label_field_name="weather")
    assert len(filtered) == 2

    # Filter by multiple labels
    filtered = ds.filter_by_labels(["rainy", "cloudy"], label_field_name="weather")
    assert len(filtered) == 3

    # Filter by mixed
    filtered = ds.filter_by_labels(["rainy", 2], label_field_name="weather")
    assert len(filtered) == 3


def test_filter_by_labels_list_and_multi_label():
    """Filter a dataset whose LabelField has both is_list=True and multi_label=True (List(List(UInt)))."""
    categories = LabelCategories(labels=("a", "b", "c"))
    schema = Schema(
        attributes={
            "tags": AttributeInfo(
                type=list,
                field=label_field(dtype=pl.UInt8(), is_list=True, multi_label=True),
                categories=categories,
            )
        }
    )
    ds = Dataset(schema)
    ds.append(Sample(tags=[[0, 1], [2]]))  # [a,b], [c]
    ds.append(Sample(tags=[[1]]))  # [b]
    ds.append(Sample(tags=[[2], [0, 2]]))  # [c], [a,c]

    filtered = ds.filter_by_labels(["a"], label_field_name="tags")
    assert len(filtered) == 2  # rows 0, 2

    filtered = ds.filter_by_labels(["b"], label_field_name="tags")
    assert len(filtered) == 2  # rows 0, 1

    filtered = ds.filter_by_labels(["c"], label_field_name="tags")
    assert len(filtered) == 2  # rows 0, 2


# ── auto-detection tests ────────────────────────────────────────────────────


def test_filter_by_labels_auto_detect_single_label_field():
    """When schema has exactly one LabelField, label_field_name can be omitted."""
    categories = LabelCategories(labels=("x", "y"))
    schema = Schema(
        attributes={
            "label": AttributeInfo(type=int, field=label_field(), categories=categories),
            "image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.UInt8(), format="RGB")),
        }
    )
    ds = Dataset(schema)
    ds.append(Sample(label=0, image=np.array([[[1, 2, 3]]], dtype=np.uint8)))
    ds.append(Sample(label=1, image=np.array([[[4, 5, 6]]], dtype=np.uint8)))

    # Should auto-detect "label" as the LabelField
    filtered = ds.filter_by_labels(["x"])
    assert len(filtered) == 1
    assert filtered[0].label == 0


def test_filter_by_labels_auto_detect_no_label_field():
    """RuntimeError when schema has no LabelField and label_field_name is not given."""
    schema = Schema(
        attributes={
            "image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.UInt8(), format="RGB")),
        }
    )
    ds = Dataset(schema)
    ds.append(Sample(image=np.array([[[1, 2, 3]]], dtype=np.uint8)))

    with pytest.raises(RuntimeError, match="does not contain any LabelField"):
        ds.filter_by_labels(["anything"])


def test_filter_by_labels_auto_detect_multiple_label_fields():
    """RuntimeError when schema has multiple LabelFields and label_field_name is not given."""
    categories = LabelCategories(labels=("a", "b"))
    schema = Schema(
        attributes={
            "primary": AttributeInfo(type=int, field=label_field(semantic="primary"), categories=categories),
            "secondary": AttributeInfo(type=int, field=label_field(semantic="secondary"), categories=categories),
        }
    )
    ds = Dataset(schema)
    ds.append(Sample(primary=0, secondary=1))

    with pytest.raises(RuntimeError, match="multiple LabelField"):
        ds.filter_by_labels(["a"])

    # But specifying explicitly should work
    filtered = ds.filter_by_labels(["a"], label_field_name="primary")
    assert len(filtered) == 1


# ── HierarchicalLabelCategories tests ───────────────────────────────────────


def test_filter_by_labels_with_hierarchical_categories():
    """filter_by_labels works with HierarchicalLabelCategories."""
    from datumaro.experimental.categories import HierarchicalLabelCategories, HierarchicalLabelCategory

    items = (
        HierarchicalLabelCategory(name="animal"),
        HierarchicalLabelCategory(name="cat", parent="animal"),
        HierarchicalLabelCategory(name="dog", parent="animal"),
    )
    categories = HierarchicalLabelCategories(items=items)

    schema = Schema(attributes={"label": AttributeInfo(type=int, field=label_field(), categories=categories)})
    ds = Dataset(schema)
    ds.append(Sample(label=0))  # animal
    ds.append(Sample(label=1))  # cat
    ds.append(Sample(label=2))  # dog

    filtered = ds.filter_by_labels(["cat", "dog"], label_field_name="label")
    assert len(filtered) == 2
    assert {filtered[0].label, filtered[1].label} == {1, 2}


# ── immutability / original-unmodified tests ────────────────────────────────


def test_filter_by_labels_does_not_mutate_original():
    """The original dataset must not be modified by filtering."""
    categories = LabelCategories(labels=("cat", "dog"))
    schema = Schema(attributes={"label": AttributeInfo(type=int, field=label_field(), categories=categories)})
    ds = Dataset(schema)
    ds.append(Sample(label=0))
    ds.append(Sample(label=1))
    ds.append(Sample(label=0))

    original_len = len(ds)
    _ = ds.filter_by_labels(["cat"])
    assert len(ds) == original_len


# ── validation/error tests ──────────────────────────────────────────────────


def test_filter_by_labels_validation_errors():
    """Test various validation errors: out of range index, negative index, invalid type."""
    categories = LabelCategories(labels=("cat", "dog"))
    schema = Schema(attributes={"label": AttributeInfo(type=int, field=label_field(), categories=categories)})
    ds = Dataset(schema)
    ds.append(Sample(label=0))

    # Index out of range
    with pytest.raises(ValueError, match="out of range"):
        ds.filter_by_labels([5])

    # Negative index
    with pytest.raises(ValueError, match="out of range"):
        ds.filter_by_labels([-1])

    # Invalid type (not string or int)
    with pytest.raises(TypeError, match=r"must be a string.*or int"):
        ds.filter_by_labels([1.5])

    # Unknown label name
    with pytest.raises(ValueError, match="not found in categories"):
        ds.filter_by_labels(["elephant"], label_field_name="label")


def test_filter_by_labels_schema_errors():
    """Test errors related to schema: field not found, wrong field type, missing/wrong categories."""
    categories = LabelCategories(labels=("cat", "dog"))
    schema = Schema(attributes={"label": AttributeInfo(type=int, field=label_field(), categories=categories)})
    ds = Dataset(schema)
    ds.append(Sample(label=0))

    # Field not found
    with pytest.raises(KeyError, match="not_a_field"):
        ds.filter_by_labels(["cat"], label_field_name="not_a_field")

    # Field is not a LabelField
    schema2 = Schema(
        attributes={"image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.UInt8(), format="RGB"))}
    )
    ds2 = Dataset(schema2)
    ds2.append(Sample(image=np.array([[[1, 2, 3]]], dtype=np.uint8)))
    with pytest.raises(TypeError, match="not a LabelField"):
        ds2.filter_by_labels(["cat"], label_field_name="image")

    # No categories attached
    schema3 = Schema(attributes={"label": AttributeInfo(type=int, field=label_field())})
    df = pl.DataFrame({"label": pl.Series([0, 1], dtype=pl.UInt8)})
    ds3 = Dataset.from_dataframe(df, dtype_or_schema=schema3)
    with pytest.raises(ValueError, match="does not have LabelCategories"):
        ds3.filter_by_labels(["cat"], label_field_name="label")

    # Wrong categories type
    mask_cats = MaskCategories(labels=["bg", "fg"])
    schema4 = Schema(attributes={"label": AttributeInfo(type=int, field=label_field(), categories=mask_cats)})
    df = pl.DataFrame({"label": pl.Series([0, 1], dtype=pl.UInt8)})
    ds4 = Dataset.from_dataframe(df, dtype_or_schema=schema4)
    with pytest.raises(ValueError, match="does not have LabelCategories"):
        ds4.filter_by_labels(["bg"], label_field_name="label")


def test_filter_by_labels_empty_labels_raises_error():
    """Filtering with empty labels list raises ValueError."""
    categories = LabelCategories(labels=("cat", "dog", "bird"))
    schema = Schema(attributes={"label": AttributeInfo(type=int, field=label_field(), categories=categories)})
    ds = Dataset(schema)

    with pytest.raises(ValueError, match="No labels provided to filter"):
        ds.filter_by_labels([])
