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
    # Should not raise
    valid_sample.validate()


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

    dataset = Dataset(TestSample)

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
