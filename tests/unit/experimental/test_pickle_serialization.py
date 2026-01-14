"""Unit tests for pickle serialization of Dataset, Transform and mask loader classes."""

import pickle
from typing import Annotated, Callable

import numpy as np
import polars as pl

from datumaro.components.annotation import AnnotationType, ExtractedMask, LabelCategories
from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields.images import image_callable_field
from datumaro.experimental.legacy import ForwardMaskAnnotationConverter, convert_from_legacy


# Module-level sample class for pickle tests (local classes can't be pickled)
class PicklableSample(Sample):
    """Sample class defined at module level for pickle testing."""

    image: Annotated[Callable, image_callable_field()]


# Module-level callable for pickle tests (local functions can't be pickled by default pickle)
def module_level_image_loader():
    """A module-level callable that can be pickled."""
    return np.zeros((10, 10, 3), dtype=np.uint8)


class SemanticMaskLoaderTest:
    """Tests for the SemanticMaskLoader picklable callable class."""

    def test_semantic_mask_loader_empty(self):
        """Test SemanticMaskLoader with empty mask list."""
        from datumaro.experimental.legacy.annotation_converters import SemanticMaskLoader

        loader = SemanticMaskLoader([])
        result = loader()
        assert result is None

    def test_semantic_mask_loader_single_mask(self):
        """Test SemanticMaskLoader with a single mask."""
        from datumaro.experimental.legacy.annotation_converters import SemanticMaskLoader

        mask = np.array([[True, False], [False, True]], dtype=bool)
        mask_data_list = [(mask, 1)]

        loader = SemanticMaskLoader(mask_data_list)
        result = loader()

        assert result is not None
        assert result.shape == (2, 2)
        assert result.dtype == np.uint8
        assert result[0, 0] == 1
        assert result[0, 1] == 0
        assert result[1, 0] == 0
        assert result[1, 1] == 1

    def test_semantic_mask_loader_multiple_masks(self):
        """Test SemanticMaskLoader with multiple masks."""
        from datumaro.experimental.legacy.annotation_converters import SemanticMaskLoader

        mask1 = np.array([[True, False], [False, False]], dtype=bool)
        mask2 = np.array([[False, True], [True, False]], dtype=bool)
        mask_data_list = [(mask1, 1), (mask2, 2)]

        loader = SemanticMaskLoader(mask_data_list)
        result = loader()

        assert result is not None
        assert result[0, 0] == 1  # From mask1
        assert result[0, 1] == 2  # From mask2
        assert result[1, 0] == 2  # From mask2
        assert result[1, 1] == 0  # Background

    def test_semantic_mask_loader_is_picklable(self):
        """Test that SemanticMaskLoader can be pickled and unpickled."""
        from datumaro.experimental.legacy.annotation_converters import SemanticMaskLoader

        mask = np.array([[True, False], [False, True]], dtype=bool)
        mask_data_list = [(mask, 1)]

        loader = SemanticMaskLoader(mask_data_list)

        # Pickle and unpickle
        pickled = pickle.dumps(loader)
        restored_loader = pickle.loads(pickled)

        # Verify the restored loader works correctly
        result = restored_loader()
        assert result is not None
        assert result[0, 0] == 1
        assert result[1, 1] == 1


class InstanceMaskLoaderTest:
    """Tests for the InstanceMaskLoader picklable callable class."""

    def test_instance_mask_loader_empty(self):
        """Test InstanceMaskLoader with empty mask list."""
        from datumaro.experimental.legacy.annotation_converters import InstanceMaskLoader

        loader = InstanceMaskLoader([])
        result = loader()

        assert result.shape == (0, 0, 0)
        assert result.dtype == bool

    def test_instance_mask_loader_single_mask(self):
        """Test InstanceMaskLoader with a single mask."""
        from datumaro.experimental.legacy.annotation_converters import InstanceMaskLoader

        mask = np.array([[True, False], [False, True]], dtype=bool)
        mask_images = [mask]

        loader = InstanceMaskLoader(mask_images)
        result = loader()

        assert result.shape == (1, 2, 2)
        assert result.dtype == bool
        assert result[0, 0, 0] == True  # noqa: E712
        assert result[0, 0, 1] == False  # noqa: E712

    def test_instance_mask_loader_multiple_masks(self):
        """Test InstanceMaskLoader with multiple masks."""
        from datumaro.experimental.legacy.annotation_converters import InstanceMaskLoader

        mask1 = np.array([[True, False], [False, False]], dtype=bool)
        mask2 = np.array([[False, True], [True, False]], dtype=bool)
        mask_images = [mask1, mask2]

        loader = InstanceMaskLoader(mask_images)
        result = loader()

        assert result.shape == (2, 2, 2)
        assert np.array_equal(result[0], mask1)
        assert np.array_equal(result[1], mask2)

    def test_instance_mask_loader_is_picklable(self):
        """Test that InstanceMaskLoader can be pickled and unpickled."""
        from datumaro.experimental.legacy.annotation_converters import InstanceMaskLoader

        mask1 = np.array([[True, False], [False, False]], dtype=bool)
        mask2 = np.array([[False, True], [True, False]], dtype=bool)
        mask_images = [mask1, mask2]

        loader = InstanceMaskLoader(mask_images)

        # Pickle and unpickle
        pickled = pickle.dumps(loader)
        restored_loader = pickle.loads(pickled)

        # Verify the restored loader works correctly
        result = restored_loader()
        assert result.shape == (2, 2, 2)
        assert np.array_equal(result[0], mask1)
        assert np.array_equal(result[1], mask2)


class DatasetPickleSerializationTest:
    """Tests for Dataset pickle serialization with Object columns."""

    def test_dataset_with_object_columns_is_picklable(self):
        """Test that a Dataset with Object columns can be pickled and unpickled."""
        # Use module-level sample class (local classes can't be pickled)
        ds = Dataset(PicklableSample)

        # Add a sample with a module-level callable (local functions can't be pickled)
        sample = PicklableSample(image=module_level_image_loader)
        ds.append(sample)

        # Verify the dataset has Object columns
        assert pl.Object in ds.df.schema.values()

        # Pickle and unpickle
        pickled = pickle.dumps(ds)
        restored_ds = pickle.loads(pickled)

        # Verify the restored dataset works
        assert len(restored_ds) == 1
        # Verify the callable was restored and works
        restored_sample = restored_ds[0]
        result = restored_sample.image()
        assert result.shape == (10, 10, 3)

    def test_empty_dataset_with_object_schema_is_picklable(self):
        """Test that an empty Dataset with Object column schema can be pickled."""
        # Use module-level sample class
        ds = Dataset(PicklableSample)

        # Pickle and unpickle empty dataset
        pickled = pickle.dumps(ds)
        restored_ds = pickle.loads(pickled)

        assert len(restored_ds) == 0


class TransformPickleSerializationTest:
    """Tests for Transform pickle serialization with Object columns."""

    def test_identity_transform_with_object_columns_is_picklable(self):
        """Test that IdentityTransform with Object columns can be pickled."""
        from datumaro.experimental.fields.images import ImageCallableField
        from datumaro.experimental.schema import AttributeInfo, Schema
        from datumaro.experimental.transform import IdentityTransform

        # Create a DataFrame with Object column using module-level callable
        df = pl.DataFrame({"data": pl.Series([module_level_image_loader], dtype=pl.Object())})

        # Create schema
        schema = Schema(attributes={"data": AttributeInfo(type=callable, field=ImageCallableField())})

        transform = IdentityTransform(df, schema)

        # Pickle and unpickle
        pickled = pickle.dumps(transform)
        restored_transform = pickle.loads(pickled)

        # Verify the restored transform works
        assert len(restored_transform) == 1
        result_df = restored_transform.apply(["data"])
        assert "data" in result_df.columns


class ForwardMaskAnnotationConverterPickleTest:
    """Tests for ForwardMaskAnnotationConverter producing picklable results."""

    def test_semantic_mask_conversion_is_picklable(self):
        """Test that semantic mask conversion results are picklable."""
        # Create a legacy dataset with semantic segmentation masks
        index_mask_data = np.array([[0, 1], [1, 2]], dtype=np.uint8)

        legacy_categories = LabelCategories()
        legacy_categories.add("background")
        legacy_categories.add("foreground")
        legacy_categories.add("other")

        mask1 = ExtractedMask(index_mask=index_mask_data, index=1, label=1)
        mask2 = ExtractedMask(index_mask=index_mask_data, index=2, label=2)
        item = DatasetItem(
            id="test",
            media=Image.from_numpy(np.zeros((2, 2, 3), dtype=np.uint8)),
            annotations=[mask1, mask2],
        )

        legacy_dataset = LegacyDataset.from_iterable([item], categories={AnnotationType.label: legacy_categories})

        # Create the converter
        converter = ForwardMaskAnnotationConverter.create(legacy_dataset)
        assert converter is not None
        assert converter.is_semantic is True

        # Convert the annotations
        result = converter.convert_annotations(item.annotations, item)

        # The result should contain a picklable callable
        mask_callable = result["mask_callable"]
        assert callable(mask_callable)

        # Pickle and unpickle the callable
        pickled = pickle.dumps(mask_callable)
        restored_callable = pickle.loads(pickled)

        # Verify the restored callable works
        output_mask = restored_callable()
        assert output_mask is not None

    def test_instance_mask_conversion_is_picklable(self):
        """Test that instance mask conversion results are picklable."""
        # Create a legacy dataset with instance segmentation masks
        index_mask_data = np.array([[0, 1], [1, 2]], dtype=np.uint8)

        legacy_categories = LabelCategories()
        legacy_categories.add("object")

        # Instance segmentation: indices don't match labels
        mask1 = ExtractedMask(index_mask=index_mask_data, index=1, label=0)
        mask2 = ExtractedMask(index_mask=index_mask_data, index=2, label=0)

        item = DatasetItem(
            id="test",
            media=Image.from_numpy(np.zeros((2, 2, 3), dtype=np.uint8)),
            annotations=[mask1, mask2],
        )

        legacy_dataset = LegacyDataset.from_iterable([item], categories={AnnotationType.label: legacy_categories})

        # Create the converter
        converter = ForwardMaskAnnotationConverter.create(legacy_dataset)
        assert converter is not None
        assert converter.is_semantic is False

        # Convert the annotations
        result = converter.convert_annotations(item.annotations, item)

        # The result should contain a picklable callable
        instance_mask_callable = result["instance_mask_callable"]
        assert callable(instance_mask_callable)

        # Pickle and unpickle the callable
        pickled = pickle.dumps(instance_mask_callable)
        restored_callable = pickle.loads(pickled)

        # Verify the restored callable works
        output_masks = restored_callable()
        assert output_masks.shape == (2, 2, 2)


class ConvertFromLegacyPickleTest:
    """Tests for convert_from_legacy producing picklable datasets."""

    def test_converted_segmentation_dataset_is_picklable(self):
        """Test that a converted segmentation dataset can be pickled."""
        # Create a simple semantic segmentation dataset
        index_mask_data = np.array([[0, 1], [1, 2]], dtype=np.uint8)

        legacy_categories = LabelCategories()
        legacy_categories.add("background")
        legacy_categories.add("foreground")
        legacy_categories.add("other")

        mask1 = ExtractedMask(index_mask=index_mask_data, index=1, label=1)
        mask2 = ExtractedMask(index_mask=index_mask_data, index=2, label=2)
        item = DatasetItem(
            id="test",
            media=Image.from_numpy(np.zeros((2, 2, 3), dtype=np.uint8)),
            annotations=[mask1, mask2],
        )

        legacy_dataset = LegacyDataset.from_iterable([item], categories={AnnotationType.label: legacy_categories})

        # Convert the dataset
        new_dataset = convert_from_legacy(legacy_dataset)

        # Pickle and unpickle the dataset
        pickled = pickle.dumps(new_dataset)
        restored_dataset = pickle.loads(pickled)

        # Verify the restored dataset works
        assert len(restored_dataset) == 1
