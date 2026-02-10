"""Unit tests for legacy dataset conversion functionality."""

import io
import pickle
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from PIL import Image as PILImage
from typing_extensions import Annotated

import datumaro.experimental.categories as exp_categories
from datumaro.components.annotation import AnnotationType, Bbox, ExtractedMask, LabelCategories
from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, ImageFromData, ImageFromFile, Video
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import bbox_field, image_path_field, label_field, rotated_bbox_field, tensor_field
from datumaro.experimental.legacy import (
    BackwardBboxAnnotationConverter,
    BackwardImageMediaConverter,
    ForwardBboxAnnotationConverter,
    ForwardImageMediaConverter,
    analyze_experimental_dataset,
    analyze_legacy_dataset,
    convert_from_legacy,
    convert_to_legacy,
    register_forward_annotation_converter,
    register_forward_media_converter,
)
from datumaro.experimental.legacy.annotation_converters import get_forward_annotation_converter
from datumaro.experimental.legacy.dataset_converters import _attributes_to_dict, _has_derived_labels
from datumaro.experimental.legacy.register_legacy_converters import get_forward_media_converter
from datumaro.util.image import encode_image


class ConvertFromLegacyTest:
    """Tests for convert_from_legacy functionality."""

    def test_convert_from_legacy_with_image_bytes(self):
        """Test full conversion pipeline with ImageFromData images."""
        # Create test images
        test_image1 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        test_image2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        # Encode images as bytes
        image_bytes1 = encode_image(test_image1, ".png")
        image_bytes2 = encode_image(test_image2, ".jpg")

        # Create dataset items with ImageFromBytes media
        items = [
            DatasetItem(id="item1", media=Image.from_bytes(image_bytes1), annotations=[]),
            DatasetItem(id="item2", media=Image.from_bytes(image_bytes2), annotations=[]),
        ]

        # Create legacy dataset
        legacy_dataset = LegacyDataset.from_iterable(items)

        # Convert to v2 format
        experimental_dataset = convert_from_legacy(legacy_dataset)

        # Verify conversion
        assert "image_bytes" in experimental_dataset.schema.attributes
        assert "image_path" not in experimental_dataset.schema.attributes

        # Verify samples
        assert len(experimental_dataset) == 2

        sample1 = experimental_dataset[0]
        sample2 = experimental_dataset[1]

        # Check that image_bytes are present
        assert hasattr(sample1, "image_bytes")
        assert hasattr(sample2, "image_bytes")

    def test_convert_from_legacy_with_image_paths(self):
        """Test that file-based images still use ImagePathField correctly."""
        # Create dataset items with ImageFromFile media
        items = [
            DatasetItem(id="item1", media=Image.from_file("test1.jpg"), annotations=[]),
            DatasetItem(id="item2", media=Image.from_file("test2.png"), annotations=[]),
        ]

        # Create legacy dataset
        legacy_dataset = LegacyDataset.from_iterable(items)

        # Convert to v2 format
        experimental_dataset = convert_from_legacy(legacy_dataset)

        # Verify conversion
        assert "image_path" in experimental_dataset.schema.attributes
        assert "image_bytes" not in experimental_dataset.schema.attributes

        # Verify samples
        assert len(experimental_dataset) == 2

        sample1 = experimental_dataset[0]
        sample2 = experimental_dataset[1]

        # Check that image_path are present
        assert hasattr(sample1, "image_path")
        assert hasattr(sample2, "image_path")

    def test_convert_from_legacy_with_callable_image_data(self):
        """Test conversion with ImageFromData containing callable _data."""

        # Create test image bytes
        test_image1 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        test_image2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        # Create bytes using PIL
        def create_image_bytes(img_array):
            pil_image = PILImage.fromarray(img_array)
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format="PNG")
            return img_bytes.getvalue()

        image_bytes1 = create_image_bytes(test_image1)
        image_bytes2 = create_image_bytes(test_image2)

        # Create callables that return these bytes
        def get_image_bytes1():
            return image_bytes1

        def get_image_bytes2():
            return image_bytes2

        # Create dataset items with ImageFromData containing callable _data
        items = [
            DatasetItem(id="item1", media=ImageFromData(data=get_image_bytes1), annotations=[]),
            DatasetItem(id="item2", media=ImageFromData(data=get_image_bytes2), annotations=[]),
        ]

        # Create legacy dataset
        legacy_dataset = LegacyDataset.from_iterable(items)

        # Convert to v2 format
        experimental_dataset = convert_from_legacy(legacy_dataset)

        # Verify conversion uses callable field instead of bytes field
        assert "image_callable" in experimental_dataset.schema.attributes
        assert "image_bytes" not in experimental_dataset.schema.attributes
        assert "image_path" not in experimental_dataset.schema.attributes

        # Verify samples
        assert len(experimental_dataset) == 2

        sample1 = experimental_dataset[0]
        sample2 = experimental_dataset[1]

        # Check that image_callable are present and work
        assert hasattr(sample1, "image_callable")
        assert hasattr(sample2, "image_callable")
        assert callable(sample1.image_callable)
        assert callable(sample2.image_callable)

        # Test that callables return proper image arrays
        result1 = sample1.image_callable()
        result2 = sample2.image_callable()

        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
        assert result1.dtype == np.uint8
        assert result2.dtype == np.uint8
        assert result1.shape == (32, 32, 3)
        assert result2.shape == (64, 64, 3)

    def test_image_converter_with_mixed_callable_and_bytes_data(self):
        """Test that mixed callable and non-callable data uses callable field."""

        # Create test image bytes
        test_image = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        pil_image = PILImage.fromarray(test_image)
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="PNG")
        image_bytes = img_bytes.getvalue()

        def get_image_bytes():
            return image_bytes

        # Create dataset with mixed callable and non-callable data
        items = [
            DatasetItem(id="item1", media=ImageFromData(data=get_image_bytes), annotations=[]),  # callable
            DatasetItem(id="item2", media=ImageFromData(data=image_bytes), annotations=[]),  # non-callable
        ]

        legacy_dataset = LegacyDataset.from_iterable(items)
        experimental_dataset = convert_from_legacy(legacy_dataset)

        # Should use callable field (callable takes precedence)
        assert "image_callable" in experimental_dataset.schema.attributes
        assert "image_bytes" not in experimental_dataset.schema.attributes

        # Both samples should have working callables
        sample1 = experimental_dataset[0]
        sample2 = experimental_dataset[1]

        assert hasattr(sample1, "image_callable")
        assert hasattr(sample2, "image_callable")

        result1 = sample1.image_callable()
        result2 = sample2.image_callable()

        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
        assert result1.shape == (16, 16, 3)
        assert result2.shape == (16, 16, 3)


class ConverterRegistryTest:
    """Tests for converter registry functionality."""

    def test_register_and_get_media_converter(self):
        """Test media converter registration and retrieval."""
        register_forward_media_converter(ForwardImageMediaConverter)

        # Create a dataset with file-based images
        items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
        dataset = LegacyDataset.from_iterable(items)

        retrieved = get_forward_media_converter(dataset)
        assert retrieved is not None
        assert isinstance(retrieved, ForwardImageMediaConverter)

    def test_register_and_get_annotation_converter(self):
        """Test annotation converter registration and retrieval."""
        register_forward_annotation_converter(ForwardBboxAnnotationConverter)

        # Create a dataset with empty categories
        dataset = LegacyDataset.from_iterable([], categories={})
        retrieved = get_forward_annotation_converter(AnnotationType.bbox, dataset)
        assert retrieved is not None
        assert isinstance(retrieved, ForwardBboxAnnotationConverter)

    def test_get_nonexistent_media_converter(self):
        """Test getting converter for unsupported dataset."""
        # Create a dataset with Video media (not supported by current converters)
        items = [DatasetItem(id="test", media=Video("/path/to/video.mp4"))]
        dataset = LegacyDataset.from_iterable(items, media_type=Video)

        retrieved = get_forward_media_converter(dataset)
        assert retrieved is None

    def test_get_nonexistent_annotation_converter(self):
        """Test getting converter for unregistered annotation type."""
        dataset = LegacyDataset.from_iterable([], categories={})
        converter = get_forward_annotation_converter(AnnotationType.unknown, dataset)
        assert converter is None


class AnalyzeLegacyDatasetTest:
    """Tests for analyze_legacy_dataset functionality."""

    def test_analyze_image_and_bbox_dataset(self):
        """Test analysis of dataset with images and bboxes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            image_path = str(temp_path / "image1.jpg")
            image_media = Image.from_file(image_path, size=(480, 640))
            bbox = Bbox(10, 20, 30, 40, label=1)
            item = DatasetItem(id="item1", media=image_media, annotations=[bbox])

            # Create dataset with label categories
            label_categories = LabelCategories()
            label_categories.add("background")
            label_categories.add("class_1")

            dataset = LegacyDataset.from_iterable(
                [item],
                ann_types={AnnotationType.bbox},
                categories={AnnotationType.label: label_categories},
            )

            analysis_result = analyze_legacy_dataset(dataset)

            # Should have image and bbox attributes
            assert "image_path" in analysis_result.schema.attributes
            assert "bboxes" in analysis_result.schema.attributes
            assert "labels" in analysis_result.schema.attributes

    def test_analyze_image_only_dataset(self):
        """Test analysis of dataset with only images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            image_path = str(temp_path / "image1.jpg")
            image_media = Image.from_file(image_path, size=(480, 640))
            item = DatasetItem(id="item1", media=image_media, annotations=[])

            dataset = LegacyDataset.from_iterable([item])

            analysis_result = analyze_legacy_dataset(dataset)

            # Should have only image attributes
            assert "image_path" in analysis_result.schema.attributes
            assert "bboxes" not in analysis_result.schema.attributes
            assert "bbox_labels" not in analysis_result.schema.attributes

    def test_analyze_unknown_annotation_type(self):
        """Test analysis with unknown annotation type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            image_path = str(temp_path / "image1.jpg")
            image_media = Image.from_file(image_path, size=(480, 640))

            class UnknownAnnotation(Bbox):
                _type = AnnotationType.unknown

            bbox = UnknownAnnotation(10, 20, 30, 40, label=1)
            item = DatasetItem(id="item1", media=image_media, annotations=[bbox])

            dataset = LegacyDataset.from_iterable([item])

            analysis_result = analyze_legacy_dataset(dataset)

            # Should skip unknown annotation type
            assert "image_path" in analysis_result.schema.attributes
            assert "image_info" in analysis_result.schema.attributes
            assert len(analysis_result.schema.attributes) == 3  # Only image_path, image_info, and subset fields


class ConvertDatasetTest:
    """Tests for convert dataset functionality."""

    def test_convert_simple_bbox_dataset(self):
        """Test conversion of a simple dataset with bboxes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            image_path = str(temp_path / "image1.jpg")
            image_media = Image.from_file(image_path, size=(480, 640))

            bbox1 = Bbox(10, 20, 30, 40, label=1)
            bbox2 = Bbox(50, 60, 70, 80, label=2)

            item1 = DatasetItem(id="item1", media=image_media, annotations=[bbox1, bbox2])

            # Create dataset with label categories so bbox_labels will be present
            label_categories = LabelCategories()
            label_categories.add("background")
            label_categories.add("class_1")
            label_categories.add("class_2")

            dataset = LegacyDataset.from_iterable(
                [item1],
                ann_types={AnnotationType.bbox},
                categories={AnnotationType.label: label_categories},
            )

            # Convert dataset
            experimental_ds = convert_from_legacy(dataset)

            assert len(experimental_ds.df) == 1

            sample = experimental_ds[0]

            # Check attributes
            assert hasattr(sample, "image_path")
            assert Path(getattr(sample, "image_path")) == Path(image_path)
            assert hasattr(sample, "bboxes")
            assert hasattr(sample, "labels")

            expected_bboxes = np.array(
                [
                    [10, 20, 40, 60],  # First bbox in x1,y1,x2,y2 format
                    [50, 60, 120, 140],  # Second bbox in x1,y1,x2,y2 format
                ],
                dtype=np.float32,
            )
            expected_labels = np.array([1, 2], dtype=np.int32)

            np.testing.assert_array_equal(getattr(sample, "bboxes"), expected_bboxes)
            np.testing.assert_array_equal(getattr(sample, "labels"), expected_labels)

    def test_convert_mask_dataset(self):
        """Test conversion of dataset with semantic segmentation masks."""
        # Create a simple mask annotation
        mask_data = np.array([[0, 1], [1, 2]], dtype=np.uint8)
        mask1 = ExtractedMask(index_mask=mask_data, index=1, label=1)
        mask2 = ExtractedMask(index_mask=mask_data, index=2, label=2)

        # Create annotations list
        annotations = [mask1, mask2]

        # Create label categories
        categories = {AnnotationType.label: LabelCategories.from_iterable(["bg", "cat", "dog"])}

        # Create dataset item with the mask
        item = DatasetItem(id="item1", annotations=annotations)
        dataset = LegacyDataset.from_iterable([item], categories=categories)

        # Convert to v2 dataset
        experimental_ds = convert_from_legacy(dataset)

        # Check conversion results
        assert len(experimental_ds.df) == 1
        sample = experimental_ds[0]

        # Should have mask_callable and labels attributes
        assert hasattr(sample, "mask_callable")
        assert callable(getattr(sample, "mask_callable"))
        assert not hasattr(sample, "labels")

        # Verify mask data
        mask_callable = getattr(sample, "mask_callable")
        output_mask_data = mask_callable()
        np.testing.assert_array_equal(mask_data, output_mask_data)

    def test_convert_empty_dataset(self):
        """Test conversion of empty dataset."""
        dataset = LegacyDataset.from_iterable([])

        experimental_ds = convert_from_legacy(dataset)
        assert len(experimental_ds.df) == 0

    def test_convert_image_only_dataset(self):
        """Test conversion of dataset with only images, no annotations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            image_path = str(temp_path / "image1.jpg")
            image_media = Image.from_file(image_path, size=(480, 640))
            item = DatasetItem(id="item1", media=image_media, annotations=[])

            dataset = LegacyDataset.from_iterable([item])

            experimental_ds = convert_from_legacy(dataset)

            assert len(experimental_ds.df) == 1
            sample = experimental_ds[0]

            # Should have image attributes only
            assert hasattr(sample, "image_path")
            assert Path(getattr(sample, "image_path")) == Path(image_path)
            assert not hasattr(sample, "bboxes")
            assert not hasattr(sample, "bbox_labels")

    def test_builtin_converters_registration(self):
        """Test that built-in converters are registered on import."""
        # Create a dataset with file-based images
        items = [DatasetItem(id="test", media=Image.from_file("/path/to/image.jpg"))]
        dataset = LegacyDataset.from_iterable(items)

        image_converter = get_forward_media_converter(dataset)
        assert image_converter is not None
        assert isinstance(image_converter, ForwardImageMediaConverter)

        bbox_converter = get_forward_annotation_converter(AnnotationType.bbox, dataset)
        assert bbox_converter is not None
        assert isinstance(bbox_converter, ForwardBboxAnnotationConverter)


# Define a sample schema for testing convert_to_legacy
class DetectionSample(Sample):
    image_path: Annotated[str, image_path_field()]
    bboxes: Annotated[np.ndarray[Any, np.dtype[np.float32]], bbox_field(dtype=pl.Float32(), format="x1y1x2y2")]
    bbox_labels: Annotated[np.ndarray[Any, np.dtype[np.uint32]], label_field(dtype=pl.UInt32(), is_list=True)]


class RotatedDetectionSample(Sample):
    image_path: Annotated[str, image_path_field()]
    rotated_bboxes: Annotated[np.ndarray[Any, np.dtype[np.float32]], rotated_bbox_field(dtype=pl.Float32())]
    rotated_bbox_labels: Annotated[np.ndarray[Any, np.dtype[np.uint32]], label_field(dtype=pl.UInt32(), is_list=True)]


class ConvertToLegacyTest:
    """Tests for convert_to_legacy functionality."""

    def test_convert_to_legacy_simple(self):
        """Test basic convert_to_legacy functionality."""
        # Create v2 dataset with sample data
        experimental_dataset = Dataset(
            dtype_or_schema=DetectionSample,
            categories={"bbox_labels": exp_categories.LabelCategories(labels=("1", "2", "3"))},
        )

        # Add sample data
        sample1 = DetectionSample(
            image_path="/path/to/image1.jpg",
            bboxes=np.array([[10, 20, 50, 60], [100, 120, 150, 160]], dtype=np.float32),
            bbox_labels=np.array([0, 1], dtype=np.int32),
        )

        sample2 = DetectionSample(
            image_path="/path/to/image2.jpg",
            bboxes=np.array([[5, 15, 45, 55]], dtype=np.float32),
            bbox_labels=np.array([1], dtype=np.int32),
        )

        experimental_dataset.append(sample1)
        experimental_dataset.append(sample2)

        # Convert to legacy format
        legacy_dataset = convert_to_legacy(experimental_dataset)  # type: ignore

        # Verify conversion
        assert len(legacy_dataset) == 2

        # Check first item
        items = list(legacy_dataset)
        first_item = items[0]

        # Verify media
        assert isinstance(first_item.media, Image)  # type: ignore[reportUnknownMemberType]
        assert getattr(first_item.media, "path", None) == "/path/to/image1.jpg"

        # Verify annotations
        assert len(first_item.annotations) == 2

        # Check first bbox: [10, 20, 50, 60] -> Bbox(x=10, y=20, w=40, h=40)
        bbox1 = first_item.annotations[0]
        assert isinstance(bbox1, Bbox)
        assert bbox1.x == 10
        assert bbox1.y == 20
        assert bbox1.w == 40  # 50 - 10
        assert bbox1.h == 40  # 60 - 20
        assert bbox1.label == 0

        # Check second bbox: [100, 120, 150, 160] -> Bbox(x=100, y=120, w=50, h=40)
        bbox2 = first_item.annotations[1]
        assert isinstance(bbox2, Bbox)
        assert bbox2.x == 100
        assert bbox2.y == 120
        assert bbox2.w == 50  # 150 - 100
        assert bbox2.h == 40  # 160 - 120
        assert bbox2.label == 1

        # Check second item
        second_item = items[1]
        second_item_media = cast("Any", second_item.media)
        assert isinstance(second_item_media, ImageFromFile)
        assert second_item_media.path == "/path/to/image2.jpg"
        assert len(second_item.annotations) == 1

        # Check the bbox in second item: [5, 15, 45, 55] -> Bbox(x=5, y=15, w=40, h=40)
        bbox3 = second_item.annotations[0]
        assert isinstance(bbox3, Bbox)
        assert bbox3.x == 5
        assert bbox3.y == 15
        assert bbox3.w == 40  # 45 - 5
        assert bbox3.h == 40  # 55 - 15
        assert bbox3.label == 1

    def test_convert_to_legacy_empty_dataset(self):
        """Test convert_to_legacy with empty v2 dataset."""
        experimental_dataset = Dataset(DetectionSample)

        legacy_dataset = convert_to_legacy(experimental_dataset)  # type: ignore

        assert len(legacy_dataset) == 0


class AnalyzeExperimentalDatasetTest:
    """Tests for analyze_experimental_dataset functionality."""

    def test_analyze_experimental_dataset(self):
        """Test analysis of v2 dataset for backward conversion."""
        experimental_dataset = Dataset(
            dtype_or_schema=DetectionSample,
            categories={"bbox_labels": exp_categories.LabelCategories(labels=("1", "2", "3"))},
        )

        # Add sample data
        sample = DetectionSample(
            image_path="/test/image.jpg",
            bboxes=np.array([[10, 20, 50, 60]], dtype=np.float32),
            bbox_labels=np.array([1], dtype=np.int32),
        )
        experimental_dataset.append(sample)

        # Analyze dataset
        analysis_result = analyze_experimental_dataset(experimental_dataset)  # type: ignore

        # Check media type and converter
        assert analysis_result.media_type == Image
        assert analysis_result.media_converter is not None
        assert isinstance(analysis_result.media_converter, BackwardImageMediaConverter)

        # Check annotation types and converters
        assert AnnotationType.bbox in analysis_result.ann_types
        assert AnnotationType.bbox in analysis_result.ann_converters
        assert isinstance(analysis_result.ann_converters[AnnotationType.bbox], BackwardBboxAnnotationConverter)

        # Check categories
        assert AnnotationType.label in analysis_result.categories

    def test_analyze_experimental_dataset_no_compatible_converters(self):
        """Test analysis with dataset that has no compatible backward converters."""

        # Create a custom sample type without image_path or bbox fields
        class CustomSample(Sample):
            some_data: Annotated[np.ndarray[Any, np.dtype[np.float32]], tensor_field(dtype=pl.Float32())]

        experimental_dataset = Dataset(CustomSample)

        # Add sample
        sample = CustomSample(some_data=np.array([1, 2, 3], dtype=np.float32))
        experimental_dataset.append(sample)

        # Analyze dataset
        analysis_result = analyze_experimental_dataset(experimental_dataset)  # type: ignore

        # Should have no compatible converters
        assert analysis_result.media_type is None
        assert analysis_result.media_converter is None
        assert len(analysis_result.ann_types) == 0
        assert len(analysis_result.ann_converters) == 0
        assert len(analysis_result.categories) == 0

    def test_convert_to_legacy_with_only_images(self):
        """Test convert_to_legacy with dataset containing only images."""

        # Define schema with only image_path
        class ImageOnlySample(Sample):
            image_path: Annotated[str, image_path_field()]

        experimental_dataset = Dataset(ImageOnlySample)

        # Add samples
        sample1 = ImageOnlySample(image_path="/path/to/image1.jpg")
        sample2 = ImageOnlySample(image_path="/path/to/image2.jpg")

        experimental_dataset.append(sample1)
        experimental_dataset.append(sample2)

        # Convert to legacy
        legacy_dataset = convert_to_legacy(experimental_dataset)  # type: ignore

        assert len(legacy_dataset) == 2

        items = list(legacy_dataset)

        # Check first item
        first_item = items[0]
        first_item_media = cast("Any", first_item.media)
        assert isinstance(first_item_media, ImageFromFile)
        assert first_item_media.path == "/path/to/image1.jpg"
        assert len(first_item.annotations) == 0  # No annotations

        # Check second item
        second_item = items[1]
        second_item_media = cast("Any", second_item.media)
        assert isinstance(second_item_media, ImageFromFile)
        assert second_item_media.path == "/path/to/image2.jpg"
        assert len(second_item.annotations) == 0


class HelperFunctionsTest:
    """Tests for helper functions in legacy conversion."""

    def test_attributes_to_dict(self):
        """Test _attributes_to_dict helper function."""
        # Test with valid attribute format
        attributes = ["some_attr", "__color__red", "__type__human", "invalid_format"]
        result = _attributes_to_dict(attributes)

        expected = {"color": "red", "type": "human"}
        assert result == expected

        # Test with empty attributes
        result = _attributes_to_dict([])
        assert result == {}

    def test_has_derived_labels(self):
        """Test _has_derived_labels helper function."""
        # Test with hierarchical labels (derived labels exist)
        labels = ["animal", "animal__dog", "animal__cat", "vehicle"]
        assert _has_derived_labels(labels) is True

        # Test with more complex hierarchy
        labels = ["root", "root__child", "root__child__grandchild"]
        assert _has_derived_labels(labels) is True

        # Test with no hierarchical structure
        labels = ["cat", "dog", "bird"]
        assert _has_derived_labels(labels) is False

        # Test with empty list
        assert _has_derived_labels([]) is False

        # Test with single label
        assert _has_derived_labels(["single"]) is False

        # Test with labels that look similar but aren't hierarchical
        labels = ["test", "testing", "tester"]
        assert _has_derived_labels(labels) is False


class HierarchicalLegacyDatasetTest:
    """Tests for hierarchical legacy dataset analysis."""

    def test_analyze_legacy_dataset_hierarchical(self):
        """Test analyze_legacy_dataset with hierarchical labels."""
        from datumaro.components.annotation import Label
        from datumaro.components.annotation import LabelCategories as LegacyLabelCategories
        from datumaro.components.dataset import Dataset as LegacyDataset
        from datumaro.components.dataset_base import DatasetItem
        from datumaro.experimental.categories import HierarchicalLabelCategories

        # Create legacy label categories with hierarchical structure
        legacy_categories = LegacyLabelCategories()
        legacy_categories.add("animal")
        legacy_categories.add("animal__dog")
        legacy_categories.add("animal__cat")
        legacy_categories.add("vehicle")

        # Create item with label annotation
        item = DatasetItem(id="test", annotations=[Label(label=1)])
        legacy_dataset = LegacyDataset.from_iterable([item], categories={AnnotationType.label: legacy_categories})

        # Analyze the dataset
        analysis_result = analyze_legacy_dataset(legacy_dataset)

        # Should detect hierarchical structure
        assert analysis_result.is_hierarchical is True
        assert analysis_result.is_anomaly is False

        # Check that hierarchical categories were created
        label_attr = analysis_result.schema.attributes[AnnotationType.label.name]
        assert isinstance(label_attr.categories, HierarchicalLabelCategories)
        assert label_attr.field.is_list is True

    def test_analyze_legacy_dataset_non_hierarchical(self):
        """Test analyze_legacy_dataset with non-hierarchical labels."""
        from datumaro.components.annotation import LabelCategories as LegacyLabelCategories
        from datumaro.components.dataset import Dataset as LegacyDataset
        from datumaro.components.dataset_base import DatasetItem

        # Create legacy label categories without hierarchical structure
        legacy_categories = LegacyLabelCategories()
        legacy_categories.add("cat")
        legacy_categories.add("dog")
        legacy_categories.add("bird")

        # Create a simple item
        item = DatasetItem(id="test", annotations=[])
        legacy_dataset = LegacyDataset.from_iterable([item], categories={AnnotationType.label: legacy_categories})

        # Analyze the dataset
        analysis_result = analyze_legacy_dataset(legacy_dataset)

        # Should not detect hierarchical structure
        assert analysis_result.is_hierarchical is False


class ConvertFromLegacyPickleTest:
    """Tests for convert_from_legacy producing picklable datasets."""

    def test_converted_segmentation_dataset_is_picklable(self):
        """Test that a converted segmentation dataset can be pickled."""
        # Create a simple semantic segmentation dataset
        mask_data = np.array([[1, 0], [0, 1]], dtype=np.uint8)

        legacy_categories = LabelCategories()
        legacy_categories.add("background")
        legacy_categories.add("foreground")

        mask = ExtractedMask(index_mask=mask_data, index=1, label=1)
        item = DatasetItem(
            id="test",
            media=Image.from_numpy(np.zeros((2, 2, 3), dtype=np.uint8)),
            annotations=[mask],
        )

        legacy_dataset = LegacyDataset.from_iterable([item], categories={AnnotationType.label: legacy_categories})

        # Convert the dataset
        new_dataset = convert_from_legacy(legacy_dataset)

        # Pickle and unpickle the dataset
        pickled = pickle.dumps(new_dataset)
        restored_dataset = pickle.loads(pickled)

        # Verify the restored dataset works
        assert len(restored_dataset) == 1
