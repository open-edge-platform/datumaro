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
from datumaro.components.media import Image, ImageFromData, ImageFromFile, MediaElement, Video, VideoFrame
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import (
    bbox_field,
    image_path_field,
    label_field,
    media_path_field,
    rotated_bbox_field,
    tensor_field,
    video_frame_path_field,
)
from datumaro.experimental.legacy import (
    BackwardBboxAnnotationConverter,
    BackwardImageMediaConverter,
    BackwardMixedMediaConverter,
    ForwardBboxAnnotationConverter,
    ForwardImageMediaConverter,
    ForwardMixedMediaConverter,
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
from datumaro.experimental.media import LazyImage, LazyVideoFrame
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

    def test_convert_hierarchical_dataset_getitem(self):
        """Test that __getitem__ works on a converted hierarchical dataset.

        Regression test: multi_label label fields were stored with type=int,
        causing int(Series) to fail when reading back from Polars.
        """
        from datumaro.components.annotation import Label
        from datumaro.components.annotation import LabelCategories as LegacyLabelCategories

        legacy_categories = LegacyLabelCategories()
        legacy_categories.add("animal")
        legacy_categories.add("animal__dog")
        legacy_categories.add("animal__cat")

        item = DatasetItem(id="test", annotations=[Label(label=1)])
        legacy_dataset = LegacyDataset.from_iterable([item], categories={AnnotationType.label: legacy_categories})

        dataset = convert_from_legacy(legacy_dataset)

        sample = dataset[0]
        assert isinstance(sample.label, np.ndarray)
        assert 1 in sample.label

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
        assert label_attr.field.multi_label is True

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


class GetForwardMediaConverterStrategyTest:
    """Tests for the get_forward_media_converter selection strategy.

    Strategy:
    - Image-only datasets → ForwardImageMediaConverter (image_path_field)
    - Video-only datasets → ForwardMixedMediaConverter (media_path_field)
    - Mixed image+video datasets → ForwardMixedMediaConverter (media_path_field)
    - Empty datasets → None
    - Whole Video media (not VideoFrame) → None
    """

    def test_image_only_dataset_selects_image_converter(self):
        """Image-only datasets should use ForwardImageMediaConverter."""
        items = [
            DatasetItem(id="img1", media=Image.from_file("/path/to/image1.jpg")),
            DatasetItem(id="img2", media=Image.from_file("/path/to/image2.jpg")),
        ]
        dataset = LegacyDataset.from_iterable(items)

        converter = get_forward_media_converter(dataset)
        assert converter is not None
        assert isinstance(converter, ForwardImageMediaConverter)

    def test_video_only_dataset_selects_mixed_converter(self):
        """Video-only (VideoFrame) datasets should use ForwardMixedMediaConverter."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
            DatasetItem(id="frame_1", media=VideoFrame(video=video, index=1)),
            DatasetItem(id="frame_5", media=VideoFrame(video=video, index=5)),
        ]
        dataset = LegacyDataset.from_iterable(items, media_type=VideoFrame)

        converter = get_forward_media_converter(dataset)
        assert converter is not None
        assert isinstance(converter, ForwardMixedMediaConverter)

    def test_mixed_image_and_video_dataset_selects_mixed_converter(self):
        """Mixed (Image + VideoFrame) datasets should use ForwardMixedMediaConverter."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="img1", media=Image.from_file("/path/to/image1.jpg")),
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
            DatasetItem(id="img2", media=Image.from_file("/path/to/image2.jpg")),
            DatasetItem(id="frame_5", media=VideoFrame(video=video, index=5)),
        ]
        dataset = LegacyDataset.from_iterable(items)

        converter = get_forward_media_converter(dataset)
        assert converter is not None
        assert isinstance(converter, ForwardMixedMediaConverter)

    def test_empty_dataset_returns_none(self):
        """Empty datasets should return None."""
        dataset = LegacyDataset.from_iterable([])

        converter = get_forward_media_converter(dataset)
        assert converter is None

    def test_whole_video_media_returns_none(self):
        """Datasets with whole Video media (not VideoFrame) should return None."""
        items = [DatasetItem(id="vid", media=Video("/path/to/video.mp4"))]
        dataset = LegacyDataset.from_iterable(items, media_type=Video)

        converter = get_forward_media_converter(dataset)
        assert converter is None

    def test_video_only_converter_produces_media_path_field(self):
        """Verify that the converter for video datasets produces media_path_field schema."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
        ]
        dataset = LegacyDataset.from_iterable(items, media_type=VideoFrame)

        converter = get_forward_media_converter(dataset)
        assert converter is not None

        attributes = converter.get_schema_attributes()
        assert "media" in attributes
        from datumaro.experimental.fields import MediaPathField

        assert isinstance(attributes["media"].field, MediaPathField)

    def test_mixed_converter_produces_media_path_field(self):
        """Verify that the converter for mixed datasets produces media_path_field schema."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="img1", media=Image.from_file("/path/to/image.jpg")),
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
        ]
        dataset = LegacyDataset.from_iterable(items)

        converter = get_forward_media_converter(dataset)
        assert converter is not None

        attributes = converter.get_schema_attributes()
        assert "media" in attributes
        from datumaro.experimental.fields import MediaPathField

        assert isinstance(attributes["media"].field, MediaPathField)


class AnalyzeLegacyVideoDatasetTest:
    """Tests for analyze_legacy_dataset with video and mixed media."""

    def test_analyze_video_only_dataset(self):
        """Test analysis of dataset with only VideoFrame media."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
            DatasetItem(id="frame_1", media=VideoFrame(video=video, index=1)),
        ]
        dataset = LegacyDataset.from_iterable(items, media_type=VideoFrame)

        analysis_result = analyze_legacy_dataset(dataset)

        # Should use media_path_field (via ForwardMixedMediaConverter)
        assert "media" in analysis_result.schema.attributes
        assert analysis_result.media_converter is not None
        assert isinstance(analysis_result.media_converter, ForwardMixedMediaConverter)

    def test_analyze_mixed_image_and_video_dataset(self):
        """Test analysis of dataset with both Image and VideoFrame media."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="img1", media=Image.from_file("/path/to/image.jpg")),
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
        ]
        dataset = LegacyDataset.from_iterable(items)

        analysis_result = analyze_legacy_dataset(dataset)

        assert "media" in analysis_result.schema.attributes
        # Should NOT have image_path (not image-only)
        assert "image_path" not in analysis_result.schema.attributes
        assert isinstance(analysis_result.media_converter, ForwardMixedMediaConverter)

    def test_analyze_video_dataset_with_bboxes(self):
        """Test analysis of video dataset with bbox annotations."""
        label_categories = LabelCategories()
        label_categories.add("car")
        label_categories.add("person")

        video = Video(path="/path/to/video.mp4")
        bbox = Bbox(10, 20, 30, 40, label=0)
        items = [
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0), annotations=[bbox]),
        ]
        dataset = LegacyDataset.from_iterable(
            items,
            media_type=VideoFrame,
            ann_types={AnnotationType.bbox},
            categories={AnnotationType.label: label_categories},
        )

        analysis_result = analyze_legacy_dataset(dataset)

        # Should have media and bbox attributes
        assert "media" in analysis_result.schema.attributes
        assert "bboxes" in analysis_result.schema.attributes
        assert "labels" in analysis_result.schema.attributes


class ConvertFromLegacyVideoTest:
    """Tests for convert_from_legacy with video and mixed media datasets."""

    def test_convert_video_only_dataset(self):
        """Test conversion of dataset with only VideoFrame items."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
            DatasetItem(id="frame_10", media=VideoFrame(video=video, index=10)),
            DatasetItem(id="frame_20", media=VideoFrame(video=video, index=20)),
        ]
        dataset = LegacyDataset.from_iterable(items, media_type=VideoFrame)

        experimental_ds = convert_from_legacy(dataset)

        assert len(experimental_ds.df) == 3

        # Check first sample
        sample0 = experimental_ds[0]
        assert hasattr(sample0, "media")
        media0 = getattr(sample0, "media")
        assert isinstance(media0, LazyVideoFrame)
        assert str(media0.video_path) == "/path/to/video.mp4"
        assert media0.frame_index == 0

        # Check second sample
        sample1 = experimental_ds[1]
        media1 = getattr(sample1, "media")
        assert isinstance(media1, LazyVideoFrame)
        assert media1.frame_index == 10

        # Check third sample
        sample2 = experimental_ds[2]
        media2 = getattr(sample2, "media")
        assert isinstance(media2, LazyVideoFrame)
        assert media2.frame_index == 20

    def test_convert_mixed_image_and_video_dataset(self):
        """Test conversion of dataset with both Image and VideoFrame items."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="img1", media=Image.from_file("/path/to/image1.jpg")),
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
            DatasetItem(id="img2", media=Image.from_file("/path/to/image2.jpg")),
            DatasetItem(id="frame_5", media=VideoFrame(video=video, index=5)),
        ]
        dataset = LegacyDataset.from_iterable(items)

        experimental_ds = convert_from_legacy(dataset)

        assert len(experimental_ds.df) == 4

        # Check image sample
        sample0 = experimental_ds[0]
        assert hasattr(sample0, "media")
        media0 = getattr(sample0, "media")
        assert isinstance(media0, LazyImage)
        assert str(media0.path) == "/path/to/image1.jpg"

        # Check video frame sample
        sample1 = experimental_ds[1]
        media1 = getattr(sample1, "media")
        assert isinstance(media1, LazyVideoFrame)
        assert str(media1.video_path) == "/path/to/video.mp4"
        assert media1.frame_index == 0

        # Check second image sample
        sample2 = experimental_ds[2]
        media2 = getattr(sample2, "media")
        assert isinstance(media2, LazyImage)
        assert str(media2.path) == "/path/to/image2.jpg"

        # Check second video frame sample
        sample3 = experimental_ds[3]
        media3 = getattr(sample3, "media")
        assert isinstance(media3, LazyVideoFrame)
        assert media3.frame_index == 5

    def test_convert_video_dataset_with_annotations(self):
        """Test conversion of video dataset that also has bbox annotations."""
        label_categories = LabelCategories()
        label_categories.add("car")
        label_categories.add("person")

        video = Video(path="/path/to/video.mp4")
        bbox1 = Bbox(10, 20, 30, 40, label=0)
        bbox2 = Bbox(50, 60, 70, 80, label=1)

        items = [
            DatasetItem(
                id="frame_0",
                media=VideoFrame(video=video, index=0),
                annotations=[bbox1],
            ),
            DatasetItem(
                id="frame_10",
                media=VideoFrame(video=video, index=10),
                annotations=[bbox2],
            ),
        ]
        dataset = LegacyDataset.from_iterable(
            items,
            media_type=VideoFrame,
            ann_types={AnnotationType.bbox},
            categories={AnnotationType.label: label_categories},
        )

        experimental_ds = convert_from_legacy(dataset)

        assert len(experimental_ds.df) == 2

        # Check media
        sample0 = experimental_ds[0]
        assert hasattr(sample0, "media")
        media0 = getattr(sample0, "media")
        assert isinstance(media0, LazyVideoFrame)
        assert media0.frame_index == 0

        # Check annotations
        assert hasattr(sample0, "bboxes")
        assert hasattr(sample0, "labels")

    def test_convert_multiple_videos(self):
        """Test conversion with frames from multiple different videos."""
        video1 = Video(path="/path/to/video1.mp4")
        video2 = Video(path="/path/to/video2.mp4")
        items = [
            DatasetItem(id="v1_frame_0", media=VideoFrame(video=video1, index=0)),
            DatasetItem(id="v2_frame_0", media=VideoFrame(video=video2, index=0)),
            DatasetItem(id="v1_frame_5", media=VideoFrame(video=video1, index=5)),
            DatasetItem(id="v2_frame_10", media=VideoFrame(video=video2, index=10)),
        ]
        dataset = LegacyDataset.from_iterable(items, media_type=VideoFrame)

        experimental_ds = convert_from_legacy(dataset)

        assert len(experimental_ds.df) == 4

        # Check that different video paths are preserved
        media0 = getattr(experimental_ds[0], "media")
        media1 = getattr(experimental_ds[1], "media")
        assert isinstance(media0, LazyVideoFrame)
        assert isinstance(media1, LazyVideoFrame)
        assert str(media0.video_path) == "/path/to/video1.mp4"
        assert str(media1.video_path) == "/path/to/video2.mp4"

    def test_convert_single_video_frame_dataset(self):
        """Test conversion with just one video frame."""
        video = Video(path="/path/to/video.mp4")
        items = [DatasetItem(id="frame_42", media=VideoFrame(video=video, index=42))]
        dataset = LegacyDataset.from_iterable(items, media_type=VideoFrame)

        experimental_ds = convert_from_legacy(dataset)

        assert len(experimental_ds.df) == 1
        sample = experimental_ds[0]
        media = getattr(sample, "media")
        assert isinstance(media, LazyVideoFrame)
        assert media.frame_index == 42


class ConvertToLegacyVideoTest:
    """Tests for convert_to_legacy with mixed/video media datasets."""

    def test_convert_to_legacy_mixed_media_dataset(self):
        """Test converting a dataset with MediaPathField back to legacy format."""

        class MixedMediaSample(Sample):
            media: Annotated[LazyImage | LazyVideoFrame, media_path_field()]

        experimental_dataset = Dataset(MixedMediaSample)

        # Add an image sample
        experimental_dataset.append(MixedMediaSample(media=LazyImage(path="/path/to/image.jpg")))
        # Add a video frame sample
        experimental_dataset.append(
            MixedMediaSample(media=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=10))
        )

        legacy_dataset = convert_to_legacy(experimental_dataset)  # type: ignore

        assert len(legacy_dataset) == 2

        items = list(legacy_dataset)

        # First item should be an Image
        first_media = items[0].media
        assert isinstance(first_media, Image)
        assert getattr(first_media, "path", None) == "/path/to/image.jpg"

        # Second item should be a VideoFrame
        second_media = items[1].media
        assert isinstance(second_media, VideoFrame)
        assert second_media.video.path == "/path/to/video.mp4"
        assert second_media.index == 10

    def test_convert_to_legacy_video_only_media_dataset(self):
        """Test converting a video-only dataset with MediaPathField back to legacy."""

        class VideoMediaSample(Sample):
            media: Annotated[LazyImage | LazyVideoFrame, media_path_field()]

        experimental_dataset = Dataset(VideoMediaSample)
        experimental_dataset.append(
            VideoMediaSample(media=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=0))
        )
        experimental_dataset.append(
            VideoMediaSample(media=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=5))
        )

        legacy_dataset = convert_to_legacy(experimental_dataset)  # type: ignore

        assert len(legacy_dataset) == 2

        items = list(legacy_dataset)
        for item in items:
            assert isinstance(item.media, VideoFrame)

        assert items[0].media.index == 0
        assert items[1].media.index == 5
        # Should share same Video object (caching)
        assert items[0].media.video.path == items[1].media.video.path

    def test_convert_to_legacy_video_frame_path_field(self):
        """Test converting a dataset with VideoFramePathField back to legacy."""

        class VideoFrameSample(Sample):
            video_frame: Annotated[LazyVideoFrame, video_frame_path_field()]

        experimental_dataset = Dataset(VideoFrameSample)
        experimental_dataset.append(
            VideoFrameSample(
                video_frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=42),
            )
        )

        legacy_dataset = convert_to_legacy(experimental_dataset)  # type: ignore

        assert len(legacy_dataset) == 1
        item = next(iter(legacy_dataset))
        assert isinstance(item.media, VideoFrame)
        assert item.media.video.path == "/path/to/video.mp4"
        assert item.media.index == 42


class AnalyzeExperimentalVideoDatasetTest:
    """Tests for analyze_experimental_dataset with video/mixed media schemas."""

    def test_analyze_mixed_media_dataset(self):
        """Test analysis of dataset with MediaPathField schema."""

        class MixedMediaSample(Sample):
            media: Annotated[LazyImage | LazyVideoFrame, media_path_field()]

        experimental_dataset = Dataset(MixedMediaSample)
        experimental_dataset.append(MixedMediaSample(media=LazyImage(path="/path/to/image.jpg")))

        analysis_result = analyze_experimental_dataset(experimental_dataset)  # type: ignore

        assert analysis_result.media_type == MediaElement
        assert analysis_result.media_converter is not None
        assert isinstance(analysis_result.media_converter, BackwardMixedMediaConverter)

    def test_analyze_video_frame_path_dataset(self):
        """Test analysis of dataset with VideoFramePathField schema."""

        class VideoFrameSample(Sample):
            video_frame: Annotated[LazyVideoFrame, video_frame_path_field()]

        experimental_dataset = Dataset(VideoFrameSample)
        experimental_dataset.append(
            VideoFrameSample(
                video_frame=LazyVideoFrame(video_path="/path/to/video.mp4", frame_index=0),
            )
        )

        from datumaro.experimental.legacy import BackwardVideoMediaConverter

        analysis_result = analyze_experimental_dataset(experimental_dataset)  # type: ignore

        assert analysis_result.media_type == VideoFrame
        assert analysis_result.media_converter is not None
        assert isinstance(analysis_result.media_converter, BackwardVideoMediaConverter)


class ConvertVideoRoundTripTest:
    """Round-trip tests: legacy → experimental → legacy for video datasets."""

    def test_video_only_round_trip(self):
        """Test round-trip conversion for a video-only dataset."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
            DatasetItem(id="frame_10", media=VideoFrame(video=video, index=10)),
        ]
        legacy_ds = LegacyDataset.from_iterable(items, media_type=VideoFrame)

        # Forward: legacy → experimental
        experimental_ds = convert_from_legacy(legacy_ds)
        assert len(experimental_ds.df) == 2

        # Backward: experimental → legacy
        restored_ds = convert_to_legacy(experimental_ds)
        assert len(restored_ds) == 2

        restored_items = list(restored_ds)
        for item in restored_items:
            assert isinstance(item.media, (VideoFrame, Image))

    def test_mixed_media_round_trip(self):
        """Test round-trip conversion for a mixed image+video dataset."""
        video = Video(path="/path/to/video.mp4")
        items = [
            DatasetItem(id="img1", media=Image.from_file("/path/to/image.jpg")),
            DatasetItem(id="frame_0", media=VideoFrame(video=video, index=0)),
        ]
        legacy_ds = LegacyDataset.from_iterable(items)

        # Forward: legacy → experimental
        experimental_ds = convert_from_legacy(legacy_ds)
        assert len(experimental_ds.df) == 2

        # Check that experimental dataset has both types
        sample0 = experimental_ds[0]
        sample1 = experimental_ds[1]
        assert isinstance(getattr(sample0, "media"), LazyImage)
        assert isinstance(getattr(sample1, "media"), LazyVideoFrame)

        # Backward: experimental → legacy
        restored_ds = convert_to_legacy(experimental_ds)
        assert len(restored_ds) == 2

        restored_items = list(restored_ds)
        # First should be Image, second should be VideoFrame
        assert isinstance(restored_items[0].media, Image)
        assert isinstance(restored_items[1].media, VideoFrame)
        assert getattr(restored_items[0].media, "path", None) == "/path/to/image.jpg"
        assert restored_items[1].media.video.path == "/path/to/video.mp4"
        assert restored_items[1].media.index == 0
