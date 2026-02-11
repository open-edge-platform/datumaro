"""Unit tests for legacy dataset conversion functionality."""

import json
import math
import pickle
import tempfile
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl

import datumaro.experimental.categories as exp_categories
from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Ellipse,
    ExtractedMask,
    LabelCategories,
    Points,
    Polygon,
    RotatedBbox,
)
from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import CategoriesInfo, DatasetItem
from datumaro.components.media import Image
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import bbox_field, image_path_field, label_field, polygon_field, rotated_bbox_field
from datumaro.experimental.legacy import (
    BackwardBboxAnnotationConverter,
    BackwardPolygonAnnotationConverter,
    BackwardRotatedBboxAnnotationConverter,
    ForwardBboxAnnotationConverter,
    ForwardEllipseAnnotationConverter,
    ForwardKeypointAnnotationConverter,
    ForwardMaskAnnotationConverter,
    ForwardPolygonAnnotationConverter,
    ForwardRotatedBboxAnnotationConverter,
    convert_from_legacy,
    convert_to_legacy,
)
from datumaro.experimental.schema import AttributeInfo, Schema


# Define a sample schema for testing convert_to_legacy
class DetectionSample(Sample):
    image_path: Annotated[str, image_path_field()]
    bboxes: Annotated[np.ndarray[Any, np.dtype[np.float32]], bbox_field(dtype=pl.Float32(), format="x1y1x2y2")]
    bbox_labels: Annotated[np.ndarray[Any, np.dtype[np.uint32]], label_field(dtype=pl.UInt32(), is_list=True)]


class RotatedDetectionSample(Sample):
    image_path: Annotated[str, image_path_field()]
    rotated_bboxes: Annotated[np.ndarray[Any, np.dtype[np.float32]], rotated_bbox_field(dtype=pl.Float32())]
    rotated_bbox_labels: Annotated[np.ndarray[Any, np.dtype[np.uint32]], label_field(dtype=pl.UInt32(), is_list=True)]


class ForwardBboxAnnotationConverterTest:
    """Tests for ForwardBboxAnnotationConverter."""

    def test_bbox_annotation_converter_get_schema_attributes(self):
        """Test schema attribute generation for bboxes."""
        # Create a dataset with empty categories
        dataset = LegacyDataset.from_iterable([], categories={})
        converter = ForwardBboxAnnotationConverter.create(dataset)
        assert converter is not None

        attributes = converter.get_schema_attributes()

        assert "bboxes" in attributes
        # bbox_labels should not be present when there are no label categories
        assert "bbox_labels" not in attributes
        assert attributes["bboxes"].type == np.ndarray

    def test_bbox_annotation_converter_convert_annotations_single_bbox(self):
        """Test conversion of single bbox annotation."""

        # Create categories with labels to test with labels
        label_categories = LabelCategories()
        label_categories.add("class_1")
        categories: CategoriesInfo = {AnnotationType.label: label_categories}

        # Create a dataset with the categories
        dataset = LegacyDataset.from_iterable([], categories=categories)
        converter = ForwardBboxAnnotationConverter.create(dataset)
        assert converter is not None

        bbox = Bbox(10, 20, 30, 40, label=1)  # x=10, y=20, w=30, h=40
        item = DatasetItem(id="test")

        result = converter.convert_annotations([bbox], item)

        expected_bbox = np.array([[10, 20, 40, 60]], dtype=np.float32)  # x1,y1,x2,y2 format
        expected_labels = np.array([1], dtype=np.int32)

        np.testing.assert_array_equal(result["bboxes"], expected_bbox)
        np.testing.assert_array_equal(result["labels"], expected_labels)

    def test_bbox_annotation_converter_convert_annotations_multiple_bboxes(self):
        """Test conversion of multiple bbox annotations."""

        # Create categories with labels
        label_categories = LabelCategories()
        label_categories.add("class_1")
        label_categories.add("class_2")
        categories: CategoriesInfo = {AnnotationType.label: label_categories}

        # Create a dataset with the categories
        dataset = LegacyDataset.from_iterable([], categories=categories)
        converter = ForwardBboxAnnotationConverter.create(dataset)
        assert converter is not None

        bbox1 = Bbox(10, 20, 30, 40, label=1)
        bbox2 = Bbox(50, 60, 70, 80, label=2)
        item = DatasetItem(id="test")

        result = converter.convert_annotations([bbox1, bbox2], item)

        expected_bboxes = np.array(
            [
                [10, 20, 40, 60],  # x1,y1,x2,y2 for first bbox
                [50, 60, 120, 140],  # x1,y1,x2,y2 for second bbox
            ],
            dtype=np.float32,
        )
        expected_labels = np.array([1, 2], dtype=np.int32)

        np.testing.assert_array_equal(result["bboxes"], expected_bboxes)
        np.testing.assert_array_equal(result["labels"], expected_labels)

    def test_bbox_annotation_converter_convert_annotations_empty_list(self):
        """Test conversion of empty annotation list."""
        # Create a dataset with empty categories
        dataset = LegacyDataset.from_iterable([], categories={})
        converter = ForwardBboxAnnotationConverter.create(dataset)
        assert converter is not None
        item = DatasetItem(id="test")

        result = converter.convert_annotations([], item)

        # Empty arrays with proper shapes
        assert result["bboxes"].shape == (0, 4)
        assert result["bboxes"].dtype == np.float32
        # No bbox_labels should be present when there are no categories
        assert "bbox_labels" not in result


class ForwardEllipseAnnotationConverterTest:
    """Tests for ForwardEllipseAnnotationConverter."""

    def test_ellipse_annotation_converter_get_schema_attributes(self):
        """Test schema attribute generation for ellipses."""
        # Create a dataset with empty categories
        dataset = LegacyDataset.from_iterable([], categories={})
        converter = ForwardEllipseAnnotationConverter.create(dataset)
        assert converter is not None

        attributes = converter.get_schema_attributes()

        assert "ellipses" in attributes
        # ellipses_labels should not be present when there are no label categories
        assert "ellipses_labels" not in attributes
        assert attributes["ellipses"].type == np.ndarray

    def test_ellipses_annotation_converter_convert_annotations_single_ellipse(self):
        """Test conversion of single ellipse annotation."""

        # Create categories with labels to test with labels
        label_categories = LabelCategories()
        label_categories.add("class_1")
        categories: CategoriesInfo = {AnnotationType.label: label_categories}

        # Create a dataset with the categories
        dataset = LegacyDataset.from_iterable([], categories=categories)
        converter = ForwardEllipseAnnotationConverter.create(dataset)
        assert converter is not None

        ellipse = Ellipse(10, 20, 30, 40, label=1)  # x=10, y=20, w=30, h=40
        item = DatasetItem(id="test")

        result = converter.convert_annotations([ellipse], item)

        expected_ellipse = np.array([[10, 20, 30, 40]], dtype=np.float32)  # x1,y1,x2,y2 format
        expected_labels = np.array([1], dtype=np.int32)

        np.testing.assert_array_equal(result["ellipses"], expected_ellipse)
        np.testing.assert_array_equal(result["labels"], expected_labels)

    def test_ellipse_annotation_converter_convert_annotations_multiple_ellipses(self):
        """Test conversion of multiple ellipse annotations."""

        # Create categories with labels
        label_categories = LabelCategories()
        label_categories.add("class_1")
        label_categories.add("class_2")
        categories: CategoriesInfo = {AnnotationType.label: label_categories}

        # Create a dataset with the categories
        dataset = LegacyDataset.from_iterable([], categories=categories)
        converter = ForwardEllipseAnnotationConverter.create(dataset)
        assert converter is not None

        ellipse1 = Ellipse(10, 20, 30, 40, label=1)
        ellipse2 = Ellipse(50, 60, 70, 80, label=2)
        item = DatasetItem(id="test")

        result = converter.convert_annotations([ellipse1, ellipse2], item)

        expected_ellipses = np.array(
            [
                [10, 20, 30, 40],  # x1,y1,x2,y2 for first ellipse
                [50, 60, 70, 80],  # x1,y1,x2,y2 for second ellipse
            ],
            dtype=np.float32,
        )
        expected_labels = np.array([1, 2], dtype=np.int32)

        np.testing.assert_array_equal(result["ellipses"], expected_ellipses)
        np.testing.assert_array_equal(result["labels"], expected_labels)

    def test_ellipse_annotation_converter_convert_annotations_empty_list(self):
        """Test conversion of empty annotation list."""
        # Create a dataset with empty categories
        dataset = LegacyDataset.from_iterable([], categories={})
        converter = ForwardEllipseAnnotationConverter.create(dataset)
        assert converter is not None
        item = DatasetItem(id="test")

        result = converter.convert_annotations([], item)

        # Empty arrays with proper shapes
        assert result["ellipses"].shape == (0, 4)
        assert result["ellipses"].dtype == np.float32
        # No labels should be present when there are no categories
        assert "labels" not in result


class BackwardBboxAnnotationConverterTest:
    """Tests for BackwardBboxAnnotationConverter."""

    def test_backward_bbox_annotation_converter_create_from_schema(self):
        """Test BackwardBboxAnnotationConverter.create_from_schema method."""
        # Create v2 dataset to get schema
        experimental_dataset = Dataset(DetectionSample)
        schema = experimental_dataset.schema

        # Test that converter can be created from schema with bbox fields
        converter = BackwardBboxAnnotationConverter.create_from_schema(schema)
        assert converter is not None
        assert isinstance(converter, BackwardBboxAnnotationConverter)
        assert converter.bboxes_attr == "bboxes"
        assert converter.bbox_labels_attr == "bbox_labels"

        # Test get_annotation_type
        assert converter.get_annotation_type() == AnnotationType.bbox

    def test_backward_bbox_annotation_converter_create_from_schema_missing_fields(self):
        """Test BackwardBboxAnnotationConverter with incomplete schema."""

        # Create schema without bbox fields
        schema = Schema(attributes={"image_path": AttributeInfo(type=str, field=image_path_field())})

        converter = BackwardBboxAnnotationConverter.create_from_schema(schema)
        assert converter is None


class ForwardMaskAnnotationConverterTest:
    """Tests for ForwardMaskAnnotationConverter."""

    def test_forward_instance_mask_annotation_converter(self):
        """Test instance mask annotation forward conversion."""
        # Create a sample mask and index
        index_mask_data = np.array([[0, 1], [1, 2]], dtype=np.uint8)
        mask1 = ExtractedMask(index_mask=index_mask_data, index=1, label=0)
        mask2 = ExtractedMask(index_mask=index_mask_data, index=2, label=1)

        # Create annotations list
        annotations = [mask1, mask2]

        # Create test item
        item = DatasetItem(id="test", annotations=annotations)

        # Create label categories
        label_categories = LabelCategories()
        label_categories.add("class_0")
        label_categories.add("class_1")
        categories = {AnnotationType.label: label_categories}

        # Create a dataset with the categories
        dataset = LegacyDataset.from_iterable([item], categories=categories)
        converter = ForwardMaskAnnotationConverter.create(dataset)
        assert converter is not None

        # Get schema attributes
        attributes = converter.get_schema_attributes()
        assert "instance_mask_callable" in attributes
        assert "labels" in attributes

        # Convert annotations
        result = converter.convert_annotations(annotations, item)

        # Check result
        assert "instance_mask_callable" in result
        assert "labels" in result

        # Test instance mask callables
        assert callable(result["instance_mask_callable"])  # One callable for shared index mask
        instance_masks = result["instance_mask_callable"]()

        # Verify shape and content of instance masks
        assert instance_masks.shape == (2, 2, 2)  # (N=2 instances, H=2, W=2)
        assert np.array_equal(instance_masks[0], index_mask_data == 1)  # First instance
        assert np.array_equal(instance_masks[1], index_mask_data == 2)  # Second instance

        # Verify labels
        assert np.array_equal(result["labels"], np.array([0, 1], dtype=np.int32))

    def test_forward_mask_annotation_converter_empty(self):
        """Test instance mask annotation forward conversion with no masks."""
        # Create empty annotations list
        annotations: list = []

        # Create test item
        item = DatasetItem(id="test", annotations=annotations)

        # Create label categories
        label_categories = LabelCategories()
        label_categories.add("class_0")
        categories = {AnnotationType.label: label_categories}

        # Create a dataset with the categories
        dataset = LegacyDataset.from_iterable([], categories=categories)
        converter = ForwardMaskAnnotationConverter.create(dataset)
        assert converter is not None

        # Convert annotations
        result = converter.convert_annotations(annotations, item)

        # Check result
        assert "instance_mask_callable" not in result
        assert "mask_callable" in result
        assert "labels" not in result

        mask = result["mask_callable"]()

        assert mask is None  # No callables for empty masks


class BackwardBboxAnnotationConverterAdvancedTest:
    """Advanced tests for BackwardBboxAnnotationConverter."""

    def test_backward_bbox_annotation_converter_convert_to_legacy_annotations(self):
        """Test annotation conversion from v2 to legacy."""
        experimental_dataset = Dataset(DetectionSample)
        schema = experimental_dataset.schema

        converter = BackwardBboxAnnotationConverter.create_from_schema(schema)
        assert converter is not None

        # Create sample with bboxes
        sample = DetectionSample(
            image_path="/test/image.jpg",
            bboxes=np.array([[10, 20, 50, 60], [100, 150, 200, 250]], dtype=np.float32),
            bbox_labels=np.array([1, 2], dtype=np.int32),
        )

        # Convert to legacy annotations
        categories: CategoriesInfo = {}  # Empty categories for this test
        legacy_annotations = converter.convert_to_legacy_annotations(sample, categories)

        assert len(legacy_annotations) == 2

        # Check first bbox: [10, 20, 50, 60] -> Bbox(x=10, y=20, w=40, h=40)
        bbox1 = legacy_annotations[0]
        assert isinstance(bbox1, Bbox)
        assert bbox1.x == 10
        assert bbox1.y == 20
        assert bbox1.w == 40  # 50 - 10
        assert bbox1.h == 40  # 60 - 20
        assert bbox1.label == 1

        # Check second bbox: [100, 150, 200, 250] -> Bbox(x=100, y=150, w=100, h=100)
        bbox2 = legacy_annotations[1]
        assert isinstance(bbox2, Bbox)
        assert bbox2.x == 100
        assert bbox2.y == 150
        assert bbox2.w == 100  # 200 - 100
        assert bbox2.h == 100  # 250 - 150
        assert bbox2.label == 2

    def test_backward_bbox_annotation_converter_convert_empty_annotations(self):
        """Test bbox converter with empty arrays."""
        experimental_dataset = Dataset(DetectionSample)
        schema = experimental_dataset.schema

        converter = BackwardBboxAnnotationConverter.create_from_schema(schema)
        assert converter is not None

        # Create sample with empty bboxes
        sample = DetectionSample(
            image_path="/test/image.jpg",
            bboxes=np.array([], dtype=np.float32).reshape(0, 4),
            bbox_labels=np.array([], dtype=np.int32),
        )

        # Convert to legacy annotations
        categories: CategoriesInfo = {}
        legacy_annotations = converter.convert_to_legacy_annotations(sample, categories)

        assert len(legacy_annotations) == 0

    def test_backward_bbox_annotation_converter_infer_categories(self):
        """Test category inference from v2 dataset."""
        experimental_dataset = Dataset(
            dtype_or_schema=DetectionSample,
            categories={"bbox_labels": exp_categories.LabelCategories(labels=("1", "2", "3"))},
        )

        # Add samples with different labels
        sample1 = DetectionSample(
            image_path="/test/image1.jpg",
            bboxes=np.array([[10, 20, 50, 60]], dtype=np.float32),
            bbox_labels=np.array([0], dtype=np.int32),
        )

        sample2 = DetectionSample(
            image_path="/test/image2.jpg",
            bboxes=np.array([[100, 150, 200, 250], [50, 75, 100, 125]], dtype=np.float32),
            bbox_labels=np.array([2, 1], dtype=np.int32),
        )

        experimental_dataset.append(sample1)
        experimental_dataset.append(sample2)

        schema = experimental_dataset.schema
        converter = BackwardBboxAnnotationConverter.create_from_schema(schema)
        assert converter is not None

        # Infer categories
        categories = converter.infer_categories(experimental_dataset)  # type: ignore

        # Should have label categories
        assert AnnotationType.label in categories
        # Basic check that categories were created - detailed verification is complex due to legacy types
        assert categories[AnnotationType.label] is not None


class ForwardPolygonAnnotationConverterTest:
    """Tests for ForwardPolygonAnnotationConverter."""

    def test_forward_polygon_annotation_converter_get_schema_attributes(self):
        """Test schema attribute generation for polygons."""
        # Create a dataset with empty categories
        dataset = LegacyDataset.from_iterable([], categories={})
        converter = ForwardPolygonAnnotationConverter.create(dataset)
        assert converter is not None
        attributes = converter.get_schema_attributes()

        assert "polygons" in attributes
        # polygon_labels should not be present when there are no label categories
        assert "polygon_labels" not in attributes
        assert attributes["polygons"].type == np.ndarray

    def test_forward_polygon_annotation_converter_convert_annotations(self):
        """Test polygon annotation conversion."""

        # Create categories with labels
        label_categories = LabelCategories()
        label_categories.add("class_1")
        label_categories.add("class_2")
        categories: CategoriesInfo = {AnnotationType.label: label_categories}

        # Create a dataset with the categories
        dataset = LegacyDataset.from_iterable([], categories=categories)
        converter = ForwardPolygonAnnotationConverter.create(dataset)
        assert converter is not None

        # Create polygon annotations with flat coordinates
        triangle = Polygon(points=[0, 0, 10, 0, 5, 10], label=1)  # Triangle as flat list
        rectangle = Polygon(points=[20, 20, 30, 20, 30, 30, 20, 30], label=2)  # Rectangle as flat list

        annotations = [triangle, rectangle]
        item = DatasetItem(id="test")

        result = converter.convert_annotations(annotations, item)

        assert "polygons" in result
        assert "labels" in result

        # Check polygon data
        polygons = result["polygons"]
        assert len(polygons) == 2

        # First polygon (triangle): [0,0,10,0,5,10] -> [0,0,10,0,5,10]
        assert np.all(polygons[0] == [[0, 0], [10, 0], [5, 10]])

        # Second polygon (rectangle): [20,20,30,20,30,30,20,30] -> [20,20,30,20,30,30,20,30]
        assert np.all(polygons[1] == [[20, 20], [30, 20], [30, 30], [20, 30]])

        # Check labels
        labels = result["labels"]
        assert len(labels) == 2
        assert labels[0] == 1
        assert labels[1] == 2


class BackwardPolygonAnnotationConverterTest:
    """Tests for BackwardPolygonAnnotationConverter."""

    def test_backward_polygon_annotation_converter_create_from_schema(self):
        """Test BackwardPolygonAnnotationConverter schema detection."""

        # Create schema with polygon fields
        schema = Schema(
            attributes={
                "polygons": AttributeInfo(type=list, field=polygon_field(dtype=pl.Float32())),
                "polygon_labels": AttributeInfo(
                    type=np.ndarray, field=label_field(dtype=pl.UInt32(), multi_label=True)
                ),
            }
        )

        converter = BackwardPolygonAnnotationConverter.create_from_schema(schema)
        assert converter is not None
        assert converter.polygons_attr == "polygons"
        assert converter.polygon_labels_attr == "polygon_labels"

    def test_backward_polygon_annotation_converter_convert_to_legacy(self):
        """Test conversion from v2 to legacy polygon annotations."""
        converter = BackwardPolygonAnnotationConverter("polygons", "polygon_labels")

        # Create sample with polygon data
        class TestSample(Sample):
            pass

        sample = TestSample()
        # Simulate polygon data: triangle and rectangle
        sample.polygons = np.array(
            [
                np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float32),
                np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float32),
            ],
            dtype=object,
        )
        sample.polygon_labels = np.array([1, 2], dtype=np.int32)

        categories: CategoriesInfo = {}
        result = converter.convert_to_legacy_annotations(sample, categories)

        assert len(result) == 2

        # Check first polygon (triangle)
        poly1 = result[0]
        assert isinstance(poly1, Polygon)
        assert poly1.points == [0.0, 0.0, 10.0, 0.0, 5.0, 10.0]  # Flat coordinate format
        assert poly1.label == 1

        # Check second polygon (rectangle)
        poly2 = result[1]
        assert isinstance(poly2, Polygon)
        assert poly2.points == [
            20.0,
            20.0,
            30.0,
            20.0,
            30.0,
            30.0,
            20.0,
            30.0,
        ]  # Flat coordinate format
        assert poly2.label == 2


class PolygonConversionTest:
    """Tests for polygon conversion between legacy and v2 formats."""

    def test_polygon_conversion_with_labels(self):
        """Test polygon conversion between legacy and v2 formats with label categories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an image file for the test
            image_path = str(temp_path / "image1.jpg")
            image_media = Image.from_file(image_path, size=(480, 640))

            # Create polygon annotations with different shapes
            triangle = Polygon(points=[10, 20, 30, 25, 20, 40], label=1)  # Triangle
            rectangle = Polygon(points=[50, 60, 80, 60, 80, 90, 50, 90], label=2)  # Rectangle

            item = DatasetItem(id="polygon_test", media=image_media, annotations=[triangle, rectangle])

            # Create label categories

            label_categories = LabelCategories()
            label_categories.add("background")
            label_categories.add("triangle_class")
            label_categories.add("rectangle_class")

            # Create legacy dataset with categories
            legacy_dataset = LegacyDataset.from_iterable(
                [item],
                ann_types={AnnotationType.polygon},
                categories={AnnotationType.label: label_categories},
            )

            # Convert to v2 format
            experimental_dataset = convert_from_legacy(legacy_dataset)

            # Verify v2 dataset structure
            assert len(experimental_dataset) == 1
            exp_sample = experimental_dataset[0]

            # Check that polygons and labels are present
            assert hasattr(exp_sample, "polygons")
            assert hasattr(exp_sample, "labels")

            # Check polygon data
            assert len(exp_sample.polygons) == 2
            assert np.all(exp_sample.polygons[0] == [[10, 20], [30, 25], [20, 40]])  # Triangle
            assert np.all(exp_sample.polygons[1] == [[50, 60], [80, 60], [80, 90], [50, 90]])  # Rectangle

            # Check labels
            np.testing.assert_array_equal(exp_sample.labels, [1, 2])

            # Convert back to legacy format
            restored_legacy_dataset = convert_to_legacy(experimental_dataset)

            # Verify restored dataset
            restored_items = list(restored_legacy_dataset)
            assert len(restored_items) == 1

            restored_item = restored_items[0]
            # Expect 4 annotations: 2 polygons + 2 bboxes (derived from polygon bounds)
            assert len(restored_item.annotations) == 4

            # Check restored polygons
            polygon_anns = [ann for ann in restored_item.annotations if isinstance(ann, Polygon)]
            assert len(polygon_anns) == 2

            # Sort by label for consistent comparison
            polygon_anns.sort(key=lambda x: x.label)

            # Verify triangle
            assert polygon_anns[0].points == [10, 20, 30, 25, 20, 40]
            assert polygon_anns[0].label == 1

            # Verify rectangle
            assert polygon_anns[1].points == [50, 60, 80, 60, 80, 90, 50, 90]
            assert polygon_anns[1].label == 2

    def test_polygon_conversion_without_labels(self):
        """Test polygon conversion when no label categories are present."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an image file for the test
            image_path = str(temp_path / "image1.jpg")
            image_media = Image.from_file(image_path, size=(480, 640))

            # Create polygon annotation without label categories
            triangle = Polygon(points=[10, 20, 30, 25, 20, 40])  # No label

            item = DatasetItem(id="polygon_no_labels", media=image_media, annotations=[triangle])

            # Create legacy dataset without label categories
            legacy_dataset = LegacyDataset.from_iterable([item], ann_types={AnnotationType.polygon})

            # Convert to v2 format
            experimental_dataset = convert_from_legacy(legacy_dataset)

            # Verify v2 dataset structure
            assert len(experimental_dataset) == 1
            exp_sample = experimental_dataset[0]

            # Check that polygons is present but polygon_labels is not
            assert hasattr(exp_sample, "polygons")
            assert not hasattr(exp_sample, "polygon_labels")

            # Check polygon data
            assert len(exp_sample.polygons) == 1
            assert np.all(exp_sample.polygons[0] == [[10, 20], [30, 25], [20, 40]])

        restored_legacy_dataset = convert_to_legacy(experimental_dataset)

        # Verify restored dataset
        restored_items = list(restored_legacy_dataset)
        assert len(restored_items) == 1

        restored_item = restored_items[0]
        assert len(restored_item.annotations) == 1

        # Check restored polygon
        polygon_anns = [ann for ann in restored_item.annotations if isinstance(ann, Polygon)]
        assert len(polygon_anns) == 1

        restored_polygon = polygon_anns[0]
        assert restored_polygon.points == [10, 20, 30, 25, 20, 40]
        assert restored_polygon.label is None

    def test_two_polygons_inside_single_bounding_box(self):
        """Test legacy COCO dataset with two polygons in the same annotation (multi-part segmentation)."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Build a COCO instances dataset on disk
            (temp_path / "annotations").mkdir()
            (temp_path / "images").mkdir()
            coco_json = {
                "images": [{"id": 1, "width": 640, "height": 480, "file_name": "img.jpg"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [
                            [10, 10, 20, 10, 15, 30],
                            [25, 10, 35, 10, 30, 30],
                        ],
                        "area": 200.0,
                        "bbox": [10, 10, 25, 20],
                        "iscrowd": 0,
                    },
                ],
                "categories": [{"id": 1, "name": "object"}],
            }
            (temp_path / "annotations" / "instances_train.json").write_text(json.dumps(coco_json))

            # Import as a legacy COCO dataset
            legacy_dataset = LegacyDataset.import_from(str(temp_path), "coco_instances")

            # Verify the legacy dataset loaded both polygons
            legacy_items = list(legacy_dataset)
            assert len(legacy_items) == 1
            polygon_anns = [a for a in legacy_items[0].annotations if isinstance(a, Polygon)]
            assert len(polygon_anns) == 2

            # Convert to v2 format
            experimental_dataset = convert_from_legacy(legacy_dataset)

            assert len(experimental_dataset) == 1
            exp_sample = experimental_dataset[0]

            # Both polygons should be preserved as separate entries
            assert hasattr(exp_sample, "polygons")
            assert len(exp_sample.polygons) == 2

            # Verify polygon coordinates are preserved
            poly_points = [exp_sample.polygons[i].tolist() for i in range(2)]
            assert [[10, 10], [20, 10], [15, 30]] in poly_points
            assert [[25, 10], [35, 10], [30, 30]] in poly_points


class ForwardRotatedBboxAnnotationConverterTest:
    """Tests for ForwardRotatedBboxAnnotationConverter."""

    def test_forward_rotated_bbox_annotation_converter_get_schema_attributes(self):
        """Test schema attribute generation for rotated bboxes."""
        # Create a dataset with empty categories
        dataset = LegacyDataset.from_iterable([], categories={})
        converter = ForwardRotatedBboxAnnotationConverter.create(dataset)
        assert converter is not None
        attributes = converter.get_schema_attributes()

        assert "rotated_bboxes" in attributes
        # rotated_bbox_labels should not be present when there are no label categories
        assert "rotated_bbox_labels" not in attributes
        assert attributes["rotated_bboxes"].type == np.ndarray

    def test_forward_rotated_bbox_annotation_converter_convert_annotations(self):
        """Test rotated bbox annotation conversion."""

        # Create categories with labels
        label_categories = LabelCategories()
        label_categories.add("class_1")
        label_categories.add("class_2")
        categories: CategoriesInfo = {AnnotationType.label: label_categories}

        # Create a dataset with the categories
        dataset = LegacyDataset.from_iterable([], categories=categories)
        converter = ForwardRotatedBboxAnnotationConverter.create(dataset)
        assert converter is not None

        # Create rotated bbox annotations: RotatedBbox(cx, cy, w, h, r_degrees, ...)
        rotated_bbox1 = RotatedBbox(50, 60, 30, 20, 45.0, label=0)  # 45 degrees
        rotated_bbox2 = RotatedBbox(100, 120, 40, 25, -30.0, label=1)  # -30 degrees

        annotations = [rotated_bbox1, rotated_bbox2]
        item = DatasetItem(id="test")

        result = converter.convert_annotations(annotations, item)

        assert "rotated_bboxes" in result
        assert "labels" in result

        # Check rotated bbox data (should be converted to radians)
        rotated_bboxes = result["rotated_bboxes"]
        assert len(rotated_bboxes) == 2

        # First rotated bbox: (50, 60, 30, 20, 45°) -> (50, 60, 30, 20, π/4 radians)
        assert abs(rotated_bboxes[0, 0] - 50.0) < 1e-6
        assert abs(rotated_bboxes[0, 1] - 60.0) < 1e-6
        assert abs(rotated_bboxes[0, 2] - 30.0) < 1e-6
        assert abs(rotated_bboxes[0, 3] - 20.0) < 1e-6
        assert abs(rotated_bboxes[0, 4] - math.radians(45.0)) < 1e-6

        # Second rotated bbox: (100, 120, 40, 25, -30°) -> (100, 120, 40, 25, -π/6 radians)
        assert abs(rotated_bboxes[1, 0] - 100.0) < 1e-6
        assert abs(rotated_bboxes[1, 1] - 120.0) < 1e-6
        assert abs(rotated_bboxes[1, 2] - 40.0) < 1e-6
        assert abs(rotated_bboxes[1, 3] - 25.0) < 1e-6
        assert abs(rotated_bboxes[1, 4] - math.radians(-30.0)) < 1e-6

        # Check labels
        labels = result["labels"]
        assert len(labels) == 2
        assert labels[0] == 0
        assert labels[1] == 1


class BackwardRotatedBboxAnnotationConverterTest:
    """Tests for BackwardRotatedBboxAnnotationConverter."""

    def test_backward_rotated_bbox_annotation_converter_create_from_schema(self):
        """Test backward rotated bbox converter creation from schema."""

        # Create schema with rotated bbox attributes
        attributes = {
            "rotated_bboxes": AttributeInfo(
                type=np.ndarray,
                field=rotated_bbox_field(dtype=pl.Float32()),
            ),
            "rotated_bbox_labels": AttributeInfo(
                type=np.ndarray,
                field=label_field(is_list=True),
            ),
        }
        schema = Schema(attributes=attributes)

        converter = BackwardRotatedBboxAnnotationConverter.create_from_schema(schema)
        assert converter is not None
        assert converter.rotated_bboxes_attr == "rotated_bboxes"
        assert converter.rotated_bbox_labels_attr == "rotated_bbox_labels"

    def test_backward_rotated_bbox_annotation_converter_convert_to_legacy(self):
        """Test backward conversion from v2 to legacy rotated bbox format."""

        # Create v2 dataset with rotated bbox data
        rotated_bbox_data = np.array(
            [
                [50.0, 60.0, 30.0, 20.0, math.radians(45.0)],  # 45 degrees in radians
                [100.0, 120.0, 40.0, 25.0, math.radians(-30.0)],  # -30 degrees in radians
            ],
            dtype=np.float32,
        )

        label_data = np.array([0, 1], dtype=np.int32)

        # Create v2 dataset and add sample with rotated bbox data
        experimental_dataset = Dataset(
            dtype_or_schema=RotatedDetectionSample,
            categories={
                "rotated_bbox_labels": exp_categories.LabelCategories(labels=("label_1", "label_2", "label_3"))
            },
        )

        sample = RotatedDetectionSample(
            image_path="/path/to/test.jpg",
            rotated_bboxes=rotated_bbox_data,
            rotated_bbox_labels=label_data,
        )
        experimental_dataset.append(sample)

        # Create categories for testing
        categories: CategoriesInfo = {
            AnnotationType.label: LabelCategories(),
        }

        # Create converter
        attributes = {
            "rotated_bboxes": AttributeInfo(
                type=np.ndarray,
                field=rotated_bbox_field(dtype=pl.Float32()),
            ),
            "rotated_bbox_labels": AttributeInfo(
                type=np.ndarray,
                field=label_field(is_list=True),
            ),
        }
        schema = Schema(attributes=attributes)

        converter = BackwardRotatedBboxAnnotationConverter.create_from_schema(schema)
        assert converter is not None

        # Convert to legacy annotations
        annotations = converter.convert_to_legacy_annotations(sample, categories)

        # Check results
        assert len(annotations) == 2

        # Check first rotated bbox (should be converted back to degrees)
        rotated_bbox1 = annotations[0]
        assert isinstance(rotated_bbox1, RotatedBbox)
        assert abs(rotated_bbox1.cx - 50.0) < 1e-5
        assert abs(rotated_bbox1.cy - 60.0) < 1e-5
        assert abs(rotated_bbox1.w - 30.0) < 1e-5
        assert abs(rotated_bbox1.h - 20.0) < 1e-5
        assert abs(rotated_bbox1.r - 45.0) < 1e-5  # Back to degrees
        assert rotated_bbox1.label == 0

        # Check second rotated bbox
        rotated_bbox2 = annotations[1]
        assert isinstance(rotated_bbox2, RotatedBbox)
        assert abs(rotated_bbox2.cx - 100.0) < 1e-5
        assert abs(rotated_bbox2.cy - 120.0) < 1e-5
        assert abs(rotated_bbox2.w - 40.0) < 1e-5
        assert abs(rotated_bbox2.h - 25.0) < 1e-5
        assert abs(rotated_bbox2.r - (-30.0)) < 1e-5  # Back to degrees
        assert rotated_bbox2.label == 1


class RotatedBboxConversionTest:
    """Tests for rotated bbox conversion between legacy and v2 formats."""

    def test_rotated_bbox_conversion_with_labels(self):
        """Test end-to-end rotated bbox conversion with labels."""

        # Create legacy dataset with rotated bbox and label categories
        label_categories = LabelCategories()
        label_categories.add("person")
        label_categories.add("car")

        items = [
            DatasetItem(
                id="test_item",
                media=Image.from_file("test.jpg", size=(200, 150)),
                annotations=[
                    RotatedBbox(75, 50, 40, 30, 30.0, label=0),  # person
                    RotatedBbox(125, 100, 60, 40, -45.0, label=1),  # car
                ],
            )
        ]

        legacy_dataset = LegacyDataset.from_iterable(
            items,
            ann_types={AnnotationType.rotated_bbox},
            categories={AnnotationType.label: label_categories},
        )

        # Convert to v2 format
        experimental_dataset = convert_from_legacy(legacy_dataset)

        # Check v2 dataset
        assert len(experimental_dataset) == 1
        exp_sample = experimental_dataset[0]

        # Check rotated bbox data
        assert hasattr(exp_sample, "rotated_bboxes")
        assert hasattr(exp_sample, "labels")

        rotated_bboxes = exp_sample.rotated_bboxes
        labels = exp_sample.labels

        assert len(rotated_bboxes) == 2
        assert len(labels) == 2

        # Verify first rotated bbox (converted to radians)
        assert abs(rotated_bboxes[0, 0] - 75.0) < 1e-6  # cx
        assert abs(rotated_bboxes[0, 1] - 50.0) < 1e-6  # cy
        assert abs(rotated_bboxes[0, 2] - 40.0) < 1e-6  # w
        assert abs(rotated_bboxes[0, 3] - 30.0) < 1e-6  # h
        assert abs(rotated_bboxes[0, 4] - math.radians(30.0)) < 1e-6  # r in radians
        assert labels[0] == 0

        # Verify second rotated bbox
        assert abs(rotated_bboxes[1, 0] - 125.0) < 1e-6  # cx
        assert abs(rotated_bboxes[1, 1] - 100.0) < 1e-6  # cy
        assert abs(rotated_bboxes[1, 2] - 60.0) < 1e-6  # w
        assert abs(rotated_bboxes[1, 3] - 40.0) < 1e-6  # h
        assert abs(rotated_bboxes[1, 4] - math.radians(-45.0)) < 1e-6  # r in radians
        assert labels[1] == 1

        # Convert back to legacy format
        restored_legacy_dataset = convert_to_legacy(experimental_dataset)

        # Verify restored dataset
        restored_items = list(restored_legacy_dataset)
        assert len(restored_items) == 1

        restored_item = restored_items[0]
        rotated_bbox_anns = [ann for ann in restored_item.annotations if isinstance(ann, RotatedBbox)]
        assert len(rotated_bbox_anns) == 2

        # Check restored first rotated bbox (should be back to degrees)
        restored_bbox1 = rotated_bbox_anns[0]
        assert abs(restored_bbox1.cx - 75.0) < 1e-5
        assert abs(restored_bbox1.cy - 50.0) < 1e-5
        assert abs(restored_bbox1.w - 40.0) < 1e-5
        assert abs(restored_bbox1.h - 30.0) < 1e-5
        assert abs(restored_bbox1.r - 30.0) < 1e-5  # Back to degrees
        assert restored_bbox1.label == 0

        # Check restored second rotated bbox
        restored_bbox2 = rotated_bbox_anns[1]
        assert abs(restored_bbox2.cx - 125.0) < 1e-5
        assert abs(restored_bbox2.cy - 100.0) < 1e-5
        assert abs(restored_bbox2.w - 60.0) < 1e-5
        assert abs(restored_bbox2.h - 40.0) < 1e-5
        assert abs(restored_bbox2.r - (-45.0)) < 1e-5  # Back to degrees
        assert restored_bbox2.label == 1

    def test_rotated_bbox_conversion_without_labels(self):
        """Test rotated bbox conversion without label categories."""

        # Create legacy dataset with rotated bbox but no label categories
        items = [
            DatasetItem(
                id="test_item",
                media=Image.from_file("test.jpg", size=(100, 100)),
                annotations=[
                    RotatedBbox(50, 50, 20, 30, 0.0),  # No label, no rotation
                    RotatedBbox(25, 75, 10, 15, 90.0),  # No label, 90 degrees
                ],
            )
        ]

        legacy_dataset = LegacyDataset.from_iterable(items, ann_types={AnnotationType.rotated_bbox})

        # Convert to v2 format
        experimental_dataset = convert_from_legacy(legacy_dataset)

        # Check v2 dataset
        assert len(experimental_dataset) == 1
        exp_sample = experimental_dataset[0]

        # Should have rotated_bboxes but not rotated_bbox_labels
        assert hasattr(exp_sample, "rotated_bboxes")
        assert not hasattr(exp_sample, "labels")

        rotated_bboxes = exp_sample.rotated_bboxes
        assert len(rotated_bboxes) == 2

        # Verify rotated bbox data
        assert abs(rotated_bboxes[0, 0] - 50.0) < 1e-6  # cx
        assert abs(rotated_bboxes[0, 1] - 50.0) < 1e-6  # cy
        assert abs(rotated_bboxes[1, 4] - math.radians(90.0)) < 1e-6  # 90 degrees in radians

        # Convert back to legacy format
        restored_legacy_dataset = convert_to_legacy(experimental_dataset)

        # Verify restored dataset
        restored_items = list(restored_legacy_dataset)
        assert len(restored_items) == 1

        restored_item = restored_items[0]
        rotated_bbox_anns = [ann for ann in restored_item.annotations if isinstance(ann, RotatedBbox)]
        assert len(rotated_bbox_anns) == 2

        # Check that labels are None (since we didn't have label categories)
        for bbox in rotated_bbox_anns:
            assert bbox.label is None


class ForwardKeypointAnnotationConverterTest:
    """Tests for ForwardKeypointAnnotationConverter."""

    def test_keypoint_annotation_converter_convert_annotations_with_labels(self):
        """Test keypoint annotation conversion with label categories."""
        # Create label categories
        label_categories = LabelCategories()
        label_categories.add("person")
        label_categories.add("bicycle")

        categories: CategoriesInfo = {AnnotationType.label: label_categories}
        # Create a dataset with the categories
        dataset = LegacyDataset.from_iterable([], categories=categories)
        converter = ForwardKeypointAnnotationConverter.create(dataset)
        assert converter is not None

        # Create test Points annotation with label and keypoint_label_ids attribute
        points_data = [100.0, 200.0, 300.0, 400.0]  # 2 keypoints
        visibility = [2, 1]  # visible, hidden
        points_annotation = Points(
            points_data,
            visibility,
            label=0,
            attributes={"keypoint_label_ids": [0, 1]},  # person, bicycle
        )

        # Mock DatasetItem
        item = DatasetItem(id="test", annotations=[points_annotation])

        # Convert annotations
        result = converter.convert_annotations([points_annotation], item)

        assert "keypoints" in result
        assert "labels" in result
        assert result["keypoints"] == points_annotation
        assert result["labels"] == [0, 1]

    def test_keypoint_annotation_converter_get_annotation_type(self):
        """Test that keypoint converter returns correct annotation type."""
        annotation_types = ForwardKeypointAnnotationConverter.get_supported_annotation_types()
        assert annotation_types == [AnnotationType.points]


class SemanticMaskLoaderTest:
    """Tests for the SemanticMaskLoader picklable callable class."""

    def test_semantic_mask_loader_functionality(self):
        """Test SemanticMaskLoader with empty, single, and multiple masks."""
        from datumaro.experimental.legacy.annotation_converters import SemanticMaskLoader

        # Test empty mask list
        loader_empty = SemanticMaskLoader([])
        assert loader_empty() is None

        # Test single mask
        mask = np.array([[True, False], [False, True]], dtype=bool)
        loader_single = SemanticMaskLoader([(mask, 1)])
        result_single = loader_single()
        assert result_single is not None
        assert result_single.shape == (2, 2)
        assert result_single.dtype == np.uint8
        assert result_single[0, 0] == 1
        assert result_single[0, 1] == 0
        assert result_single[1, 0] == 0
        assert result_single[1, 1] == 1

        # Test multiple masks
        mask1 = np.array([[True, False], [False, False]], dtype=bool)
        mask2 = np.array([[False, True], [True, False]], dtype=bool)
        loader_multi = SemanticMaskLoader([(mask1, 1), (mask2, 2)])
        result_multi = loader_multi()
        assert result_multi is not None
        assert result_multi[0, 0] == 1  # From mask1
        assert result_multi[0, 1] == 2  # From mask2
        assert result_multi[1, 0] == 2  # From mask2
        assert result_multi[1, 1] == 0  # Background

    def test_semantic_mask_loader_is_picklable(self):
        """Test that SemanticMaskLoader can be pickled and unpickled."""
        from datumaro.experimental.legacy.annotation_converters import SemanticMaskLoader

        mask = np.array([[True, False], [False, True]], dtype=bool)
        loader = SemanticMaskLoader([(mask, 1)])

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

    def test_instance_mask_loader_functionality(self):
        """Test InstanceMaskLoader with empty, single, and multiple masks."""
        from datumaro.experimental.legacy.annotation_converters import InstanceMaskLoader

        # Test empty mask list
        loader_empty = InstanceMaskLoader([])
        result_empty = loader_empty()
        assert result_empty.shape == (0, 0, 0)
        assert result_empty.dtype == bool

        # Test single mask
        mask = np.array([[True, False], [False, True]], dtype=bool)
        loader_single = InstanceMaskLoader([mask])
        result_single = loader_single()
        assert result_single.shape == (1, 2, 2)
        assert result_single.dtype == bool
        assert result_single[0, 0, 0] == True  # noqa: E712
        assert result_single[0, 0, 1] == False  # noqa: E712

        # Test multiple masks
        mask1 = np.array([[True, False], [False, False]], dtype=bool)
        mask2 = np.array([[False, True], [True, False]], dtype=bool)
        loader_multi = InstanceMaskLoader([mask1, mask2])
        result_multi = loader_multi()
        assert result_multi.shape == (2, 2, 2)
        assert np.array_equal(result_multi[0], mask1)
        assert np.array_equal(result_multi[1], mask2)

    def test_instance_mask_loader_is_picklable(self):
        """Test that InstanceMaskLoader can be pickled and unpickled."""
        from datumaro.experimental.legacy.annotation_converters import InstanceMaskLoader

        mask1 = np.array([[True, False], [False, False]], dtype=bool)
        mask2 = np.array([[False, True], [True, False]], dtype=bool)
        loader = InstanceMaskLoader([mask1, mask2])

        # Pickle and unpickle
        pickled = pickle.dumps(loader)
        restored_loader = pickle.loads(pickled)

        # Verify the restored loader works correctly
        result = restored_loader()
        assert result.shape == (2, 2, 2)
        assert np.array_equal(result[0], mask1)
        assert np.array_equal(result[1], mask2)


class ForwardMaskAnnotationConverterPickleTest:
    """Tests for ForwardMaskAnnotationConverter producing picklable results."""

    def test_semantic_mask_conversion_is_picklable(self):
        """Test that semantic mask conversion results are picklable."""
        # Create a legacy dataset with semantic segmentation masks
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
        index_mask_data = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        legacy_categories = LabelCategories()
        legacy_categories.add("object")

        # Instance segmentation: indices don't match labels
        mask1 = ExtractedMask(index_mask=index_mask_data, index=0, label=0)
        mask2 = ExtractedMask(index_mask=index_mask_data, index=1, label=0)

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
