"""
Unit tests for schema and field system.
"""

from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from datumaro.experimental.dataset import Sample
from datumaro.experimental.fields import (
    BBoxField,
    ImageBytesField,
    ImageCallableField,
    ImageField,
    ImageInfo,
    ImageInfoField,
    ImagePathField,
    PolygonField,
    TensorField,
    bbox_field,
    image_bytes_field,
    image_callable_field,
    image_field,
    image_info_field,
    image_path_field,
    polygon_field,
    tensor_field,
)
from datumaro.experimental.schema import AttributeInfo, Schema, Semantic
from datumaro.util.image import decode_image, encode_image


def test_tensor_field_creation():
    """Test TensorField creation and properties."""
    field = tensor_field(dtype=pl.Float32, semantic=Semantic.Default)

    assert isinstance(field, TensorField)
    assert field.dtype == pl.Float32
    assert field.semantic == Semantic.Default


def test_tensor_field_polars_schema():
    """Test TensorField Polars schema generation."""
    field = tensor_field(dtype=pl.Float32)
    schema = field.to_polars_schema("test_tensor")

    expected = {
        "test_tensor": pl.List(pl.Float32()),
        "test_tensor_shape": pl.List(pl.Int32()),
    }
    assert schema == expected


def test_tensor_field_polars_conversion():
    """Test TensorField to/from Polars conversion."""
    field = cast(TensorField, tensor_field(dtype=pl.Float32))
    test_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # Test to_polars
    polars_data = field.to_polars("test_tensor", test_tensor)
    assert "test_tensor" in polars_data
    assert isinstance(polars_data["test_tensor"], pl.Series)

    # Create DataFrame and test from_polars
    df = pl.DataFrame(polars_data)
    reconstructed = cast(np.ndarray[Any, Any], field.from_polars("test_tensor", 0, df, np.ndarray))

    assert isinstance(reconstructed, np.ndarray)
    assert np.allclose(reconstructed, test_tensor)


def test_image_field_creation():
    """Test ImageField creation and properties."""
    field = image_field(dtype=pl.UInt8, format="RGB", semantic=Semantic.Left)

    assert isinstance(field, ImageField)
    assert field.dtype == pl.UInt8
    assert field.format == "RGB"
    assert field.semantic == Semantic.Left


def test_image_field_polars_schema():
    """Test ImageField Polars schema generation."""
    field = image_field(dtype=pl.UInt8, format="RGB")
    schema = field.to_polars_schema("image")

    expected = {"image": pl.List(pl.UInt8()), "image_shape": pl.List(pl.Int32())}
    assert schema == expected


def test_image_bytes_field_creation():
    """Test ImageBytesField creation and properties."""
    field = image_bytes_field(semantic=Semantic.Left)

    assert isinstance(field, ImageBytesField)
    assert field.semantic == Semantic.Left


def test_image_bytes_field_polars_schema():
    """Test ImageBytesField Polars schema generation."""
    field = image_bytes_field()
    schema = field.to_polars_schema("image_bytes")

    expected = {"image_bytes": pl.Binary()}
    assert schema == expected


def test_image_bytes_field_polars_conversion_with_numpy():
    """Test ImageBytesField to/from Polars conversion with numpy arrays."""
    field = image_bytes_field()
    image_bytes = np.array(np.random.bytes(30))

    # Test to_polars with numpy array
    polars_data = field.to_polars("image_bytes", image_bytes)
    assert "image_bytes" in polars_data
    assert isinstance(polars_data["image_bytes"][0], bytes)
    assert len(polars_data["image_bytes"][0]) > 0

    # Create DataFrame and test from_polars
    df = pl.DataFrame(polars_data)
    reconstructed = field.from_polars("image_bytes", 0, df, np.ndarray)

    assert isinstance(reconstructed, np.ndarray)
    assert np.all(reconstructed == image_bytes)


def test_image_bytes_field_polars_conversion_with_bytes():
    """Test ImageBytesField to/from Polars conversion with direct bytes."""
    field = image_bytes_field()
    image_bytes = np.array(np.random.bytes(30))

    # Test to_polars with bytes
    polars_data = field.to_polars("image_bytes", image_bytes)
    assert "image_bytes" in polars_data
    assert polars_data["image_bytes"][0] == image_bytes

    # Create DataFrame and test from_polars
    df = pl.DataFrame(polars_data)
    reconstructed = field.from_polars("image_bytes", 0, df, bytes)

    assert isinstance(reconstructed, bytes)
    assert reconstructed == image_bytes


def test_bbox_field_creation():
    """Test BBoxField creation and properties."""
    field = bbox_field(
        dtype=pl.Float32, format="x1y1x2y2", normalize=True, semantic=Semantic.Default
    )

    assert isinstance(field, BBoxField)
    assert field.dtype == pl.Float32
    assert field.format == "x1y1x2y2"
    assert field.normalize is True
    assert field.semantic == Semantic.Default


def test_bbox_field_polars_schema():
    """Test BBoxField Polars schema generation."""
    field = bbox_field(dtype=pl.Float32)
    schema = field.to_polars_schema("bbox")

    expected = {"bbox": pl.List(pl.Array(pl.Float32, 4))}
    assert schema == expected


def test_bbox_field_polars_conversion():
    """Test BBoxField to/from Polars conversion."""
    field = cast(BBoxField, bbox_field(dtype=pl.Float32, normalize=False))
    test_bbox = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)

    # Test to_polars
    polars_data = field.to_polars("bbox", test_bbox)
    assert "bbox" in polars_data
    assert isinstance(polars_data["bbox"], pl.Series)

    # Create DataFrame and test from_polars
    df = pl.DataFrame(polars_data)
    reconstructed = cast(np.ndarray[Any, Any], field.from_polars("bbox", 0, df, np.ndarray))

    assert isinstance(reconstructed, np.ndarray)
    assert np.allclose(reconstructed, test_bbox)


def test_image_info_creation():
    """Test ImageInfo creation."""
    info = ImageInfo(width=640, height=480)

    assert info.width == 640
    assert info.height == 480


def test_image_info_field_creation():
    """Test ImageInfoField creation."""
    field = image_info_field(semantic=Semantic.Left)

    assert isinstance(field, ImageInfoField)
    assert field.semantic == Semantic.Left


def test_image_info_field_polars_schema():
    """Test ImageInfoField Polars schema generation."""
    field = image_info_field()
    schema = field.to_polars_schema("image_info")

    expected = {
        "image_info": pl.Struct([pl.Field("width", pl.Int32()), pl.Field("height", pl.Int32())])
    }
    assert schema == expected


def test_image_info_field_polars_conversion():
    """Test ImageInfoField to/from Polars conversion."""
    field = image_info_field()
    test_info = ImageInfo(width=640, height=480)

    # Test to_polars
    polars_data = field.to_polars("image_info", test_info)
    assert "image_info" in polars_data
    assert isinstance(polars_data["image_info"], pl.Series)

    # Create DataFrame and test from_polars
    df = pl.DataFrame(polars_data)
    reconstructed = field.from_polars("image_info", 0, df, ImageInfo)

    assert isinstance(reconstructed, ImageInfo)
    assert reconstructed.width == 640
    assert reconstructed.height == 480


def test_image_path_field_creation():
    """Test ImagePathField creation."""
    field = image_path_field(semantic=Semantic.Default)

    assert isinstance(field, ImagePathField)
    assert field.semantic == Semantic.Default


def test_image_path_field_polars_schema():
    """Test ImagePathField Polars schema generation."""
    field = image_path_field()
    schema = field.to_polars_schema("image_path")

    expected = {"image_path": pl.String()}
    assert schema == expected


def test_image_path_field_polars_conversion():
    """Test ImagePathField to/from Polars conversion."""
    field = image_path_field()
    test_path = "/path/to/image.jpg"

    # Test to_polars
    polars_data = field.to_polars("image_path", test_path)
    assert "image_path" in polars_data
    assert isinstance(polars_data["image_path"], pl.Series)

    # Create DataFrame and test from_polars
    df = pl.DataFrame(polars_data)
    reconstructed = field.from_polars("image_path", 0, df, str)

    assert isinstance(reconstructed, str)
    assert reconstructed == test_path


def test_image_callable_field_creation():
    """Test ImageCallableField creation and properties."""
    field = image_callable_field(format="RGB", semantic=Semantic.Default)

    assert isinstance(field, ImageCallableField)
    assert field.format == "RGB"
    assert field.semantic == Semantic.Default

    # Test with different format
    field_bgr = image_callable_field(format="BGR", semantic=Semantic.Left)
    assert field_bgr.format == "BGR"
    assert field_bgr.semantic == Semantic.Left


def test_image_callable_field_polars_schema():
    """Test ImageCallableField Polars schema generation."""
    field = image_callable_field()
    schema = field.to_polars_schema("image_callable")

    expected = {"image_callable": pl.Object}
    assert schema == expected


def test_image_callable_field_polars_conversion():
    """Test ImageCallableField to/from Polars conversion."""
    import numpy as np

    field = image_callable_field()  # Create a test callable

    def test_image_generator():
        return np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)

    # Test to_polars
    polars_data = field.to_polars("image_callable", test_image_generator)
    assert "image_callable" in polars_data
    assert isinstance(polars_data["image_callable"], pl.Series)

    # Create DataFrame and test from_polars
    df = pl.DataFrame(polars_data)
    reconstructed = field.from_polars("image_callable", 0, df, callable)

    assert callable(reconstructed)
    assert reconstructed == test_image_generator

    # Test that the callable produces the expected output
    result = reconstructed()
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8


def test_image_callable_field_error_handling():
    """Test ImageCallableField error handling."""
    field = image_callable_field()

    # Test with non-callable object
    with pytest.raises(TypeError, match="Expected callable"):
        field.to_polars("image_callable", "not_a_callable")

    # Test with non-callable in from_polars
    df = pl.DataFrame({"image_callable": ["not_a_callable"]})
    with pytest.raises(TypeError, match="Expected callable in column"):
        field.from_polars("image_callable", 0, df, callable)


def test_image_callable_field_complex_callable():
    """Test ImageCallableField with more complex callables."""
    import numpy as np

    field = image_callable_field(format="RGBA")  # Test with lambda
    lambda_callable = lambda: np.zeros((5, 5, 4), dtype=np.uint8)

    polars_data = field.to_polars("image_callable", lambda_callable)
    df = pl.DataFrame(polars_data)
    reconstructed = field.from_polars("image_callable", 0, df, callable)

    result = reconstructed()
    assert result.shape == (5, 5, 4)
    assert result.dtype == np.uint8
    assert np.all(result == 0)

    # Test with class method
    class ImageGenerator:
        def __init__(self, size):
            self.size = size

        def generate(self):
            return np.ones((self.size, self.size, 3), dtype=np.uint8) * 128

    generator = ImageGenerator(3)
    polars_data2 = field.to_polars("image_callable", generator.generate)
    df2 = pl.DataFrame(polars_data2)
    reconstructed2 = field.from_polars("image_callable", 0, df2, callable)

    result2 = reconstructed2()
    assert result2.shape == (3, 3, 3)
    assert np.all(result2 == 128)


def test_attribute_info_creation():
    """Test AttributeInfo creation."""
    field = tensor_field(dtype=pl.Float32)
    attr_info = AttributeInfo(type=np.ndarray, annotation=field)

    assert attr_info.type == np.ndarray
    assert attr_info.annotation == field


def test_schema_creation():
    """Test Schema creation."""
    attributes = {
        "image": AttributeInfo(
            type=np.ndarray, annotation=image_field(dtype=pl.UInt8, format="RGB")
        ),
        "bbox": AttributeInfo(
            type=np.ndarray, annotation=bbox_field(dtype=pl.Float32, normalize=False)
        ),
    }

    schema = Schema(attributes=attributes)

    assert len(schema.attributes) == 2
    assert "image" in schema.attributes
    assert "bbox" in schema.attributes
    assert schema.attributes["image"].type == np.ndarray
    assert schema.attributes["bbox"].type == np.ndarray


def test_field_equality():
    """Test field equality comparison."""
    field1 = tensor_field(dtype=pl.Float32, semantic=Semantic.Default)
    field2 = tensor_field(dtype=pl.Float32, semantic=Semantic.Default)
    field3 = tensor_field(dtype=pl.Int32, semantic=Semantic.Default)

    # Same configuration should be equal
    assert field1 == field2

    # Different dtype should not be equal
    assert field1 != field3


def test_field_semantic_variations():
    """Test fields with different semantic values."""
    left_field = image_field(dtype=pl.UInt8, format="RGB", semantic=Semantic.Left)
    right_field = image_field(dtype=pl.UInt8, format="RGB", semantic=Semantic.Right)
    default_field = image_field(dtype=pl.UInt8, format="RGB", semantic=Semantic.Default)

    assert left_field != right_field
    assert left_field != default_field
    assert right_field != default_field

    assert left_field.semantic == Semantic.Left
    assert right_field.semantic == Semantic.Right
    assert default_field.semantic == Semantic.Default


def test_schema_duplicate_field_type_assertion():
    """Test that schema creation fails with assertion when two fields have the same field type."""

    # This should fail because we have two ImageFields with the same semantic context
    with pytest.raises(ValueError):

        class InvalidSample(Sample):
            image1: np.ndarray[Any, Any] = image_field(
                dtype=pl.UInt8, format="RGB", semantic=Semantic.Default
            )
            image2: np.ndarray[Any, Any] = image_field(
                dtype=pl.UInt8, format="RGB", semantic=Semantic.Default
            )

        # This should trigger the assertion error when schema is inferred
        InvalidSample.infer_schema()

    # This should work because the fields have different semantic contexts
    class ValidSample(Sample):
        left_image: np.ndarray[Any, Any] = image_field(
            dtype=pl.UInt8, format="RGB", semantic=Semantic.Left
        )
        right_image: np.ndarray[Any, Any] = image_field(
            dtype=pl.UInt8, format="RGB", semantic=Semantic.Right
        )

    # This should not raise an assertion error
    schema = ValidSample.infer_schema()
    assert len(schema.attributes) == 2
    assert "left_image" in schema.attributes
    assert "right_image" in schema.attributes


def test_polygon_field_creation():
    """Test PolygonField creation and properties."""
    field = polygon_field(dtype=pl.Float32, format="xy", normalize=True, semantic=Semantic.Default)

    assert isinstance(field, PolygonField)
    assert field.dtype == pl.Float32
    assert field.format == "xy"
    assert field.normalize is True
    assert field.semantic == Semantic.Default


def test_polygon_field_creation_defaults():
    """Test PolygonField creation with default values."""
    field = polygon_field(dtype=pl.Float32)

    assert isinstance(field, PolygonField)
    assert field.dtype == pl.Float32
    assert field.format == "xy"  # Default format
    assert field.normalize is False  # Default normalization
    assert field.semantic == Semantic.Default  # Default semantic


def test_polygon_field_polars_schema():
    """Test PolygonField Polars schema generation."""
    field = polygon_field(dtype=pl.Float32)
    schema = field.to_polars_schema("polygon")

    expected = {"polygon": pl.List(pl.List(pl.Array(pl.Float32, 2)))}
    assert schema == expected


def test_polygon_field_polars_conversion():
    """Test PolygonField to/from Polars conversion with simple polygon."""
    field = cast(PolygonField, polygon_field(dtype=pl.Float32, normalize=False))
    # Triangle: (0,0) -> (10,0) -> (5,10) -> (0,0)
    test_polygon_1 = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]], dtype=np.float32)
    test_polygon_2 = np.array([[2.0, 2.0], [10.0, 2.0], [5.0, 10.0], [6.0, 11.0]], dtype=np.float32)
    polygons = np.array([test_polygon_1, test_polygon_2], dtype=object)

    # Test to_polars
    polars_data = field.to_polars("polygon", polygons)
    assert "polygon" in polars_data
    assert isinstance(polars_data["polygon"], pl.Series)

    # Create DataFrame and test from_polars
    df = pl.DataFrame(polars_data)
    reconstructed = cast(np.ndarray[Any, Any], field.from_polars("polygon", 0, df, np.ndarray))

    assert len(reconstructed) == 2
    assert np.all(reconstructed[0] == test_polygon_1)
    assert np.all(reconstructed[1] == test_polygon_2)
