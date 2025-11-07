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
    InstanceMaskCallableField,
    InstanceMaskField,
    MaskCallableField,
    MaskField,
    PolygonField,
    RotatedBBoxField,
    TensorField,
    bbox_field,
    image_bytes_field,
    image_callable_field,
    image_field,
    image_info_field,
    image_path_field,
    instance_mask_callable_field,
    instance_mask_field,
    mask_callable_field,
    mask_field,
    polygon_field,
    rotated_bbox_field,
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


def test_instance_mask_field_creation():
    """Test InstanceMaskField creation and properties."""
    field = instance_mask_field(dtype=pl.Boolean)

    assert isinstance(field, InstanceMaskField)
    assert field.dtype == pl.Boolean
    assert field.semantic == Semantic.Default


def test_instance_mask_field_polars_schema():
    """Test InstanceMaskField Polars schema generation."""
    field = instance_mask_field(dtype=pl.Boolean)
    schema = field.to_polars_schema("instance_mask")

    expected = {
        "instance_mask": pl.List(pl.Boolean()),
        "instance_mask_shape": pl.List(pl.Int32()),
    }
    assert schema == expected


def test_instance_mask_field_polars_conversion():
    """Test InstanceMaskField to/from Polars conversion."""
    field = cast(InstanceMaskField, instance_mask_field(dtype=pl.Boolean))
    test_mask = np.array(
        [[[True, False], [False, True]], [[False, True], [True, False]]], dtype=bool
    )  # (2,2,2)

    # Test to_polars
    polars_data = field.to_polars("instance_mask", test_mask)
    assert "instance_mask" in polars_data
    assert isinstance(polars_data["instance_mask"], pl.Series)

    # Create DataFrame and test from_polars
    df = pl.DataFrame(polars_data)
    reconstructed = cast(
        np.ndarray[Any, Any], field.from_polars("instance_mask", 0, df, np.ndarray)
    )

    assert isinstance(reconstructed, np.ndarray)
    assert np.array_equal(reconstructed, test_mask)


def test_instance_mask_callable_field_creation():
    """Test InstanceMaskCallableField creation and properties."""
    field = instance_mask_callable_field(dtype=pl.Boolean, semantic=Semantic.Default)

    assert isinstance(field, InstanceMaskCallableField)
    assert field.dtype == pl.Boolean
    assert field.semantic == Semantic.Default


def test_instance_mask_callable_field_polars_schema():
    """Test InstanceMaskCallableField Polars schema generation."""
    field = instance_mask_callable_field(dtype=pl.Boolean, semantic=Semantic.Default)
    schema = field.to_polars_schema("instance_mask_callable")

    expected = {
        "instance_mask_callable": pl.Object(),
    }
    assert schema == expected


def test_instance_mask_callable_field_polars_conversion():
    """Test InstanceMaskCallableField to/from Polars conversion."""
    field = cast(
        InstanceMaskCallableField,
        instance_mask_callable_field(dtype=pl.Boolean, semantic=Semantic.Default),
    )

    def get_instance_masks():
        # Return a (2,2,2) array of boolean masks
        return np.array(
            [[[True, False], [False, True]], [[False, True], [True, False]]], dtype=bool
        )

    # Test to_polars
    polars_data = field.to_polars("instance_mask_callable", get_instance_masks)
    assert "instance_mask_callable" in polars_data
    assert callable(polars_data["instance_mask_callable"][0])

    # Create DataFrame and test from_polars
    df = pl.DataFrame(polars_data)
    reconstructed = field.from_polars("instance_mask_callable", 0, df, callable)

    assert callable(reconstructed)
    assert np.array_equal(reconstructed(), get_instance_masks())


def test_mask_callable_field_creation():
    """Test MaskCallableField creation and properties."""
    field = mask_callable_field(dtype=pl.UInt8, semantic=Semantic.Default)

    assert isinstance(field, MaskCallableField)
    assert field.dtype == pl.UInt8
    assert field.semantic == Semantic.Default


def test_mask_callable_field_polars_schema():
    """Test MaskCallableField Polars schema generation."""
    field = mask_callable_field(dtype=pl.UInt8, semantic=Semantic.Default)
    schema = field.to_polars_schema("mask_callable")

    expected = {
        "mask_callable": pl.Object(),
    }
    assert schema == expected


def test_mask_callable_field_polars_conversion():
    """Test MaskCallableField to/from Polars conversion."""
    field = cast(
        MaskCallableField,
        mask_callable_field(dtype=pl.UInt8, semantic=Semantic.Default),
    )

    def get_mask():
        # Return a (2,2) array of uint8 mask with category IDs
        return np.array([[1, 2], [2, 1]], dtype=np.uint8)

    # Test to_polars
    polars_data = field.to_polars("mask_callable", get_mask)
    assert "mask_callable" in polars_data
    assert callable(polars_data["mask_callable"][0])


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


def test_rotated_bbox_field_creation():
    """Test RotatedBBoxField creation and properties."""
    field = rotated_bbox_field(
        dtype=pl.Float32, format="cxcywhr", normalize=True, semantic=Semantic.Default
    )

    assert isinstance(field, RotatedBBoxField)
    assert field.dtype == pl.Float32
    assert field.format == "cxcywhr"
    assert field.normalize is True
    assert field.semantic == Semantic.Default


def test_rotated_bbox_field_creation_defaults():
    """Test RotatedBBoxField creation with default values."""
    field = rotated_bbox_field(dtype=pl.Float32)

    assert isinstance(field, RotatedBBoxField)
    assert field.dtype == pl.Float32
    assert field.format == "cxcywhr"  # Default format
    assert field.normalize is False  # Default normalization
    assert field.semantic == Semantic.Default  # Default semantic


def test_rotated_bbox_field_polars_schema():
    """Test RotatedBBoxField Polars schema generation."""
    field = rotated_bbox_field(dtype=pl.Float32)
    schema = field.to_polars_schema("rotated_bbox")

    expected = {"rotated_bbox": pl.List(pl.Array(pl.Float32, 5))}
    assert schema == expected


def test_rotated_bbox_field_polars_conversion():
    """Test RotatedBBoxField to/from Polars conversion."""
    field = cast(RotatedBBoxField, rotated_bbox_field(dtype=pl.Float32, normalize=False))
    # Test with rotated bboxes: [cx, cy, w, h, r]
    test_rotated_bbox = np.array(
        [[50.0, 60.0, 30.0, 20.0, 0.785], [100.0, 120.0, 40.0, 25.0, 1.57]], dtype=np.float32
    )

    # Test to_polars
    polars_data = field.to_polars("rotated_bbox", test_rotated_bbox)
    assert "rotated_bbox" in polars_data
    assert isinstance(polars_data["rotated_bbox"], pl.Series)

    # Create DataFrame and test from_polars
    df = pl.DataFrame(polars_data)
    reconstructed = cast(np.ndarray[Any, Any], field.from_polars("rotated_bbox", 0, df, np.ndarray))

    assert isinstance(reconstructed, np.ndarray)
    assert np.allclose(reconstructed, test_rotated_bbox)


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
    attr_info = AttributeInfo(type=np.ndarray, field=field)

    assert attr_info.type == np.ndarray
    assert attr_info.field == field


def test_schema_creation():
    """Test Schema creation."""
    attributes = {
        "image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.UInt8, format="RGB")),
        "bbox": AttributeInfo(type=np.ndarray, field=bbox_field(dtype=pl.Float32, normalize=False)),
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


# ==================== Serialization Tests ====================


def test_tensor_field_serialization():
    """Test TensorField to_dict/from_dict serialization."""
    field = TensorField(dtype=pl.Float32(), semantic=Semantic.Left, channels_first=True)

    # Serialize to dict
    field_dict = field.to_dict()
    assert field_dict["type"] == "TensorField"
    assert field_dict["semantic"] == "Left"
    assert field_dict["channels_first"] is True
    assert "Float32" in str(field_dict["dtype"])

    # Deserialize from dict
    from datumaro.experimental.schema import Field

    reconstructed = Field.from_dict(field_dict)
    assert isinstance(reconstructed, TensorField)

    # Compare with original
    assert reconstructed.semantic == field.semantic
    assert reconstructed.channels_first == field.channels_first
    assert str(reconstructed.dtype) == str(field.dtype)


def test_image_field_serialization():
    """Test ImageField to_dict/from_dict serialization."""
    field = image_field(dtype=pl.UInt8, format="RGB", semantic=Semantic.Right)

    # Serialize to dict
    field_dict = field.to_dict()
    assert field_dict["type"] == "ImageField"
    assert field_dict["semantic"] == "Right"
    assert field_dict["format"] == "RGB"
    assert "UInt8" in str(field_dict["dtype"])

    # Deserialize from dict
    from datumaro.experimental.schema import Field

    reconstructed = Field.from_dict(field_dict)
    assert isinstance(reconstructed, ImageField)

    # Compare with original
    assert reconstructed.semantic == field.semantic
    assert reconstructed.format == field.format
    assert str(reconstructed.dtype) == str(field.dtype)


def test_bbox_field_serialization():
    """Test BBoxField to_dict/from_dict serialization."""
    field = bbox_field(dtype=pl.Float32, format="xywh", normalize=True, semantic=Semantic.Default)

    # Serialize to dict
    field_dict = field.to_dict()
    assert field_dict["type"] == "BBoxField"
    assert field_dict["semantic"] == "Default"
    assert field_dict["format"] == "xywh"
    assert field_dict["normalize"] is True

    # Deserialize from dict
    from datumaro.experimental.schema import Field

    reconstructed = Field.from_dict(field_dict)
    assert isinstance(reconstructed, BBoxField)

    # Compare with original
    assert reconstructed.format == field.format
    assert reconstructed.normalize == field.normalize
    assert reconstructed.semantic == field.semantic
    assert str(reconstructed.dtype) == str(field.dtype)


def test_rotated_bbox_field_serialization():
    """Test RotatedBBoxField to_dict/from_dict serialization."""
    field = rotated_bbox_field(dtype=pl.Float32, format="xywha")

    # Serialize to dict
    field_dict = field.to_dict()
    assert field_dict["type"] == "RotatedBBoxField"
    assert field_dict["format"] == "xywha"

    # Deserialize from dict
    from datumaro.experimental.schema import Field

    reconstructed = Field.from_dict(field_dict)
    assert isinstance(reconstructed, RotatedBBoxField)

    # Compare with original
    assert reconstructed.format == field.format
    assert reconstructed.semantic == field.semantic
    assert str(reconstructed.dtype) == str(field.dtype)


def test_polygon_field_serialization():
    """Test PolygonField to_dict/from_dict serialization."""
    field = polygon_field(dtype=pl.Float64, format="xy", normalize=True)

    # Serialize to dict
    field_dict = field.to_dict()
    assert field_dict["type"] == "PolygonField"
    assert field_dict["format"] == "xy"
    assert field_dict["normalize"] is True

    # Deserialize from dict
    from datumaro.experimental.schema import Field

    reconstructed = Field.from_dict(field_dict)
    assert isinstance(reconstructed, PolygonField)

    # Compare with original
    assert reconstructed.format == field.format
    assert reconstructed.normalize == field.normalize
    assert reconstructed.semantic == field.semantic
    assert str(reconstructed.dtype) == str(field.dtype)


def test_mask_field_serialization():
    """Test MaskField to_dict/from_dict serialization."""
    field = mask_field(dtype=pl.UInt8)

    # Serialize to dict
    field_dict = field.to_dict()
    assert field_dict["type"] == "MaskField"
    assert "UInt8" in str(field_dict["dtype"])

    # Deserialize from dict
    from datumaro.experimental.schema import Field

    reconstructed = Field.from_dict(field_dict)
    assert isinstance(reconstructed, MaskField)

    # Compare with original
    assert reconstructed.semantic == field.semantic
    assert str(reconstructed.dtype) == str(field.dtype)


def test_instance_mask_field_serialization():
    """Test InstanceMaskField to_dict/from_dict serialization."""
    field = instance_mask_field(dtype=pl.Boolean, semantic=Semantic.Anomaly)

    # Serialize to dict
    field_dict = field.to_dict()
    assert field_dict["type"] == "InstanceMaskField"
    assert field_dict["semantic"] == "Anomaly"

    # Deserialize from dict
    from datumaro.experimental.schema import Field

    reconstructed = Field.from_dict(field_dict)
    assert isinstance(reconstructed, InstanceMaskField)

    # Compare with original
    assert reconstructed.semantic == field.semantic
    assert str(reconstructed.dtype) == str(field.dtype)


def test_image_path_field_serialization():
    """Test ImagePathField to_dict/from_dict serialization."""
    field = image_path_field(semantic=Semantic.Left)

    # Serialize to dict
    field_dict = field.to_dict()
    assert field_dict["type"] == "ImagePathField"
    assert field_dict["semantic"] == "Left"

    # Deserialize from dict
    from datumaro.experimental.schema import Field

    reconstructed = Field.from_dict(field_dict)
    assert isinstance(reconstructed, ImagePathField)

    # Compare with original
    assert reconstructed.semantic == field.semantic


def test_image_bytes_field_serialization():
    """Test ImageBytesField to_dict/from_dict serialization."""
    field = image_bytes_field(semantic=Semantic.Right)

    # Serialize to dict
    field_dict = field.to_dict()
    assert field_dict["type"] == "ImageBytesField"
    assert field_dict["semantic"] == "Right"

    # Deserialize from dict
    from datumaro.experimental.schema import Field

    reconstructed = Field.from_dict(field_dict)
    assert isinstance(reconstructed, ImageBytesField)

    # Compare with original
    assert reconstructed.semantic == field.semantic


def test_image_info_field_serialization():
    """Test ImageInfoField to_dict/from_dict serialization."""
    field = image_info_field(semantic=Semantic.Default)

    # Serialize to dict
    field_dict = field.to_dict()
    assert field_dict["type"] == "ImageInfoField"
    assert field_dict["semantic"] == "Default"

    # Deserialize from dict
    from datumaro.experimental.schema import Field

    reconstructed = Field.from_dict(field_dict)
    assert isinstance(reconstructed, ImageInfoField)

    # Compare with original
    assert reconstructed.semantic == field.semantic


def test_schema_serialization_simple():
    """Test Schema to_dict/from_dict with simple attributes."""
    from datumaro.experimental.schema import AttributeInfo, Schema

    # Create a simple schema
    schema = Schema(
        attributes={
            "image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.UInt8)),
            "bbox": AttributeInfo(type=np.ndarray, field=bbox_field(dtype=pl.Float32)),
        }
    )

    # Serialize to dict
    schema_dict = schema.to_dict()
    assert "attributes" in schema_dict
    assert "categories" in schema_dict
    assert "image" in schema_dict["attributes"]
    assert "bbox" in schema_dict["attributes"]
    assert schema_dict["attributes"]["image"]["field"]["type"] == "ImageField"
    assert schema_dict["attributes"]["bbox"]["field"]["type"] == "BBoxField"

    # Deserialize from dict
    reconstructed = Schema.from_dict(schema_dict)
    assert isinstance(reconstructed, Schema)

    # Compare with original
    assert set(reconstructed.attributes.keys()) == set(schema.attributes.keys())
    assert isinstance(reconstructed.attributes["image"].field, ImageField)
    assert isinstance(reconstructed.attributes["bbox"].field, BBoxField)
    assert type(reconstructed.attributes["image"].type) == type(schema.attributes["image"].type)
    assert type(reconstructed.attributes["bbox"].type) == type(schema.attributes["bbox"].type)


def test_schema_serialization_with_categories():
    """Test Schema to_dict/from_dict with categories."""
    from datumaro.experimental.categories import LabelCategories
    from datumaro.experimental.schema import AttributeInfo, Schema

    # Create schema with categories
    label_cats = LabelCategories(labels=("cat", "dog", "bird"))
    schema = Schema(
        attributes={
            "label": AttributeInfo(
                type=str, field=tensor_field(dtype=pl.Int32), categories=label_cats
            )
        }
    )

    # Serialize to dict
    schema_dict = schema.to_dict()
    assert "label" in schema_dict["attributes"]
    assert "label" in schema_dict["categories"]
    assert schema_dict["categories"]["label"]["type"] == "LabelCategories"
    assert schema_dict["categories"]["label"]["labels"] == ["cat", "dog", "bird"]

    # Deserialize from dict
    reconstructed = Schema.from_dict(schema_dict)

    # Compare with original
    assert "label" in reconstructed.attributes
    assert reconstructed.attributes["label"].categories is not None
    assert isinstance(reconstructed.attributes["label"].categories, LabelCategories)
    assert (
        reconstructed.attributes["label"].categories.labels
        == schema.attributes["label"].categories.labels
    )
    assert reconstructed.attributes["label"].categories.labels == ("cat", "dog", "bird")


def test_schema_serialization_builtin_types():
    """Test Schema serialization handles built-in types correctly."""
    from datumaro.experimental.schema import AttributeInfo, Schema

    # Create schema with built-in Python types and different semantics to avoid conflicts
    schema = Schema(
        attributes={
            "score": AttributeInfo(
                type=float, field=tensor_field(dtype=pl.Float32, semantic=Semantic.Default)
            ),
            "count": AttributeInfo(
                type=int, field=tensor_field(dtype=pl.Int32, semantic=Semantic.Left)
            ),
            "name": AttributeInfo(
                type=str, field=tensor_field(dtype=pl.Utf8, semantic=Semantic.Right)
            ),
        }
    )

    # Serialize and deserialize
    schema_dict = schema.to_dict()
    reconstructed = Schema.from_dict(schema_dict)

    # Compare with original
    assert set(reconstructed.attributes.keys()) == set(schema.attributes.keys())
    assert "score" in reconstructed.attributes
    assert "count" in reconstructed.attributes
    assert "name" in reconstructed.attributes
    # Verify semantics are preserved
    assert (
        reconstructed.attributes["score"].field.semantic
        == schema.attributes["score"].field.semantic
    )
    assert (
        reconstructed.attributes["count"].field.semantic
        == schema.attributes["count"].field.semantic
    )
    assert (
        reconstructed.attributes["name"].field.semantic == schema.attributes["name"].field.semantic
    )


def test_schema_with_categories_method():
    """Test Schema.with_categories method with serialization."""
    from datumaro.experimental.categories import LabelCategories
    from datumaro.experimental.schema import AttributeInfo, Schema

    # Create base schema without categories (use different semantics to avoid conflicts)
    schema = Schema(
        attributes={
            "label": AttributeInfo(
                type=str, field=tensor_field(dtype=pl.Int32, semantic=Semantic.Default)
            ),
            "mask": AttributeInfo(
                type=np.ndarray, field=mask_field(dtype=pl.UInt8, semantic=Semantic.Left)
            ),
        }
    )

    # Add categories
    label_cats = LabelCategories(labels=("cat", "dog"))
    schema_with_cats = schema.with_categories({"label": label_cats})

    # Serialize and deserialize
    schema_dict = schema_with_cats.to_dict()
    reconstructed = Schema.from_dict(schema_dict)

    # Compare with original
    assert reconstructed.attributes["label"].categories is not None
    assert isinstance(reconstructed.attributes["label"].categories, LabelCategories)
    assert (
        reconstructed.attributes["label"].categories.labels
        == schema_with_cats.attributes["label"].categories.labels
    )
    assert reconstructed.attributes["label"].categories.labels == ("cat", "dog")
    assert reconstructed.attributes["mask"].categories is None
