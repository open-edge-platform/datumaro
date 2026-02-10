"""
Unit tests for converter registry and converter implementations.
"""

import os
import tempfile

import numpy as np
import polars as pl
import pytest

from datumaro.experimental.converters import (
    ChannelsFirstConverter,
    ImageBytesToImageConverter,
    ImageCallableToImageConverter,
    ImagePathToImageConverter,
    RedBlueColorConverter,
    UInt8ToFloat32Converter,
    find_conversion_path,
)
from datumaro.experimental.fields import ImageBytesField, ImageCallableField, ImageField, ImageInfoField, ImagePathField
from datumaro.experimental.schema import AttributeInfo, AttributeSpec, Schema


def test_rgb_to_bgr_converter():
    """Test RGB to BGR format conversion."""
    converter_instance = RedBlueColorConverter()  # type: ignore[call-arg]

    # Create test data
    rgb_data = np.array([255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128])
    df = pl.DataFrame(
        {"image": [rgb_data], "image_shape": [[2, 2, 3]]},
        schema=pl.Schema({"image": pl.List(pl.UInt8()), "image_shape": pl.List(pl.Int64)}),
    )

    # Set up converter attributes
    input_field = ImageField(dtype=pl.UInt8(), format="RGB")
    output_field = ImageField(dtype=pl.UInt8(), format="BGR")

    setattr(
        converter_instance,
        "input_image",
        AttributeSpec(name="image", field=input_field),
    )
    setattr(
        converter_instance,
        "output_image",
        AttributeSpec(name="image", field=output_field),
    )

    # Test filter - should return True for RGB->BGR conversion
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "image" in result_df.columns
    # Result should have BGR format (channels swapped)
    result_data = result_df["image"][0]
    # First pixel: RGB [255, 0, 0] -> BGR [0, 0, 255]
    assert (result_data.reshape((2, 2, 3))[0][0] == [0, 0, 255]).all()


def test_uint8_to_float32_converter():
    """Test UInt8 to Float32 data type conversion."""
    converter_instance = UInt8ToFloat32Converter()  # type: ignore[call-arg]

    # Create test data with UInt8 values
    uint8_data = [255, 128, 0, 64, 192, 32]
    df = pl.DataFrame(
        {"image": [uint8_data], "image_shape": [[2, 3]]},
        schema=pl.Schema({"image": pl.List(pl.UInt8()), "image_shape": pl.List(pl.Int64)}),
    )

    # Set up converter attributes
    input_field = ImageField(dtype=pl.UInt8(), format="RGB")
    output_field = ImageField(dtype=pl.Float32(), format="RGB")

    setattr(
        converter_instance,
        "input_image",
        AttributeSpec(name="image", field=input_field),
    )
    setattr(
        converter_instance,
        "output_image",
        AttributeSpec(name="image", field=output_field),
    )

    # Test filter - should return True for UInt8->Float32 conversion
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "image" in result_df.columns
    result_data = result_df["image"][0]

    # Check that values are normalized to [0, 1] range
    assert (result_data >= 0).all()
    assert (result_data <= 1).all()


def test_image_path_to_image_converter():
    """Test lazy loading of images from file paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test image file
        import numpy as np
        from PIL import Image as PILImage

        test_image_path = os.path.join(temp_dir, "test.png")
        img_array = np.random.randint(0, 255, (50, 75, 3), dtype=np.uint8)
        test_img = PILImage.fromarray(img_array)
        test_img.save(test_image_path)

        converter_instance = ImagePathToImageConverter()  # type: ignore[call-arg]

        # Create test data
        df = pl.DataFrame({"image_path": [test_image_path]})

        # Set up converter attributes
        input_field = ImagePathField()
        output_field = ImageField(dtype=pl.UInt8(), format="RGB")
        output_info_field = ImageInfoField()

        setattr(
            converter_instance,
            "input_path",
            AttributeSpec(name="image_path", field=input_field),
        )
        setattr(
            converter_instance,
            "output_image",
            AttributeSpec(name="image", field=output_field),
        )
        setattr(
            converter_instance,
            "output_info",
            AttributeSpec(name="image_info", field=output_info_field),
        )

        # Test filter - should return True for path->image conversion
        assert converter_instance.filter_output_spec() is True

        # Test conversion
        result_df = converter_instance.convert(df)

        assert "image" in result_df.columns
        assert "image_shape" in result_df.columns

        # Check that image was loaded correctly
        result_shape = list(result_df["image_shape"][0])
        assert result_shape == [50, 75, 3]  # height, width, channels


def test_image_bytes_to_image_converter():
    """Test ImageBytesToImageConverter functionality."""
    import numpy as np

    from datumaro.util.image import encode_image

    converter_instance = ImageBytesToImageConverter()  # type: ignore[call-arg]

    # Create test image data
    test_image = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
    image_bytes = encode_image(test_image, ".png")

    # Create test data
    df = pl.DataFrame({"image_bytes": [image_bytes]})

    # Set up converter attributes

    input_field = ImageBytesField()
    output_field = ImageField(dtype=pl.UInt8(), format="RGB")
    output_info_field = ImageInfoField()

    setattr(
        converter_instance,
        "input_bytes",
        AttributeSpec(name="image_bytes", field=input_field),
    )
    setattr(
        converter_instance,
        "output_image",
        AttributeSpec(name="image", field=output_field),
    )
    setattr(
        converter_instance,
        "output_info",
        AttributeSpec(name="image_info", field=output_info_field),
    )

    # Test filter - should return True for bytes->image conversion
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "image" in result_df.columns
    assert "image_shape" in result_df.columns

    # Check that image was loaded correctly
    result_shape = tuple(result_df["image_shape"][0])
    assert result_shape == (32, 48, 3)  # height, width, channels

    # Check that the actual image data is correct (approximately, since PNG compression may cause slight differences)
    result_image = result_df["image"][0].to_numpy().reshape(result_shape)
    assert result_image.shape == test_image.shape
    assert result_image.dtype == np.uint8


def test_image_callable_to_image_converter():
    """Test ImageCallableToImageConverter functionality."""
    import numpy as np

    # Create a callable that returns a test image
    def create_test_image():
        return np.array(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
                [[128, 128, 128], [64, 64, 64], [192, 192, 192]],
            ],
            dtype=np.uint8,
        )

    # Create input DataFrame with callable
    df = pl.DataFrame({"my_callable": [create_test_image]}, schema={"my_callable": pl.Object})

    # Create converter instance
    converter_instance = ImageCallableToImageConverter()  # type: ignore[call-arg]

    # Set up converter attributes
    input_field = ImageCallableField(format="RGB")
    output_field = ImageField(dtype=pl.UInt8(), format="RGB")
    output_info_field = ImageInfoField()

    setattr(
        converter_instance,
        "input_callable",
        AttributeSpec(name="my_callable", field=input_field),
    )
    setattr(
        converter_instance,
        "output_image",
        AttributeSpec(name="output_image", field=output_field),
    )
    setattr(
        converter_instance,
        "output_info",
        AttributeSpec(name="output_info", field=output_info_field),
    )

    # Test filter - should return True
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    # Check expected columns exist
    assert "output_image" in result_df.columns
    assert "output_image_shape" in result_df.columns

    # Check image shape
    image_shape = result_df["output_image_shape"][0]
    assert list(image_shape) == [3, 3, 3]  # height, width, channels

    # Check image data can be reconstructed
    image_data = result_df["output_image"][0]
    reconstructed = np.array(image_data, dtype=np.uint8).reshape(list(image_shape))
    assert reconstructed.shape == (3, 3, 3)
    assert reconstructed.dtype == np.uint8


def test_image_callable_converter_error_handling():
    """Test error handling in ImageCallableToImageConverter."""
    import numpy as np

    # Test with callable that returns non-numpy array
    def bad_callable_1():
        return "not an array"

    # Test with callable that returns wrong shape
    def bad_callable_2():
        return np.array([1, 2, 3, 4])  # 1D array

    # Test with callable that raises exception
    def bad_callable_3():
        raise ValueError("Something went wrong")

    # Create converter instance
    converter_instance = ImageCallableToImageConverter()  # type: ignore[call-arg]

    # Set up converter attributes
    input_field = ImageCallableField(format="RGB")
    output_field = ImageField(dtype=pl.UInt8(), format="RGB")
    output_info_field = ImageInfoField()

    setattr(
        converter_instance,
        "input_callable",
        AttributeSpec(name="my_callable", field=input_field),
    )
    setattr(
        converter_instance,
        "output_image",
        AttributeSpec(name="output_image", field=output_field),
    )
    setattr(
        converter_instance,
        "output_info",
        AttributeSpec(name="output_info", field=output_info_field),
    )

    # Test TypeError for non-numpy return
    df1 = pl.DataFrame({"my_callable": [bad_callable_1]}, schema={"my_callable": pl.Object})
    with pytest.raises(RuntimeError, match="must return numpy\.ndarray"):
        converter_instance.convert(df1)

    # Test ValueError for wrong shape
    df2 = pl.DataFrame({"my_callable": [bad_callable_2]}, schema={"my_callable": pl.Object})
    with pytest.raises(RuntimeError, match="must be 3D"):
        converter_instance.convert(df2)

    # Test exception propagation
    df3 = pl.DataFrame({"my_callable": [bad_callable_3]}, schema={"my_callable": pl.Object})
    with pytest.raises(RuntimeError, match="Error executing callable"):
        converter_instance.convert(df3)


def test_image_callable_converter_dtype_handling():
    """Test dtype handling in ImageCallableToImageConverter."""
    import numpy as np

    # Test with uint8 image (should work)
    def uint8_callable():
        return np.array([[[255, 0, 128]]], dtype=np.uint8)

    # Test with float32 image
    def float32_callable():
        return np.array([[[1.0, 0.5, 0.25]]], dtype=np.float32)

    # Create converter instance
    converter_instance = ImageCallableToImageConverter()  # type: ignore[call-arg]

    # Set up converter attributes for uint8
    input_field = ImageCallableField(format="RGB")
    output_field = ImageField(dtype=pl.UInt8(), format="RGB")
    output_info_field = ImageInfoField()

    setattr(converter_instance, "input_callable", AttributeSpec(name="my_callable", field=input_field))
    setattr(converter_instance, "output_image", AttributeSpec(name="output_image", field=output_field))
    setattr(
        converter_instance,
        "output_info",
        AttributeSpec(name="output_info", field=output_info_field),
    )

    # Test uint8 image
    df1 = pl.DataFrame({"my_callable": [uint8_callable]}, schema={"my_callable": pl.Object})
    result1 = converter_instance.convert(df1)
    image_data1 = np.array(result1["output_image"][0], dtype=np.uint8).reshape([1, 1, 3])
    assert image_data1.dtype == np.uint8
    assert image_data1[0, 0, 0] == 255
    assert image_data1[0, 0, 1] == 0
    assert image_data1[0, 0, 2] == 128

    # Test float32 image with different output field
    output_field_f32 = ImageField(dtype=pl.Float32(), format="RGB")
    setattr(
        converter_instance,
        "output_image",
        AttributeSpec(name="output_image", field=output_field_f32),
    )

    df2 = pl.DataFrame({"my_callable": [float32_callable]}, schema={"my_callable": pl.Object})
    result2 = converter_instance.convert(df2)
    image_data2 = np.array(result2["output_image"][0], dtype=np.float32).reshape([1, 1, 3])
    assert image_data2.dtype == np.float32
    assert image_data2[0, 0, 0] == 1.0
    assert image_data2[0, 0, 1] == 0.5
    assert image_data2[0, 0, 2] == 0.25


def test_channels_first_converter():
    """Test ChannelsFirstConverter updates metadata without data transposition."""

    converter_instance = ChannelsFirstConverter()

    # Create test data with channels-first image (the data is stored flattened)
    # Shape would be (3, 2, 2) for channels-first
    image_data = list(range(12))  # 3 channels * 2 height * 2 width
    df = pl.DataFrame(
        {
            "image": [image_data],
            "image_shape": [[3, 2, 2]],  # channels-first shape (C, H, W)
        },
        schema=pl.Schema({"image": pl.List(pl.UInt8), "image_shape": pl.List(pl.Int32)}),
    )

    # Set up converter attributes
    input_field = ImageField(dtype=pl.UInt8(), format="RGB", channels_first=True)
    output_field = ImageField(dtype=pl.UInt8(), format="RGB", channels_first=False)

    setattr(converter_instance, "input_image", AttributeSpec(name="image", field=input_field))
    setattr(converter_instance, "output_image", AttributeSpec(name="image", field=output_field))

    # Test filter - should return True for channels_first change
    assert converter_instance.filter_output_spec() is True

    # Test conversion - should just copy the data (transposition handled by from_polars)
    result_df = converter_instance.convert(df)

    assert "image" in result_df.columns
    assert "image_shape" in result_df.columns

    # Data should be unchanged
    assert result_df["image"][0].to_list() == image_data
    # Shape should be unchanged (the field handles transposition on read)
    assert result_df["image_shape"][0].to_list() == [3, 2, 2]


def test_channels_first_converter_same_channels_first():
    """Test ChannelsFirstConverter returns False when channels_first is the same."""

    converter_instance = ChannelsFirstConverter()

    # Set up converter attributes with same channels_first
    input_field = ImageField(dtype=pl.UInt8(), format="RGB", channels_first=True)
    output_field = ImageField(dtype=pl.UInt8(), format="RGB", channels_first=True)

    setattr(converter_instance, "input_image", AttributeSpec(name="image", field=input_field))
    setattr(converter_instance, "output_image", AttributeSpec(name="image", field=output_field))

    # Test filter - should return False when channels_first is the same
    assert converter_instance.filter_output_spec() is False


def test_channels_first_converter_schema_conversion():
    """Test schema conversion for channels_first change using find_conversion_path."""

    # Create source schema with channels_first=True
    source_schema = Schema(
        attributes={
            "image": AttributeInfo(
                type=np.ndarray,
                field=ImageField(dtype=pl.UInt8(), format="RGB", channels_first=True),
            )
        }
    )

    # Create target schema with channels_first=False
    target_schema = Schema(
        attributes={
            "image": AttributeInfo(
                type=np.ndarray,
                field=ImageField(dtype=pl.UInt8(), format="RGB", channels_first=False),
            )
        }
    )

    # Find conversion path
    conversion_paths, _ = find_conversion_path(source_schema, target_schema)

    # Should have exactly one converter
    assert len(conversion_paths.converters["image"]) == 1
    assert type(conversion_paths.converters["image"][0]) is ChannelsFirstConverter
