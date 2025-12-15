"""
Unit tests for converter registry and converter implementations.
"""

import os
import tempfile
from dataclasses import field
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from datumaro.experimental.categories import LabelCategories, MaskCategories
from datumaro.experimental.converters import (
    AttributeRemapperConverter,
    BBoxCoordinateConverter,
    BBoxFormatConverter,
    BBoxToPolygonConverter,
    ChannelsFirstConverter,
    ConversionError,
    Converter,
    ConverterRegistry,
    EllipseDtypeConverter,
    EllipseToBBoxConverter,
    ImageBytesToImageConverter,
    ImageCallableToImageConverter,
    ImagePathToImageConverter,
    InstanceMaskCallableToInstanceMaskConverter,
    KeypointsCoordinateConverter,
    KeypointsDtypeConverter,
    KeypointsToBBoxConverter,
    LabelIndexConverter,
    MaskCallableToMaskConverter,
    PolygonToBBoxConverter,
    PolygonToInstanceMaskConverter,
    PolygonToMaskConverter,
    RedBlueColorConverter,
    RotatedBBoxCoordinateConverter,
    RotatedBBoxToBBoxConverter,
    RotatedBBoxToPolygonConverter,
    UInt8ToFloat32Converter,
    converter,
    find_conversion_path,
)
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import (
    BBoxField,
    EllipseField,
    Field,
    ImageBytesField,
    ImageCallableField,
    ImageField,
    ImageInfoField,
    ImagePathField,
    InstanceMaskCallableField,
    InstanceMaskField,
    KeypointsField,
    LabelField,
    MaskCallableField,
    MaskField,
    PolygonField,
    RotatedBBoxField,
    bbox_field,
    image_field,
    image_info_field,
)
from datumaro.experimental.schema import AttributeInfo, AttributeSpec, Schema


def test_converter_decorator(request: pytest.FixtureRequest):
    """Test the @converter decorator functionality."""

    # Create a simple converter for testing
    @converter
    class TestConverter(Converter):  # type: ignore[reportUnusedClass]
        input_field: AttributeSpec[ImageField]
        output_field: AttributeSpec[ImageField]

        def filter_output_spec(self) -> bool:
            return False

        def convert(self, df: pl.DataFrame) -> pl.DataFrame:
            return df

    registry = ConverterRegistry()
    request.addfinalizer(lambda: registry.remove_converter(TestConverter))

    # Check that it was registered
    converters = registry.list_converters()

    # Should find our test converter in the registry
    test_converters = [c for c in converters if c == TestConverter]
    assert len(test_converters) > 0


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


def test_bbox_coordinate_converter():
    """Test bounding box coordinate normalization/denormalization."""
    converter_instance = BBoxCoordinateConverter()  # type: ignore[call-arg]

    # Create test data with absolute coordinates and image dimensions
    df = pl.DataFrame(
        {
            "bbox": [[[100.0, 150.0, 200.0, 250.0]]],  # One bbox: x1,y1,x2,y2
            "image_shape": [[300, 400, 3]],  # height=300, width=400
        },
        schema=pl.Schema({"bbox": pl.List(pl.Array(pl.Float32, 4)), "image_shape": pl.List(pl.Int64())}),
    )

    # Set up converter for absolute to normalized conversion
    input_bbox_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)
    output_bbox_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=True)
    input_image_field = ImageField(dtype=pl.UInt8(), format="RGB")

    setattr(
        converter_instance,
        "input_bbox",
        AttributeSpec(name="bbox", field=input_bbox_field),
    )
    setattr(
        converter_instance,
        "output_bbox",
        AttributeSpec(name="bbox", field=output_bbox_field),
    )
    setattr(
        converter_instance,
        "input_image",
        AttributeSpec(name="image", field=input_image_field),
    )

    # Test filter - should return True for normalization change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "bbox" in result_df.columns
    result_bbox = result_df["bbox"][0][0]  # First bbox

    # Check normalization: divide by width for x coords, height for y coords
    # x1: 100/400 = 0.25, y1: 150/300 = 0.5, x2: 200/400 = 0.5, y2: 250/300 = 0.833...
    expected = [100 / 400, 150 / 300, 200 / 400, 250 / 300]
    assert np.allclose(result_bbox.to_numpy(), expected)


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


def test_find_conversion_path():
    """Test the find_conversion_path function."""

    # Create simple source and target schemas
    source_schema = Schema(
        attributes={"image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.UInt8(), format="RGB"))}
    )

    target_schema = Schema(
        attributes={"image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.Float32(), format="RGB"))}
    )

    # This should find a conversion path (UInt8 -> Float32)
    path, _ = find_conversion_path(source_schema, target_schema)
    assert len(path.converters["image"]) == 1
    assert type(path.converters["image"][0]) is UInt8ToFloat32Converter


def test_convert_dataframe():
    """Test getting conversion path and applying it manually."""

    # Create test DataFrame
    df = pl.DataFrame(
        {"image": [[255, 0, 0, 0, 255, 0]], "image_shape": [[2, 3]]},
        schema=pl.Schema({"image": pl.List(pl.UInt8()), "image_shape": pl.List(pl.Int64)}),
    )

    source_schema = Schema(
        attributes={"image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.UInt8(), format="RGB"))}
    )

    target_schema = Schema(
        attributes={"image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.Float32(), format="BGR"))}
    )

    # Get conversion path and apply it manually
    conversion_paths, _ = find_conversion_path(source_schema, target_schema)

    # Apply batch converters first
    result_df = df
    for conv in conversion_paths.converters["image"]:
        result_df = conv.convert(result_df)

    # For this test with identical schemas, there should be no converters needed
    # or the result should be equivalent to the input
    assert result_df is not None

    image = result_df["image"][0].to_numpy()
    assert np.all(image == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def test_lazy_converter():
    """Test lazy converter functionality."""
    registry = ConverterRegistry()
    lazy_converters = [c for c in registry.list_converters() if c.lazy]

    # Should have at least one lazy converter (ImagePathToImageConverter)
    assert len(lazy_converters) > 0

    # Check that ImagePathToImageConverter is marked as lazy
    image_path_converters = [c for c in lazy_converters if issubclass(c, ImagePathToImageConverter) and c.lazy]
    assert len(image_path_converters) > 0


def test_converter_with_auxiliary_fields():
    """Test converters that require auxiliary fields."""
    # This would test converters like bbox normalization that need image size
    # The exact implementation depends on how auxiliary fields are handled

    converter_instance = BBoxCoordinateConverter()  # type: ignore[call-arg]

    # BBox converter needs image data as auxiliary
    input_bbox_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)
    output_bbox_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=True)
    input_image_field = ImageField(dtype=pl.UInt8(), format="RGB")

    setattr(
        converter_instance,
        "input_bbox",
        AttributeSpec(name="bbox", field=input_bbox_field),
    )
    setattr(
        converter_instance,
        "output_bbox",
        AttributeSpec(name="bbox", field=output_bbox_field),
    )
    setattr(
        converter_instance,
        "input_image",
        AttributeSpec(name="image", field=input_image_field),
    )

    # Should require auxiliary image data for bbox normalization
    assert hasattr(converter_instance, "input_image")


def test_multiple_converter_chaining():
    """Test that multiple converters can be chained together."""
    # This tests the A* search functionality for finding conversion paths

    # Create a complex conversion scenario
    source_schema = Schema(
        attributes={
            "image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.UInt8(), format="RGB")),
            "bbox": AttributeInfo(
                type=np.ndarray,
                field=bbox_field(dtype=pl.Float32(), normalize=False),
            ),
        }
    )

    target_schema = Schema(
        attributes={
            "image": AttributeInfo(type=np.ndarray, field=image_field(dtype=pl.Float32(), format="BGR")),
            "bbox": AttributeInfo(type=np.ndarray, field=bbox_field(dtype=pl.Float32(), normalize=True)),
        }
    )

    # This would require multiple conversions:
    # 1. RGB -> BGR (format)
    # 2. UInt8 -> Float32 (dtype)
    # 3. absolute -> normalized bbox (with image as auxiliary)

    path, _ = find_conversion_path(source_schema, target_schema)
    # If successful, should have multiple steps
    assert len(path.converters["image"]) == 2
    assert type(path.converters["image"][0]) is UInt8ToFloat32Converter
    assert type(path.converters["image"][1]) is RedBlueColorConverter

    # FIXME(gdlg): the BBoxCoordinateConverter needs an image
    # and it does not matter if the image is 8 bits or 32 bits,
    # so converting the image then the bbox is correct, hence the dependency.
    # The problem is that this is not desirable as it creates a spurious dependency
    # on the 32 bits conversion even though it is not needed.
    # To fix this, we need to adjust the weights to favour applying image conversions
    # after the bbox ones.
    assert len(path.converters["bbox"]) == 2
    assert type(path.converters["bbox"][0]) is UInt8ToFloat32Converter
    assert type(path.converters["bbox"][1]) is BBoxCoordinateConverter


def test_astar_direct_conversion():
    """Test direct conversion without chaining using A* search."""
    from dataclasses import dataclass

    # Mock field types for testing
    @dataclass(frozen=True)
    class TestImageField(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Binary, init=False)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Binary()}

    @dataclass(frozen=True)
    class TestImageSizeField(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Int32, init=False)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {f"{name}_width": pl.Int32(), f"{name}_height": pl.Int32()}

    # Register test converter as a class
    @converter
    class ExtractImageSizeConverter(Converter):
        input_image: AttributeSpec[TestImageField]
        output_image_size: AttributeSpec[TestImageSizeField]

        def filter_output_spec(self) -> bool:
            return True

        def convert(self, df: pl.DataFrame) -> pl.DataFrame:
            """Extract image size from image data."""
            return df.with_columns(
                [
                    pl.lit(640).alias("image_size_width"),
                    pl.lit(480).alias("image_size_height"),
                ]
            )

    # Create schemas
    from_schema = Schema(
        {
            "image": AttributeInfo(type=bytes, field=TestImageField()),
        }
    )

    to_schema = Schema(
        {
            "image_size": AttributeInfo(type=tuple, field=TestImageSizeField()),
        }
    )

    # Test finding conversion path
    path, _ = find_conversion_path(from_schema, to_schema)
    assert len(path.converters["image_size"]) == 1
    assert type(path.converters["image_size"][0]) is ExtractImageSizeConverter


def test_astar_chained_conversion():
    """Test chained conversion using A* search."""
    from dataclasses import dataclass

    # Mock field types for testing
    @dataclass(frozen=True)
    class TestImageField(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Binary, init=False)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Binary()}

    @dataclass(frozen=True)
    class TestImageSizeField(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Int32, init=False)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {f"{name}_width": pl.Int32(), f"{name}_height": pl.Int32()}

    @dataclass(frozen=True)
    class TestNormalizedBboxField(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Float32, init=False)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.List(pl.Float32())}

    @dataclass(frozen=True)
    class TestAbsoluteBboxField(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Int32, init=False)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.List(pl.Int32())}

    # Register test converters as classes
    @converter
    class ExtractImageSizeChainConverter(Converter):
        input_image: AttributeSpec[TestImageField]
        output_image_size: AttributeSpec[TestImageSizeField]

        def filter_output_spec(self) -> bool:
            return True

        def convert(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                [
                    pl.lit(640).alias("image_size_width"),
                    pl.lit(480).alias("image_size_height"),
                ]
            )

    @converter
    class NormalizeToAbsoluteBboxConverter(Converter):
        input_bbox: AttributeSpec[TestNormalizedBboxField]
        input_image_size: AttributeSpec[TestImageSizeField]
        output_absolute_bbox: AttributeSpec[TestAbsoluteBboxField]

        def filter_output_spec(self) -> bool:
            return True

        def convert(self, df: pl.DataFrame) -> pl.DataFrame:
            """Convert normalized bbox to absolute coordinates."""
            width = df.select("image_size_width").item()
            height = df.select("image_size_height").item()

            return df.with_columns(
                [
                    pl.col("bbox")
                    .list.eval(
                        pl.when(pl.int_range(pl.len()) % 2 == 0)
                        .then(pl.element() * width)
                        .otherwise(pl.element() * height)
                    )
                    .alias("absolute_bbox")
                ]
            ).drop("bbox")

    # Create schemas that require chaining
    from_schema = Schema(
        {
            "image": AttributeInfo(type=bytes, field=TestImageField()),
            "bbox": AttributeInfo(type=list, field=TestNormalizedBboxField()),
        }
    )

    to_schema = Schema(
        {
            "absolute_bbox": AttributeInfo(type=list, field=TestAbsoluteBboxField()),
        }
    )

    # Test finding conversion path
    path, _ = find_conversion_path(from_schema, to_schema)
    # Should need 2 converters: ImageField -> ImageSizeField, then NormalizedBbox -> AbsoluteBbox
    assert len(path.converters["absolute_bbox"]) == 2
    assert type(path.converters["absolute_bbox"][0]) is ExtractImageSizeChainConverter
    assert type(path.converters["absolute_bbox"][1]) is NormalizeToAbsoluteBboxConverter


def test_astar_no_conversion_needed():
    """Test when no conversion is needed using A* search."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class TestField(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Utf8, init=False)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    # Identical schemas - no conversion needed
    schema = Schema(
        {
            "data": AttributeInfo(type=str, field=TestField()),
        }
    )

    path, _ = find_conversion_path(schema, schema)
    assert len(path.converters) == 0, f"Expected 0 converters for identical schemas, got {len(path.converters)}"


def test_astar_impossible_conversion():
    """Test conversion that should fail using A* search."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FieldA(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Utf8)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @dataclass(frozen=True)
    class FieldB(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Utf8)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    # No converters registered between these types - should fail
    from_schema = Schema(
        {
            "field_a": AttributeInfo(type=str, field=FieldA()),
        }
    )

    to_schema = Schema(
        {
            "field_b": AttributeInfo(type=str, field=FieldB()),
        }
    )

    # Should raise ConversionError
    with pytest.raises(ConversionError):
        find_conversion_path(from_schema, to_schema)


def test_optimal_path_selection():
    """Test that A* chooses the shortest path when multiple paths exist."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FieldA(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Utf8)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @dataclass(frozen=True)
    class FieldB(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Utf8)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @dataclass(frozen=True)
    class FieldC(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Utf8, init=False)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    # Register converters: A->B, B->C, and A->C (direct path)
    @converter
    class AToBConverter(Converter):  # pyright: ignore [reportUnusedClass]
        input_field_a: AttributeSpec[FieldA]
        output_field_b: AttributeSpec[FieldB]

        def filter_output_spec(self) -> bool:
            return True

        def convert(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.col("field_a").alias("field_b")).drop("field_a")

    @converter
    class BToCConverter(Converter):  # pyright: ignore [reportUnusedClass]
        input_field_b: AttributeSpec[FieldB]
        output_field_c: AttributeSpec[FieldC]

        def filter_output_spec(self) -> bool:
            return True

        def convert(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.col("field_b").alias("field_c")).drop("field_b")

    @converter  # Direct path (should be preferred)
    class AToCDirectConverter(Converter):
        input_field_a: AttributeSpec[FieldA]
        output_field_c: AttributeSpec[FieldC]

        def filter_output_spec(self) -> bool:
            return True

        def convert(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.col("field_a").alias("field_c")).drop("field_a")

    from_schema = Schema(
        {
            "field_a": AttributeInfo(type=str, field=FieldA()),
        }
    )

    to_schema = Schema(
        {
            "field_c": AttributeInfo(type=str, field=FieldC()),
        }
    )

    path, _ = find_conversion_path(from_schema, to_schema)
    assert len(path.converters["field_c"]) == 1
    assert type(path.converters["field_c"][0]) is AToCDirectConverter


def test_generator_converter():
    """Test converter that generates fields from nothing."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FieldB(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Utf8)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @converter  # Generator converter (no inputs)
    class GenerateBConverter(Converter):
        output_field_b: AttributeSpec[FieldB]

        def filter_output_spec(self) -> bool:
            return True

        def convert(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.lit("generated_b").alias("field_b"))

    from_schema = Schema({})  # Empty schema

    to_schema = Schema(
        {
            "field_b": AttributeInfo(type=str, field=FieldB()),
        }
    )

    path, _ = find_conversion_path(from_schema, to_schema)
    assert len(path.converters["field_b"]) == 1
    assert type(path.converters["field_b"][0]) is GenerateBConverter


def test_multiple_output_converter():
    """Test converters that produce multiple outputs."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FieldA(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Utf8)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @dataclass(frozen=True)
    class MultiField1(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Utf8, init=False)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @dataclass(frozen=True)
    class MultiField2(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Utf8, init=False)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @converter
    class AToMultiConverter(Converter):
        input_field_a: AttributeSpec[FieldA]
        output_multi1: AttributeSpec[MultiField1]
        output_multi2: AttributeSpec[MultiField2]

        def filter_output_spec(self) -> bool:
            return True

        def convert(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns([pl.col("field_a").alias("multi1"), pl.col("field_a").alias("multi2")]).drop(
                "field_a"
            )

    from_schema = Schema(
        {
            "field_a": AttributeInfo(type=str, field=FieldA()),
        }
    )

    to_schema = Schema(
        {
            "multi1": AttributeInfo(type=str, field=MultiField1()),
            "multi2": AttributeInfo(type=str, field=MultiField2()),
        }
    )

    path, _ = find_conversion_path(from_schema, to_schema)
    assert len(path.converters["multi1"]) == 1
    assert type(path.converters["multi1"][0]) is AToMultiConverter
    assert len(path.converters["multi2"]) == 1
    assert type(path.converters["multi2"][0]) is AToMultiConverter


def test_partial_schema_matching():
    """Test when target schema is a subset of what converters can produce."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FieldA(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Utf8)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @dataclass(frozen=True)
    class FieldC(Field):
        semantic: str = "default"
        dtype: pl.DataType = field(default_factory=pl.Utf8, init=False)

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @converter
    class AToCPartialConverter(Converter):
        input_field_a: AttributeSpec[FieldA]
        output_field_c: AttributeSpec[FieldC]

        def filter_output_spec(self) -> bool:
            return True

        def convert(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(pl.col("field_a").alias("field_c")).drop("field_a")

    from_schema = Schema(
        {
            "field_a": AttributeInfo(type=str, field=FieldA()),
            "other_field": AttributeInfo(
                type=str, field=FieldA(semantic="bbox")
            ),  # Extra field with different semantic
        }
    )

    to_schema = Schema(
        {
            "field_c": AttributeInfo(type=str, field=FieldC()),
            # Only requesting field_c, not other_field
        }
    )

    path, _ = find_conversion_path(from_schema, to_schema)
    assert len(path.converters["field_c"]) == 1
    assert type(path.converters["field_c"][0]) is AToCPartialConverter


def test_attribute_renaming():
    """Test attribute renaming functionality using special converters."""

    # Create source schema with an image field named "input_image"
    source_schema = Schema(
        {
            "input_image": AttributeInfo(type=str, field=image_field(dtype=pl.UInt8(), format="RGB")),
        }
    )

    # Create target schema with the same field type but different name "output_image"
    target_schema = Schema(
        {
            "output_image": AttributeInfo(type=str, field=image_field(dtype=pl.UInt8(), format="RGB")),
        }
    )

    # Find conversion path - should include a remapper converter
    path, _ = find_conversion_path(source_schema, target_schema)

    # Should have exactly one batch converter for renaming
    assert len(path.converters["output_image"]) == 1
    assert isinstance(path.converters["output_image"][0], AttributeRemapperConverter)

    # Test the remapper converter functionality
    remapper_converter = path.converters["output_image"][0]

    # Create test DataFrame with the source column
    test_data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    df = pl.DataFrame({"input_image": [test_data.flatten()], "input_image_shape": [test_data.shape]})

    # Apply the remapper converter
    result_df = remapper_converter.convert(df)

    # Check that columns were renamed correctly
    assert "output_image" in result_df.columns
    assert "output_image_shape" in result_df.columns
    assert "input_image" not in result_df.columns
    assert "input_image_shape" not in result_df.columns

    # Check that data was preserved
    assert result_df["output_image"][0].to_list() == test_data.flatten().tolist()
    assert result_df["output_image_shape"][0].to_list() == list(test_data.shape)


def test_combined_rename_and_delete():
    """Test scenario with both renaming and deletion operations."""

    # Create source schema with three fields
    source_schema = Schema(
        {
            "old_image": AttributeInfo(type=str, field=image_field(dtype=pl.UInt8(), format="RGB")),
            "bbox": AttributeInfo(type=str, field=bbox_field(dtype=pl.Float32(), format="x1y1x2y2")),
            "extra_field": AttributeInfo(type=str, field=image_info_field()),
        }
    )

    # Create target schema with renamed image and no bbox or extra_field
    target_schema = Schema(
        {
            "new_image": AttributeInfo(type=str, field=image_field(dtype=pl.UInt8(), format="RGB")),
        }
    )

    # Find conversion path - should include a single remapper converter that handles both operations
    path, _ = find_conversion_path(source_schema, target_schema)

    # Should have exactly one batch converter that handles both renaming and deletion
    assert len(path.converters["new_image"]) == 1
    assert isinstance(path.converters["new_image"][0], AttributeRemapperConverter)

    # Check that the remapper handles the conversion correctly by testing on sample data
    remapper_converter = path.converters["new_image"][0]

    # Create test DataFrame
    test_df = pl.DataFrame(
        {
            "old_image": [[1, 2, 3, 4, 5, 6]],  # Sample image data
            "old_image_shape": [[2, 3]],  # Sample shape data
            "other_field": ["should_not_change"],  # Should not be changed
        }
    )

    # Apply the converter
    result_df = remapper_converter.convert(test_df)

    # Check that only the renamed fields and the unmodified ones are present
    expected_columns = {"new_image", "new_image_shape", "other_field"}
    assert set(result_df.columns) == expected_columns

    # Check that data is preserved correctly
    assert result_df["new_image"].to_list() == [[1, 2, 3, 4, 5, 6]]
    assert result_df["new_image_shape"].to_list() == [[2, 3]]


def test_attribute_remapping_converter():
    """Test AttributeRemapperConverter with multiple attributes."""
    import polars as pl

    # Create test DataFrame with multiple columns including auxiliary columns
    df = pl.DataFrame(
        {
            "old_image": [[1, 2, 3]],
            "old_image_shape": [[3]],
            "old_bbox": [[[1, 2, 3, 4]]],
            "other_field": ["test"],
        }
    )

    # Create attribute mappings using AttributeSpec
    from_image_attr = AttributeSpec(name="old_image", field=image_field(dtype=pl.Int32()))
    to_image_attr = AttributeSpec(name="new_image", field=image_field(dtype=pl.Int32()))

    from_bbox_attr = AttributeSpec(name="old_bbox", field=bbox_field(dtype=pl.Int32()))
    to_bbox_attr = AttributeSpec(name="new_bbox", field=bbox_field(dtype=pl.Int32()))

    attr_mappings = [(from_image_attr, to_image_attr), (from_bbox_attr, to_bbox_attr)]

    # Create and apply converter
    converter = AttributeRemapperConverter(attr_mappings=attr_mappings)
    result_df = converter.convert(df)

    # Check that columns were remapped correctly
    expected_columns = {"new_image", "new_image_shape", "new_bbox", "other_field"}
    assert set(result_df.columns) == expected_columns

    # Check that data is preserved
    assert result_df["new_image"].to_list() == [[1, 2, 3]]
    assert result_df["new_image_shape"].to_list() == [[3]]
    assert result_df["new_bbox"].to_list() == [[[1, 2, 3, 4]]]


def test_polygon_to_mask_converter():
    """Test conversion from polygon coordinates to mask format."""
    converter_instance = PolygonToMaskConverter()  # type: ignore[call-arg]

    # Create test data with polygon coordinates and labels
    # Triangle polygon: (10,10) -> (30,10) -> (20,30) -> (10,10)
    polygon_coords1 = [[10.0, 10.0], [30.0, 10.0], [20.0, 30.0]]

    # Rectangle polygon: (40,40) -> (60,40) -> (60,60) -> (40,60) -> (40,40)
    polygon_coords2 = [[40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0], [40.0, 40.0]]

    # Pentagon polygon: (70,10) -> (85,5) -> (90,20) -> (80,35) -> (65,25)
    polygon_coords3 = [[70.0, 10.0], [85.0, 5.0], [90.0, 20.0], [80.0, 35.0], [65.0, 25.0]]

    polygon_series = pl.Series(
        [polygon_coords1, polygon_coords2, polygon_coords3], dtype=pl.List(pl.Array(pl.Float32, 2))
    )

    df = pl.DataFrame(
        {
            "polygons": [polygon_series],  # List of three polygons
            "labels": [[0, 1, 2]],  # Corresponding labels for each polygon
            "image_info": [{"width": 100, "height": 100}],  # Image dimensions
        }
    )

    # Set up converter attributes
    input_polygon_field = PolygonField(dtype=pl.Float32(), format="xy", normalize=False)
    input_labels_field = LabelField(dtype=pl.Int32(), multi_label=True)
    image_info_field = ImageInfoField()
    output_mask_field = MaskField(dtype=pl.UInt8())

    setattr(
        converter_instance,
        "input_polygon",
        AttributeSpec(name="polygons", field=input_polygon_field),
    )
    setattr(
        converter_instance,
        "input_labels",
        AttributeSpec(name="labels", field=input_labels_field),
    )
    setattr(
        converter_instance,
        "input_image_info",
        AttributeSpec(name="image_info", field=image_info_field),
    )
    setattr(
        converter_instance,
        "output_mask",
        AttributeSpec(name="mask", field=output_mask_field),
    )

    # Test filter - should return True when we have valid input
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    # Check that mask column was created
    assert "mask" in result_df.columns
    assert "mask_shape" in result_df.columns

    # Get the mask data and reshape it
    mask_data = np.array(result_df["mask"][0])
    mask_shape = result_df["mask_shape"][0]
    mask = mask_data.reshape(mask_shape)

    # Check mask properties
    assert mask.shape == (100, 100)  # Should match image dimensions
    assert mask.dtype == np.uint8

    # Check that polygons were filled with correct labels
    # Triangle should have label 1, rectangle should have label 2, pentagon should have label 3
    # Background should be 0

    # Check that triangle area has label 0 (stored as mask value 1)
    assert mask[15, 20] == 1  # Point inside triangle

    # Check that rectangle area has label 1 (stored as mask value 2)
    assert mask[50, 50] == 2  # Point inside rectangle

    # Check that pentagon area has label 2 (stored as mask value 3)
    assert mask[20, 75] == 3  # Point inside pentagon

    # Check background area has label 0
    assert mask[5, 5] == 0  # Point outside all polygons
    assert mask[95, 95] == 0  # Another background point

    # Check that mask contains the expected label values
    unique_labels = np.unique(mask)
    assert 0 in unique_labels  # Background
    assert 1 in unique_labels  # First polygon label (triangle)
    assert 2 in unique_labels  # Second polygon label (rectangle)
    assert 3 in unique_labels  # Third polygon label (pentagon)


def test_polygon_to_mask_converter_normalized():
    """Test conversion with normalized polygon coordinates."""
    converter_instance = PolygonToMaskConverter()  # type: ignore[call-arg]

    # Create test data with normalized coordinates (0.0 to 1.0 range)
    # Small triangle in normalized coordinates
    polygon_coords = [[0.1, 0.1], [0.3, 0.1], [0.2, 0.3]]  # Normalized coordinates
    polygon_series = pl.Series([polygon_coords], dtype=pl.List(pl.Array(pl.Float32, 2)))

    df = pl.DataFrame(
        {
            "polygons": [polygon_series],
            "labels": [[5]],  # Label 5 for this polygon
            "image_info": [{"width": 100, "height": 100}],
        }
    )

    # Set up converter attributes with normalization enabled
    input_polygon_field = PolygonField(
        dtype=pl.Float32(),
        format="xy",
        normalize=True,  # Enable normalization
    )
    input_labels_field = LabelField(dtype=pl.Int32(), multi_label=True)
    image_info_field = ImageInfoField()
    output_mask_field = MaskField(dtype=pl.UInt8())

    setattr(
        converter_instance,
        "input_polygon",
        AttributeSpec(name="polygons", field=input_polygon_field),
    )
    setattr(converter_instance, "input_labels", AttributeSpec(name="labels", field=input_labels_field))
    setattr(
        converter_instance,
        "input_image_info",
        AttributeSpec(name="image_info", field=image_info_field),
    )
    setattr(converter_instance, "output_mask", AttributeSpec(name="mask", field=output_mask_field))

    # Test conversion
    result_df = converter_instance.convert(df)

    # Get the mask and check it
    mask_data = np.array(result_df["mask"][0])
    mask_shape = result_df["mask_shape"][0]
    mask = mask_data.reshape(mask_shape)

    # Check that polygon was filled with label 5 (stored as mask value 6)
    # Normalized coordinates should be scaled: 0.2 * 100 = 20, 0.1 * 100 = 10, etc.
    assert mask[15, 20] == 6  # Point inside the scaled triangle (5+1=6)
    assert mask[5, 5] == 0  # Background point


def test_find_conversion_path_inferred_categories():
    """Test that find_conversion_path returns inferred categories."""

    # Create test data for polygon to mask conversion
    pl.DataFrame(
        {
            "polygons": [[[10, 20, 30, 25, 20, 40]]],  # Triangle coordinates
            "labels": [[2]],  # Label 2
            "image_info": [{"width": 100, "height": 100}],
        },
        schema=pl.Schema(
            {
                "polygons": pl.List(pl.List(pl.Float32())),
                "labels": pl.List(pl.Int32()),
                "image_info": pl.Struct({"width": pl.Int32, "height": pl.Int32}),
            }
        ),
    )

    # Create source schema with label categories
    label_categories = LabelCategories(labels=("cat", "dog", "bird"))
    source_schema = Schema(
        attributes={
            "polygons": AttributeInfo(
                type=list,
                field=PolygonField(dtype=pl.Float32()),
                categories=None,
            ),
            "labels": AttributeInfo(
                type=list,
                field=LabelField(),
                categories=label_categories,
            ),
            "image_info": AttributeInfo(type=dict, field=ImageInfoField()),
        }
    )

    # Create target schema (polygon to mask conversion)
    target_schema = Schema(attributes={"mask": AttributeInfo(type=np.ndarray, field=MaskField(dtype=pl.UInt8()))})

    # Get conversion path and check inferred categories
    conversion_paths, inferred_categories = find_conversion_path(source_schema, target_schema)

    # Should have converters for polygon to mask conversion
    assert len(conversion_paths.converters) > 0

    # Check that mask categories were inferred
    assert "mask" in inferred_categories
    mask_categories = inferred_categories["mask"]
    assert isinstance(mask_categories, MaskCategories)

    # Check that the mask categories include background + original labels
    expected_labels = ("background", "cat", "dog", "bird")
    assert mask_categories.labels == expected_labels


def test_polygon_to_instance_mask_converter():
    """Test conversion from polygon coordinates to instance mask format."""
    # Create test data with triangle, rectangle, and pentagon polygons
    polygon_coords1 = [[10.0, 10.0], [20.0, 10.0], [15.0, 20.0]]
    polygon_coords2 = [[30.0, 30.0], [40.0, 30.0], [40.0, 40.0], [30.0, 40.0]]
    polygon_coords3 = [[50.0, 50.0], [60.0, 50.0], [65.0, 60.0], [55.0, 70.0], [45.0, 60.0]]

    polygon_series = pl.Series(
        [polygon_coords1, polygon_coords2, polygon_coords3], dtype=pl.List(pl.Array(pl.Float32, 2))
    )

    df = pl.DataFrame(
        {
            "polygons": [polygon_series],
            "image_info": [{"width": 100, "height": 100}],
        }
    )

    # Create converter instance
    converter_instance = PolygonToInstanceMaskConverter()

    # Set up field specs
    input_polygon_field = PolygonField(dtype=pl.Float32(), format="xy", normalize=False)
    image_info_field = ImageInfoField()
    output_instance_mask_field = InstanceMaskField(dtype=pl.Boolean())

    setattr(
        converter_instance,
        "input_polygon",
        AttributeSpec(name="polygons", field=input_polygon_field),
    )
    setattr(
        converter_instance,
        "input_image_info",
        AttributeSpec(name="image_info", field=image_info_field),
    )
    setattr(
        converter_instance,
        "output_instance_mask",
        AttributeSpec(name="instance_mask", field=output_instance_mask_field),
    )

    # Test filter - should return True when we have valid input
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    # Check that instance mask column was created
    assert "instance_mask" in result_df.columns
    assert "instance_mask_shape" in result_df.columns

    # Get the mask data and reshape it
    mask_data = np.array(result_df["instance_mask"][0])
    mask_shape = result_df["instance_mask_shape"][0]
    masks = mask_data.reshape(mask_shape)

    # Check mask properties
    assert masks.shape == (3, 100, 100)  # 3 instances, 100x100 image
    assert masks.dtype == bool

    # Check that each instance is properly filled
    # Triangle should be in first mask
    assert masks[0, 15, 15]  # Point inside triangle
    assert not masks[0, 5, 5]  # Point outside triangle

    # Rectangle should be in second mask
    assert masks[1, 35, 35]  # Point inside rectangle
    assert not masks[1, 5, 5]  # Point outside rectangle

    # Pentagon should be in third mask
    assert masks[2, 55, 55]  # Point inside pentagon
    assert not masks[2, 5, 5]  # Point outside pentagon

    # No overlap between instances
    assert not np.any(masks[0] & masks[1])  # Triangle and rectangle don't overlap
    assert not np.any(masks[0] & masks[2])  # Triangle and pentagon don't overlap
    assert not np.any(masks[1] & masks[2])  # Rectangle and pentagon don't overlap


def test_polygon_to_instance_mask_converter_normalized():
    """Test conversion with normalized polygon coordinates."""
    # Create test data with normalized coordinates (0-1 range)
    polygon_coords1 = [[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]]
    polygon_coords2 = [[0.3, 0.3], [0.4, 0.3], [0.4, 0.4], [0.3, 0.4]]

    polygon_series = pl.Series([polygon_coords1, polygon_coords2], dtype=pl.List(pl.Array(pl.Float32, 2)))

    df = pl.DataFrame(
        {
            "polygons": [polygon_series],
            "image_info": [{"width": 100, "height": 100}],
        }
    )

    # Create converter instance with normalized coordinates
    converter_instance = PolygonToInstanceMaskConverter()

    # Set up field specs
    input_polygon_field = PolygonField(dtype=pl.Float32(), format="xy", normalize=True)
    image_info_field = ImageInfoField()
    output_instance_mask_field = InstanceMaskField(dtype=pl.Boolean())

    setattr(
        converter_instance,
        "input_polygon",
        AttributeSpec(name="polygons", field=input_polygon_field),
    )
    setattr(
        converter_instance,
        "input_image_info",
        AttributeSpec(name="image_info", field=image_info_field),
    )
    setattr(
        converter_instance,
        "output_instance_mask",
        AttributeSpec(name="instance_mask", field=output_instance_mask_field),
    )

    # Test conversion
    result_df = converter_instance.convert(df)

    # Get the mask and check it
    mask_data = np.array(result_df["instance_mask"][0])
    mask_shape = result_df["instance_mask_shape"][0]
    masks = mask_data.reshape(mask_shape)

    # Check mask properties
    assert masks.shape == (2, 100, 100)

    # Check that polygons were filled correctly after denormalization
    # Triangle: 0.1 * 100 = 10, 0.2 * 100 = 20, etc.
    assert masks[0, 15, 15]  # Point inside the scaled triangle
    assert not masks[0, 5, 5]  # Background point

    # Rectangle: 0.3 * 100 = 30, 0.4 * 100 = 40, etc.
    assert masks[1, 35, 35]  # Point inside the scaled rectangle
    assert not masks[1, 5, 5]  # Background point


def test_instance_mask_callable_to_instance_mask_converter():
    """Test InstanceMaskCallableToInstanceMaskConverter conversion."""
    converter_instance = InstanceMaskCallableToInstanceMaskConverter()  # type: ignore[call-arg]

    # Create a test callable that returns instance masks
    def get_instance_masks():
        return np.array([[[True, False], [False, True]], [[False, True], [True, False]]], dtype=bool)  # (2,2,2)

    df = pl.DataFrame(
        {
            "instance_mask_callable": [get_instance_masks],
        },
        schema=pl.Schema({"instance_mask_callable": pl.Object}),
    )

    # Set up converter attributes
    input_field = InstanceMaskCallableField(dtype=pl.Boolean())
    output_field = InstanceMaskField(dtype=pl.Boolean())

    setattr(
        converter_instance,
        "input_callable",
        AttributeSpec(
            name="instance_mask_callable",
            field=input_field,
        ),
    )
    setattr(
        converter_instance,
        "output_mask",
        AttributeSpec(
            name="instance_mask",
            field=output_field,
        ),
    )

    # Convert
    result_df = converter_instance.convert(df)

    # Check result
    assert "instance_mask" in result_df.columns
    assert "instance_mask_shape" in result_df.columns

    # Verify shape
    expected_shape = [2, 2, 2]  # N, H, W
    assert result_df["instance_mask_shape"][0].to_list() == expected_shape

    # Check instance masks
    expected_masks = get_instance_masks()
    result_masks = np.array(result_df["instance_mask"][0]).reshape(expected_shape)
    assert np.array_equal(result_masks, expected_masks)


def test_instance_mask_callable_to_instance_mask_converter_validation():
    """Test validation in InstanceMaskCallableToInstanceMaskConverter."""
    converter_instance = InstanceMaskCallableToInstanceMaskConverter()  # type: ignore[call-arg]

    # Create an invalid test callable that returns wrong shape
    def get_invalid_masks():
        return np.array([[True, False], [False, True]], dtype=bool)  # 2D instead of 3D

    df = pl.DataFrame(
        {
            "instance_mask_callable": [get_invalid_masks],
        },
        schema=pl.Schema({"instance_mask_callable": pl.Object}),
    )

    # Set up converter attributes
    input_field = InstanceMaskCallableField(dtype=pl.Boolean())
    output_field = InstanceMaskField(dtype=pl.Boolean())

    setattr(
        converter_instance,
        "input_callable",
        AttributeSpec(
            name="instance_mask_callable",
            field=input_field,
        ),
    )
    setattr(
        converter_instance,
        "output_mask",
        AttributeSpec(
            name="instance_mask",
            field=output_field,
        ),
    )

    # Conversion should raise error due to wrong shape
    with pytest.raises(ValueError):
        converter_instance.convert(df)


def test_mask_callable_to_mask_converter():
    """Test MaskCallableToMaskConverter conversion."""
    converter_instance = MaskCallableToMaskConverter()  # type: ignore[call-arg]

    # Create a test callable that returns a mask with category IDs
    def get_mask():
        return np.array([[1, 2], [2, 1]], dtype=np.uint8)  # (2,2)

    df = pl.DataFrame(
        {
            "mask_callable": [get_mask],
        },
        schema=pl.Schema({"mask_callable": pl.Object}),
    )

    # Set up converter attributes
    input_field = MaskCallableField(dtype=pl.UInt8())
    output_field = MaskField(dtype=pl.UInt8())

    setattr(
        converter_instance,
        "input_callable",
        AttributeSpec(
            name="mask_callable",
            field=input_field,
        ),
    )
    setattr(
        converter_instance,
        "output_mask",
        AttributeSpec(
            name="mask",
            field=output_field,
        ),
    )

    # Convert
    result_df = converter_instance.convert(df)

    # Check result
    assert "mask" in result_df.columns
    assert "mask_shape" in result_df.columns

    # Verify shape
    expected_shape = [2, 2]  # H, W
    assert result_df["mask_shape"][0].to_list() == expected_shape

    # Check mask
    expected_mask = get_mask()
    result_mask = np.array(result_df["mask"][0]).reshape(expected_shape)
    assert np.array_equal(result_mask, expected_mask)


def test_mask_callable_to_mask_converter_validation():
    """Test validation in MaskCallableToMaskConverter."""
    converter_instance = MaskCallableToMaskConverter()  # type: ignore[call-arg]

    # Create an invalid test callable that returns wrong shape
    def get_invalid_mask():
        return np.array([[[True, False], [False, True]]], dtype=np.uint8)

    df = pl.DataFrame(
        {
            "mask_callable": [get_invalid_mask],
        },
        schema=pl.Schema({"mask_callable": pl.Object}),
    )

    # Set up converter attributes
    input_field = MaskCallableField(dtype=pl.Boolean())
    output_field = InstanceMaskField(dtype=pl.Boolean())

    setattr(
        converter_instance,
        "input_callable",
        AttributeSpec(
            name="mask_callable",
            field=input_field,
        ),
    )
    setattr(
        converter_instance,
        "output_mask",
        AttributeSpec(
            name="mask",
            field=output_field,
        ),
    )

    # Check that it raises error for invalid shape
    with pytest.raises(ValueError, match="Mask array must be 2D \(H,W\), got shape \(1, 2, 2\)"):
        converter_instance.convert(df)


def test_polygon_to_bbox_converter():
    """Test conversion from polygon coordinates to bounding box format."""
    # Create test data with triangle and rectangle polygons
    polygon_coords1 = [[10.0, 10.0], [20.0, 10.0], [15.0, 20.0]]
    polygon_coords2 = [[30.0, 30.0], [40.0, 30.0], [40.0, 40.0], [30.0, 40.0]]

    polygon_series = pl.Series([polygon_coords1, polygon_coords2], dtype=pl.List(pl.Array(pl.Float32, 2)))

    df = pl.DataFrame(
        {
            "polygons": [polygon_series],
        }
    )

    # Create converter instance
    converter_instance = PolygonToBBoxConverter()

    # Set up field specs
    input_polygon_field = PolygonField(dtype=pl.Float32(), format="xy", normalize=False)
    output_bbox_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)

    setattr(
        converter_instance,
        "input_polygon",
        AttributeSpec(name="polygons", field=input_polygon_field),
    )
    setattr(
        converter_instance,
        "output_bbox",
        AttributeSpec(name="bboxes", field=output_bbox_field),
    )

    # Test filter - should return True when we have valid input
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    # Check that bbox column was created
    assert "bboxes" in result_df.columns

    # Get the bbox data
    bboxes = result_df["bboxes"][0]

    # Check that we have 2 bounding boxes (triangle and rectangle)
    assert len(bboxes) == 2

    # Check triangle bbox (x1y1x2y2 format)
    triangle_bbox = bboxes[0]
    assert triangle_bbox[0] == 10.0  # x1 (min x)
    assert triangle_bbox[1] == 10.0  # y1 (min y)
    assert triangle_bbox[2] == 20.0  # x2 (max x)
    assert triangle_bbox[3] == 20.0  # y2 (max y)

    # Check rectangle bbox
    rectangle_bbox = bboxes[1]
    assert rectangle_bbox[0] == 30.0  # x1
    assert rectangle_bbox[1] == 30.0  # y1
    assert rectangle_bbox[2] == 40.0  # x2
    assert rectangle_bbox[3] == 40.0  # y2


def test_polygon_to_bbox_converter_xywh():
    """Test conversion to xywh bbox format."""
    # Create test data with rectangle polygon
    polygon_coords = [[30.0, 30.0], [40.0, 30.0], [40.0, 40.0], [30.0, 40.0]]

    polygon_series = pl.Series([polygon_coords], dtype=pl.List(pl.Array(pl.Float32, 2)))

    df = pl.DataFrame(
        {
            "polygons": [polygon_series],
        }
    )

    # Create converter instance
    converter_instance = PolygonToBBoxConverter()

    # Set up field specs for xywh format
    input_polygon_field = PolygonField(dtype=pl.Float32(), format="xy", normalize=False)
    output_bbox_field = BBoxField(dtype=pl.Float32(), format="xywh", normalize=False)

    setattr(
        converter_instance,
        "input_polygon",
        AttributeSpec(name="polygons", field=input_polygon_field),
    )
    setattr(
        converter_instance,
        "output_bbox",
        AttributeSpec(name="bboxes", field=output_bbox_field),
    )

    # Test conversion
    result_df = converter_instance.convert(df)

    # Get the bbox data
    bboxes = result_df["bboxes"][0]

    # Check rectangle bbox in xywh format
    rectangle_bbox = bboxes[0]
    assert rectangle_bbox[0] == 30.0  # x (min x)
    assert rectangle_bbox[1] == 30.0  # y (min y)
    assert rectangle_bbox[2] == 10.0  # w (width)
    assert rectangle_bbox[3] == 10.0  # h (height)


def test_polygon_to_bbox_converter_normalized():
    """Test conversion with normalized polygon coordinates."""
    # Create test data with normalized coordinates (0-1 range)
    polygon_coords = [[0.3, 0.3], [0.4, 0.3], [0.4, 0.4], [0.3, 0.4]]

    polygon_series = pl.Series([polygon_coords], dtype=pl.List(pl.Array(pl.Float32, 2)))

    df = pl.DataFrame(
        {
            "polygons": [polygon_series],
        }
    )

    # Create converter instance
    converter_instance = PolygonToBBoxConverter()

    # Set up field specs with normalized coordinates
    input_polygon_field = PolygonField(dtype=pl.Float32(), format="xy", normalize=True)
    output_bbox_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=True)

    setattr(
        converter_instance,
        "input_polygon",
        AttributeSpec(name="polygons", field=input_polygon_field),
    )
    setattr(
        converter_instance,
        "output_bbox",
        AttributeSpec(name="bboxes", field=output_bbox_field),
    )

    # Test conversion
    result_df = converter_instance.convert(df)

    # Get the bbox data
    bboxes = result_df["bboxes"][0]

    # Check rectangle bbox with normalized coordinates
    rectangle_bbox = bboxes[0]
    assert abs(rectangle_bbox[0] - 0.3) < 1e-6  # x1 (normalized)
    assert abs(rectangle_bbox[1] - 0.3) < 1e-6  # y1 (normalized)
    assert abs(rectangle_bbox[2] - 0.4) < 1e-6  # x2 (normalized)
    assert abs(rectangle_bbox[3] - 0.4) < 1e-6  # y2 (normalized)


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


def test_label_index_converter():
    """Test LabelIndexConverter functionality for remapping label indices."""

    # Create input and output specs with different label orders
    input_categories = LabelCategories(labels=("cat", "dog", "bird"))
    output_categories = LabelCategories(labels=("bird", "cat", "dog"))  # Different order

    input_spec = AttributeSpec(
        name="label",
        field=LabelField(dtype=pl.Int32(), multi_label=False),
        categories=input_categories,
    )

    output_spec = AttributeSpec(
        name="label",
        field=LabelField(dtype=pl.Int32(), multi_label=False),
        categories=output_categories,
    )

    # Create converter
    converter = LabelIndexConverter(input_labels=input_spec, output_labels=output_spec)

    # Test filter - should return True for valid category remapping
    assert converter.filter_output_spec() is True

    # Test data with original label indices
    test_df = pl.DataFrame({"label": [0, 1, 2, 0, 1]})  # cat=0, dog=1, bird=2 in input

    # Convert
    result_df = converter.convert(test_df)

    # Verify the mapping: cat(0->1), dog(1->2), bird(2->0)
    expected = [1, 2, 0, 1, 2]  # cat=1, dog=2, bird=0 in output
    actual = result_df["label"].to_list()

    assert actual == expected


def test_label_index_converter_multi_label():
    """Test LabelIndexConverter functionality for multi-label scenarios."""

    # Create input and output specs with different label orders
    input_categories = LabelCategories(labels=("cat", "dog", "bird"))
    output_categories = LabelCategories(labels=("bird", "cat", "dog"))  # Different order

    input_spec = AttributeSpec(
        name="labels",
        field=LabelField(dtype=pl.Int32(), multi_label=True),
        categories=input_categories,
    )

    output_spec = AttributeSpec(
        name="labels",
        field=LabelField(dtype=pl.Int32(), multi_label=True),
        categories=output_categories,
    )

    # Create converter
    converter = LabelIndexConverter(input_labels=input_spec, output_labels=output_spec)

    # Test filter - should return True for valid category remapping
    assert converter.filter_output_spec() is True

    # Test multi-label data
    test_df = pl.DataFrame({"labels": [[0, 1], [2], [0, 2], [1]]})  # Multiple labels per row

    # Convert
    result_df = converter.convert(test_df)

    # Verify multi-label mapping
    expected = [[1, 2], [0], [1, 0], [2]]
    actual = result_df["labels"].to_list()

    assert actual == expected


def test_label_index_converter_same_categories():
    """Test LabelIndexConverter with identical categories (should not apply)."""

    # Create identical input and output categories
    categories = LabelCategories(labels=("cat", "dog", "bird"))

    input_spec = AttributeSpec(
        name="label",
        field=LabelField(dtype=pl.Int32(), multi_label=False),
        categories=categories,
    )

    output_spec = AttributeSpec(
        name="label",
        field=LabelField(dtype=pl.Int32(), multi_label=False),
        categories=categories,
    )

    # Create converter
    converter = LabelIndexConverter(input_labels=input_spec, output_labels=output_spec)

    # Test filter - should return False for identical categories
    assert converter.filter_output_spec() is False


def test_label_index_converter_different_labels():
    """Test LabelIndexConverter with different label sets (should not apply)."""

    # Create categories with different label sets
    input_categories = LabelCategories(labels=("cat", "dog", "bird"))
    output_categories = LabelCategories(labels=("horse", "cow", "sheep"))  # Different labels

    input_spec = AttributeSpec(
        name="label",
        field=LabelField(dtype=pl.Int32(), multi_label=False),
        categories=input_categories,
    )

    output_spec = AttributeSpec(
        name="label",
        field=LabelField(dtype=pl.Int32(), multi_label=False),
        categories=output_categories,
    )

    # Create converter
    converter = LabelIndexConverter(input_labels=input_spec, output_labels=output_spec)

    # Test filter - should return False for different label sets
    assert converter.filter_output_spec() is False


def test_label_index_converter_missing_categories():
    """Test LabelIndexConverter with missing categories (should not apply)."""

    # Create specs where one is missing categories
    input_categories = LabelCategories(labels=("cat", "dog", "bird"))

    input_spec = AttributeSpec(
        name="label",
        field=LabelField(dtype=pl.Int32(), multi_label=False),
        categories=input_categories,
    )

    output_spec = AttributeSpec(
        name="label",
        field=LabelField(dtype=pl.Int32(), multi_label=False),
        categories=None,  # Missing categories
    )

    # Create converter
    converter = LabelIndexConverter(input_labels=input_spec, output_labels=output_spec)

    # Test filter - should return False when categories are missing
    assert converter.filter_output_spec() is False


def test_label_index_converter_unmapped_labels():
    """Test LabelIndexConverter with unmapped labels using None default."""

    # Create input and output specs where input has extra labels not in output
    input_categories = LabelCategories(labels=("cat", "dog", "bird", "fish"))
    output_categories = LabelCategories(labels=("bird", "cat"))  # Missing dog and fish

    input_spec = AttributeSpec(
        name="label",
        field=LabelField(dtype=pl.Int32(), multi_label=False),
        categories=input_categories,
    )

    output_spec = AttributeSpec(
        name="label",
        field=LabelField(dtype=pl.Int32(), multi_label=False),
        categories=output_categories,
    )

    # Create converter
    converter = LabelIndexConverter(input_labels=input_spec, output_labels=output_spec)

    # This should return False because the label sets are different
    assert converter.filter_output_spec() is False


def test_rotated_bbox_to_polygon_converter():
    """Test conversion from rotated bounding box to polygon format."""
    import math

    # Create test data with rotated bboxes: [cx, cy, w, h, r]
    rotated_bbox_coords1 = [50.0, 60.0, 30.0, 20.0, 0.0]  # No rotation
    rotated_bbox_coords2 = [100.0, 120.0, 40.0, 25.0, math.pi / 4]  # 45 degrees

    rotated_bbox_series = pl.Series([rotated_bbox_coords1, rotated_bbox_coords2], dtype=pl.Array(pl.Float32, 5))

    df = pl.DataFrame(
        {
            "rotated_bboxes": [rotated_bbox_series],
        }
    )

    # Create converter instance
    converter_instance = RotatedBBoxToPolygonConverter()

    # Set up field specs
    input_rotated_bbox_field = RotatedBBoxField(dtype=pl.Float32(), format="cxcywhr", normalize=False)
    output_polygon_field = PolygonField(dtype=pl.Float32(), format="xy", normalize=False)

    setattr(
        converter_instance,
        "input_rotated_bbox",
        AttributeSpec(name="rotated_bboxes", field=input_rotated_bbox_field),
    )
    setattr(
        converter_instance,
        "output_polygon",
        AttributeSpec(name="polygons", field=output_polygon_field),
    )

    # Test filter - should return True when we have valid input
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    # Check that polygon column was created
    assert "polygons" in result_df.columns

    # Get the polygon data
    polygons = result_df["polygons"][0]

    # Check that we have 2 polygons
    assert len(polygons) == 2

    # Check first polygon (no rotation) - should be axis-aligned rectangle
    polygon1 = polygons[0]
    assert len(polygon1) == 4  # Four corners

    # For no rotation, corners should be at predictable positions
    expected_corners = [
        [35.0, 50.0],  # bottom-left
        [65.0, 50.0],  # bottom-right
        [65.0, 70.0],  # top-right
        [35.0, 70.0],  # top-left
    ]

    for expected, actual in zip(expected_corners, polygon1):
        assert abs(actual[0] - expected[0]) < 1e-5
        assert abs(actual[1] - expected[1]) < 1e-5

    # Check second polygon (45-degree rotation)
    polygon2 = polygons[1]
    assert len(polygon2) == 4  # Four corners


def test_bbox_dtype_converter_int_to_float():
    """Test BBoxDtypeConverter converting Int32 to Float32."""
    from datumaro.experimental.converters import BBoxDtypeConverter

    converter_instance = BBoxDtypeConverter()

    # Create test data with Int32 bboxes
    df = pl.DataFrame(
        {"bbox": [[[5, 5, 20, 20], [25, 30, 50, 60]]]},
        schema=pl.Schema({"bbox": pl.List(pl.Array(pl.Int32, 4))}),
    )

    # Set up converter attributes
    input_bbox_field = BBoxField(dtype=pl.Int32(), format="x1y1x2y2", normalize=False)
    output_bbox_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)

    setattr(converter_instance, "input_bbox", AttributeSpec(name="bbox", field=input_bbox_field))
    setattr(converter_instance, "output_bbox", AttributeSpec(name="bbox", field=output_bbox_field))

    # Test filter - should return True for dtype change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "bbox" in result_df.columns
    # Check dtype conversion
    result_bbox = result_df["bbox"][0]
    assert result_bbox[0].to_numpy().dtype == np.float32
    assert result_bbox[1].to_numpy().dtype == np.float32

    # Check values preserved
    np.testing.assert_array_almost_equal(result_bbox[0].to_numpy(), [5.0, 5.0, 20.0, 20.0])
    np.testing.assert_array_almost_equal(result_bbox[1].to_numpy(), [25.0, 30.0, 50.0, 60.0])


def test_bbox_dtype_converter_float_to_int():
    """Test BBoxDtypeConverter converting Float64 to Int32."""
    from datumaro.experimental.converters import BBoxDtypeConverter

    converter_instance = BBoxDtypeConverter()

    # Create test data with Float64 bboxes
    df = pl.DataFrame(
        {"bbox": [[[10.5, 20.7, 30.2, 40.9]]]},
        schema=pl.Schema({"bbox": pl.List(pl.Array(pl.Float64, 4))}),
    )

    # Set up converter attributes
    input_bbox_field = BBoxField(dtype=pl.Float64(), format="x1y1x2y2", normalize=False)
    output_bbox_field = BBoxField(dtype=pl.Int32(), format="x1y1x2y2", normalize=False)

    setattr(converter_instance, "input_bbox", AttributeSpec(name="bbox", field=input_bbox_field))
    setattr(converter_instance, "output_bbox", AttributeSpec(name="bbox", field=output_bbox_field))

    # Test filter - should return True for dtype change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "bbox" in result_df.columns
    # Check dtype conversion
    result_bbox = result_df["bbox"][0]
    assert result_bbox[0].to_numpy().dtype == np.int32
    # Values should be truncated
    np.testing.assert_array_equal(result_bbox[0].to_numpy(), [10, 20, 30, 40])


def test_bbox_dtype_converter_same_dtype():
    """Test BBoxDtypeConverter returns False when dtypes are the same."""
    from datumaro.experimental.converters import BBoxDtypeConverter

    converter_instance = BBoxDtypeConverter()

    # Set up converter attributes with same dtype
    input_bbox_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)
    output_bbox_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)

    setattr(converter_instance, "input_bbox", AttributeSpec(name="bbox", field=input_bbox_field))
    setattr(converter_instance, "output_bbox", AttributeSpec(name="bbox", field=output_bbox_field))

    # Test filter - should return False when dtypes are the same
    assert converter_instance.filter_output_spec() is False


def test_rotated_bbox_dtype_converter_int_to_float():
    """Test RotatedBBoxDtypeConverter converting Int32 to Float32."""
    from datumaro.experimental.converters import RotatedBBoxDtypeConverter

    converter_instance = RotatedBBoxDtypeConverter()

    # Create test data with Int32 rotated bboxes (cx, cy, w, h, r)
    df = pl.DataFrame(
        {"rotated_bbox": [[[50, 60, 30, 20, 0], [100, 120, 40, 25, 1]]]},
        schema=pl.Schema({"rotated_bbox": pl.List(pl.Array(pl.Int32, 5))}),
    )

    # Set up converter attributes
    input_field = RotatedBBoxField(dtype=pl.Int32(), format="cxcywhr", normalize=False)
    output_field = RotatedBBoxField(dtype=pl.Float32(), format="cxcywhr", normalize=False)

    setattr(converter_instance, "input_rotated_bbox", AttributeSpec(name="rotated_bbox", field=input_field))
    setattr(converter_instance, "output_rotated_bbox", AttributeSpec(name="rotated_bbox", field=output_field))

    # Test filter - should return True for dtype change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "rotated_bbox" in result_df.columns
    # Check dtype conversion
    result_rotated_bbox = result_df["rotated_bbox"][0]
    assert result_rotated_bbox[0].to_numpy().dtype == np.float32

    # Check values preserved
    np.testing.assert_array_almost_equal(result_rotated_bbox[0].to_numpy(), [50.0, 60.0, 30.0, 20.0, 0.0])
    np.testing.assert_array_almost_equal(result_rotated_bbox[1].to_numpy(), [100.0, 120.0, 40.0, 25.0, 1.0])


def test_rotated_bbox_dtype_converter_same_dtype():
    """Test RotatedBBoxDtypeConverter returns False when dtypes are the same."""
    from datumaro.experimental.converters import RotatedBBoxDtypeConverter

    converter_instance = RotatedBBoxDtypeConverter()

    # Set up converter attributes with same dtype
    input_field = RotatedBBoxField(dtype=pl.Float32(), format="cxcywhr", normalize=False)
    output_field = RotatedBBoxField(dtype=pl.Float32(), format="cxcywhr", normalize=False)

    setattr(converter_instance, "input_rotated_bbox", AttributeSpec(name="rotated_bbox", field=input_field))
    setattr(converter_instance, "output_rotated_bbox", AttributeSpec(name="rotated_bbox", field=output_field))

    # Test filter - should return False when dtypes are the same
    assert converter_instance.filter_output_spec() is False


def test_label_dtype_converter_int32_to_uint8():
    """Test LabelDtypeConverter converting Int32 to UInt8."""
    from datumaro.experimental.converters import LabelDtypeConverter

    converter_instance = LabelDtypeConverter()

    # Create test data with Int32 label
    df = pl.DataFrame(
        {"label": [5]},
        schema=pl.Schema({"label": pl.Int32()}),
    )

    # Set up converter attributes
    input_field = LabelField(dtype=pl.Int32(), multi_label=False, is_list=False)
    output_field = LabelField(dtype=pl.UInt8(), multi_label=False, is_list=False)

    setattr(converter_instance, "input_label", AttributeSpec(name="label", field=input_field))
    setattr(converter_instance, "output_label", AttributeSpec(name="label", field=output_field))

    # Test filter - should return True for dtype change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "label" in result_df.columns
    assert result_df["label"].dtype == pl.UInt8
    assert result_df["label"][0] == 5


def test_label_dtype_converter_list_labels():
    """Test LabelDtypeConverter with is_list=True."""
    from datumaro.experimental.converters import LabelDtypeConverter

    converter_instance = LabelDtypeConverter()

    # Create test data with list of Int32 labels
    df = pl.DataFrame(
        {"labels": [[1, 2, 3, 255]]},
        schema=pl.Schema({"labels": pl.List(pl.Int32())}),
    )

    # Set up converter attributes
    input_field = LabelField(dtype=pl.Int32(), multi_label=False, is_list=True)
    output_field = LabelField(dtype=pl.UInt8(), multi_label=False, is_list=True)

    setattr(converter_instance, "input_label", AttributeSpec(name="labels", field=input_field))
    setattr(converter_instance, "output_label", AttributeSpec(name="labels", field=output_field))

    # Test filter - should return True for dtype change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "labels" in result_df.columns
    assert result_df["labels"].dtype == pl.List(pl.UInt8)
    assert result_df["labels"][0].to_list() == [1, 2, 3, 255]


def test_label_dtype_converter_same_dtype():
    """Test LabelDtypeConverter returns False when dtypes are the same."""
    from datumaro.experimental.converters import LabelDtypeConverter

    converter_instance = LabelDtypeConverter()

    # Set up converter attributes with same dtype
    input_field = LabelField(dtype=pl.Int32(), multi_label=False, is_list=False)
    output_field = LabelField(dtype=pl.Int32(), multi_label=False, is_list=False)

    setattr(converter_instance, "input_label", AttributeSpec(name="label", field=input_field))
    setattr(converter_instance, "output_label", AttributeSpec(name="label", field=output_field))

    # Test filter - should return False when dtypes are the same
    assert converter_instance.filter_output_spec() is False


def test_polygon_dtype_converter_int_to_float():
    """Test PolygonDtypeConverter converting Int32 to Float32."""
    from datumaro.experimental.converters import PolygonDtypeConverter

    converter_instance = PolygonDtypeConverter()

    # Create test data with Int32 polygons: List[List[Array[2]]]
    # Each polygon is a list of (x, y) points
    polygon1 = [[10, 10], [30, 10], [20, 30]]  # Triangle
    polygon2 = [[40, 40], [60, 40], [60, 60], [40, 60]]  # Rectangle

    df = pl.DataFrame(
        {"polygon": [[polygon1, polygon2]]},
        schema=pl.Schema({"polygon": pl.List(pl.List(pl.Array(pl.Int32, 2)))}),
    )

    # Set up converter attributes
    input_field = PolygonField(dtype=pl.Int32(), format="xy", normalize=False)
    output_field = PolygonField(dtype=pl.Float32(), format="xy", normalize=False)

    setattr(converter_instance, "input_polygon", AttributeSpec(name="polygon", field=input_field))
    setattr(converter_instance, "output_polygon", AttributeSpec(name="polygon", field=output_field))

    # Test filter - should return True for dtype change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "polygon" in result_df.columns
    # Check dtype conversion - get the inner arrays and check their dtype
    result_polygons = result_df["polygon"][0]
    assert len(result_polygons) == 2

    # Check first polygon (triangle)
    result_polygon1 = result_polygons[0]
    assert len(result_polygon1) == 3
    np.testing.assert_array_almost_equal(result_polygon1[0].to_numpy(), [10.0, 10.0])
    np.testing.assert_array_almost_equal(result_polygon1[1].to_numpy(), [30.0, 10.0])
    np.testing.assert_array_almost_equal(result_polygon1[2].to_numpy(), [20.0, 30.0])

    # Check second polygon (rectangle)
    result_polygon2 = result_polygons[1]
    assert len(result_polygon2) == 4


def test_polygon_dtype_converter_same_dtype():
    """Test PolygonDtypeConverter returns False when dtypes are the same."""
    from datumaro.experimental.converters import PolygonDtypeConverter

    converter_instance = PolygonDtypeConverter()

    # Set up converter attributes with same dtype
    input_field = PolygonField(dtype=pl.Float32(), format="xy", normalize=False)
    output_field = PolygonField(dtype=pl.Float32(), format="xy", normalize=False)

    setattr(converter_instance, "input_polygon", AttributeSpec(name="polygon", field=input_field))
    setattr(converter_instance, "output_polygon", AttributeSpec(name="polygon", field=output_field))

    # Test filter - should return False when dtypes are the same
    assert converter_instance.filter_output_spec() is False


def test_bbox_dtype_converter_preserves_array_structure():
    """Test that BBoxDtypeConverter preserves the fixed-size array structure."""
    from datumaro.experimental.converters import BBoxDtypeConverter

    converter_instance = BBoxDtypeConverter()

    # Create test data with multiple bboxes
    df = pl.DataFrame(
        {"bbox": [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]},
        schema=pl.Schema({"bbox": pl.List(pl.Array(pl.Int32, 4))}),
    )

    # Set up converter attributes
    input_bbox_field = BBoxField(dtype=pl.Int32(), format="x1y1x2y2", normalize=False)
    output_bbox_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)

    setattr(converter_instance, "input_bbox", AttributeSpec(name="bbox", field=input_bbox_field))
    setattr(converter_instance, "output_bbox", AttributeSpec(name="bbox", field=output_bbox_field))

    converter_instance.filter_output_spec()

    # Test conversion
    result_df = converter_instance.convert(df)

    # Check that the result maintains the List[Array[4]] structure
    assert result_df.schema["bbox"] == pl.List(pl.Array(pl.Float32, 4))

    # Check that all bboxes are correctly converted
    result_bboxes = result_df["bbox"][0]
    assert len(result_bboxes) == 3
    np.testing.assert_array_almost_equal(result_bboxes[0].to_numpy(), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result_bboxes[1].to_numpy(), [5.0, 6.0, 7.0, 8.0])
    np.testing.assert_array_almost_equal(result_bboxes[2].to_numpy(), [9.0, 10.0, 11.0, 12.0])


def test_create_fixed_array_cast_expr():
    """Test the _create_fixed_array_cast_expr helper function."""
    from datumaro.experimental.converters.annotation_converters import _create_fixed_array_cast_expr

    # Test with 4-element array (bbox)
    arr_data = np.array([1, 2, 3, 4], dtype=np.int32)
    df = pl.DataFrame(
        {"arr": [[arr_data]]},
        schema=pl.Schema({"arr": pl.List(pl.Array(pl.Int32(), 4))}),
    )

    # Apply the expression
    result = df.select(pl.col("arr").list.eval(_create_fixed_array_cast_expr(pl.Float64(), 4)))

    # Check result
    assert result["arr"].dtype == pl.List(pl.Array(pl.Float64(), 4))
    np.testing.assert_array_almost_equal(result["arr"][0][0].to_numpy(), [1.0, 2.0, 3.0, 4.0])


def test_convert_fixed_array_dtype():
    """Test the _convert_fixed_array_dtype helper function."""
    from datumaro.experimental.converters.annotation_converters import _convert_fixed_array_dtype

    # Test with 5-element array (rotated bbox)
    df = pl.DataFrame(
        {"input": [[[10, 20, 30, 40, 50]]]},
        schema=pl.Schema({"input": pl.List(pl.Array(pl.Int32(), 5))}),
    )

    # Apply the conversion
    result_df = _convert_fixed_array_dtype(df, "input", "output", pl.Float32(), array_size=5)

    # Check result
    assert "output" in result_df.columns
    assert result_df["output"].dtype == pl.List(pl.Array(pl.Float32(), 5))
    np.testing.assert_array_almost_equal(result_df["output"][0][0].to_numpy(), [10.0, 20.0, 30.0, 40.0, 50.0])


def test_bbox_dtype_conversion_numpy_dtype():
    class DetectionSampleFloat(Sample):
        bboxes: npt.NDArray[np.floating[Any]] = bbox_field(dtype=pl.Float32())

    class DetectionSampleInt(Sample):
        bboxes: npt.NDArray[np.integer[Any]] = bbox_field(dtype=pl.Int32())

    # Create source dataset with int32 bboxes
    dataset_int = Dataset(DetectionSampleInt)
    di_int = DetectionSampleInt(bboxes=np.array([[5, 5, 20, 20], [25, 30, 50, 60]], dtype=np.int32))
    dataset_int.append(di_int)

    # Convert to float32 schema
    dataset_float = dataset_int.convert_to_schema(DetectionSampleFloat)

    # Verify original dataset is unchanged
    assert dataset_int[0].bboxes.dtype == np.int32
    np.testing.assert_array_equal(dataset_int[0].bboxes, [[5, 5, 20, 20], [25, 30, 50, 60]])

    # The dtype should be float32, not object
    assert dataset_float[0].bboxes.dtype == np.float32

    # Verify the values are correct
    expected_values = np.array([[5.0, 5.0, 20.0, 20.0], [25.0, 30.0, 50.0, 60.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(dataset_float[0].bboxes, expected_values)


def test_keypoints_dtype_converter():
    """Test KeypointsDtypeConverter converting Float32 to Float64."""
    converter_instance = KeypointsDtypeConverter()

    # Create test data with Float32 keypoints [x, y, visibility]
    df = pl.DataFrame(
        {"keypoints": [[[10.0, 20.0, 2.0], [30.0, 40.0, 1.0], [50.0, 60.0, 0.0]]]},
        schema=pl.Schema({"keypoints": pl.List(pl.Array(pl.Float32, 3))}),
    )

    # Set up converter attributes
    input_field = KeypointsField(dtype=pl.Float32(), normalize=False)
    output_field = KeypointsField(dtype=pl.Float64(), normalize=False)

    setattr(converter_instance, "input_keypoints", AttributeSpec(name="keypoints", field=input_field))
    setattr(converter_instance, "output_keypoints", AttributeSpec(name="keypoints", field=output_field))

    # Test filter - should return True for dtype change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "keypoints" in result_df.columns
    assert result_df.schema["keypoints"] == pl.List(pl.Array(pl.Float64, 3))

    result_kpts = result_df["keypoints"][0]
    np.testing.assert_array_almost_equal(result_kpts[0].to_numpy(), [10.0, 20.0, 2.0])
    np.testing.assert_array_almost_equal(result_kpts[1].to_numpy(), [30.0, 40.0, 1.0])
    np.testing.assert_array_almost_equal(result_kpts[2].to_numpy(), [50.0, 60.0, 0.0])


def test_keypoints_coordinate_converter_normalize():
    """Test KeypointsCoordinateConverter normalizing absolute coordinates."""

    converter_instance = KeypointsCoordinateConverter()

    # Create test data with absolute coordinates
    df = pl.DataFrame(
        {
            "keypoints": [[[100.0, 150.0, 2.0], [200.0, 300.0, 1.0]]],
            "image": [[0] * 100],  # dummy image data
            "image_shape": [[400, 500, 3]],  # height=400, width=500
        },
        schema=pl.Schema(
            {
                "keypoints": pl.List(pl.Array(pl.Float32, 3)),
                "image": pl.List(pl.UInt8),
                "image_shape": pl.List(pl.Int32),
            }
        ),
    )

    # Set up converter attributes
    input_keypoints_field = KeypointsField(dtype=pl.Float32(), normalize=False)
    output_keypoints_field = KeypointsField(dtype=pl.Float32(), normalize=True)
    image_field = ImageField(dtype=pl.UInt8())

    setattr(converter_instance, "input_keypoints", AttributeSpec(name="keypoints", field=input_keypoints_field))
    setattr(converter_instance, "output_keypoints", AttributeSpec(name="keypoints", field=output_keypoints_field))
    setattr(converter_instance, "input_image", AttributeSpec(name="image", field=image_field))

    # Test filter - should return True for normalization change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    result_kpts = result_df["keypoints"][0]
    # x is normalized by width (500), y by height (400)
    np.testing.assert_array_almost_equal(result_kpts[0].to_numpy(), [100.0 / 500, 150.0 / 400, 2.0], decimal=5)
    np.testing.assert_array_almost_equal(result_kpts[1].to_numpy(), [200.0 / 500, 300.0 / 400, 1.0], decimal=5)


def test_ellipse_dtype_converter():
    """Test EllipseDtypeConverter converting Int32 to Float32."""

    converter_instance = EllipseDtypeConverter()

    # Create test data with Int32 ellipses [x1, y1, x2, y2]
    df = pl.DataFrame(
        {"ellipse": [[[10, 20, 30, 40], [50, 60, 70, 80]]]},
        schema=pl.Schema({"ellipse": pl.List(pl.Array(pl.Int32, 4))}),
    )

    # Set up converter attributes
    input_field = EllipseField(dtype=pl.Int32(), format="x1y1x2y2", normalize=False)
    output_field = EllipseField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)

    setattr(converter_instance, "input_ellipse", AttributeSpec(name="ellipse", field=input_field))
    setattr(converter_instance, "output_ellipse", AttributeSpec(name="ellipse", field=output_field))

    # Test filter - should return True for dtype change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "ellipse" in result_df.columns
    assert result_df.schema["ellipse"] == pl.List(pl.Array(pl.Float32, 4))

    result_ellipses = result_df["ellipse"][0]
    np.testing.assert_array_almost_equal(result_ellipses[0].to_numpy(), [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_array_almost_equal(result_ellipses[1].to_numpy(), [50.0, 60.0, 70.0, 80.0])


def test_bbox_format_converter_x1y1x2y2_to_xywh():
    """Test BBoxFormatConverter converting x1y1x2y2 to xywh."""

    converter_instance = BBoxFormatConverter()

    # Create test data with x1y1x2y2 format
    df = pl.DataFrame(
        {"bbox": [[[10.0, 20.0, 30.0, 50.0], [100.0, 150.0, 200.0, 250.0]]]},
        schema=pl.Schema({"bbox": pl.List(pl.Array(pl.Float32, 4))}),
    )

    # Set up converter attributes
    input_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)
    output_field = BBoxField(dtype=pl.Float32(), format="xywh", normalize=False)

    setattr(converter_instance, "input_bbox", AttributeSpec(name="bbox", field=input_field))
    setattr(converter_instance, "output_bbox", AttributeSpec(name="bbox", field=output_field))

    # Test filter - should return True for format change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    result_bboxes = result_df["bbox"][0]
    # x1y1x2y2 [10, 20, 30, 50] -> xywh [10, 20, 20, 30] (w=30-10=20, h=50-20=30)
    np.testing.assert_array_almost_equal(result_bboxes[0].to_numpy(), [10.0, 20.0, 20.0, 30.0])
    # x1y1x2y2 [100, 150, 200, 250] -> xywh [100, 150, 100, 100]
    np.testing.assert_array_almost_equal(result_bboxes[1].to_numpy(), [100.0, 150.0, 100.0, 100.0])


def test_bbox_format_converter_xywh_to_cxcywh():
    """Test BBoxFormatConverter converting xywh to cxcywh."""

    converter_instance = BBoxFormatConverter()

    # Create test data with xywh format
    df = pl.DataFrame(
        {"bbox": [[[10.0, 20.0, 20.0, 30.0]]]},  # x, y, w, h
        schema=pl.Schema({"bbox": pl.List(pl.Array(pl.Float32, 4))}),
    )

    # Set up converter attributes
    input_field = BBoxField(dtype=pl.Float32(), format="xywh", normalize=False)
    output_field = BBoxField(dtype=pl.Float32(), format="cxcywh", normalize=False)

    setattr(converter_instance, "input_bbox", AttributeSpec(name="bbox", field=input_field))
    setattr(converter_instance, "output_bbox", AttributeSpec(name="bbox", field=output_field))

    # Test filter - should return True for format change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    result_bboxes = result_df["bbox"][0]
    # xywh [10, 20, 20, 30] -> cxcywh [20, 35, 20, 30] (cx=10+20/2=20, cy=20+30/2=35)
    np.testing.assert_array_almost_equal(result_bboxes[0].to_numpy(), [20.0, 35.0, 20.0, 30.0])


def test_bbox_format_converter_cxcywh_to_x1y1x2y2():
    """Test BBoxFormatConverter converting cxcywh to x1y1x2y2."""

    converter_instance = BBoxFormatConverter()

    # Create test data with cxcywh format
    df = pl.DataFrame(
        {"bbox": [[[20.0, 35.0, 20.0, 30.0]]]},  # cx, cy, w, h
        schema=pl.Schema({"bbox": pl.List(pl.Array(pl.Float32, 4))}),
    )

    # Set up converter attributes
    input_field = BBoxField(dtype=pl.Float32(), format="cxcywh", normalize=False)
    output_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)

    setattr(converter_instance, "input_bbox", AttributeSpec(name="bbox", field=input_field))
    setattr(converter_instance, "output_bbox", AttributeSpec(name="bbox", field=output_field))

    # Test filter - should return True for format change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    result_bboxes = result_df["bbox"][0]
    # cxcywh [20, 35, 20, 30] -> x1y1x2y2 [10, 20, 30, 50]
    np.testing.assert_array_almost_equal(result_bboxes[0].to_numpy(), [10.0, 20.0, 30.0, 50.0])


def test_bbox_to_polygon_converter():
    """Test BBoxToPolygonConverter converting bbox to polygon."""

    converter_instance = BBoxToPolygonConverter()

    # Create test data with x1y1x2y2 format
    df = pl.DataFrame(
        {"bbox": [[[10.0, 20.0, 30.0, 50.0]]]},  # x1, y1, x2, y2
        schema=pl.Schema({"bbox": pl.List(pl.Array(pl.Float32, 4))}),
    )

    # Set up converter attributes
    input_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)
    output_field = PolygonField(dtype=pl.Float32(), format="xy", normalize=False)

    setattr(converter_instance, "input_bbox", AttributeSpec(name="bbox", field=input_field))
    setattr(converter_instance, "output_polygon", AttributeSpec(name="polygon", field=output_field))

    # Test filter - should return True
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "polygon" in result_df.columns
    result_polygons = result_df["polygon"][0]

    # Check that we have 1 polygon with 4 corners
    assert len(result_polygons) == 1
    polygon = result_polygons[0]
    assert len(polygon) == 4

    # Check corners: top-left, top-right, bottom-right, bottom-left
    np.testing.assert_array_almost_equal(polygon[0].to_numpy(), [10.0, 20.0])  # top-left
    np.testing.assert_array_almost_equal(polygon[1].to_numpy(), [30.0, 20.0])  # top-right
    np.testing.assert_array_almost_equal(polygon[2].to_numpy(), [30.0, 50.0])  # bottom-right
    np.testing.assert_array_almost_equal(polygon[3].to_numpy(), [10.0, 50.0])  # bottom-left


def test_ellipse_to_bbox_converter():
    """Test EllipseToBBoxConverter converting ellipse to bbox."""

    converter_instance = EllipseToBBoxConverter()

    # Create test data with x1y1x2y2 format ellipse
    df = pl.DataFrame(
        {"ellipse": [[[10.0, 20.0, 30.0, 50.0]]]},
        schema=pl.Schema({"ellipse": pl.List(pl.Array(pl.Float32, 4))}),
    )

    # Set up converter attributes
    input_field = EllipseField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)
    output_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)

    setattr(converter_instance, "input_ellipse", AttributeSpec(name="ellipse", field=input_field))
    setattr(converter_instance, "output_bbox", AttributeSpec(name="bbox", field=output_field))

    # Test filter - should return True
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "bbox" in result_df.columns
    result_bboxes = result_df["bbox"][0]

    # Ellipse and bbox use same format, should be identical
    np.testing.assert_array_almost_equal(result_bboxes[0].to_numpy(), [10.0, 20.0, 30.0, 50.0])


def test_keypoints_to_bbox_converter():
    """Test KeypointsToBBoxConverter converting keypoints to enclosing bbox."""

    converter_instance = KeypointsToBBoxConverter()

    # Create test data with keypoints [x, y, visibility]
    # visibility > 0 means visible
    df = pl.DataFrame(
        {
            "keypoints": [
                [[10.0, 20.0, 2.0], [30.0, 50.0, 1.0], [25.0, 35.0, 2.0], [5.0, 5.0, 0.0]]  # invisible
            ]
        },
        schema=pl.Schema({"keypoints": pl.List(pl.Array(pl.Float32, 3))}),
    )

    # Set up converter attributes
    input_field = KeypointsField(dtype=pl.Float32(), normalize=False)
    output_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)

    setattr(converter_instance, "input_keypoints", AttributeSpec(name="keypoints", field=input_field))
    setattr(converter_instance, "output_bbox", AttributeSpec(name="bbox", field=output_field))

    # Test filter - should return True
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "bbox" in result_df.columns
    result_bboxes = result_df["bbox"][0]

    # Bbox should enclose only visible keypoints: (10,20), (30,50), (25,35)
    # Not (5,5) which has visibility=0
    # min_x=10, min_y=20, max_x=30, max_y=50
    np.testing.assert_array_almost_equal(result_bboxes[0].to_numpy(), [10.0, 20.0, 30.0, 50.0])


def test_rotated_bbox_to_bbox_converter():
    """Test RotatedBBoxToBBoxConverter converting rotated bbox to AABB."""

    converter_instance = RotatedBBoxToBBoxConverter()

    # Create test data with rotated bboxes [cx, cy, w, h, r]
    # No rotation - should give same as regular bbox
    df = pl.DataFrame(
        {"rotated_bbox": [[[50.0, 60.0, 20.0, 10.0, 0.0]]]},  # cx=50, cy=60, w=20, h=10, r=0
        schema=pl.Schema({"rotated_bbox": pl.List(pl.Array(pl.Float32, 5))}),
    )

    # Set up converter attributes
    input_field = RotatedBBoxField(dtype=pl.Float32(), format="cxcywhr", normalize=False)
    output_field = BBoxField(dtype=pl.Float32(), format="x1y1x2y2", normalize=False)

    setattr(converter_instance, "input_rotated_bbox", AttributeSpec(name="rotated_bbox", field=input_field))
    setattr(converter_instance, "output_bbox", AttributeSpec(name="bbox", field=output_field))

    # Test filter - should return True
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "bbox" in result_df.columns
    result_bboxes = result_df["bbox"][0]

    # With r=0: cx=50, cy=60, w=20, h=10 -> x1=40, y1=55, x2=60, y2=65
    np.testing.assert_array_almost_equal(result_bboxes[0].to_numpy(), [40.0, 55.0, 60.0, 65.0])


def test_rotated_bbox_coordinate_converter():
    """Test RotatedBBoxCoordinateConverter normalizing coordinates."""

    converter_instance = RotatedBBoxCoordinateConverter()

    # Create test data with absolute coordinates
    df = pl.DataFrame(
        {
            "rotated_bbox": [[[250.0, 200.0, 100.0, 80.0, 0.5]]],  # cx, cy, w, h, r
            "image": [[0] * 100],  # dummy image data
            "image_shape": [[400, 500, 3]],  # height=400, width=500
        },
        schema=pl.Schema(
            {
                "rotated_bbox": pl.List(pl.Array(pl.Float32, 5)),
                "image": pl.List(pl.UInt8),
                "image_shape": pl.List(pl.Int32),
            }
        ),
    )

    # Set up converter attributes
    input_rbbox_field = RotatedBBoxField(dtype=pl.Float32(), format="cxcywhr", normalize=False)
    output_rbbox_field = RotatedBBoxField(dtype=pl.Float32(), format="cxcywhr", normalize=True)
    image_field = ImageField(dtype=pl.UInt8())

    setattr(converter_instance, "input_rotated_bbox", AttributeSpec(name="rotated_bbox", field=input_rbbox_field))
    setattr(converter_instance, "output_rotated_bbox", AttributeSpec(name="rotated_bbox", field=output_rbbox_field))
    setattr(converter_instance, "input_image", AttributeSpec(name="image", field=image_field))

    # Test filter - should return True for normalization change
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    result_rbbox = result_df["rotated_bbox"][0][0].to_numpy()
    # cx and w normalized by width (500), cy and h normalized by height (400), r unchanged
    expected = [250.0 / 500, 200.0 / 400, 100.0 / 500, 80.0 / 400, 0.5]
    np.testing.assert_array_almost_equal(result_rbbox, expected, decimal=5)


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
    from datumaro.experimental.converters import find_conversion_path

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
