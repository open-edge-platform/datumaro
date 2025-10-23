"""
Unit tests for converter registry and converter implementations.
"""

import os
import tempfile

import numpy as np
import polars as pl
import pytest

from datumaro.experimental.categories import LabelCategories, MaskCategories
from datumaro.experimental.converter_registry import (
    AttributeRemapperConverter,
    AttributeSpec,
    ConversionError,
    Converter,
    ConverterRegistry,
    converter,
    find_conversion_path,
)
from datumaro.experimental.converters import (
    BBoxCoordinateConverter,
    BBoxFormatConverter,
    EllipseFormatConverter,
    ImageBytesToImageConverter,
    ImageCallableToImageConverter,
    ImagePathToImageConverter,
    InstanceMaskCallableToInstanceMaskConverter,
    LabelIndexConverter,
    MaskCallableToMaskConverter,
    PolygonFormatConverter,
    PolygonToBBoxConverter,
    PolygonToInstanceMaskConverter,
    PolygonToMaskConverter,
    RGBToBGRConverter,
    RotatedBBoxFormatConverter,
    RotatedBBoxToPolygonConverter,
    UInt8ToFloat32Converter,
)
from datumaro.experimental.fields import (
    BBoxField,
    BBoxFormat,
    EllipseField,
    EllipseFormat,
    Field,
    ImageBytesField,
    ImageCallableField,
    ImageField,
    ImageFormat,
    ImageInfoField,
    ImagePathField,
    InstanceMaskCallableField,
    InstanceMaskField,
    LabelField,
    MaskCallableField,
    MaskField,
    PolygonField,
    PolygonFormat,
    RotatedBBoxField,
    RotatedBBoxFormat,
    bbox_field,
    image_field,
    image_info_field,
)
from datumaro.experimental.schema import AttributeInfo, Schema, Semantic


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
    converter_instance = RGBToBGRConverter()  # type: ignore[call-arg]

    # Create test data
    rgb_data = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [128, 128, 128]]])
    df = pl.DataFrame(
        {"image": [rgb_data.reshape(-1)], "image_shape": [[2, 2, 3]]},
        schema=pl.Schema({"image": pl.List(pl.UInt8), "image_shape": pl.List(pl.Int64)}),
    )

    # Set up converter attributes
    input_field = ImageField(dtype=pl.UInt8, format="RGB", semantic=Semantic.Default)
    output_field = ImageField(dtype=pl.UInt8, format="BGR", semantic=Semantic.Default)

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
        schema=pl.Schema({"image": pl.List(pl.UInt8), "image_shape": pl.List(pl.Int64)}),
    )

    # Set up converter attributes
    input_field = ImageField(dtype=pl.UInt8, format=ImageFormat.RGB, semantic=Semantic.Default)
    output_field = ImageField(dtype=pl.Float32, format=ImageFormat.RGB, semantic=Semantic.Default)

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
        schema=pl.Schema(
            {"bbox": pl.List(pl.Array(pl.Float32, 4)), "image_shape": pl.List(pl.Int64)}
        ),
    )

    # Set up converter for absolute to normalized conversion
    input_bbox_field = BBoxField(
        dtype=pl.Float32, format=BBoxFormat.X1Y1X2Y2, normalize=False, semantic=Semantic.Default
    )
    output_bbox_field = BBoxField(
        dtype=pl.Float32, format=BBoxFormat.X1Y1X2Y2, normalize=True, semantic=Semantic.Default
    )
    input_image_field = ImageField(
        dtype=pl.UInt8, format=ImageFormat.RGB, semantic=Semantic.Default
    )

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
        input_field = ImagePathField(semantic=Semantic.Default)
        output_field = ImageField(dtype=pl.UInt8, format=ImageFormat.RGB, semantic=Semantic.Default)
        output_info_field = ImageInfoField(semantic=Semantic.Default)

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

    input_field = ImageBytesField(semantic=Semantic.Default)
    output_field = ImageField(dtype=pl.UInt8, format="RGB", semantic=Semantic.Default)
    output_info_field = ImageInfoField(semantic=Semantic.Default)

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
        attributes={
            "image": AttributeInfo(
                type=np.ndarray, annotation=image_field(dtype=pl.UInt8, format="RGB")
            )
        }
    )

    target_schema = Schema(
        attributes={
            "image": AttributeInfo(
                type=np.ndarray, annotation=image_field(dtype=pl.Float32, format="RGB")
            )
        }
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
        schema=pl.Schema({"image": pl.List(pl.UInt8), "image_shape": pl.List(pl.Int64)}),
    )

    source_schema = Schema(
        attributes={
            "image": AttributeInfo(
                type=np.ndarray, annotation=image_field(dtype=pl.UInt8, format="RGB")
            )
        }
    )

    target_schema = Schema(
        attributes={
            "image": AttributeInfo(
                type=np.ndarray, annotation=image_field(dtype=pl.Float32, format="BGR")
            )
        }
    )

    # Get conversion path and apply it manually
    conversion_paths, _ = find_conversion_path(source_schema, target_schema)

    # Apply batch converters first
    result_df = df
    for converter in conversion_paths.converters["image"]:
        result_df = converter.convert(result_df)

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
    image_path_converters = [
        c for c in lazy_converters if issubclass(c, ImagePathToImageConverter) and c.lazy
    ]
    assert len(image_path_converters) > 0


def test_converter_with_auxiliary_fields():
    """Test converters that require auxiliary fields."""
    # This would test converters like bbox normalization that need image size
    # The exact implementation depends on how auxiliary fields are handled

    converter_instance = BBoxCoordinateConverter()  # type: ignore[call-arg]

    # BBox converter needs image data as auxiliary
    input_bbox_field = BBoxField(
        dtype=pl.Float32, format="x1y1x2y2", normalize=False, semantic=Semantic.Default
    )
    output_bbox_field = BBoxField(
        dtype=pl.Float32, format="x1y1x2y2", normalize=True, semantic=Semantic.Default
    )
    input_image_field = ImageField(dtype=pl.UInt8, format="RGB", semantic=Semantic.Default)

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
            "image": AttributeInfo(
                type=np.ndarray, annotation=image_field(dtype=pl.UInt8, format="RGB")
            ),
            "bbox": AttributeInfo(
                type=np.ndarray,
                annotation=bbox_field(dtype=pl.Float32, normalize=False),
            ),
        }
    )

    target_schema = Schema(
        attributes={
            "image": AttributeInfo(
                type=np.ndarray, annotation=image_field(dtype=pl.Float32, format="BGR")
            ),
            "bbox": AttributeInfo(
                type=np.ndarray, annotation=bbox_field(dtype=pl.Float32, normalize=True)
            ),
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
    assert type(path.converters["image"][1]) is RGBToBGRConverter

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
        semantic: Semantic = Semantic.Default

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Binary()}

    @dataclass(frozen=True)
    class TestImageSizeField(Field):
        semantic: Semantic = Semantic.Default

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
            "image": AttributeInfo(
                type=bytes, annotation=TestImageField(semantic=Semantic.Default)
            ),
        }
    )

    to_schema = Schema(
        {
            "image_size": AttributeInfo(
                type=tuple, annotation=TestImageSizeField(semantic=Semantic.Default)
            ),
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
        semantic: Semantic = Semantic.Default

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Binary()}

    @dataclass(frozen=True)
    class TestImageSizeField(Field):
        semantic: Semantic = Semantic.Default

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {f"{name}_width": pl.Int32(), f"{name}_height": pl.Int32()}

    @dataclass(frozen=True)
    class TestNormalizedBboxField(Field):
        semantic: Semantic = Semantic.Default

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.List(pl.Float32())}

    @dataclass(frozen=True)
    class TestAbsoluteBboxField(Field):
        semantic: Semantic = Semantic.Default

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
            "image": AttributeInfo(
                type=bytes, annotation=TestImageField(semantic=Semantic.Default)
            ),
            "bbox": AttributeInfo(
                type=list, annotation=TestNormalizedBboxField(semantic=Semantic.Default)
            ),
        }
    )

    to_schema = Schema(
        {
            "absolute_bbox": AttributeInfo(
                type=list, annotation=TestAbsoluteBboxField(semantic=Semantic.Default)
            ),
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
        semantic: Semantic = Semantic.Default

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    # Identical schemas - no conversion needed
    schema = Schema(
        {
            "data": AttributeInfo(type=str, annotation=TestField(semantic=Semantic.Default)),
        }
    )

    path, _ = find_conversion_path(schema, schema)
    assert (
        len(path.converters) == 0
    ), f"Expected 0 converters for identical schemas, got {len(path.converters)}"


def test_astar_impossible_conversion():
    """Test conversion that should fail using A* search."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FieldA(Field):
        semantic: Semantic = Semantic.Default

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @dataclass(frozen=True)
    class FieldB(Field):
        semantic: Semantic = Semantic.Default

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    # No converters registered between these types - should fail
    from_schema = Schema(
        {
            "field_a": AttributeInfo(type=str, annotation=FieldA(semantic=Semantic.Default)),
        }
    )

    to_schema = Schema(
        {
            "field_b": AttributeInfo(type=str, annotation=FieldB(semantic=Semantic.Default)),
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
        semantic: Semantic = Semantic.Default

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @dataclass(frozen=True)
    class FieldB(Field):
        semantic: Semantic = Semantic.Default

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @dataclass(frozen=True)
    class FieldC(Field):
        semantic: Semantic = Semantic.Default

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
            "field_a": AttributeInfo(type=str, annotation=FieldA(semantic=Semantic.Default)),
        }
    )

    to_schema = Schema(
        {
            "field_c": AttributeInfo(type=str, annotation=FieldC(semantic=Semantic.Default)),
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
        semantic: Semantic = Semantic.Default

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
            "field_b": AttributeInfo(type=str, annotation=FieldB(semantic=Semantic.Default)),
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
        semantic: Semantic = Semantic.Default

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @dataclass(frozen=True)
    class MultiField1(Field):
        semantic: Semantic = Semantic.Default

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @dataclass(frozen=True)
    class MultiField2(Field):
        semantic: Semantic = Semantic.Default

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
            return df.with_columns(
                [pl.col("field_a").alias("multi1"), pl.col("field_a").alias("multi2")]
            ).drop("field_a")

    from_schema = Schema(
        {
            "field_a": AttributeInfo(type=str, annotation=FieldA(semantic=Semantic.Default)),
        }
    )

    to_schema = Schema(
        {
            "multi1": AttributeInfo(type=str, annotation=MultiField1(semantic=Semantic.Default)),
            "multi2": AttributeInfo(type=str, annotation=MultiField2(semantic=Semantic.Default)),
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
        semantic: Semantic = Semantic.Default

        def to_polars_schema(self, name: str) -> dict[str, pl.DataType]:
            return {name: pl.Utf8()}

    @dataclass(frozen=True)
    class FieldC(Field):
        semantic: Semantic = Semantic.Default

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
            "field_a": AttributeInfo(type=str, annotation=FieldA(semantic=Semantic.Default)),
            "other_field": AttributeInfo(
                type=str, annotation=FieldA(semantic=Semantic.Left)
            ),  # Extra field with different semantic
        }
    )

    to_schema = Schema(
        {
            "field_c": AttributeInfo(type=str, annotation=FieldC(semantic=Semantic.Default)),
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
            "input_image": AttributeInfo(
                type=str, annotation=image_field(dtype=pl.UInt8(), format=ImageFormat.RGB)
            ),
        }
    )

    # Create target schema with the same field type but different name "output_image"
    target_schema = Schema(
        {
            "output_image": AttributeInfo(
                type=str, annotation=image_field(dtype=pl.UInt8(), format=ImageFormat.RGB)
            ),
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
    df = pl.DataFrame(
        {"input_image": [test_data.flatten()], "input_image_shape": [test_data.shape]}
    )

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
            "old_image": AttributeInfo(
                type=str, annotation=image_field(dtype=pl.UInt8(), format=ImageFormat.RGB)
            ),
            "bbox": AttributeInfo(
                type=str, annotation=bbox_field(dtype=pl.Float32(), format=BBoxFormat.X1Y1X2Y2)
            ),
            "extra_field": AttributeInfo(type=str, annotation=image_info_field()),
        }
    )

    # Create target schema with renamed image and no bbox or extra_field
    target_schema = Schema(
        {
            "new_image": AttributeInfo(
                type=str, annotation=image_field(dtype=pl.UInt8(), format=ImageFormat.RGB)
            ),
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
    input_polygon_field = PolygonField(
        dtype=pl.Float32, format=PolygonFormat.XY, normalize=False, semantic=Semantic.Default
    )
    input_labels_field = LabelField(dtype=pl.Int32, semantic=Semantic.Default, multi_label=True)
    image_info_field = ImageInfoField(semantic=Semantic.Default)
    output_mask_field = MaskField(dtype=pl.UInt8, semantic=Semantic.Default)

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
        dtype=pl.Float32,
        format=PolygonFormat.XY,
        normalize=True,  # Enable normalization
        semantic=Semantic.Default,
    )
    input_labels_field = LabelField(dtype=pl.Int32, semantic=Semantic.Default, multi_label=True)
    image_info_field = ImageInfoField(semantic=Semantic.Default)
    output_mask_field = MaskField(dtype=pl.UInt8, semantic=Semantic.Default)

    setattr(
        converter_instance,
        "input_polygon",
        AttributeSpec(name="polygons", field=input_polygon_field),
    )
    setattr(
        converter_instance, "input_labels", AttributeSpec(name="labels", field=input_labels_field)
    )
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
    df = pl.DataFrame(
        {
            "polygons": [[[10, 20, 30, 25, 20, 40]]],  # Triangle coordinates
            "labels": [[2]],  # Label 2
            "image_info": [{"width": 100, "height": 100}],
        },
        schema=pl.Schema(
            {
                "polygons": pl.List(pl.List(pl.Float32)),
                "labels": pl.List(pl.Int32),
                "image_info": pl.Struct({"width": pl.Int32, "height": pl.Int32}),
            }
        ),
    )

    # Create source schema with label categories
    label_categories = LabelCategories(labels=["cat", "dog", "bird"])
    source_schema = Schema(
        attributes={
            "polygons": AttributeInfo(
                type=list,
                annotation=PolygonField(dtype=pl.Float32, semantic=Semantic.Default),
                categories=None,
            ),
            "labels": AttributeInfo(
                type=list,
                annotation=LabelField(semantic=Semantic.Default),
                categories=label_categories,
            ),
            "image_info": AttributeInfo(
                type=dict, annotation=ImageInfoField(semantic=Semantic.Default)
            ),
        }
    )

    # Create target schema (polygon to mask conversion)
    target_schema = Schema(
        attributes={
            "mask": AttributeInfo(
                type=np.ndarray, annotation=MaskField(dtype=pl.UInt8, semantic=Semantic.Default)
            )
        }
    )

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
    input_polygon_field = PolygonField(
        dtype=pl.Float32, format=PolygonFormat.XY, normalize=False, semantic=Semantic.Default
    )
    image_info_field = ImageInfoField(semantic=Semantic.Default)
    output_instance_mask_field = InstanceMaskField(dtype=pl.Boolean, semantic=Semantic.Default)

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
    assert masks[0, 15, 15] == True  # Point inside triangle
    assert masks[0, 5, 5] == False  # Point outside triangle

    # Rectangle should be in second mask
    assert masks[1, 35, 35] == True  # Point inside rectangle
    assert masks[1, 5, 5] == False  # Point outside rectangle

    # Pentagon should be in third mask
    assert masks[2, 55, 55] == True  # Point inside pentagon
    assert masks[2, 5, 5] == False  # Point outside pentagon

    # No overlap between instances
    assert not np.any(masks[0] & masks[1])  # Triangle and rectangle don't overlap
    assert not np.any(masks[0] & masks[2])  # Triangle and pentagon don't overlap
    assert not np.any(masks[1] & masks[2])  # Rectangle and pentagon don't overlap


def test_polygon_to_instance_mask_converter_normalized():
    """Test conversion with normalized polygon coordinates."""
    # Create test data with normalized coordinates (0-1 range)
    polygon_coords1 = [[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]]
    polygon_coords2 = [[0.3, 0.3], [0.4, 0.3], [0.4, 0.4], [0.3, 0.4]]

    polygon_series = pl.Series(
        [polygon_coords1, polygon_coords2], dtype=pl.List(pl.Array(pl.Float32, 2))
    )

    df = pl.DataFrame(
        {
            "polygons": [polygon_series],
            "image_info": [{"width": 100, "height": 100}],
        }
    )

    # Create converter instance with normalized coordinates
    converter_instance = PolygonToInstanceMaskConverter()

    # Set up field specs
    input_polygon_field = PolygonField(
        dtype=pl.Float32, format=PolygonFormat.XY, normalize=True, semantic=Semantic.Default
    )
    image_info_field = ImageInfoField(semantic=Semantic.Default)
    output_instance_mask_field = InstanceMaskField(dtype=pl.Boolean, semantic=Semantic.Default)

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
    assert masks[0, 15, 15] == True  # Point inside the scaled triangle
    assert masks[0, 5, 5] == False  # Background point

    # Rectangle: 0.3 * 100 = 30, 0.4 * 100 = 40, etc.
    assert masks[1, 35, 35] == True  # Point inside the scaled rectangle
    assert masks[1, 5, 5] == False  # Background point


def test_instance_mask_callable_to_instance_mask_converter():
    """Test InstanceMaskCallableToInstanceMaskConverter conversion."""
    converter_instance = InstanceMaskCallableToInstanceMaskConverter()  # type: ignore[call-arg]

    # Create a test callable that returns instance masks
    def get_instance_masks():
        return np.array(
            [[[True, False], [False, True]], [[False, True], [True, False]]], dtype=bool
        )  # (2,2,2)

    df = pl.DataFrame(
        {
            "instance_mask_callable": [get_instance_masks],
        },
        schema=pl.Schema({"instance_mask_callable": pl.Object}),
    )

    # Set up converter attributes
    input_field = InstanceMaskCallableField(dtype=pl.Boolean, semantic=Semantic.Default)
    output_field = InstanceMaskField(dtype=pl.Boolean, semantic=Semantic.Default)

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
    input_field = InstanceMaskCallableField(dtype=pl.Boolean, semantic=Semantic.Default)
    output_field = InstanceMaskField(dtype=pl.Boolean, semantic=Semantic.Default)

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
    input_field = MaskCallableField(dtype=pl.UInt8, semantic=Semantic.Default)
    output_field = MaskField(dtype=pl.UInt8, semantic=Semantic.Default)

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
    input_field = MaskCallableField(dtype=pl.Boolean, semantic=Semantic.Default)
    output_field = InstanceMaskField(dtype=pl.Boolean, semantic=Semantic.Default)

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

    polygon_series = pl.Series(
        [polygon_coords1, polygon_coords2], dtype=pl.List(pl.Array(pl.Float32, 2))
    )

    df = pl.DataFrame(
        {
            "polygons": [polygon_series],
        }
    )

    # Create converter instance
    converter_instance = PolygonToBBoxConverter()

    # Set up field specs
    input_polygon_field = PolygonField(
        dtype=pl.Float32, format=PolygonFormat.XY, normalize=False, semantic=Semantic.Default
    )
    output_bbox_field = BBoxField(
        dtype=pl.Float32, format=BBoxFormat.X1Y1X2Y2, normalize=False, semantic=Semantic.Default
    )

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
    input_polygon_field = PolygonField(
        dtype=pl.Float32, format=PolygonFormat.XY, normalize=False, semantic=Semantic.Default
    )
    output_bbox_field = BBoxField(
        dtype=pl.Float32, format=BBoxFormat.XYWH, normalize=False, semantic=Semantic.Default
    )

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
    input_polygon_field = PolygonField(
        dtype=pl.Float32, format=PolygonFormat.XY, normalize=True, semantic=Semantic.Default
    )
    output_bbox_field = BBoxField(
        dtype=pl.Float32, format=BBoxFormat.X1Y1X2Y2, normalize=True, semantic=Semantic.Default
    )

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
    input_field = ImageCallableField(format="RGB", semantic=Semantic.Default)
    output_field = ImageField(dtype=pl.UInt8, format="RGB", semantic=Semantic.Default)
    output_info_field = ImageInfoField(semantic=Semantic.Default)

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
    input_field = ImageCallableField(format="RGB", semantic=Semantic.Default)
    output_field = ImageField(dtype=pl.UInt8, format="RGB", semantic=Semantic.Default)
    output_info_field = ImageInfoField(semantic=Semantic.Default)

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
    with pytest.raises(RuntimeError, match="must return numpy.ndarray"):
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
    input_field = ImageCallableField(format="RGB", semantic=Semantic.Default)
    output_field = ImageField(dtype=pl.UInt8, format="RGB", semantic=Semantic.Default)
    output_info_field = ImageInfoField(semantic=Semantic.Default)

    setattr(
        converter_instance, "input_callable", AttributeSpec(name="my_callable", field=input_field)
    )
    setattr(
        converter_instance, "output_image", AttributeSpec(name="output_image", field=output_field)
    )
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
    output_field_f32 = ImageField(dtype=pl.Float32, format="RGB", semantic=Semantic.Default)
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


def test_rotated_bbox_to_polygon_converter():
    """Test conversion from RotatedBBox to Polygon format."""
    converter_instance = RotatedBBoxToPolygonConverter()

    input_bbox = np.array([[100, 200, 50, 30, 45]], dtype=np.float32)
    df = pl.DataFrame({"bbox": [input_bbox]}, schema={"bbox": pl.Array(pl.Float32, 5)})

    input_field = RotatedBBoxField(semantic=Semantic.Default)
    output_field = PolygonField(semantic=Semantic.Default)

    setattr(converter_instance, "input_bbox", AttributeSpec("bbox", input_field))
    setattr(converter_instance, "output_polygon", AttributeSpec("polygon", output_field))

    assert converter_instance.filter_output_spec() is True

    result_df = converter_instance.convert(df)
    points = result_df["polygon"][0]

    # Should output 4 points (8 coordinates)
    assert len(points) == 8


def test_label_index_converter():
    """Test LabelIndexConverter functionality for remapping label indices."""

    # Create input and output specs with different label orders
    input_categories = LabelCategories(labels=("cat", "dog", "bird"))
    output_categories = LabelCategories(labels=("bird", "cat", "dog"))  # Different order

    input_spec = AttributeSpec(
        name="label",
        field=LabelField(semantic=Semantic.Default, dtype=pl.Int32, multi_label=False),
        categories=input_categories,
    )

    output_spec = AttributeSpec(
        name="label",
        field=LabelField(semantic=Semantic.Default, dtype=pl.Int32, multi_label=False),
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
        field=LabelField(semantic=Semantic.Default, dtype=pl.Int32, multi_label=True),
        categories=input_categories,
    )

    output_spec = AttributeSpec(
        name="labels",
        field=LabelField(semantic=Semantic.Default, dtype=pl.Int32, multi_label=True),
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
        field=LabelField(semantic=Semantic.Default, dtype=pl.Int32, multi_label=False),
        categories=categories,
    )

    output_spec = AttributeSpec(
        name="label",
        field=LabelField(semantic=Semantic.Default, dtype=pl.Int32, multi_label=False),
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
        field=LabelField(semantic=Semantic.Default, dtype=pl.Int32, multi_label=False),
        categories=input_categories,
    )

    output_spec = AttributeSpec(
        name="label",
        field=LabelField(semantic=Semantic.Default, dtype=pl.Int32, multi_label=False),
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
        field=LabelField(semantic=Semantic.Default, dtype=pl.Int32, multi_label=False),
        categories=input_categories,
    )

    output_spec = AttributeSpec(
        name="label",
        field=LabelField(semantic=Semantic.Default, dtype=pl.Int32, multi_label=False),
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
        field=LabelField(semantic=Semantic.Default, dtype=pl.Int32, multi_label=False),
        categories=input_categories,
    )

    output_spec = AttributeSpec(
        name="label",
        field=LabelField(semantic=Semantic.Default, dtype=pl.Int32, multi_label=False),
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

    rotated_bbox_series = pl.Series(
        [rotated_bbox_coords1, rotated_bbox_coords2], dtype=pl.Array(pl.Float32, 5)
    )

    df = pl.DataFrame(
        {
            "rotated_bboxes": [rotated_bbox_series],
        }
    )

    # Create converter instance
    converter_instance = RotatedBBoxToPolygonConverter()

    # Set up field specs
    input_rotated_bbox_field = RotatedBBoxField(
        dtype=pl.Float32, format="cxcywhr", normalize=False, semantic=Semantic.Default
    )
    output_polygon_field = PolygonField(
        dtype=pl.Float32, format="xy", normalize=False, semantic=Semantic.Default
    )

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


def test_bbox_format_converter_x1y1x2y2_to_xywh():
    """Test BBoxFormatConverter conversion from X1Y1X2Y2 to XYWH format."""
    # Create test data: (x1, y1, x2, y2) format
    test_data = [[10.0, 20.0, 50.0, 70.0], [0.0, 0.0, 100.0, 200.0]]

    df = pl.DataFrame({"bboxes": [test_data]}, schema={"bboxes": pl.List(pl.Array(pl.Float32, 4))})

    converter_instance = BBoxFormatConverter()

    # Set up converter attributes
    input_bbox_field = BBoxField(
        dtype=pl.Float32, format=BBoxFormat.X1Y1X2Y2, normalize=False, semantic=Semantic.Default
    )
    output_bbox_field = BBoxField(
        dtype=pl.Float32, format=BBoxFormat.XYWH, normalize=False, semantic=Semantic.Default
    )

    setattr(converter_instance, "input_bbox", AttributeSpec(name="bboxes", field=input_bbox_field))
    setattr(
        converter_instance,
        "output_bbox",
        AttributeSpec(name="bboxes_xywh", field=output_bbox_field),
    )

    # Test filter
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "bboxes_xywh" in result_df.columns
    result_bboxes = result_df["bboxes_xywh"][0]

    # First bbox: (10, 20, 50, 70) -> (10, 20, 40, 50)
    expected_bbox1 = [10.0, 20.0, 40.0, 50.0]  # x, y, w=50-10, h=70-20
    # Second bbox: (0, 0, 100, 200) -> (0, 0, 100, 200)
    expected_bbox2 = [0.0, 0.0, 100.0, 200.0]  # x, y, w=100-0, h=200-0

    assert np.allclose(result_bboxes[0], expected_bbox1)
    assert np.allclose(result_bboxes[1], expected_bbox2)


def test_bbox_format_converter_xywh_to_x1y1x2y2():
    """Test BBoxFormatConverter conversion from XYWH to X1Y1X2Y2 format."""
    # Create test data: (x, y, w, h) format
    test_data = [[10.0, 20.0, 40.0, 50.0], [0.0, 0.0, 100.0, 200.0]]

    df = pl.DataFrame({"bboxes": [test_data]}, schema={"bboxes": pl.List(pl.Array(pl.Float32, 4))})

    converter_instance = BBoxFormatConverter()

    # Set up converter attributes
    input_bbox_field = BBoxField(
        dtype=pl.Float32, format=BBoxFormat.XYWH, normalize=False, semantic=Semantic.Default
    )
    output_bbox_field = BBoxField(
        dtype=pl.Float32, format=BBoxFormat.X1Y1X2Y2, normalize=False, semantic=Semantic.Default
    )

    setattr(converter_instance, "input_bbox", AttributeSpec(name="bboxes", field=input_bbox_field))
    setattr(
        converter_instance,
        "output_bbox",
        AttributeSpec(name="bboxes_xyxy", field=output_bbox_field),
    )

    # Test filter
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "bboxes_xyxy" in result_df.columns
    result_bboxes = result_df["bboxes_xyxy"][0]

    # First bbox: (10, 20, 40, 50) -> (10, 20, 50, 70)
    expected_bbox1 = [10.0, 20.0, 50.0, 70.0]  # x1, y1, x2=x+w, y2=y+h
    # Second bbox: (0, 0, 100, 200) -> (0, 0, 100, 200)
    expected_bbox2 = [0.0, 0.0, 100.0, 200.0]  # x1, y1, x2=x+w, y2=y+h

    assert np.allclose(result_bboxes[0], expected_bbox1)
    assert np.allclose(result_bboxes[1], expected_bbox2)


def test_ellipse_format_converter_x1y1x2y2_to_cxcywh():
    """Test EllipseFormatConverter conversion from X1Y1X2Y2 to CXCYWH format."""
    # Create test data: (x1, y1, x2, y2) format
    test_data = [[10.0, 70.0, 50.0, 20.0], [0.0, 200.0, 100.0, 0.0]]

    df = pl.DataFrame(
        {"ellipses": [test_data]}, schema={"ellipses": pl.List(pl.Array(pl.Float32, 4))}
    )

    converter_instance = EllipseFormatConverter()

    # Set up converter attributes
    input_ellipse_field = EllipseField(
        dtype=pl.Float32, format=EllipseFormat.X1Y1X2Y2, normalize=False, semantic=Semantic.Default
    )
    output_ellipse_field = EllipseField(
        dtype=pl.Float32, format=EllipseFormat.CXCYWH, normalize=False, semantic=Semantic.Default
    )

    setattr(
        converter_instance,
        "input_ellipse",
        AttributeSpec(name="ellipses", field=input_ellipse_field),
    )
    setattr(
        converter_instance,
        "output_ellipse",
        AttributeSpec(name="ellipse_cxcywh", field=output_ellipse_field),
    )

    # Test filter
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "ellipses_cxcywh" in result_df.columns
    result_ellipses = result_df["ellipses_cxcywh"][0]

    # First ellipse: (10, 70, 50, 20) -> (10, 20, 40, 50)
    expected_ellipse1 = [20.0, 45.0, 40.0, 50.0]  # x, y, w=50-10, h=70-20
    # Second ellipse: (0, 200, 100, 0) -> (50, 100, 100, 200)
    expected_ellipse2 = [0.0, 100.0, 100.0, 200.0]  # x, y, w=100-0, h=200-0

    assert np.allclose(result_ellipses[0], expected_ellipse1)
    assert np.allclose(result_ellipses[1], expected_ellipse2)


def test_ellipse_format_converter_cxcywh_to_x1y1x2y2():
    """Test EllipseFormatConverter conversion from CXCYWH to X1Y1X2Y2 format."""
    # Create test data: (cx, cy, w, h) format
    test_data = [[20.0, 30.0, 40.0, 50.0], [50.0, 100.0, 100.0, 200.0]]

    df = pl.DataFrame(
        {"ellipses": [test_data]}, schema={"ellipses": pl.List(pl.Array(pl.Float32, 4))}
    )

    converter_instance = EllipseFormatConverter()

    # Set up converter attributes
    input_ellipse_field = EllipseField(
        dtype=pl.Float32, format=EllipseFormat.CXCYWH, normalize=False, semantic=Semantic.Default
    )
    output_ellipse_field = EllipseField(
        dtype=pl.Float32, format=EllipseFormat.X1Y1X2Y2, normalize=False, semantic=Semantic.Default
    )

    setattr(
        converter_instance,
        "input_ellipse",
        AttributeSpec(name="ellipses", field=input_ellipse_field),
    )
    setattr(
        converter_instance,
        "output_ellipse",
        AttributeSpec(name="ellipses_x1y1x2y2", field=output_ellipse_field),
    )

    # Test filter
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "ellipses_x1y1x2y2" in result_df.columns
    result_ellipses = result_df["ellipses_x1y1x2y2"][0]

    # First ellipse: (20, 30, 40, 50) -> (0, 55, 40, 5)
    expected_ellipse1 = [0.0, 55.0, 40.0, 5.0]  # cx +/- w/2 and cy +/- h/2
    # Seond ellipse: (50, 100, 100, 200) -> (0, 200, 100, 0)
    expected_ellipse2 = [0.0, 200.0, 100.0, 0.0]  # cx +/- w/2 and cy +/- h/2

    assert np.allclose(result_ellipses[0], expected_ellipse1)
    assert np.allclose(result_ellipses[1], expected_ellipse2)


def test_rotated_bbox_format_converter_radians_to_degrees():
    """Test RotatedBBoxFormatConverter conversion from radians to degrees."""
    # Create test data: (cx, cy, w, h, r) format with radians
    test_data = [
        [50.0, 60.0, 30.0, 20.0, 0.0],  # 0 radians
        [100.0, 120.0, 40.0, 25.0, np.pi / 2],  # π/2 radians = 90 degrees
        [150.0, 180.0, 50.0, 30.0, np.pi],  # π radians = 180 degrees
    ]

    df = pl.DataFrame(
        {"rotated_bboxes": [test_data]}, schema={"rotated_bboxes": pl.List(pl.Array(pl.Float32, 5))}
    )

    converter_instance = RotatedBBoxFormatConverter()

    # Set up converter attributes
    input_field = RotatedBBoxField(
        dtype=pl.Float32,
        format=RotatedBBoxFormat.CXCYWHR,
        normalize=False,
        semantic=Semantic.Default,
    )
    output_field = RotatedBBoxField(
        dtype=pl.Float32,
        format=RotatedBBoxFormat.CXCYWHA,
        normalize=False,
        semantic=Semantic.Default,
    )

    setattr(
        converter_instance,
        "input_rotated_bbox",
        AttributeSpec(name="rotated_bboxes", field=input_field),
    )
    setattr(
        converter_instance,
        "output_rotated_bbox",
        AttributeSpec(name="rotated_bboxes_degrees", field=output_field),
    )

    # Test filter
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "rotated_bboxes_degrees" in result_df.columns
    result_bboxes = result_df["rotated_bboxes_degrees"][0]

    # Check conversions
    expected_results = [
        [50.0, 60.0, 30.0, 20.0, 0.0],  # 0 radians -> 0 degrees
        [100.0, 120.0, 40.0, 25.0, 90.0],  # π/2 radians -> 90 degrees
        [150.0, 180.0, 50.0, 30.0, 180.0],  # π radians -> 180 degrees
    ]

    for i, expected in enumerate(expected_results):
        assert np.allclose(result_bboxes[i], expected, atol=1e-6)


def test_rotated_bbox_format_converter_degrees_to_radians():
    """Test RotatedBBoxFormatConverter conversion from degrees to radians."""
    # Create test data: (cx, cy, w, h, a) format with degrees
    test_data = [
        [50.0, 60.0, 30.0, 20.0, 0.0],  # 0 degrees
        [100.0, 120.0, 40.0, 25.0, 90.0],  # 90 degrees
        [150.0, 180.0, 50.0, 30.0, 180.0],  # 180 degrees
    ]

    df = pl.DataFrame(
        {"rotated_bboxes": [test_data]}, schema={"rotated_bboxes": pl.List(pl.Array(pl.Float32, 5))}
    )

    converter_instance = RotatedBBoxFormatConverter()

    # Set up converter attributes
    input_field = RotatedBBoxField(
        dtype=pl.Float32,
        format=RotatedBBoxFormat.CXCYWHA,
        normalize=False,
        semantic=Semantic.Default,
    )
    output_field = RotatedBBoxField(
        dtype=pl.Float32,
        format=RotatedBBoxFormat.CXCYWHR,
        normalize=False,
        semantic=Semantic.Default,
    )

    setattr(
        converter_instance,
        "input_rotated_bbox",
        AttributeSpec(name="rotated_bboxes", field=input_field),
    )
    setattr(
        converter_instance,
        "output_rotated_bbox",
        AttributeSpec(name="rotated_bboxes_radians", field=output_field),
    )

    # Test filter
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "rotated_bboxes_radians" in result_df.columns
    result_bboxes = result_df["rotated_bboxes_radians"][0]

    # Check conversions
    expected_results = [
        [50.0, 60.0, 30.0, 20.0, 0.0],  # 0 degrees -> 0 radians
        [100.0, 120.0, 40.0, 25.0, np.pi / 2],  # 90 degrees -> π/2 radians
        [150.0, 180.0, 50.0, 30.0, np.pi],  # 180 degrees -> π radians
    ]

    for i, expected in enumerate(expected_results):
        assert np.allclose(result_bboxes[i], expected, atol=1e-6)


def test_polygon_format_converter_xy_to_yx():
    """Test PolygonFormatConverter conversion from XY to YX format."""
    # Create test polygon data: [(x1, y1), (x2, y2), (x3, y3)]
    polygon1 = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
    polygon2 = [[100.0, 200.0], [300.0, 400.0]]

    df = pl.DataFrame(
        {"polygons": [[polygon1, polygon2]]},
        schema={"polygons": pl.List(pl.List(pl.Array(pl.Float32, 2)))},
    )

    converter_instance = PolygonFormatConverter()

    # Set up converter attributes
    input_field = PolygonField(
        dtype=pl.Float32, format=PolygonFormat.XY, normalize=False, semantic=Semantic.Default
    )
    output_field = PolygonField(
        dtype=pl.Float32, format=PolygonFormat.YX, normalize=False, semantic=Semantic.Default
    )

    setattr(converter_instance, "input_polygon", AttributeSpec(name="polygons", field=input_field))
    setattr(
        converter_instance, "output_polygon", AttributeSpec(name="polygons_yx", field=output_field)
    )

    # Test filter
    assert converter_instance.filter_output_spec() is True

    # Test conversion
    result_df = converter_instance.convert(df)

    assert "polygons_yx" in result_df.columns
    result_polygons = result_df["polygons_yx"][0]

    # Check that coordinates are swapped: (x, y) -> (y, x)
    expected_polygon1 = [[20.0, 10.0], [40.0, 30.0], [60.0, 50.0]]  # Swapped coordinates
    expected_polygon2 = [[200.0, 100.0], [400.0, 300.0]]  # Swapped coordinates

    assert len(result_polygons) == 2
    assert np.allclose(result_polygons[0], expected_polygon1)
    assert np.allclose(result_polygons[1], expected_polygon2)
