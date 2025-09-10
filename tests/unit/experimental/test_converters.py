"""
Unit tests for converter registry and converter implementations.
"""

import os
import tempfile

import numpy as np
import polars as pl
import pytest

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
    ImagePathToImageConverter,
    PolygonToMaskConverter,
    RGBToBGRConverter,
    UInt8ToFloat32Converter,
)
from datumaro.experimental.fields import (
    BBoxField,
    Field,
    ImageField,
    ImageInfoField,
    ImagePathField,
    LabelField,
    MaskField,
    PolygonField,
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
    input_field = ImageField(dtype=pl.UInt8, format="RGB", semantic=Semantic.Default)
    output_field = ImageField(dtype=pl.Float32, format="RGB", semantic=Semantic.Default)

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
        output_field = ImageField(dtype=pl.UInt8, format="RGB", semantic=Semantic.Default)

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

        # Test filter - should return True for path->image conversion
        assert converter_instance.filter_output_spec() is True

        # Test conversion
        result_df = converter_instance.convert(df)

        assert "image" in result_df.columns
        assert "image_shape" in result_df.columns

        # Check that image was loaded correctly
        result_shape = list(result_df["image_shape"][0])
        assert result_shape == [50, 75, 3]  # height, width, channels


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
    path = find_conversion_path(source_schema, target_schema)
    assert len(path.batch_converters) == 2
    assert type(path.batch_converters[0]) is UInt8ToFloat32Converter
    assert type(path.batch_converters[1]) is AttributeRemapperConverter


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
    conversion_paths = find_conversion_path(source_schema, target_schema)

    # Apply batch converters first
    result_df = df
    for converter in conversion_paths.batch_converters:
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

    path = find_conversion_path(source_schema, target_schema)
    # If successful, should have multiple steps
    assert len(path.batch_converters) == 4
    assert type(path.batch_converters[0]) is BBoxCoordinateConverter
    assert type(path.batch_converters[1]) is RGBToBGRConverter
    assert type(path.batch_converters[2]) is UInt8ToFloat32Converter
    assert type(path.batch_converters[3]) is AttributeRemapperConverter


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
    path = find_conversion_path(from_schema, to_schema)
    assert len(path.batch_converters) == 2
    assert type(path.batch_converters[0]) is ExtractImageSizeConverter
    assert type(path.batch_converters[1]) is AttributeRemapperConverter
    assert path.lazy_converters == []


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
    path = find_conversion_path(from_schema, to_schema)
    # Should need 2 converters: ImageField -> ImageSizeField, then NormalizedBbox -> AbsoluteBbox
    assert len(path.batch_converters) == 3
    assert type(path.batch_converters[0]) is ExtractImageSizeChainConverter
    assert type(path.batch_converters[1]) is NormalizeToAbsoluteBboxConverter
    assert type(path.batch_converters[2]) is AttributeRemapperConverter
    assert len(path.lazy_converters) == 0


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

    path = find_conversion_path(schema, schema)
    total_converters = len(path.batch_converters) + len(path.lazy_converters)
    assert (
        total_converters == 0
    ), f"Expected 0 converters for identical schemas, got {total_converters}"


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

    path = find_conversion_path(from_schema, to_schema)
    assert len(path.batch_converters) == 2
    assert type(path.batch_converters[0]) is AToCDirectConverter
    assert type(path.batch_converters[1]) is AttributeRemapperConverter
    assert len(path.lazy_converters) == 0


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

    path = find_conversion_path(from_schema, to_schema)
    assert len(path.batch_converters) == 2
    assert type(path.batch_converters[0]) is GenerateBConverter
    assert type(path.batch_converters[1]) is AttributeRemapperConverter
    assert len(path.lazy_converters) == 0


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

    path = find_conversion_path(from_schema, to_schema)
    assert len(path.batch_converters) == 2
    assert type(path.batch_converters[0]) is AToMultiConverter
    assert type(path.batch_converters[1]) is AttributeRemapperConverter
    assert len(path.lazy_converters) == 0


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

    path = find_conversion_path(from_schema, to_schema)
    assert len(path.batch_converters) == 2
    assert type(path.batch_converters[0]) is AToCPartialConverter
    assert type(path.batch_converters[1]) is AttributeRemapperConverter
    assert len(path.lazy_converters) == 0


def test_attribute_renaming():
    """Test attribute renaming functionality using special converters."""

    # Create source schema with an image field named "input_image"
    source_schema = Schema(
        {
            "input_image": AttributeInfo(
                type=str, annotation=image_field(dtype=pl.UInt8(), format="RGB")
            ),
        }
    )

    # Create target schema with the same field type but different name "output_image"
    target_schema = Schema(
        {
            "output_image": AttributeInfo(
                type=str, annotation=image_field(dtype=pl.UInt8(), format="RGB")
            ),
        }
    )

    # Find conversion path - should include a remapper converter
    path = find_conversion_path(source_schema, target_schema)

    # Should have exactly one batch converter for renaming
    assert len(path.batch_converters) == 1
    assert isinstance(path.batch_converters[0], AttributeRemapperConverter)
    assert len(path.lazy_converters) == 0

    # Test the remapper converter functionality
    remapper_converter = path.batch_converters[0]

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


def test_attribute_deletion():
    """Test attribute deletion functionality using AttributeRenameConverter."""

    # Create source schema with two fields
    source_schema = Schema(
        {
            "image": AttributeInfo(
                type=str, annotation=image_field(dtype=pl.UInt8(), format="RGB")
            ),
            "bbox": AttributeInfo(
                type=str, annotation=bbox_field(dtype=pl.Float32(), format="x1y1x2y2")
            ),
        }
    )

    # Create target schema with only the image field (bbox should be deleted)
    target_schema = Schema(
        {
            "image": AttributeInfo(
                type=str, annotation=image_field(dtype=pl.UInt8(), format="RGB")
            ),
        }
    )

    # Find conversion path - should include a remapper converter that only keeps image
    path = find_conversion_path(source_schema, target_schema)

    # Should have exactly one batch converter for selection/deletion
    assert len(path.batch_converters) == 1
    assert isinstance(path.batch_converters[0], AttributeRemapperConverter)
    assert len(path.lazy_converters) == 0

    # Test the remapper converter functionality
    remapper_converter = path.batch_converters[0]

    # Create test DataFrame with both columns
    test_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    test_bbox = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)

    df = pl.DataFrame(
        {
            "image": test_image.reshape(1, -1),
            "image_shape": [list(test_image.shape)],
            "bbox": test_bbox.reshape(1, -1, 4),
        }
    )

    # Apply the remapper converter
    result_df = remapper_converter.convert(df)

    # Check that bbox column was removed but image columns remain
    assert "image" in result_df.columns
    assert "image_shape" in result_df.columns
    assert "bbox" not in result_df.columns

    # Check that image data was preserved
    assert result_df["image"][0].to_list() == test_image.flatten().tolist()
    assert result_df["image_shape"][0].to_list() == list(test_image.shape)


def test_combined_rename_and_delete():
    """Test scenario with both renaming and deletion operations."""

    # Create source schema with three fields
    source_schema = Schema(
        {
            "old_image": AttributeInfo(
                type=str, annotation=image_field(dtype=pl.UInt8(), format="RGB")
            ),
            "bbox": AttributeInfo(
                type=str, annotation=bbox_field(dtype=pl.Float32(), format="x1y1x2y2")
            ),
            "extra_field": AttributeInfo(type=str, annotation=image_info_field()),
        }
    )

    # Create target schema with renamed image and no bbox or extra_field
    target_schema = Schema(
        {
            "new_image": AttributeInfo(
                type=str, annotation=image_field(dtype=pl.UInt8(), format="RGB")
            ),
        }
    )

    # Find conversion path - should include a single remapper converter that handles both operations
    path = find_conversion_path(source_schema, target_schema)

    # Should have exactly one batch converter that handles both renaming and deletion
    assert len(path.batch_converters) == 1
    assert isinstance(path.batch_converters[0], AttributeRemapperConverter)
    assert len(path.lazy_converters) == 0

    # Check that the remapper handles the conversion correctly by testing on sample data
    remapper_converter = path.batch_converters[0]

    # Create test DataFrame
    test_df = pl.DataFrame(
        {
            "old_image": [[1, 2, 3, 4, 5, 6]],  # Sample image data
            "old_image_shape": [[2, 3]],  # Sample shape data
            "other_field": ["should_be_deleted"],  # Should be deleted
        }
    )

    # Apply the converter
    result_df = remapper_converter.convert(test_df)

    # Check that only the renamed fields are present
    expected_columns = {"new_image", "new_image_shape"}
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

    # Check that columns were remapped correctly (should only have remapped columns)
    expected_columns = {"new_image", "new_image_shape", "new_bbox"}
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
            "labels": [[1, 2, 3]],  # Corresponding labels for each polygon
            "image_info": [{"width": 100, "height": 100}],  # Image dimensions
        }
    )

    # Set up converter attributes
    input_polygon_field = PolygonField(
        dtype=pl.Float32, format="xy", normalize=False, semantic=Semantic.Default
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
        "image_info",
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

    # Check that triangle area has label 1
    assert mask[15, 20] == 1  # Point inside triangle

    # Check that rectangle area has label 2
    assert mask[50, 50] == 2  # Point inside rectangle

    # Check that pentagon area has label 3
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
        format="xy",
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
        converter_instance, "image_info", AttributeSpec(name="image_info", field=image_info_field)
    )
    setattr(converter_instance, "output_mask", AttributeSpec(name="mask", field=output_mask_field))

    # Test conversion
    result_df = converter_instance.convert(df)

    # Get the mask and check it
    mask_data = np.array(result_df["mask"][0])
    mask_shape = result_df["mask_shape"][0]
    mask = mask_data.reshape(mask_shape)

    # Check that polygon was filled with label 5
    # Normalized coordinates should be scaled: 0.2 * 100 = 20, 0.1 * 100 = 10, etc.
    assert mask[15, 20] == 5  # Point inside the scaled triangle
    assert mask[5, 5] == 0  # Background point

    # Verify the label is present in the mask
    assert 5 in np.unique(mask)
