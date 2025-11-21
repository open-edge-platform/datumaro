import polars as pl
from typing_extensions import Annotated

from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields.images import image_id_field


def test_image_id_field_int32_roundtrip():
    """Test ImageIdField with Int32 type (default)."""

    class ImageSample(Sample):
        image_id: Annotated[int, image_id_field()]

    ds = Dataset(ImageSample)

    s = ImageSample(image_id=12345)
    ds.append(s)

    assert len(ds) == 1
    out = ds[0]
    assert isinstance(out, ImageSample)
    assert out.image_id == 12345
    assert isinstance(out.image_id, int)


def test_image_id_field_int64_roundtrip():
    """Test ImageIdField with Int64 type."""

    class ImageSample(Sample):
        image_id: Annotated[int, image_id_field(dtype=pl.Int64())]

    ds = Dataset(ImageSample)

    # Test with a large 64-bit integer
    large_id = 9223372036854775807  # Max int64 value
    s = ImageSample(image_id=large_id)
    ds.append(s)

    assert len(ds) == 1
    out = ds[0]
    assert isinstance(out, ImageSample)
    assert out.image_id == large_id


def test_image_id_field_string_roundtrip():
    """Test ImageIdField with string type."""

    class ImageSample(Sample):
        image_id: Annotated[str, image_id_field(dtype=pl.Utf8())]

    ds = Dataset(ImageSample)

    s = ImageSample(image_id="ILSVRC2012_val_00001234")
    ds.append(s)

    assert len(ds) == 1
    out = ds[0]
    assert isinstance(out, ImageSample)
    assert out.image_id == "ILSVRC2012_val_00001234"
    assert isinstance(out.image_id, str)


def test_image_id_field_optional_int_with_value():
    """Test optional ImageIdField with a valid integer value."""

    class ImageSample(Sample):
        image_id: Annotated[int | None, image_id_field(dtype=pl.Int32())]

    ds = Dataset(ImageSample)

    s = ImageSample(image_id=999)
    ds.append(s)

    assert len(ds) == 1
    out = ds[0]
    assert isinstance(out, ImageSample)
    assert out.image_id == 999


def test_image_id_field_optional_int_with_none():
    """Test optional ImageIdField with None value."""

    class ImageSample(Sample):
        image_id: Annotated[int | None, image_id_field(dtype=pl.Int32())]

    ds = Dataset(ImageSample)

    s = ImageSample(image_id=None)
    ds.append(s)

    assert len(ds) == 1
    out = ds[0]
    assert isinstance(out, ImageSample)
    assert out.image_id is None


def test_image_id_field_optional_string_with_value():
    """Test optional string ImageIdField with a valid value."""

    class ImageSample(Sample):
        image_id: Annotated[str | None, image_id_field(dtype=pl.Utf8())]

    ds = Dataset(ImageSample)

    s = ImageSample(image_id="img_001")
    ds.append(s)

    assert len(ds) == 1
    out = ds[0]
    assert isinstance(out, ImageSample)
    assert out.image_id == "img_001"


def test_image_id_field_optional_string_with_none():
    """Test optional string ImageIdField with None value."""

    class ImageSample(Sample):
        image_id: Annotated[str | None, image_id_field(dtype=pl.Utf8())]

    ds = Dataset(ImageSample)

    s = ImageSample(image_id=None)
    ds.append(s)

    assert len(ds) == 1
    out = ds[0]
    assert isinstance(out, ImageSample)
    assert out.image_id is None
