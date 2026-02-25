"""
Tests for export and import functionality.

This test suite covers:
- Exporting datasets with various field types (ImageCallableField, ImagePathField, InstanceMaskCallableField)
- Importing datasets from directories and ZIP files
- Image export in different formats (PNG, JPEG preservation)
- Metadata serialization and deserialization
- Schema and categories preservation
- Object column handling
"""

import json
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import pytest
from PIL import Image as PILImage

from datumaro.experimental.categories import LabelCategories, MaskCategories
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.export_import import (
    DATAFRAME_FILE,
    IMAGES_DIR,
    METADATA_FILE,
    VERSION,
    _export_images_from_dataset,
    _get_registered_samples,
    _match_dtype_from_schema,
    _sample_registry,
    export_dataset,
    import_dataset,
    register_sample,
)
from datumaro.experimental.fields import (
    ImageInfo,
    Subset,
    TileInfo,
    bbox_field,
    image_callable_field,
    image_info_field,
    image_path_field,
    instance_mask_callable_field,
    keypoints_field,
    label_field,
    mask_callable_field,
    numeric_field,
    rotated_bbox_field,
    subset_field,
    tensor_field,
    tile_field,
)

LABEL_CATEGORIES = LabelCategories(labels=("apple", "orange", "pear", "mango", "coconut"))
MASK_CATEGORIES = MaskCategories.generate(size=256)


def test_export_no_image_fields(tmp_path):
    """Test that datasets without image fields return the original dataframe."""

    class SimpleSample(Sample):
        label: int = label_field()

    dataset = Dataset(
        dtype_or_schema=SimpleSample,
        categories={"label": LABEL_CATEGORIES},
    )
    dataset.append(SimpleSample(label=1))

    output_dir = tmp_path / "output"
    result = _export_images_from_dataset(dataset, output_dir)

    assert result.equals(dataset.df)


def test_export_image_callable_field(tmp_path):
    """Test exporting datasets with ImageCallableField."""

    class CallableSample(Sample):
        image: Callable[[], np.ndarray] = image_callable_field(format="RGB")

    # Create sample images as callables
    def make_image_callable(idx):
        def load_image():
            # Create a simple test image
            img = np.zeros((50, 60, 3), dtype=np.uint8)
            img[:, :, 0] = idx * 50  # Varying red channel
            return img

        return load_image

    dataset = Dataset(CallableSample)
    dataset.append(CallableSample(image=make_image_callable(0)))
    dataset.append(CallableSample(image=make_image_callable(1)))
    dataset.append(CallableSample(image=make_image_callable(2)))

    output_dir = tmp_path / "output"
    result = _export_images_from_dataset(dataset, output_dir)

    # Check that images were saved
    assert "image" in result
    assert len(result["image"]) == 3
    for idx in range(3):
        rel_path = f"image_{idx:06d}.png"
        assert rel_path in result["image"]

        # Verify file exists and is valid
        img_file = output_dir / rel_path
        assert img_file.exists()

        # Load and verify image content
        loaded_img = np.array(PILImage.open(img_file))
        assert loaded_img.shape == (50, 60, 3)
        assert loaded_img[0, 0, 0] == idx * 50  # Check red channel


def test_export_image_path_field(tmp_path):
    """Test exporting datasets with ImagePathField (copies files directly)."""

    class PathSample(Sample):
        image_path: str = image_path_field()

    # Create source images
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    image_paths = []
    for idx in range(3):
        # Create images with different formats
        ext = [".png", ".jpg", ".jpeg"][idx % 3]
        img_path = source_dir / f"image_{idx}{ext}"

        img = np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)
        pil_img = PILImage.fromarray(img)
        if ext == ".jpg" or ext == ".jpeg":
            pil_img.save(img_path, "JPEG")
        else:
            pil_img.save(img_path)

        image_paths.append(str(img_path))

    dataset = Dataset(PathSample)
    for path in image_paths:
        dataset.append(PathSample(image_path=path))

    output_dir = tmp_path / "output"
    result = _export_images_from_dataset(dataset, output_dir)

    # Check that images were copied
    assert "image_path" in result
    assert len(result["image_path"]) == 3

    for idx in range(3):
        rel_path = result["image_path"][idx]

        # Verify the file exists and format is preserved
        img_file = output_dir / rel_path
        assert img_file.exists()

        # Check that the file extension is preserved
        expected_ext = [".png", ".jpg", ".jpeg"][idx % 3]
        assert img_file.suffix == expected_ext


def test_export_instance_mask_callable_field(tmp_path):
    """Test exporting datasets with InstanceMaskCallableField."""

    class MaskSample(Sample):
        mask: Callable[[], np.ndarray] = instance_mask_callable_field()

    def make_mask_callable(idx):
        def load_mask():
            # Create a simple mask
            mask = np.zeros((30, 40), dtype=np.uint8)
            mask[5:25, 5:35] = idx + 1  # Different values per mask
            return mask

        return load_mask

    dataset = Dataset(dtype_or_schema=MaskSample, categories={"mask": MASK_CATEGORIES})
    for idx in range(3):
        dataset.append(MaskSample(mask=make_mask_callable(idx)))

    output_dir = tmp_path / "output"
    result = _export_images_from_dataset(dataset, output_dir)

    # Check that masks were saved
    assert "mask" in result
    assert len(result["mask"]) == 3

    for idx in range(3):
        rel_path = result["mask"][idx]
        assert rel_path == f"mask_{idx:06d}.png"

        # Verify file exists
        mask_file = output_dir / rel_path
        assert mask_file.exists()

        # Load and verify mask content
        loaded_mask = np.array(PILImage.open(mask_file))
        assert loaded_mask.shape == (30, 40)
        # Check that the mask has the expected value in center region
        assert loaded_mask[10, 10] == idx + 1


def test_export_mixed_fields(tmp_path):
    """Test exporting datasets with multiple image field types."""

    class MixedSample(Sample):
        image: Callable[[], np.ndarray] = image_callable_field()
        image_path: str = image_path_field()
        mask: Callable[[], np.ndarray] = instance_mask_callable_field()

    # Create source image
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    img_path = source_dir / "source.png"
    PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(img_path)

    def make_image():
        return np.ones((20, 20, 3), dtype=np.uint8) * 100

    def make_mask():
        return np.ones((15, 15), dtype=np.uint8) * 255

    dataset = Dataset(dtype_or_schema=MixedSample, categories={"mask": MASK_CATEGORIES})
    dataset.append(
        MixedSample(
            image=make_image,
            image_path=str(img_path),
            mask=make_mask,
        )
    )

    output_dir = tmp_path / "output"
    result = _export_images_from_dataset(dataset, output_dir)

    # Check all three fields were exported
    assert "image" in result
    assert "image_path" in result
    assert "mask" in result
    assert len(result["image"]) == 1
    assert len(result["image_path"]) == 1
    assert len(result["mask"]) == 1


def test_export_with_none_values(tmp_path):
    """Test exporting datasets with None values in image fields."""

    class OptionalImageSample(Sample):
        image: Callable[[], np.ndarray] | None = image_callable_field()

    def make_image():
        return np.zeros((10, 10, 3), dtype=np.uint8)

    dataset = Dataset(OptionalImageSample)
    dataset.append(OptionalImageSample(image=make_image))
    dataset.append(OptionalImageSample(image=None))
    dataset.append(OptionalImageSample(image=make_image))

    output_dir = tmp_path / "output"
    result = _export_images_from_dataset(dataset, output_dir, ignore_missing_images=True)

    # Only indices 0 and 2 should be in result
    assert "image" in result
    assert len(result["image"]) == 2
    assert "image_000000.png" in result["image"]
    assert "image_000001.png" not in result["image"]
    assert "image_000002.png" in result["image"]


def test_export_images_error_all_image_fields_null(tmp_path):
    """Error when a row has null values for all image fields and skip_missing_images=False."""

    class AllOptionalImagesSample(Sample):
        image: Callable[[], np.ndarray] | None = image_callable_field()
        image_path: str | None = image_path_field()

    dataset = Dataset(AllOptionalImagesSample)
    # Single row where all image-related fields are None
    dataset.append(AllOptionalImagesSample(image=None, image_path=None))

    output_dir = tmp_path / "output_all_null"
    with pytest.raises(ValueError, match="image"):
        _export_images_from_dataset(dataset, output_dir)


def test_export_images_error_missing_image_path_file(tmp_path):
    """Error when an ImagePathField points to a non-existent file and skip_missing_images=False."""

    class PathImageSample(Sample):
        image_path: str = image_path_field()

    dataset = Dataset(PathImageSample)
    missing_path = tmp_path / "does_not_exist.png"
    dataset.append(PathImageSample(image_path=str(missing_path)))

    output_dir = tmp_path / "output_missing_path"
    with pytest.raises(ValueError, match="image"):
        _export_images_from_dataset(dataset, output_dir)


def test_export_images_error_failing_image_callable(tmp_path):
    """Error when an ImageCallableField cannot generate an image and skip_missing_images=False."""

    class CallableImageSample(Sample):
        image: Callable[[], np.ndarray] = image_callable_field()

    def bad_image():
        raise RuntimeError("failed to generate image")

    dataset = Dataset(CallableImageSample)
    dataset.append(CallableImageSample(image=bad_image))

    output_dir = tmp_path / "output_bad_callable"
    with pytest.raises(ValueError, match="image"):
        _export_images_from_dataset(dataset, output_dir)


def test_export_basic_dataset_to_directory(tmp_path):
    """Test basic export to directory."""

    class SimpleSample(Sample):
        label: int = label_field()
        score: float = numeric_field(dtype=pl.Float32(), semantic="score")

    dataset = Dataset(SimpleSample, categories={"label": LABEL_CATEGORIES})
    dataset.append(SimpleSample(label=1, score=0.9))
    dataset.append(SimpleSample(label=2, score=0.8))

    output_dir = tmp_path / "export"
    export_dataset(dataset, output_dir, export_images=False, as_zip=False)

    # Check directory structure
    assert output_dir.exists()
    assert (output_dir / METADATA_FILE).exists()
    assert (output_dir / DATAFRAME_FILE).exists()

    # Check metadata content
    with open(output_dir / METADATA_FILE) as f:
        metadata = json.load(f)
    assert metadata["version"] == VERSION
    assert "schema" in metadata

    # Check dataframe can be loaded
    df = pl.read_parquet(output_dir / DATAFRAME_FILE)
    assert len(df) == 2
    assert "label" in df.columns
    assert "score" in df.columns


def test_export_dataset_with_images_to_directory(tmp_path):
    """Test export with images to directory."""

    class ImageSample(Sample):
        image: Callable[[], np.ndarray] = image_callable_field()
        label: int = label_field()

    def make_image(idx):
        def load_image():
            img = np.zeros((30, 40, 3), dtype=np.uint8)
            img[:, :, 0] = idx * 50
            return img

        return load_image

    dataset = Dataset(ImageSample, categories={"label": LABEL_CATEGORIES})
    dataset.append(ImageSample(image=make_image(0), label=1))
    dataset.append(ImageSample(image=make_image(1), label=2))

    output_dir = tmp_path / "export"
    export_dataset(dataset, output_dir, export_images=True, as_zip=False)

    # Check directory structure
    assert output_dir.exists()
    assert (output_dir / METADATA_FILE).exists()
    assert (output_dir / DATAFRAME_FILE).exists()
    assert (output_dir / IMAGES_DIR).exists()

    # Check images were exported
    images_dir = output_dir / IMAGES_DIR
    assert len(list(images_dir.glob("image_*.png"))) == 2


def test_export_dataset_as_zip(tmp_path):
    """Test export as ZIP file."""

    class SimpleSample(Sample):
        label: int = label_field()

    dataset = Dataset(SimpleSample, categories={"label": LABEL_CATEGORIES})
    dataset.append(SimpleSample(label=1))

    output_zip = tmp_path / "export.zip"
    export_dataset(dataset, output_zip, export_images=False, as_zip=True)

    # Check ZIP file was created
    assert output_zip.exists()
    assert output_zip.suffix == ".zip"

    # Extract and verify contents

    with zipfile.ZipFile(output_zip) as zf:
        namelist = zf.namelist()
        assert METADATA_FILE in namelist
        assert DATAFRAME_FILE in namelist


def test_export_with_object_columns(tmp_path):
    """Test that object columns are tracked in metadata."""

    class CallableSample(Sample):
        image: Callable[[], np.ndarray] = image_callable_field()
        label: int = label_field()

    def make_image():
        return np.zeros((10, 10, 3), dtype=np.uint8)

    dataset = Dataset(CallableSample, categories={"label": LABEL_CATEGORIES})
    dataset.append(CallableSample(image=make_image, label=1))

    output_dir = tmp_path / "export"
    export_dataset(dataset, output_dir, export_images=False, as_zip=False)

    # Check metadata tracks object columns
    with open(output_dir / METADATA_FILE) as f:
        metadata = json.load(f)
    assert "object_columns" in metadata
    assert "image" in metadata["object_columns"]


def test_export_dataset_with_categories(tmp_path):
    """Test export preserves category information."""

    class LabeledSample(Sample):
        label: int = label_field()

    # Create dataset with categories
    categories = LabelCategories(labels=("cat", "dog"))

    dataset = Dataset(LabeledSample, categories={"label": categories})
    dataset.append(LabeledSample(label=0))
    dataset.append(LabeledSample(label=1))

    output_dir = tmp_path / "export"
    export_dataset(dataset, output_dir, export_images=False, as_zip=False)

    # Check schema is preserved
    with open(output_dir / METADATA_FILE) as f:
        metadata = json.load(f)
    assert "schema" in metadata
    schema_dict = metadata["schema"]
    assert "label" in schema_dict["categories"]


def test_export_images_false_skips_image_export(tmp_path):
    """Test that export_images=False skips image export."""

    class ImageSample(Sample):
        image: Callable[[], np.ndarray] = image_callable_field()

    def make_image():
        return np.zeros((10, 10, 3), dtype=np.uint8)

    dataset = Dataset(ImageSample)
    dataset.append(ImageSample(image=make_image))

    output_dir = tmp_path / "export"
    export_dataset(dataset, output_dir, export_images=False, as_zip=False)

    # Images directory should not exist
    assert not (output_dir / IMAGES_DIR).exists()


def test_import_basic_dataset_from_directory(tmp_path):
    """Test importing a basic dataset from directory."""

    class SimpleSample(Sample):
        label: int = label_field()
        score: float = numeric_field(dtype=pl.Float32(), semantic="score")

    # Export first
    original_dataset = Dataset(SimpleSample, categories={"label": LABEL_CATEGORIES})
    original_dataset.append(SimpleSample(label=1, score=0.9))
    original_dataset.append(SimpleSample(label=2, score=0.8))

    export_dir = tmp_path / "export"
    export_dataset(original_dataset, export_dir, export_images=False, as_zip=False)

    # Import back
    imported_dataset = import_dataset(export_dir, dtype=SimpleSample)

    # Verify dataset
    assert len(imported_dataset) == 2
    assert imported_dataset[0].label == 1
    assert imported_dataset[0].score == pytest.approx(0.9)
    assert imported_dataset[1].label == 2
    assert imported_dataset[1].score == pytest.approx(0.8)


def test_import_dataset_from_zip(tmp_path):
    """Test importing dataset from ZIP file."""

    class SimpleSample(Sample):
        label: int = label_field()

    # Export as ZIP
    original_dataset = Dataset(SimpleSample, categories={"label": LABEL_CATEGORIES})
    original_dataset.append(SimpleSample(label=1))
    original_dataset.append(SimpleSample(label=2))

    export_zip = tmp_path / "export.zip"
    export_dataset(original_dataset, export_zip, export_images=False, as_zip=True)

    # Import from ZIP
    imported_dataset = import_dataset(export_zip, dtype=SimpleSample)

    # Verify dataset
    assert len(imported_dataset) == 2
    assert imported_dataset[0].label == 1
    assert imported_dataset[1].label == 2


def test_import_dataset_with_image_callables(tmp_path):
    """Test importing dataset with ImageCallableField reconstructs callables."""

    class ImageSample(Sample):
        image: Callable[[], np.ndarray] = image_callable_field()
        label: int = label_field()

    def make_image(idx):
        def load_image():
            img = np.zeros((30, 40, 3), dtype=np.uint8)
            img[:, :, 0] = idx * 50
            return img

        return load_image

    # Export dataset
    original_dataset = Dataset(ImageSample, categories={"label": LABEL_CATEGORIES})
    original_dataset.append(ImageSample(image=make_image(0), label=1))
    original_dataset.append(ImageSample(image=make_image(1), label=2))

    export_dir = tmp_path / "export"
    export_dataset(original_dataset, export_dir, export_images=True, as_zip=False)

    # Import back
    imported_dataset = import_dataset(export_dir, dtype=ImageSample)

    # Verify dataset
    assert len(imported_dataset) == 2

    # Check that images are callables and work correctly
    sample0 = imported_dataset[0]
    assert callable(sample0.image)
    img0 = sample0.image()
    assert isinstance(img0, np.ndarray)
    assert img0.shape == (30, 40, 3)
    assert img0[0, 0, 0] == 0  # First image has red=0

    sample1 = imported_dataset[1]
    assert callable(sample1.image)
    img1 = sample1.image()
    assert img1.shape == (30, 40, 3)
    assert img1[0, 0, 0] == 50  # Second image has red=50


def test_import_dataset_with_image_paths(tmp_path):
    """Test importing dataset with ImagePathField updates paths."""

    class PathSample(Sample):
        image_path: str = image_path_field()
        label: int = label_field()

    # Create source image
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    img_path = source_dir / "test.png"
    PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(img_path)

    # Export dataset
    original_dataset = Dataset(PathSample, categories={"label": LABEL_CATEGORIES})
    original_dataset.append(PathSample(image_path=str(img_path), label=1))

    export_dir = tmp_path / "export"
    export_dataset(original_dataset, export_dir, export_images=True, as_zip=False)

    # Import back
    imported_dataset = import_dataset(export_dir, dtype=PathSample)

    # Verify dataset
    assert len(imported_dataset) == 1
    sample = imported_dataset[0]

    # Check that path points to exported image location
    assert sample.image_path is not None
    path = Path(sample.image_path)
    assert path.exists()
    assert IMAGES_DIR in path.parts


def test_import_dataset_with_instance_masks(tmp_path):
    """Test importing dataset with InstanceMaskCallableField."""

    class MaskSample(Sample):
        mask: Callable[[], np.ndarray] = instance_mask_callable_field()
        label: int = label_field()

    def make_mask(idx):
        def load_mask():
            mask = np.zeros((20, 30), dtype=np.uint8)
            mask[5:15, 5:25] = (idx + 1) * 100
            return mask

        return load_mask

    # Export dataset
    original_dataset = Dataset(MaskSample, categories={"label": LABEL_CATEGORIES, "mask": MASK_CATEGORIES})
    original_dataset.append(MaskSample(mask=make_mask(0), label=1))
    original_dataset.append(MaskSample(mask=make_mask(1), label=2))

    export_dir = tmp_path / "export"
    export_dataset(original_dataset, export_dir, export_images=True, as_zip=False)

    # Import back
    imported_dataset = import_dataset(export_dir, dtype=MaskSample)

    # Verify dataset
    assert len(imported_dataset) == 2

    # Check masks are callables and work correctly
    sample0 = imported_dataset[0]
    assert callable(sample0.mask)
    mask0 = sample0.mask()
    assert isinstance(mask0, np.ndarray)
    assert mask0.shape == (20, 30)
    assert mask0[10, 10] == 100  # First mask

    sample1 = imported_dataset[1]
    mask1 = sample1.mask()
    assert mask1[10, 10] == 200  # Second mask


def test_import_missing_metadata_raises_error(tmp_path):
    """Test that missing metadata file raises FileNotFoundError."""

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        import_dataset(empty_dir)


def test_import_missing_dataframe_raises_error(tmp_path):
    """Test that missing dataframe file raises FileNotFoundError."""

    export_dir = tmp_path / "export"
    export_dir.mkdir()

    # Create metadata but no dataframe
    metadata = {"version": VERSION, "schema": {}}
    with open(export_dir / METADATA_FILE, "w") as f:
        json.dump(metadata, f)

    with pytest.raises(FileNotFoundError, match="DataFrame file not found"):
        import_dataset(export_dir)


def test_import_preserves_categories(tmp_path):
    """Test that import preserves category information."""

    class LabeledSample(Sample):
        label: int = label_field()

    # Create Dataset with categories
    categories = LabelCategories(labels=("cat", "dog"))
    original_dataset = Dataset(LabeledSample, categories={"label": categories})
    original_dataset.append(LabeledSample(label=0))

    export_dir = tmp_path / "export"
    export_dataset(original_dataset, export_dir, export_images=False, as_zip=False)

    # Import back
    imported_dataset = import_dataset(export_dir, dtype=LabeledSample)

    # Verify categories are preserved
    assert imported_dataset.schema.attributes["label"] is not None
    assert len(imported_dataset.schema.attributes["label"].categories.labels) == 2


def test_import_with_none_image_values(tmp_path):
    """Test importing dataset with None values in image fields."""

    class OptionalImageSample(Sample):
        image: Callable[[], np.ndarray] | None = image_callable_field()
        label: int = label_field()

    def make_image():
        return np.zeros((10, 10, 3), dtype=np.uint8)

    # Export dataset with mixed None values
    original_dataset = Dataset(OptionalImageSample, categories={"label": LABEL_CATEGORIES})
    original_dataset.append(OptionalImageSample(image=make_image, label=1))
    original_dataset.append(OptionalImageSample(image=None, label=2))
    original_dataset.append(OptionalImageSample(image=make_image, label=3))

    export_dir = tmp_path / "export"
    export_dataset(original_dataset, export_dir, export_images=True, as_zip=False, ignore_missing_images=True)

    # Import back
    imported_dataset = import_dataset(export_dir, dtype=OptionalImageSample)

    # Verify dataset
    assert len(imported_dataset) == 2
    assert callable(imported_dataset[0].image)
    assert callable(imported_dataset[1].image)


def test_roundtrip_preserves_data_integrity(tmp_path):
    """Test that export-import roundtrip preserves data integrity."""

    class ComplexSample(Sample):
        label: int = label_field()
        score: float = numeric_field(dtype=pl.Float32(), semantic="score")
        image: Callable[[], np.ndarray] = image_callable_field()

    def make_image(value):
        def load_image():
            return np.full((20, 20, 3), value, dtype=np.uint8)

        return load_image

    # Create dataset
    original_dataset = Dataset(ComplexSample, categories={"label": LABEL_CATEGORIES})
    for i in range(5):
        original_dataset.append(ComplexSample(label=i, score=i * 0.2, image=make_image(i * 50)))

    # Export and import
    export_dir = tmp_path / "export"
    export_dataset(original_dataset, export_dir, export_images=True, as_zip=False)
    imported_dataset = import_dataset(export_dir, dtype=ComplexSample)

    # Verify complete data integrity
    assert len(imported_dataset) == len(original_dataset)
    for i in range(5):
        orig_sample = original_dataset[i]
        imported_sample = imported_dataset[i]

        assert orig_sample.label == imported_sample.label
        assert orig_sample.score == pytest.approx(imported_sample.score)

        # Check images produce the same values
        orig_img = orig_sample.image()
        imported_img = imported_sample.image()
        np.testing.assert_array_equal(orig_img, imported_img)


def test_export_empty_dataset(tmp_path):
    """Test exporting empty dataset."""

    class SimpleSample(Sample):
        label: int = label_field()

    dataset = Dataset(SimpleSample)

    output_dir = tmp_path / "export"
    export_dataset(dataset, output_dir, export_images=False, as_zip=False)

    # Should succeed and create valid structure
    assert (output_dir / METADATA_FILE).exists()
    assert (output_dir / DATAFRAME_FILE).exists()

    # Import should work
    imported = import_dataset(output_dir, dtype=SimpleSample)
    assert len(imported) == 0


def test_export_large_dataset(tmp_path):
    """Test exporting larger dataset (performance check)."""

    class SimpleSample(Sample):
        value: int = numeric_field(dtype=pl.UInt16(), semantic="value")

    dataset = Dataset(SimpleSample)
    for i in range(1000):
        dataset.append(SimpleSample(value=i))

    output_dir = tmp_path / "export"
    export_dataset(dataset, output_dir, export_images=False, as_zip=False)

    # Import and verify
    imported = import_dataset(output_dir, dtype=SimpleSample)
    assert len(imported) == 1000
    assert imported[0].value == 0
    assert imported[999].value == 999


def test_export_with_special_characters_in_paths(tmp_path):
    """Test handling of special characters in file paths."""

    class PathSample(Sample):
        image_path: str = image_path_field()

    # Create image with special characters in name
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    img_path = source_dir / "image with spaces & special.png"
    PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(img_path)

    dataset = Dataset(PathSample)
    dataset.append(PathSample(image_path=str(img_path)))

    output_dir = tmp_path / "export"
    export_dataset(dataset, output_dir, export_images=True, as_zip=False)

    # Should handle gracefully
    imported = import_dataset(output_dir, dtype=PathSample)
    assert len(imported) == 1


def test_zip_path_without_zip_extension(tmp_path):
    """Test that as_zip=True adds .zip extension if missing."""

    class SimpleSample(Sample):
        label: int = label_field()

    dataset = Dataset(SimpleSample, categories={"label": LABEL_CATEGORIES})
    dataset.append(SimpleSample(label=1))

    output_path = tmp_path / "export_no_ext"
    export_dataset(dataset, output_path, export_images=False, as_zip=True)

    # Should create .zip file
    expected_zip = tmp_path / "export_no_ext/dataset.zip"
    assert expected_zip.exists()
    assert expected_zip.suffix == ".zip"


def test_export_import_different_field_types(tmp_path):
    """Test export/import with a sample containing one of every field type."""

    # Create source image for image_path field
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_img_path = source_dir / "source_image.png"
    PILImage.fromarray(np.ones((20, 30, 3), dtype=np.uint8) * 100).save(source_img_path)

    class ComprehensiveSample(Sample):
        label: int = label_field()
        score: float = numeric_field(semantic="score")
        subset: Subset = subset_field()

        tensor: np.ndarray = tensor_field(dtype=pl.Float32())
        bbox: np.ndarray = bbox_field(dtype=pl.Float32(), normalize=False)
        rotated_bbox: np.ndarray = rotated_bbox_field(dtype=pl.Float32())
        keypoints: np.ndarray = keypoints_field(dtype=pl.Float32())

        image_callable: Callable[[], np.ndarray] = image_callable_field()
        image_path: str = image_path_field()
        image_info: ImageInfo = image_info_field()

        mask_callable: Callable[[], np.ndarray] = mask_callable_field()
        instance_mask_callable: np.ndarray | Callable[[], np.ndarray] = instance_mask_callable_field()

        tile: TileInfo = tile_field()

    # Create helper functions for callables
    def make_image():
        return np.full((20, 30, 3), 150, dtype=np.uint8)

    def make_mask():
        return np.ones((20, 30), dtype=np.uint8) * 255

    def make_instance_mask():
        return np.ones((20, 30), dtype=np.uint8) * 128

    # Create dataset with sample containing all field types
    dataset = Dataset(
        ComprehensiveSample,
        categories={
            "label": LABEL_CATEGORIES,
            "mask_callable": MASK_CATEGORIES,
            "instance_mask_callable": MASK_CATEGORIES,
        },
    )
    sample = ComprehensiveSample(
        label=1,
        score=0.95,
        subset=Subset.TRAINING,
        tensor=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        bbox=np.array([[10.0, 20.0, 50.0, 60.0]], dtype=np.float32),
        rotated_bbox=np.array([[25.0, 35.0, 40.0, 50.0, 0.5]], dtype=np.float32),
        keypoints=np.array([[15.0, 25.0, 1.0], [35.0, 45.0, 1.0]], dtype=np.float32),
        image_callable=make_image,
        image_path=str(source_img_path),
        image_info=ImageInfo(width=30, height=20),
        mask_callable=make_mask,
        instance_mask_callable=make_instance_mask,
        tile=TileInfo(source_sample_idx=0, x=10, y=20, width=30, height=40),
    )
    dataset.append(sample)

    # Export dataset
    export_dir = tmp_path / "export"
    export_dataset(dataset, export_dir, export_images=True, as_zip=False)

    # Verify export structure
    assert (export_dir / METADATA_FILE).exists()
    assert (export_dir / DATAFRAME_FILE).exists()
    assert (export_dir / IMAGES_DIR).exists()

    # Verify images were exported
    images_dir = export_dir / IMAGES_DIR
    assert (images_dir / "image_callable_000000.png").exists()
    assert (images_dir / "image_path_000000.png").exists()
    assert (images_dir / "mask_callable_000000.png").exists()
    assert (images_dir / "instance_mask_callable_000000.png").exists()

    # Import dataset
    imported_dataset = import_dataset(export_dir, dtype=ComprehensiveSample)

    # Verify dataset length
    assert len(imported_dataset) == 1

    # Get the imported sample
    imported_sample = imported_dataset[0]

    # Verify basic fields
    assert imported_sample.label == sample.label
    assert imported_sample.score == pytest.approx(sample.score)
    assert imported_sample.subset == sample.subset

    # Verify tensor
    np.testing.assert_array_equal(sample.tensor, imported_sample.tensor)

    # Verify bounding boxes
    np.testing.assert_array_equal(sample.bbox, imported_sample.bbox)
    np.testing.assert_array_equal(sample.rotated_bbox, imported_sample.rotated_bbox)

    # Verify keypoints
    np.testing.assert_array_equal(sample.keypoints, imported_sample.keypoints)

    # Verify image callable
    assert callable(imported_sample.image_callable)
    img = imported_sample.image_callable()
    expected_img = sample.image_callable()
    np.testing.assert_array_equal(expected_img, img)

    # Verify image path was updated (content matches original source image)
    assert imported_sample.image_path is not None
    assert Path(imported_sample.image_path).exists()
    loaded_img = np.array(PILImage.open(imported_sample.image_path))
    original_loaded_img = np.array(PILImage.open(sample.image_path))
    np.testing.assert_array_equal(original_loaded_img, loaded_img)

    # Verify image info
    assert imported_sample.image_info.width == sample.image_info.width
    assert imported_sample.image_info.height == sample.image_info.height

    # Verify mask callable
    assert callable(imported_sample.mask_callable)
    mask = imported_sample.mask_callable()
    expected_mask = sample.mask_callable()
    np.testing.assert_array_equal(expected_mask, mask)

    # Verify instance mask callable
    assert callable(imported_sample.instance_mask_callable)
    inst_mask = imported_sample.instance_mask_callable()
    expected_inst_mask = sample.instance_mask_callable()
    np.testing.assert_array_equal(expected_inst_mask, inst_mask)

    # Verify tile info
    assert imported_sample.tile.source_sample_idx == sample.tile.source_sample_idx
    assert imported_sample.tile.x == sample.tile.x
    assert imported_sample.tile.y == sample.tile.y
    assert imported_sample.tile.width == sample.tile.width
    assert imported_sample.tile.height == sample.tile.height


def test_export_dataset_export_images_true_and_false(tmp_path):
    """
    Test that export_dataset stores relative paths in Parquet when export_images=True
    and original absolute paths when export_images=False.
    """
    # 1. Setup: Create a temporary source image
    source_dir = tmp_path / "source_media"
    source_dir.mkdir()
    source_img_path = source_dir / "cat1.jpg"
    PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(source_img_path)

    class SampleWithPath(Sample):
        img: str = image_path_field()

    dataset = Dataset(SampleWithPath)
    dataset.append(SampleWithPath(img=str(source_img_path)))

    # 2. Case: export_images=True
    # The path in Parquet should be relative to the export directory
    export_dir_true = tmp_path / "exported_with_images"
    export_dataset(
        dataset=dataset,
        output_path=export_dir_true,
        as_zip=False,
        export_images=True,
    )

    df_true = pl.read_parquet(export_dir_true / DATAFRAME_FILE)
    exported_path = df_true["img"][0]

    assert exported_path is not None
    assert exported_path != str(source_img_path)
    # Verify the exported file exists at the path relative to images/
    # Note: rel_path in _export_image_path_field is 'img/000000.jpg'
    assert "000000.jpg" in exported_path

    # Check if the file exists on disk (handling the replacement of / with _ in filename)
    exported_file_on_disk = export_dir_true / IMAGES_DIR / exported_path
    assert exported_file_on_disk.exists()

    # 3. Case: export_images=False
    # The path in Parquet should remain the original absolute path
    export_dir_false = tmp_path / "exported_without_images"
    export_dataset(
        dataset=dataset,
        output_path=export_dir_false,
        as_zip=False,
        export_images=False,
    )

    df_false = pl.read_parquet(export_dir_false / DATAFRAME_FILE)
    assert df_false["img"][0] == str(source_img_path)
    assert not (export_dir_false / IMAGES_DIR).exists()


# ---------------------------------------------------------------------------
# Tests for automatic dtype detection (_get_registered_samples,
# _match_dtype_from_schema, register_sample)
# ---------------------------------------------------------------------------


def test_get_registered_samples_returns_empty_without_registration():
    """_get_registered_samples should return empty list when no classes are registered."""
    # Clear any existing registrations for this test
    original_registry = _sample_registry.copy()
    _sample_registry.clear()
    try:
        result = _get_registered_samples()
        assert result == []
    finally:
        _sample_registry.update(original_registry)


def test_get_registered_samples_returns_registered_classes():
    """_get_registered_samples should return classes that were explicitly registered."""

    class RegisteredChild(Sample):
        label: int = label_field()

    _sample_registry.discard(RegisteredChild)
    try:
        # Before registration, should not be in results
        assert RegisteredChild not in _get_registered_samples()

        # After registration, should be in results
        register_sample(RegisteredChild)
        assert RegisteredChild in _get_registered_samples()
    finally:
        _sample_registry.discard(RegisteredChild)


def test_get_registered_samples_does_not_include_base_sample():
    """The base Sample class itself should not appear in the result."""
    result = _get_registered_samples()
    assert Sample not in result


def test_match_dtype_from_schema_returns_matching_subclass():
    """_match_dtype_from_schema should return the subclass whose schema matches."""

    @register_sample
    class MatchMe(Sample):
        label: int = label_field()
        score: float = numeric_field(dtype=pl.Float32(), semantic="score")
        subset: Subset = subset_field()

    try:
        schema = MatchMe.infer_schema()
        result = _match_dtype_from_schema(schema)
        assert result is MatchMe
    finally:
        _sample_registry.discard(MatchMe)


def test_match_dtype_from_schema_falls_back_to_sample():
    """When no subclass matches, _match_dtype_from_schema should return Sample."""
    from datumaro.experimental.fields.types import NumericField
    from datumaro.experimental.schema import AttributeInfo, Schema

    # Build a schema that no existing subclass will match
    schema = Schema(
        attributes={
            "unique_xyz_field": AttributeInfo(
                type=int,
                field=NumericField(dtype=pl.Int64(), semantic="unique_xyz"),
            ),
            "another_unique_abc": AttributeInfo(
                type=float,
                field=NumericField(dtype=pl.Float64(), semantic="unique_abc"),
            ),
        }
    )
    result = _match_dtype_from_schema(schema)
    assert result is Sample


def test_register_sample_makes_class_discoverable():
    """register_sample should add a class to the registry so it is discoverable."""

    class ExternalSample(Sample):
        label: int = label_field()
        score: float = numeric_field(dtype=pl.Float32(), semantic="score")

    # Remove from registry in case a previous test registered it
    _sample_registry.discard(ExternalSample)

    register_sample(ExternalSample)
    try:
        assert ExternalSample in _sample_registry
        assert ExternalSample in _get_registered_samples()
    finally:
        _sample_registry.discard(ExternalSample)


def test_register_sample_works_as_decorator():
    """register_sample should work as a class decorator and return the class."""

    @register_sample
    class DecoratedSample(Sample):
        label: int = label_field()

    try:
        assert DecoratedSample in _sample_registry
        # The decorator should return the class unchanged
        assert issubclass(DecoratedSample, Sample)
    finally:
        _sample_registry.discard(DecoratedSample)


def test_import_without_dtype_falls_back_to_sample_for_unknown_schema(tmp_path):
    """When the exported schema doesn't match any subclass, dtype should fall back to Sample."""
    from datumaro.experimental.fields.types import NumericField
    from datumaro.experimental.schema import AttributeInfo, Schema

    # Create a dataset with a unique schema that won't match any subclass
    unique_schema = Schema(
        attributes={
            "weird_field_xyz": AttributeInfo(
                type=int,
                field=NumericField(dtype=pl.Int64(), semantic="weird_xyz"),
            ),
        }
    )
    dataset = Dataset(unique_schema)
    dataset.append(Sample(weird_field_xyz=42))

    export_dir = tmp_path / "export"
    export_dataset(dataset, export_dir, export_images=False, as_zip=False)

    imported_dataset = import_dataset(export_dir, dtype=None)
    assert len(imported_dataset) == 1
    assert imported_dataset.dtype == Sample


def test_import_auto_detects_dtype_roundtrip(tmp_path):
    """Full roundtrip: export with a known Sample subclass, import without dtype, verify auto-detection."""

    @register_sample
    class AutoDetectSample(Sample):
        label: int = label_field()
        score: float = numeric_field(dtype=pl.Float32(), semantic="score")
        subset: Subset = subset_field()

    try:
        original_dataset = Dataset(AutoDetectSample, categories={"label": LABEL_CATEGORIES})
        original_dataset.append(AutoDetectSample(label=2, score=0.75, subset=Subset.TRAINING))

        export_dir = tmp_path / "export"
        export_dataset(original_dataset, export_dir, export_images=False, as_zip=False)

        imported_dataset = import_dataset(export_dir)  # no dtype
        # Should auto-detect a matching subclass (not fall back to base Sample)
        assert imported_dataset.dtype is not Sample
        assert issubclass(imported_dataset.dtype, Sample)
        assert len(imported_dataset) == 1
    finally:
        _sample_registry.discard(AutoDetectSample)


def test_match_dtype_distinguishes_similar_samples_with_different_field_configs():
    """
    _match_dtype_from_schema should correctly distinguish between Sample subclasses
    that have the same field names but different field configurations.
    """

    @register_sample
    class ClassificationSample(Sample):
        label: int | None = label_field(dtype=pl.UInt8(), is_list=False)
        confidence: float | None = numeric_field(dtype=pl.Float32(), semantic="confidence")

    @register_sample
    class MultilabelClassificationSample(Sample):
        label: list[int] = label_field(dtype=pl.UInt8(), multi_label=True)
        confidence: list[float] | None = numeric_field(dtype=pl.Float32(), is_list=True, semantic="confidence")

    try:
        classification_schema = ClassificationSample.infer_schema()
        result = _match_dtype_from_schema(classification_schema)
        assert result is ClassificationSample

        multilabel_schema = MultilabelClassificationSample.infer_schema()
        result = _match_dtype_from_schema(multilabel_schema)
        assert result is MultilabelClassificationSample

    finally:
        _sample_registry.discard(ClassificationSample)
        _sample_registry.discard(MultilabelClassificationSample)


def test_import_zip_extracts_to_directory_next_to_zip_by_default(tmp_path):
    """Test that importing a zip file extracts to a directory next to the zip with the same name."""

    class ImageSample(Sample):
        image: Callable[[], np.ndarray] = image_callable_field()
        label: int = label_field()

    def make_image():
        return np.zeros((20, 30, 3), dtype=np.uint8)

    # Create and export dataset as zip
    original_dataset = Dataset(ImageSample, categories={"label": LABEL_CATEGORIES})
    original_dataset.append(ImageSample(image=make_image, label=1))

    export_zip = tmp_path / "my_dataset.zip"
    export_dataset(original_dataset, export_zip, export_images=True, as_zip=True)

    # Import from zip (no extract_dir provided)
    imported_dataset = import_dataset(export_zip, dtype=ImageSample)

    # Verify the dataset was extracted to a directory next to the zip
    expected_extract_dir = tmp_path / "my_dataset"
    assert expected_extract_dir.exists()
    assert expected_extract_dir.is_dir()
    assert (expected_extract_dir / METADATA_FILE).exists()
    assert (expected_extract_dir / DATAFRAME_FILE).exists()
    assert (expected_extract_dir / IMAGES_DIR).exists()

    # Verify the dataset works correctly
    assert len(imported_dataset) == 1
    sample = imported_dataset[0]
    assert sample.label == 1
    assert callable(sample.image)
    img = sample.image()
    assert img.shape == (20, 30, 3)


def test_import_zip_extracts_to_custom_directory_when_extract_dir_provided(tmp_path):
    """Test that importing a zip file extracts to a custom directory when extract_dir is provided."""

    class ImageSample(Sample):
        image: Callable[[], np.ndarray] = image_callable_field()
        label: int = label_field()

    def make_image():
        return np.zeros((25, 35, 3), dtype=np.uint8)

    # Create and export dataset as zip
    original_dataset = Dataset(ImageSample, categories={"label": LABEL_CATEGORIES})
    original_dataset.append(ImageSample(image=make_image, label=2))

    export_zip = tmp_path / "dataset.zip"
    export_dataset(original_dataset, export_zip, export_images=True, as_zip=True)

    # Import with custom extract_dir
    custom_extract_dir = tmp_path / "custom" / "location"
    imported_dataset = import_dataset(export_zip, dtype=ImageSample, extract_dir=custom_extract_dir)

    # Verify the dataset was extracted to the custom directory
    assert custom_extract_dir.exists()
    assert custom_extract_dir.is_dir()
    assert (custom_extract_dir / METADATA_FILE).exists()
    assert (custom_extract_dir / DATAFRAME_FILE).exists()
    assert (custom_extract_dir / IMAGES_DIR).exists()

    # Verify the default location was NOT created
    default_extract_dir = tmp_path / "dataset"
    assert not default_extract_dir.exists()

    # Verify the dataset works correctly
    assert len(imported_dataset) == 1
    sample = imported_dataset[0]
    assert sample.label == 2
    assert callable(sample.image)
    img = sample.image()
    assert img.shape == (25, 35, 3)


def test_import_zip_images_accessible_after_import(tmp_path):
    """Test that images from a zip import are accessible after import."""

    class ImageSample(Sample):
        image: Callable[[], np.ndarray] = image_callable_field()
        label: int = label_field()

    def make_image(value):
        def load():
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            img[:, :, 0] = value
            return img

        return load

    # Create and export dataset as zip
    original_dataset = Dataset(ImageSample, categories={"label": LABEL_CATEGORIES})
    original_dataset.append(ImageSample(image=make_image(100), label=1))
    original_dataset.append(ImageSample(image=make_image(200), label=2))

    export_zip = tmp_path / "test_dataset.zip"
    export_dataset(original_dataset, export_zip, export_images=True, as_zip=True)

    # Import from zip
    imported_dataset = import_dataset(export_zip, dtype=ImageSample)

    # Access images multiple times to verify they persist
    for _ in range(2):
        img0 = imported_dataset[0].image()
        img1 = imported_dataset[1].image()
        assert img0[0, 0, 0] == 100
        assert img1[0, 0, 0] == 200


def test_import_zip_extract_dir_is_ignored_for_directory_input(tmp_path):
    """Test that extract_dir parameter is ignored when input is a directory (not a zip)."""

    class SimpleSample(Sample):
        label: int = label_field()

    # Export to directory (not zip)
    original_dataset = Dataset(SimpleSample, categories={"label": LABEL_CATEGORIES})
    original_dataset.append(SimpleSample(label=3))

    export_dir = tmp_path / "exported"
    export_dataset(original_dataset, export_dir, export_images=False, as_zip=False)

    # Import from directory with extract_dir (should be ignored)
    custom_dir = tmp_path / "should_not_be_created"
    imported_dataset = import_dataset(export_dir, dtype=SimpleSample, extract_dir=custom_dir)

    # Verify the custom directory was NOT created (since input was a directory, not zip)
    assert not custom_dir.exists()

    # Verify the dataset works correctly
    assert len(imported_dataset) == 1
    assert imported_dataset[0].label == 3
