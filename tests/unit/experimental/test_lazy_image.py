"""
Unit tests for LazyImage class and ImagePathField with LazyImage type.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import ImagePathLike, LazyImage, image_path_field


class LazyImageClassTest:
    """Tests for the LazyImage class directly."""

    @pytest.fixture
    def temp_image_path(self):
        """Create a temporary test image and return its path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_image_path = os.path.join(temp_dir, "test_image.png")
            # Create a 100x150 RGB image with distinct colors
            img_array = np.zeros((100, 150, 3), dtype=np.uint8)
            img_array[:, :, 0] = 255  # Red channel
            img_array[50:, :, 1] = 128  # Green in bottom half
            img_array[:, 75:, 2] = 64  # Blue in right half

            test_img = PILImage.fromarray(img_array)
            test_img.save(test_image_path)
            yield test_image_path

    def test_lazy_image_creation_with_string_path(self, temp_image_path):
        """Test creating LazyImage with a string path."""
        lazy_img = LazyImage(path=temp_image_path)
        assert lazy_img.path == temp_image_path
        assert lazy_img.format == "RGB"
        assert lazy_img.channels_first is False

    def test_lazy_image_creation_with_path_object(self, temp_image_path):
        """Test creating LazyImage with a Path object."""
        path_obj = Path(temp_image_path)
        lazy_img = LazyImage(path=path_obj)
        # Path should be converted to string
        assert lazy_img.path == str(path_obj)
        assert isinstance(lazy_img.path, str)

    def test_lazy_image_path_access_no_load(self, temp_image_path):
        """Test that accessing path doesn't load the image."""
        lazy_img = LazyImage(path=temp_image_path)
        _ = lazy_img.path
        # Internal cache should still be None
        assert lazy_img._cached_data is None

    def test_lazy_image_data_loads_image(self, temp_image_path):
        """Test that accessing data loads the image."""
        lazy_img = LazyImage(path=temp_image_path)
        data = lazy_img.data
        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 150, 3)
        assert data.dtype == np.uint8

    def test_lazy_image_data_caches_result(self, temp_image_path):
        """Test that data is cached after first access."""
        lazy_img = LazyImage(path=temp_image_path)
        data1 = lazy_img.data
        data2 = lazy_img.data
        # Should be the exact same object (cached)
        assert data1 is data2

    def test_lazy_image_rgb_format(self, temp_image_path):
        """Test loading image in RGB format."""
        lazy_img = LazyImage(path=temp_image_path, format="RGB")
        data = lazy_img.data
        # Check that first pixel has expected RGB values
        assert data[0, 0, 0] == 255  # Red
        assert data[0, 0, 1] == 0  # Green
        assert data[0, 0, 2] == 0  # Blue (left half)

    def test_lazy_image_bgr_format(self, temp_image_path):
        """Test loading image in BGR format."""
        lazy_img_rgb = LazyImage(path=temp_image_path, format="RGB")
        lazy_img_bgr = LazyImage(path=temp_image_path, format="BGR")

        data_rgb = lazy_img_rgb.data
        data_bgr = lazy_img_bgr.data

        # BGR should have R and B channels swapped
        assert np.array_equal(data_rgb[:, :, 0], data_bgr[:, :, 2])  # R in RGB == B in BGR
        assert np.array_equal(data_rgb[:, :, 2], data_bgr[:, :, 0])  # B in RGB == R in BGR
        assert np.array_equal(data_rgb[:, :, 1], data_bgr[:, :, 1])  # G stays same

    def test_lazy_image_channels_first(self, temp_image_path):
        """Test loading image with channels-first format."""
        lazy_img = LazyImage(path=temp_image_path, channels_first=True)
        data = lazy_img.data
        # Should be (C, H, W) instead of (H, W, C)
        assert data.shape == (3, 100, 150)

    def test_lazy_image_channels_last(self, temp_image_path):
        """Test loading image with channels-last format (default)."""
        lazy_img = LazyImage(path=temp_image_path, channels_first=False)
        data = lazy_img.data
        # Should be (H, W, C)
        assert data.shape == (100, 150, 3)

    def test_lazy_image_width_property(self, temp_image_path):
        """Test width property without loading full image."""
        lazy_img = LazyImage(path=temp_image_path)
        assert lazy_img.width == 150
        # Should not have cached the full data
        assert lazy_img._cached_data is None

    def test_lazy_image_height_property(self, temp_image_path):
        """Test height property without loading full image."""
        lazy_img = LazyImage(path=temp_image_path)
        assert lazy_img.height == 100
        # Should not have cached the full data
        assert lazy_img._cached_data is None

    def test_lazy_image_size_property(self, temp_image_path):
        """Test size property returns (width, height)."""
        lazy_img = LazyImage(path=temp_image_path)
        assert lazy_img.size == (150, 100)
        # Should not have cached the full data
        assert lazy_img._cached_data is None

    def test_lazy_image_shape_property(self, temp_image_path):
        """Test shape property (triggers data load)."""
        lazy_img = LazyImage(path=temp_image_path)
        assert lazy_img.shape == (100, 150, 3)
        # Should have loaded the data
        assert lazy_img._cached_data is not None

    def test_lazy_image_str(self, temp_image_path):
        """Test string representation returns path."""
        lazy_img = LazyImage(path=temp_image_path)
        assert str(lazy_img) == temp_image_path

    def test_lazy_image_fspath(self, temp_image_path):
        """Test __fspath__ allows use with os.path functions."""
        lazy_img = LazyImage(path=temp_image_path)
        assert os.fspath(lazy_img) == temp_image_path
        # Should work with os.path functions
        assert os.path.exists(lazy_img)
        assert os.path.basename(lazy_img) == "test_image.png"

    def test_lazy_image_file_not_found(self):
        """Test that accessing data on non-existent file raises error."""
        lazy_img = LazyImage(path="/nonexistent/path/image.png")
        with pytest.raises(FileNotFoundError):
            _ = lazy_img.data

    def test_lazy_image_grayscale(self):
        """Test loading grayscale image."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_image_path = os.path.join(temp_dir, "gray_image.png")
            # Create grayscale image
            img_array = np.random.randint(0, 255, (50, 80), dtype=np.uint8)
            test_img = PILImage.fromarray(img_array, mode="L")
            test_img.save(test_image_path)

            lazy_img = LazyImage(path=test_image_path, format="L")
            data = lazy_img.data
            assert data.shape == (50, 80)
            assert data.dtype == np.uint8


class ImagePathFieldWithLazyImageTest:
    """Tests for ImagePathField when used with LazyImage type annotation."""

    @pytest.fixture
    def temp_image_path(self):
        """Create a temporary test image and return its path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_image_path = os.path.join(temp_dir, "test_image.png")
            img_array = np.zeros((100, 150, 3), dtype=np.uint8)
            img_array[:, :, 0] = 255  # Red
            img_array[:, :, 1] = 128  # Green
            img_array[:, :, 2] = 64  # Blue

            test_img = PILImage.fromarray(img_array)
            test_img.save(test_image_path)
            yield test_image_path

    def test_sample_with_lazy_image_type(self, temp_image_path):
        """Test creating a Sample with LazyImage type annotation."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field()

        sample = TestSample(image=temp_image_path)
        assert isinstance(sample.image, LazyImage)
        assert sample.image.path == temp_image_path

    def test_sample_coerces_string_to_lazy_image(self, temp_image_path):
        """Test that string path is automatically coerced to LazyImage."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field()

        # Pass a string, should be converted to LazyImage
        sample = TestSample(image=temp_image_path)
        assert isinstance(sample.image, LazyImage)

    def test_sample_accepts_lazy_image_directly(self, temp_image_path):
        """Test that LazyImage can be passed directly."""

        class TestSample(Sample):
            image: LazyImage = image_path_field()

        lazy_img = LazyImage(path=temp_image_path)
        sample = TestSample(image=lazy_img)
        assert isinstance(sample.image, LazyImage)
        assert sample.image.path == temp_image_path

    def test_sample_with_path_object(self, temp_image_path):
        """Test creating Sample with Path object."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field()

        sample = TestSample(image=Path(temp_image_path))
        assert isinstance(sample.image, LazyImage)
        assert sample.image.path == temp_image_path

    def test_dataset_with_lazy_image(self, temp_image_path):
        """Test Dataset operations with LazyImage samples."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field()

        dataset = Dataset(TestSample)
        dataset.append(TestSample(image=temp_image_path))

        retrieved = dataset[0]
        assert isinstance(retrieved.image, LazyImage)
        assert retrieved.image.path == temp_image_path
        assert retrieved.image.data.shape == (100, 150, 3)

    def test_dataset_iteration_with_lazy_image(self, temp_image_path):
        """Test iterating over Dataset with LazyImage samples."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field()

        dataset = Dataset(TestSample)
        dataset.append(TestSample(image=temp_image_path))

        for sample in dataset:
            assert isinstance(sample.image, LazyImage)
            assert sample.image.path == temp_image_path

    def test_image_path_field_format_parameter(self, temp_image_path):
        """Test that format parameter is passed to LazyImage."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field(format="BGR")

        sample = TestSample(image=temp_image_path)
        assert sample.image.format == "BGR"

    def test_image_path_field_channels_first_parameter(self, temp_image_path):
        """Test that channels_first parameter is passed to LazyImage."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field(channels_first=True)

        sample = TestSample(image=temp_image_path)
        assert sample.image.channels_first is True
        assert sample.image.data.shape == (3, 100, 150)

    def test_dataset_from_polars_with_lazy_image(self, temp_image_path):
        """Test that LazyImage is properly reconstructed from Polars data."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field()

        dataset = Dataset(TestSample)
        dataset.append(TestSample(image=temp_image_path))

        # Access from dataset (goes through from_polars)
        retrieved = dataset[0]
        assert isinstance(retrieved.image, LazyImage)
        assert retrieved.image.data.shape == (100, 150, 3)

    def test_multiple_samples_with_lazy_image(self):
        """Test multiple samples with different images."""

        class TestSample(Sample):
            image: LazyImage = image_path_field()

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for i in range(3):
                path = os.path.join(temp_dir, f"image_{i}.png")
                img_array = np.full((50 + i * 10, 60 + i * 10, 3), i * 50, dtype=np.uint8)
                PILImage.fromarray(img_array).save(path)
                paths.append(path)

            dataset = Dataset(TestSample)
            for path in paths:
                dataset.append(TestSample(image=path))

            assert len(dataset) == 3

            for i, sample in enumerate(dataset):
                expected_shape = (50 + i * 10, 60 + i * 10, 3)
                assert sample.image.data.shape == expected_shape


class ImagePathFieldWithStringTypeTest:
    """Tests to ensure ImagePathField still works with string type annotation."""

    @pytest.fixture
    def temp_image_path(self):
        """Create a temporary test image and return its path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_image_path = os.path.join(temp_dir, "test_image.png")
            img_array = np.zeros((100, 150, 3), dtype=np.uint8)
            PILImage.fromarray(img_array).save(test_image_path)
            yield test_image_path

    def test_sample_with_string_type(self, temp_image_path):
        """Test that string type annotation still works as before."""

        class TestSample(Sample):
            image: str = image_path_field()

        sample = TestSample(image=temp_image_path)
        assert isinstance(sample.image, str)
        assert sample.image == temp_image_path

    def test_dataset_with_string_type(self, temp_image_path):
        """Test Dataset operations with string type annotation."""

        class TestSample(Sample):
            image: str = image_path_field()

        dataset = Dataset(TestSample)
        dataset.append(TestSample(image=temp_image_path))

        retrieved = dataset[0]
        assert isinstance(retrieved.image, str)
        assert retrieved.image == temp_image_path


class LazyImageEdgeCasesTest:
    """Edge case tests for LazyImage functionality."""

    def test_lazy_image_with_none_value(self):
        """Test that None values are handled correctly."""

        class TestSample(Sample):
            image: LazyImage | None = image_path_field()

        sample = TestSample(image=None)
        assert sample.image is None

    def test_lazy_image_rgba_format(self):
        """Test loading RGBA image."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_image_path = os.path.join(temp_dir, "rgba_image.png")
            img_array = np.zeros((50, 80, 4), dtype=np.uint8)
            img_array[:, :, 0] = 255  # R
            img_array[:, :, 1] = 128  # G
            img_array[:, :, 2] = 64  # B
            img_array[:, :, 3] = 200  # A
            PILImage.fromarray(img_array, mode="RGBA").save(test_image_path)

            lazy_img = LazyImage(path=test_image_path, format="RGBA")
            data = lazy_img.data
            assert data.shape == (50, 80, 4)

    def test_lazy_image_channels_first_with_bgr(self):
        """Test channels-first combined with BGR format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_image_path = os.path.join(temp_dir, "test_image.png")
            img_array = np.zeros((40, 60, 3), dtype=np.uint8)
            img_array[:, :, 0] = 100  # R
            img_array[:, :, 1] = 150  # G
            img_array[:, :, 2] = 200  # B
            PILImage.fromarray(img_array).save(test_image_path)

            lazy_img = LazyImage(path=test_image_path, format="BGR", channels_first=True)
            data = lazy_img.data
            assert data.shape == (3, 40, 60)
            # First channel should be Blue (200), last should be Red (100)
            assert data[0, 0, 0] == 200  # B
            assert data[1, 0, 0] == 150  # G
            assert data[2, 0, 0] == 100  # R

    def test_dataset_slice_with_lazy_image(self):
        """Test dataset slicing with LazyImage samples."""

        class TestSample(Sample):
            image: LazyImage = image_path_field()

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for i in range(5):
                path = os.path.join(temp_dir, f"image_{i}.png")
                img_array = np.full((50, 60, 3), i * 40, dtype=np.uint8)
                PILImage.fromarray(img_array).save(path)
                paths.append(path)

            dataset = Dataset(TestSample)
            for path in paths:
                dataset.append(TestSample(image=path))

            # Test slicing
            sliced = dataset.slice(1, 3)
            assert len(sliced) == 3

            for i, sample in enumerate(sliced):
                assert isinstance(sample.image, LazyImage)
                # Check we got the right samples (indices 1, 2, 3)
                expected_path = paths[i + 1]
                assert sample.image.path == expected_path


class ImagePathLikeTypeAliasTest:
    """Tests for the ImagePathLike type alias to avoid type checker warnings."""

    @pytest.fixture
    def temp_image_path(self):
        """Create a temporary test image and return its path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_image_path = os.path.join(temp_dir, "test_image.png")
            img_array = np.zeros((100, 150, 3), dtype=np.uint8)
            PILImage.fromarray(img_array).save(test_image_path)
            yield test_image_path

    def test_sample_with_image_path_like_type(self, temp_image_path):
        """Test creating a Sample with ImagePathLike type annotation."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field()

        # Should not raise type checker warnings when passing a string
        sample = TestSample(image=temp_image_path)
        assert isinstance(sample.image, LazyImage)
        assert sample.image.path == temp_image_path

    def test_sample_with_image_path_like_accepts_path(self, temp_image_path):
        """Test that ImagePathLike accepts Path objects."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field()

        sample = TestSample(image=Path(temp_image_path))
        assert isinstance(sample.image, LazyImage)
        assert sample.image.path == temp_image_path

    def test_sample_with_image_path_like_accepts_lazy_image(self, temp_image_path):
        """Test that ImagePathLike accepts LazyImage directly."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field()

        lazy_img = LazyImage(path=temp_image_path)
        sample = TestSample(image=lazy_img)
        assert isinstance(sample.image, LazyImage)
        assert sample.image.path == temp_image_path

    def test_dataset_with_image_path_like(self, temp_image_path):
        """Test Dataset operations with ImagePathLike type."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field()

        dataset = Dataset(TestSample)
        dataset.append(TestSample(image=temp_image_path))

        retrieved = dataset[0]
        assert isinstance(retrieved.image, LazyImage)
        assert retrieved.image.path == temp_image_path
        assert retrieved.image.data.shape == (100, 150, 3)

    def test_image_path_like_with_format_option(self, temp_image_path):
        """Test ImagePathLike with format option."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field(format="BGR")

        sample = TestSample(image=temp_image_path)
        assert sample.image.format == "BGR"

    def test_image_path_like_with_channels_first_option(self, temp_image_path):
        """Test ImagePathLike with channels_first option."""

        class TestSample(Sample):
            image: ImagePathLike = image_path_field(channels_first=True)

        sample = TestSample(image=temp_image_path)
        assert sample.image.channels_first is True
        assert sample.image.data.shape == (3, 100, 150)
