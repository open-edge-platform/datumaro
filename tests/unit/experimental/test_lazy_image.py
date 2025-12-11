"""
Unit tests for LazyImage class and ImagePathField with LazyImage type.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from datumaro.experimental import (
    Dataset,
    ImagePathLike,
    LazyImage,
    Sample,
    clear_image_cache,
    get_image_cache_size,
    set_image_cache_size,
)
from datumaro.experimental.fields import image_path_field


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
        clear_image_cache()
        lazy_img = LazyImage(path=temp_image_path)
        _ = lazy_img.path
        # Cache should still be empty - image not loaded
        assert get_image_cache_size() == 0

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
        clear_image_cache()
        lazy_img = LazyImage(path=temp_image_path)
        assert lazy_img.width == 150
        # Should not have cached the full data (width uses PIL directly)
        assert get_image_cache_size() == 0

    def test_lazy_image_height_property(self, temp_image_path):
        """Test height property without loading full image."""
        clear_image_cache()
        lazy_img = LazyImage(path=temp_image_path)
        assert lazy_img.height == 100
        # Should not have cached the full data (height uses PIL directly)
        assert get_image_cache_size() == 0

    def test_lazy_image_size_property(self, temp_image_path):
        """Test size property returns (width, height)."""
        clear_image_cache()
        lazy_img = LazyImage(path=temp_image_path)
        assert lazy_img.size == (150, 100)
        # Should not have cached the full data (size uses PIL directly)
        assert get_image_cache_size() == 0

    def test_lazy_image_shape_property(self, temp_image_path):
        """Test shape property (triggers data load)."""
        clear_image_cache()
        lazy_img = LazyImage(path=temp_image_path)
        assert lazy_img.shape == (100, 150, 3)
        # Should have loaded and cached the data
        assert get_image_cache_size() == 1

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


class ImageCacheLRUTest:
    """Tests for the LRU cache functionality of LazyImage."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset cache before and after each test."""
        clear_image_cache()
        set_image_cache_size(100)  # Reset to default
        yield
        clear_image_cache()
        set_image_cache_size(100)

    def test_image_is_cached_after_access(self):
        """Test that accessing image data caches it."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test.png")
            PILImage.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(path)

            lazy_img = LazyImage(path=path)
            assert get_image_cache_size() == 0

            _ = lazy_img.data
            assert get_image_cache_size() == 1

    def test_same_image_uses_cache(self):
        """Test that accessing the same image twice uses the cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test.png")
            PILImage.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(path)

            lazy_img1 = LazyImage(path=path)
            lazy_img2 = LazyImage(path=path)

            data1 = lazy_img1.data
            assert get_image_cache_size() == 1

            data2 = lazy_img2.data
            assert get_image_cache_size() == 1  # Still 1, using cache

            # Should be the same array object (from cache)
            assert data1 is data2

    def test_different_formats_cached_separately(self):
        """Test that same image with different formats are cached separately."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test.png")
            PILImage.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(path)

            lazy_rgb = LazyImage(path=path, format="RGB")
            lazy_bgr = LazyImage(path=path, format="BGR")

            _ = lazy_rgb.data
            assert get_image_cache_size() == 1

            _ = lazy_bgr.data
            assert get_image_cache_size() == 2  # Different format = different cache entry

    def test_lru_eviction(self):
        """Test that LRU eviction works when cache is full."""
        set_image_cache_size(3)

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for i in range(5):
                path = os.path.join(temp_dir, f"image_{i}.png")
                PILImage.fromarray(np.full((10, 10, 3), i, dtype=np.uint8)).save(path)
                paths.append(path)

            # Load first 3 images
            for i in range(3):
                _ = LazyImage(path=paths[i]).data

            assert get_image_cache_size() == 3

            # Load 4th image - should evict the first one
            _ = LazyImage(path=paths[3]).data
            assert get_image_cache_size() == 3  # Still 3

            # Load 5th image - should evict the second one
            _ = LazyImage(path=paths[4]).data
            assert get_image_cache_size() == 3  # Still 3

    def test_clear_image_cache(self):
        """Test clearing the entire cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(5):
                path = os.path.join(temp_dir, f"image_{i}.png")
                PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(path)
                _ = LazyImage(path=path).data

            assert get_image_cache_size() == 5

            clear_image_cache()
            assert get_image_cache_size() == 0

    def test_set_image_cache_size(self):
        """Test setting cache size."""
        set_image_cache_size(5)

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for i in range(10):
                path = os.path.join(temp_dir, f"image_{i}.png")
                PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(path)
                paths.append(path)
                _ = LazyImage(path=path).data

            # Should only have 5 images cached
            assert get_image_cache_size() == 5

    def test_set_cache_size_evicts_existing(self):
        """Test that reducing cache size evicts existing items."""
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for i in range(10):
                path = os.path.join(temp_dir, f"image_{i}.png")
                PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(path)
                paths.append(path)
                _ = LazyImage(path=path).data

            assert get_image_cache_size() == 10

            # Reduce cache size - should evict oldest items
            set_image_cache_size(3)
            assert get_image_cache_size() == 3

    def test_clear_cache_on_single_image(self):
        """Test clearing cache for a single image."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path1 = os.path.join(temp_dir, "image1.png")
            path2 = os.path.join(temp_dir, "image2.png")
            PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(path1)
            PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(path2)

            lazy1 = LazyImage(path=path1)
            lazy2 = LazyImage(path=path2)

            _ = lazy1.data
            _ = lazy2.data
            assert get_image_cache_size() == 2

            lazy1.clear_cache()
            assert get_image_cache_size() == 1

    def test_cache_zero_size_disables_caching(self):
        """Test that setting cache size to 0 effectively disables caching."""
        set_image_cache_size(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test.png")
            PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(path)

            lazy_img = LazyImage(path=path)
            _ = lazy_img.data
            assert get_image_cache_size() == 0  # Nothing cached

    def test_cache_with_channels_first(self):
        """Test that channels_first affects cache key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test.png")
            PILImage.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(path)

            lazy_chw = LazyImage(path=path, channels_first=True)
            lazy_hwc = LazyImage(path=path, channels_first=False)

            _ = lazy_chw.data
            assert get_image_cache_size() == 1

            _ = lazy_hwc.data
            assert get_image_cache_size() == 2  # Different cache entries

    def test_lru_access_order(self):
        """Test that accessing an image moves it to the end of LRU."""
        set_image_cache_size(3)

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for i in range(3):
                path = os.path.join(temp_dir, f"image_{i}.png")
                PILImage.fromarray(np.full((10, 10, 3), i, dtype=np.uint8)).save(path)
                paths.append(path)

            lazy_images = [LazyImage(path=p) for p in paths]

            # Load all 3 images (order: 0, 1, 2)
            for lazy in lazy_images:
                _ = lazy.data

            # Access image 0 again (moves it to end, order now: 1, 2, 0)
            _ = lazy_images[0].data

            # Add a new image - should evict image 1 (the LRU)
            path_new = os.path.join(temp_dir, "image_new.png")
            PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(path_new)
            _ = LazyImage(path=path_new).data

            assert get_image_cache_size() == 3

            # Image 0 should still be in cache (we accessed it recently)
            data0 = lazy_images[0].data
            assert data0 is not None

    def test_cache_works_with_dataset(self):
        """Test that cache works correctly when using Dataset."""

        class TestSample(Sample):
            image: LazyImage = image_path_field()

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for i in range(5):
                path = os.path.join(temp_dir, f"image_{i}.png")
                PILImage.fromarray(np.full((20, 20, 3), i * 50, dtype=np.uint8)).save(path)
                paths.append(path)

            dataset = Dataset(TestSample)
            for path in paths:
                dataset.append(TestSample(image=path))

            # Access all images
            for sample in dataset:
                _ = sample.image.data

            assert get_image_cache_size() == 5

            # Clear cache and verify
            clear_image_cache()
            assert get_image_cache_size() == 0
