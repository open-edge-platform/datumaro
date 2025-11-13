import os.path as osp
from functools import partial
from typing import Any, Dict, List, Tuple
from unittest import TestCase

import numpy as np

from datumaro.components.media import Image, RoIImage
from datumaro.util.image import (
    encode_image,
    lazy_image,
    load_image,
    load_image_meta_file,
    save_image,
    save_image_meta_file,
)
from datumaro.util.image_cache import ImageCache
from tests.utils.test_utils import TestDir


class ImageCacheTest(TestCase):
    def test_cache_works(self):
        with TestDir() as test_dir:
            image = np.ones((100, 100, 3), dtype=np.uint8)
            image_path = osp.join(test_dir, "image.jpg")
            save_image(image_path, image)

            caching_loader = lazy_image(image_path, cache=True)
            self.assertTrue(caching_loader() is caching_loader())

            non_caching_loader = lazy_image(image_path, cache=False)
            self.assertFalse(non_caching_loader() is non_caching_loader())

    def test_cache_fifo_displacement(self):
        capacity = 2
        cache = ImageCache(capacity)

        loaders = [lazy_image(None, loader=lambda p: object(), cache=cache) for _ in range(capacity + 1)]

        first_request = [loader() for loader in loaders[1:]]
        loaders[0]()  # pop something from the cache

        second_request = [loader() for loader in loaders[2:]]
        second_request.insert(0, loaders[1]())

        matches = sum([a is b for a, b in zip(first_request, second_request)])
        self.assertEqual(matches, len(first_request) - 1)

    def test_global_cache_is_accessible(self):
        loader = lazy_image(None, loader=lambda p: object())

        self.assertTrue(loader() is loader())
        self.assertEqual(ImageCache.get_instance().size(), 1)

    def setUp(self) -> None:
        ImageCache.get_instance().clear()
        return super().setUp()

    def tearDown(self) -> None:
        ImageCache.get_instance().clear()
        return super().tearDown()


class ImageTest(TestCase):
    @staticmethod
    def _gen_image_and_args_list(test_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        path = osp.join(test_dir, "path.png")
        image = np.ones([2, 4, 3])
        save_image(path, image)

        return image, [
            {"data": image},
            {"data": image},
            {"data": image, "size": (2, 4)},
            {"data": image, "ext": "png"},
            {"data": image, "ext": "png", "size": (2, 4)},
            {"data": lambda: image},
            {"data": lambda: image},
            {"data": lambda: image, "ext": "jpg"},
            {"data": partial(load_image, path)},
            {"data": partial(load_image, path), "size": (2, 4)},
            {"path": path},
            {"path": path, "size": (2, 4)},
        ]

    @staticmethod
    def _gen_bytes_image_and_args_list() -> Tuple[np.ndarray, bytes, List[Dict[str, Any]]]:
        image = np.ones([2, 4, 3])
        image_bytes = encode_image(image, "png")

        return (
            image,
            image_bytes,
            [
                {"data": image_bytes},
                {"data": lambda: image_bytes},
                {"data": lambda: image_bytes, "ext": ".jpg"},
            ],
        )

    def test_can_report_cached_size(self):
        data = np.ones((5, 6, 3))

        image = Image.from_numpy(data=lambda: data, size=(2, 4))

        self.assertEqual((2, 4), image.size)

    def test_lazy_image_shape(self):
        data = encode_image(np.ones((5, 6, 3)), "png")

        image_lazy = Image.from_bytes(data=data, size=(2, 4))
        image_eager = Image.from_bytes(data=data)

        self.assertEqual((2, 4), image_lazy.size)
        self.assertEqual((5, 6), image_eager.size)

    def test_ctors(self):
        with TestDir() as test_dir:
            image, args_list = self._gen_image_and_args_list(test_dir)
            for args in args_list:
                with self.subTest(**args):
                    if "path" in args:
                        img = Image.from_file(**args)

                        img2 = img.from_self()
                        self.assertNotEqual(id(img), id(img2))
                        self.assertEqual(img, img2)

                        img2 = img.from_self(path="other")
                        self.assertNotEqual(img.path, "other")
                        self.assertEqual(img2.path, "other")
                    else:
                        img = Image.from_numpy(**args)

                        img2 = img.from_self()
                        self.assertNotEqual(id(img), id(img2))
                        self.assertEqual(img, img2)

                        img2 = img.from_self(data=np.ones([1, 2, 3]))
                        self.assertFalse(np.array_equal(img.data, np.ones([1, 2, 3])))
                        self.assertTrue(np.array_equal(img2.data, np.ones([1, 2, 3])))

                    self.assertTrue(img.has_data)
                    np.testing.assert_array_equal(img.data, image)
                    self.assertEqual(img.size, tuple(image.shape[:2]))

            image, image_bytes, args_list = self._gen_bytes_image_and_args_list()
            for args in args_list:
                with self.subTest(**args):
                    img = Image.from_bytes(**args)
                    self.assertTrue(img.has_data)
                    np.testing.assert_array_equal(img.data, image)
                    self.assertEqual(img.bytes, image_bytes)
                    self.assertEqual(img.size, tuple(image.shape[:2]))

            with self.subTest():
                img = Image.from_file(path="somepath", size=(2, 4))
                self.assertEqual(img.size, (2, 4))

                img2 = img.from_self(path="otherpath")
                self.assertEqual(img.path, "somepath")
                self.assertEqual(img2.path, "otherpath")

    def test_ctor_errors(self):
        with self.subTest("no data specified"), self.assertRaisesRegex(Exception, "Directly initalizing"):
            Image(ext="jpg")

    def test_ext_detection(self):
        image_data = np.zeros((3, 4))

        for ext in (".bmp", ".jpg", ".png"):
            with self.subTest(ext=ext):
                image = Image.from_bytes(data=encode_image(image_data, ext))
                self.assertEqual(image.ext, ext)

    def test_ext_detection_failure(self):
        image_bytes = b"\xff" * 10  # invalid image
        image = Image.from_bytes(data=image_bytes)
        self.assertEqual(image.ext, None)

    def test_floating_image_from_numpy(self):
        image_float = np.random.rand(32, 32, 3).astype(np.float16) * 255.0
        media = Image.from_numpy(image_float)
        data = media.get_data_as_dtype(dtype=np.float16)
        self.assertTrue(np.all(image_float == data))

    def test_floating_image_from_file(self):
        import tifffile

        with TestDir() as test_dir:
            image_float = np.random.rand(32, 32, 3).astype(np.float32) * 255.0
            image_path = osp.join(test_dir, "floating_image.tiff")
            tifffile.imwrite(image_path, image_float)

            media = Image.from_file(image_path)
            data = media.get_data_as_dtype(dtype=np.float32)
            self.assertTrue(np.all(image_float == data))


class RoIImageTest(TestCase):
    def _test_ctors(self, img_ctor, args_list, test_dir, is_bytes=False):
        for args in args_list:
            # Case 1. Retrieve roi_img.data without retrieving the original image
            with self.subTest(**args):
                if "path" in args:
                    img = img_ctor.from_file(**args)
                else:
                    img = img_ctor.from_bytes(**args) if is_bytes else img_ctor.from_numpy(**args)
                h, w = img.size
                new_h, new_w = h // 2, w // 2
                roi = (0, 0, new_w, new_h)  # xywh
                roi_img = RoIImage.from_image(img, roi)

                roi_img2 = roi_img.from_self()
                self.assertNotEqual(id(roi_img), id(roi_img2))
                self.assertEqual(roi_img, roi_img2)

                roi_img2 = roi_img.from_self(roi=(0, 0, 1, 1))
                self.assertFalse(np.array_equal(roi_img.size, roi_img2.size))

                self.assertEqual(roi_img.size, (new_h, new_w))
                self.assertEqual(roi_img.data.shape[:2], (new_h, new_w))

            # Case 2. Retrieve img.data first and roi_img.data second
            with self.subTest(**args):
                if "path" in args:
                    img = img_ctor.from_file(**args)
                else:
                    img = img_ctor.from_bytes(**args) if is_bytes else img_ctor.from_numpy(**args)
                h, w = img.size

                self.assertTrue(isinstance(img.data, np.ndarray))

                new_h, new_w = h // 2, w // 2
                roi = (0, 0, new_w, new_h)  # xywh
                roi_img = RoIImage.from_image(img, roi)

                self.assertEqual(roi_img.size, (new_h, new_w))
                self.assertEqual(roi_img.data.shape[:2], (new_h, new_w))

            with self.subTest(**args):
                try:
                    roi_img.save(osp.join(test_dir, "test.png"))
                except:  # noqa: E722
                    self.fail("Cannot save RoIImage")

    def test_ctors_from_image(self):
        with TestDir() as test_dir:
            _, args_list = ImageTest._gen_image_and_args_list(test_dir)
            self._test_ctors(Image, args_list, test_dir, False)
            _, _, args_list = ImageTest._gen_bytes_image_and_args_list()
            self._test_ctors(Image, args_list, test_dir, True)


class ImageMetaTest(TestCase):
    def test_loading(self):
        meta_file_contents = r"""
        # this is a comment

        a 123 456
        'b c' 10 20 # inline comment
        """

        meta_expected = {
            "a": (123, 456),
            "b c": (10, 20),
        }

        with TestDir() as test_dir:
            meta_path = osp.join(test_dir, "images.meta")

            with open(meta_path, "w") as meta_file:
                meta_file.write(meta_file_contents)

            meta_loaded = load_image_meta_file(meta_path)

        self.assertEqual(meta_loaded, meta_expected)

    def test_saving(self):
        meta_original = {
            "a": (123, 456),
            "b c": (10, 20),
        }

        with TestDir() as test_dir:
            meta_path = osp.join(test_dir, "images.meta")

            save_image_meta_file(meta_original, meta_path)
            meta_reloaded = load_image_meta_file(meta_path)

        self.assertEqual(meta_reloaded, meta_original)
