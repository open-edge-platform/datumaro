# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from itertools import product
from unittest import TestCase

import numpy as np
import pytest

import datumaro.util.image as image_module

from tests.utils.test_utils import TestDir


class ImageOperationsTest(TestCase):
    def setUp(self):
        self.default_backend = image_module.IMAGE_BACKEND.get()

    def tearDown(self):
        image_module.IMAGE_BACKEND.set(self.default_backend)

    def test_save_and_load_backends(self):
        backends = image_module.ImageBackend
        for save_backend, load_backend, c in product(backends, backends, [1, 3]):
            with TestDir() as test_dir:
                if c == 1:
                    src_image = np.random.randint(0, 255 + 1, (2, 4))
                else:
                    src_image = np.random.randint(0, 255 + 1, (2, 4, c))
                path = osp.join(test_dir, "img.png")  # lossless

                image_module.IMAGE_BACKEND.set(save_backend)
                image_module.save_image(path, src_image, jpeg_quality=100)

                image_module.IMAGE_BACKEND.set(load_backend)
                dst_image = image_module.load_image(path)

                # If image_module.IMAGE_COLOR_CHANNEL.get() == image_module.ImageColorChannel.UNCHANGED
                # OpenCV will read an image as BGR(A), but PIL will read an image as RGB(A).
                if (
                    c == 3
                    and load_backend == image_module.ImageBackend.PIL
                    and image_module.IMAGE_COLOR_CHANNEL.get()
                    == image_module.ImageColorChannel.UNCHANGED
                ):
                    dst_image = np.flip(dst_image, -1)

                self.assertTrue(
                    np.array_equal(src_image, dst_image),
                    "save: %s, load: %s" % (save_backend, load_backend),
                )

    def test_encode_and_decode_backends(self):
        backends = image_module.ImageBackend
        for save_backend, load_backend, c in product(backends, backends, [1, 3]):
            if c == 1:
                src_image = np.random.randint(0, 255 + 1, (2, 4))
            else:
                src_image = np.random.randint(0, 255 + 1, (2, 4, c))

            image_module.IMAGE_BACKEND.set(save_backend)
            buffer = image_module.encode_image(src_image, ".png", jpeg_quality=100)  # lossless

            image_module.IMAGE_BACKEND.set(load_backend)
            dst_image = image_module.decode_image(buffer)

            # If image_module.IMAGE_COLOR_CHANNEL.get() == image_module.ImageColorChannel.UNCHANGED
            # OpenCV will read an image as BGR(A), but PIL will read an image as RGB(A).
            if (
                c == 3
                and load_backend == image_module.ImageBackend.PIL
                and image_module.IMAGE_COLOR_CHANNEL.get()
                == image_module.ImageColorChannel.UNCHANGED
            ):
                dst_image = np.flip(dst_image, -1)

            self.assertTrue(
                np.array_equal(src_image, dst_image),
                "save: %s, load: %s" % (save_backend, load_backend),
            )

    def test_save_image_to_inexistent_dir_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            image_module.save_image("some/path.jpg", np.ones((5, 4, 3)), create_dir=False)

    def test_save_image_can_create_dir(self):
        with TestDir() as test_dir:
            path = osp.join(test_dir, "some", "path.jpg")
            image_module.save_image(path, np.ones((5, 4, 3)), create_dir=True)
            self.assertTrue(osp.isfile(path))


class ImageDecodeTest:
    @pytest.fixture
    def fxt_img_four_channels(self) -> np.ndarray:
        return np.random.randint(low=0, high=256, size=(5, 4, 4), dtype=np.uint8)

    @pytest.mark.parametrize(
        "image_backend", [image_module.ImageBackend.cv2, image_module.ImageBackend.PIL]
    )
    def test_decode_image_context(
        self, fxt_img_four_channels: np.ndarray, image_backend: image_module.ImageBackend
    ):
        img_bytes = image_module.encode_image(fxt_img_four_channels, ".png")

        # 3 channels from ImageColorScale.COLOR_BGR
        with image_module.decode_image_context(
            image_backend, image_module.ImageColorChannel.COLOR_BGR
        ):
            img_decoded = image_module.decode_image(img_bytes)
            assert img_decoded.shape[-1] == 3
            assert np.allclose(fxt_img_four_channels[:, :, :3], img_decoded)

        # 3 channels from ImageColorScale.COLOR_RGB
        with image_module.decode_image_context(
            image_backend, image_module.ImageColorChannel.COLOR_RGB
        ):
            img_decoded = image_module.decode_image(img_bytes)
            assert img_decoded.shape[-1] == 3
            assert np.allclose(
                fxt_img_four_channels[:, :, :3][:, :, ::-1],  # Flip color channels of the fixture
                img_decoded,
            )

        # 4 channels from ImageColorScale.UNCHANGED
        with image_module.decode_image_context(
            image_backend, image_module.ImageColorChannel.UNCHANGED
        ):
            img_decoded = image_module.decode_image(img_bytes)
            assert img_decoded.shape[-1] == 4

            if image_backend == image_module.ImageBackend.cv2:
                assert np.allclose(fxt_img_four_channels, img_decoded)
            else:
                # PIL will return RGBA, thus we need to correct the fixture
                to_rgb = fxt_img_four_channels[:, :, :3][:, :, ::-1]
                fxt_img_four_channels[:, :, :3] = to_rgb

                assert np.allclose(fxt_img_four_channels, img_decoded)
