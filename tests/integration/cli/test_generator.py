import os
import os.path as osp
from unittest import TestCase

import pytest

import datumaro.util.image as image_module

from tests.utils.test_utils import TestDir
from tests.utils.test_utils import run_datum as run


@pytest.mark.xfail(reason="Cannot download the model file from the source")
class ImageGeneratorTest(TestCase):
    def check_images_shape(self, img_dir, expected_shape):
        exp_h, exp_w, exp_c = expected_shape
        for filename in os.listdir(img_dir):
            image = image_module.load_image(osp.join(img_dir, filename))
            h, w, c = image.shape

            self.assertEqual(h, exp_h)
            self.assertEqual(w, exp_w)
            self.assertEqual(c, exp_c)

    def test_save_image_can_create_dir(self):
        with TestDir() as test_dir:
            run(self, "generate", "-t", "image", "-o", test_dir, "-k", "2", "--shape", "224", "256")
            image_files = os.listdir(test_dir)
            self.assertEqual(len(image_files), 2)

            self.check_images_shape(test_dir, expected_shape=(224, 256, 3))

    def test_save_image_can_overwrite_dir(self):
        with TestDir() as test_dir:
            run(self, "generate", "-o", test_dir, "-k", "1", "--shape", "14", "14")

            image_files = os.listdir(test_dir)
            self.assertEqual(len(image_files), 1)
            self.check_images_shape(test_dir, expected_shape=(14, 14, 3))

            run(
                self,
                "generate",
                "-t",
                "image",
                "-o",
                test_dir,
                "-k",
                "2",
                "--shape",
                "24",
                "56",
                "--overwrite",
            )
            image_files = os.listdir(test_dir)
            self.assertEqual(len(image_files), 2)
            self.check_images_shape(test_dir, expected_shape=(24, 56, 3))
