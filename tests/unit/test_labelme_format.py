import os
import os.path as osp
from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Bbox, Mask, Polygon
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.labelme import LabelMeExporter, LabelMeImporter

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, check_save_and_load, compare_datasets


class LabelMeExporterTest(TestCase):
    def _test_save_and_load(
        self, source_dataset, converter, test_dir, target_dataset=None, importer_args=None, **kwargs
    ):
        return check_save_and_load(
            self,
            source_dataset,
            converter,
            test_dir,
            importer="label_me",
            target_dataset=target_dataset,
            importer_args=importer_args,
            **kwargs,
        )

    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="dir1/1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((16, 16, 3))),
                    annotations=[
                        Bbox(0, 4, 4, 8, label=2, group=2),
                        Polygon(
                            [0, 4, 4, 4, 5, 6],
                            label=3,
                            attributes={
                                "occluded": True,
                                "a1": "qwe",
                                "a2": True,
                                "a3": 123,
                                "a4": "42",  # must be escaped and recognized as string
                                "escaped": 'a,b. = \\= \\\\ " \\" \\, \\',
                            },
                        ),
                        Mask(
                            np.array([[0, 1], [1, 0], [1, 1]]),
                            group=2,
                            attributes={"username": "test"},
                        ),
                        Bbox(1, 2, 3, 4, group=3),
                        Mask(
                            np.array([[0, 0], [0, 0], [1, 1]]),
                            group=3,
                            attributes={"occluded": True},
                        ),
                    ],
                ),
            ],
            categories=["label_" + str(label) for label in range(10)],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="dir1/1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((16, 16, 3))),
                    annotations=[
                        Bbox(
                            0,
                            4,
                            4,
                            8,
                            label=0,
                            group=2,
                            id=0,
                            attributes={
                                "occluded": False,
                                "username": "",
                            },
                        ),
                        Polygon(
                            [0, 4, 4, 4, 5, 6],
                            label=1,
                            id=1,
                            attributes={
                                "occluded": True,
                                "username": "",
                                "a1": "qwe",
                                "a2": True,
                                "a3": 123,
                                "a4": "42",
                                "escaped": 'a,b. = \\= \\\\ " \\" \\, \\',
                            },
                        ),
                        Mask(
                            np.array([[0, 1], [1, 0], [1, 1]]),
                            group=2,
                            id=2,
                            attributes={"occluded": False, "username": "test"},
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            group=1,
                            id=3,
                            attributes={
                                "occluded": False,
                                "username": "",
                            },
                        ),
                        Mask(
                            np.array([[0, 0], [0, 0], [1, 1]]),
                            group=1,
                            id=4,
                            attributes={"occluded": True, "username": ""},
                        ),
                    ],
                ),
            ],
            categories=["label_2", "label_3"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(LabelMeExporter.convert, save_media=True),
                test_dir,
                target_dataset=target_dataset,
                require_media=True,
            )

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG")
                ),
                DatasetItem(
                    id="b/c/d/2", media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".png")
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                dataset,
                partial(LabelMeExporter.convert, save_media=True),
                test_dir,
                require_media=True,
            )

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((16, 16, 3))),
                    annotations=[Polygon([0, 4, 4, 4, 5, 6], label=3)],
                ),
            ],
            categories=["label_" + str(label) for label in range(10)],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((16, 16, 3))),
                    annotations=[
                        Polygon(
                            [0, 4, 4, 4, 5, 6],
                            label=0,
                            id=0,
                            attributes={"occluded": False, "username": ""},
                        ),
                    ],
                ),
            ],
            categories=["label_3"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(LabelMeExporter.convert, save_media=True),
                test_dir,
                target_dataset=target_dataset,
                require_media=True,
            )

    def test_relative_paths(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.ones((4, 2, 3)))),
                DatasetItem(id="subdir1/1", media=Image.from_numpy(data=np.ones((2, 6, 3)))),
                DatasetItem(id="subdir2/1", media=Image.from_numpy(data=np.ones((5, 4, 3)))),
                DatasetItem(
                    id="sub/dir3/1",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 1, 1, 0],
                                    [0, 1, 1, 0],
                                    [0, 0, 0, 0],
                                ]
                            ),
                            label=1,
                            attributes={"occluded": False, "username": "user"},
                        )
                    ],
                ),
                DatasetItem(
                    id="subdir3/1",
                    subset="a",
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[
                        Bbox(
                            1, 2, 3, 4, label=0, attributes={"occluded": False, "username": "user"}
                        )
                    ],
                ),
                DatasetItem(
                    id="subdir3/1", subset="b", media=Image.from_numpy(data=np.ones((4, 4, 3)))
                ),
            ],
            categories=["label1", "label2"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(LabelMeExporter.convert, save_media=True),
                test_dir,
                require_media=True,
            )

    def test_can_save_dataset_to_correct_dir_with_correct_filename(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="dir/a", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".jpeg")
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                dataset,
                partial(LabelMeExporter.convert, save_media=True),
                test_dir,
                require_media=True,
            )

            xml_dirpath = osp.join(test_dir, "Annotations/default/dir")
            img_dirpath = osp.join(test_dir, "Images/default/dir")
            self.assertEqual(os.listdir(osp.join(test_dir, "Annotations", "default")), ["dir"])
            self.assertEqual(os.listdir(osp.join(test_dir, "Images", "default")), ["dir"])
            self.assertEqual(set(os.listdir(xml_dirpath)), {"a.xml"})
            self.assertEqual(set(os.listdir(img_dirpath)), {"a.jpeg"})

    def test_save_and_load_with_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="sub/dir3/1",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 1, 1, 0],
                                    [0, 1, 1, 0],
                                    [0, 0, 0, 0],
                                ]
                            ),
                            label=1,
                            attributes={"occluded": False, "username": "user"},
                        )
                    ],
                ),
                DatasetItem(
                    id="subdir3/1",
                    subset="a",
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[
                        Bbox(
                            1, 2, 3, 4, label=0, attributes={"occluded": False, "username": "user"}
                        )
                    ],
                ),
            ],
            categories=["label1", "label2"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(LabelMeExporter.convert, save_media=True, save_dataset_meta=True),
                test_dir,
                require_media=True,
            )
            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))


DUMMY_DATASET_DIR = get_test_asset_path("labelme_dataset")


class LabelMeImporterTest(TestCase):
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([LabelMeImporter.NAME], detected_formats)

    def test_can_import(self):
        img1 = np.ones((77, 102, 3)) * 255
        img1[6:32, 7:41] = 0

        mask1 = np.zeros((77, 102), dtype=int)
        mask1[67:69, 58:63] = 1

        mask2 = np.zeros((77, 102), dtype=int)
        mask2[13:25, 54:71] = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="img1",
                    subset="subset1",
                    media=Image.from_numpy(data=img1),
                    annotations=[
                        Polygon(
                            [43, 34, 45, 34, 45, 37, 43, 37],
                            label=0,
                            id=0,
                            attributes={"occluded": False, "username": "admin"},
                        ),
                        Mask(
                            mask1,
                            label=1,
                            id=1,
                            attributes={"occluded": False, "username": "brussell"},
                        ),
                        Polygon(
                            [30, 12, 42, 21, 24, 26, 15, 22, 18, 14, 22, 12, 27, 12],
                            label=2,
                            group=2,
                            id=2,
                            attributes={"a1": True, "occluded": True, "username": "anonymous"},
                        ),
                        Polygon(
                            [35, 21, 43, 22, 40, 28, 28, 31, 31, 22, 32, 25],
                            label=3,
                            group=2,
                            id=3,
                            attributes={"kj": True, "occluded": False, "username": "anonymous"},
                        ),
                        Bbox(
                            13,
                            19,
                            10,
                            11,
                            label=4,
                            group=2,
                            id=4,
                            attributes={"hg": True, "occluded": True, "username": "anonymous"},
                        ),
                        Mask(
                            mask2,
                            label=5,
                            group=1,
                            id=5,
                            attributes={"d": True, "occluded": False, "username": "anonymous"},
                        ),
                        Polygon(
                            [64, 21, 74, 24, 72, 32, 62, 34, 60, 27, 62, 22],
                            label=6,
                            group=1,
                            id=6,
                            attributes={
                                "gfd lkj lkj hi": True,
                                "occluded": False,
                                "username": "anonymous",
                            },
                        ),
                    ],
                ),
            ],
            categories=[
                "window",
                "license plate",
                "o1",
                "q1",
                "b1",
                "m1",
                "hg",
            ],
        )

        parsed = Dataset.import_from(DUMMY_DATASET_DIR, "label_me")
        compare_datasets(self, expected=target_dataset, actual=parsed)

    def test_can_convert(self):
        source_dataset = Dataset.import_from(DUMMY_DATASET_DIR, "label_me")
        with TestDir() as test_dir:
            LabelMeExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "label_me")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)
