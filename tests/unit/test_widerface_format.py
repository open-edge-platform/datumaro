# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Bbox, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.widerface import WiderFaceExporter, WiderFaceImporter

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import IGNORE_ALL, TestDir, compare_datasets


class WiderFaceFormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=0),
                        Bbox(
                            0,
                            1,
                            2,
                            3,
                            label=0,
                            attributes={
                                "blur": "2",
                                "expression": "0",
                                "illumination": "0",
                                "occluded": "0",
                                "pose": "2",
                                "invalid": "0",
                            },
                        ),
                        Label(1),
                    ],
                ),
                DatasetItem(
                    id="2",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(
                            0,
                            2,
                            4,
                            2,
                            label=0,
                            attributes={
                                "blur": "2",
                                "expression": "0",
                                "illumination": "1",
                                "occluded": "0",
                                "pose": "1",
                                "invalid": "0",
                            },
                        ),
                        Bbox(
                            3,
                            3,
                            2,
                            3,
                            label=0,
                            attributes={
                                "blur": "0",
                                "expression": "1",
                                "illumination": "0",
                                "occluded": "0",
                                "pose": "2",
                                "invalid": "0",
                            },
                        ),
                        Bbox(
                            2,
                            1,
                            2,
                            3,
                            label=0,
                            attributes={
                                "blur": "2",
                                "expression": "0",
                                "illumination": "0",
                                "occluded": "0",
                                "pose": "0",
                                "invalid": "1",
                            },
                        ),
                        Label(2),
                    ],
                ),
                DatasetItem(
                    id="3",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(
                            0,
                            1.1,
                            5.3,
                            2.1,
                            label=0,
                            attributes={
                                "blur": "2",
                                "expression": "1",
                                "illumination": "0",
                                "occluded": "0",
                                "pose": "1",
                                "invalid": "0",
                            },
                        ),
                        Bbox(0, 2, 3, 2, label=0, attributes={"occluded": False}),
                        Bbox(0, 3, 4, 2, label=0, attributes={"occluded": True}),
                        Bbox(0, 2, 4, 2, label=0),
                        Bbox(
                            0,
                            7,
                            3,
                            2,
                            label=0,
                            attributes={
                                "blur": "2",
                                "expression": "1",
                                "illumination": "0",
                                "occluded": "0",
                                "pose": "1",
                                "invalid": "0",
                            },
                        ),
                    ],
                ),
                DatasetItem(id="4", subset="val", media=Image.from_numpy(data=np.ones((8, 8, 3)))),
            ],
            categories=["face", "label_0", "label_1"],
        )

        with TestDir() as test_dir:
            WiderFaceExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "wider_face")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_with_no_save_media(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=1),
                        Bbox(
                            0,
                            1,
                            2,
                            3,
                            label=0,
                            attributes={
                                "blur": "2",
                                "expression": "0",
                                "illumination": "0",
                                "occluded": "0",
                                "pose": "2",
                                "invalid": "0",
                            },
                        ),
                        Label(1),
                    ],
                )
            ],
            categories=["face", "label_0"],
        )

        with TestDir() as test_dir:
            WiderFaceExporter.convert(source_dataset, test_dir, save_media=False)
            parsed_dataset = Dataset.import_from(test_dir, "wider_face")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_no_subsets(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/b/1",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(
                            0,
                            1,
                            2,
                            3,
                            label=1,
                            attributes={
                                "blur": "2",
                                "expression": "0",
                                "illumination": "0",
                                "occluded": "0",
                                "pose": "2",
                                "invalid": "0",
                            },
                        ),
                    ],
                ),
            ],
            categories=["face", "label_0", "label_1"],
        )

        with TestDir() as test_dir:
            WiderFaceExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "wider_face")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_dataset_with_save_dataset_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/b/1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(
                            0,
                            1,
                            2,
                            3,
                            label=1,
                            attributes={
                                "blur": "2",
                                "expression": "0",
                                "illumination": "0",
                                "occluded": "0",
                                "pose": "2",
                                "invalid": "0",
                            },
                        ),
                    ],
                ),
            ],
            categories=["face", "label_0", "label_1"],
        )

        with TestDir() as test_dir:
            WiderFaceExporter.convert(
                source_dataset, test_dir, save_media=True, save_dataset_meta=True
            )
            parsed_dataset = Dataset.import_from(test_dir, "wider_face")

            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))
            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(
                            0,
                            1,
                            2,
                            3,
                            label=0,
                            attributes={
                                "blur": "2",
                                "expression": "0",
                                "illumination": "0",
                                "occluded": "0",
                                "pose": "2",
                                "invalid": "0",
                            },
                        ),
                    ],
                ),
            ],
            categories=["face"],
        )

        with TestDir() as test_dir:
            WiderFaceExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "wider_face")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_dataset_with_non_widerface_attributes(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/b/1",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=0),
                        Bbox(
                            0,
                            1,
                            2,
                            3,
                            label=0,
                            attributes={"non-widerface attribute": "0", "blur": 1, "invalid": "1"},
                        ),
                        Bbox(1, 1, 2, 2, label=0, attributes={"non-widerface attribute": "0"}),
                    ],
                ),
            ],
            categories=["face"],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/b/1",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=0),
                        Bbox(0, 1, 2, 3, label=0, attributes={"blur": "1", "invalid": "1"}),
                        Bbox(1, 1, 2, 2, label=0),
                    ],
                ),
            ],
            categories=["face"],
        )

        with TestDir() as test_dir:
            WiderFaceExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "wider_face")

            compare_datasets(self, target_dataset, parsed_dataset)

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem("q/1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG")),
                DatasetItem(
                    "a/b/c/2", media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp")
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            WiderFaceExporter.convert(dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "wider_face")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)

    def test_inplace_save_writes_only_updated_data(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="train", media=Image.from_numpy(data=np.ones((2, 4, 3)))),
                DatasetItem(2, subset="train", media=Image.from_numpy(data=np.ones((3, 2, 3)))),
            ],
            categories=[],
        )

        with TestDir() as path:
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(1, subset="train", media=Image.from_numpy(data=np.ones((2, 4, 3)))),
                    DatasetItem(
                        2, subset="train", media=Image.from_file(path="2.jpg", size=(3, 2))
                    ),
                    DatasetItem(3, subset="valid", media=Image.from_numpy(data=np.ones((2, 2, 3)))),
                ],
                categories=[],
            )
            dataset.export(path, "wider_face", save_media=True)

            dataset.put(
                DatasetItem(2, subset="train", media=Image.from_numpy(data=np.ones((3, 2, 3))))
            )
            dataset.remove(3, "valid")
            dataset.save(save_media=True)

            self.assertEqual(
                {"1.jpg", "2.jpg"},
                set(os.listdir(osp.join(path, "WIDER_train", "images", "no_label"))),
            )
            self.assertEqual(
                {"wider_face_train_bbx_gt.txt"}, set(os.listdir(osp.join(path, "wider_face_split")))
            )
            compare_datasets(
                self,
                expected,
                Dataset.import_from(path, "wider_face"),
                require_media=True,
                ignored_attrs=IGNORE_ALL,
            )


DUMMY_DATASET_DIR = get_test_asset_path("widerface_dataset")


class WiderFaceImporterTest(TestCase):
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([WiderFaceImporter.NAME], detected_formats)

    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0_Parade_image_01",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(
                            1,
                            2,
                            2,
                            2,
                            attributes={
                                "blur": "0",
                                "expression": "0",
                                "illumination": "0",
                                "occluded": "0",
                                "pose": "0",
                                "invalid": "0",
                            },
                        ),
                        Label(0),
                    ],
                ),
                DatasetItem(
                    id="1_Handshaking_image_02",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(
                            1,
                            1,
                            2,
                            2,
                            attributes={
                                "blur": "0",
                                "expression": "0",
                                "illumination": "1",
                                "occluded": "0",
                                "pose": "0",
                                "invalid": "0",
                            },
                        ),
                        Bbox(
                            5,
                            1,
                            2,
                            2,
                            attributes={
                                "blur": "0",
                                "expression": "0",
                                "illumination": "1",
                                "occluded": "0",
                                "pose": "0",
                                "invalid": "0",
                            },
                        ),
                        Label(1),
                    ],
                ),
                DatasetItem(
                    id="0_Parade_image_03",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(
                            0,
                            0,
                            1,
                            1,
                            attributes={
                                "blur": "2",
                                "expression": "0",
                                "illumination": "0",
                                "occluded": "0",
                                "pose": "2",
                                "invalid": "0",
                            },
                        ),
                        Bbox(
                            3,
                            2,
                            1,
                            2,
                            attributes={
                                "blur": "0",
                                "expression": "0",
                                "illumination": "0",
                                "occluded": "1",
                                "pose": "0",
                                "invalid": "0",
                            },
                        ),
                        Bbox(
                            5,
                            6,
                            1,
                            1,
                            attributes={
                                "blur": "2",
                                "expression": "0",
                                "illumination": "0",
                                "occluded": "0",
                                "pose": "2",
                                "invalid": "0",
                            },
                        ),
                        Label(0),
                    ],
                ),
            ],
            categories=["Parade", "Handshaking"],
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "wider_face")

        compare_datasets(self, expected_dataset, dataset)
