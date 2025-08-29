import os.path as osp
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image, PointCloud
from datumaro.plugins.data_formats.kitti_3d.importer import Kitti3dImporter

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets_3d

DUMMY_DATASET_DIR = get_test_asset_path("kitti_dataset", "kitti_3d")
DUMMY_SUBSET_DATASET_DIR = get_test_asset_path("kitti_dataset", "kitti_3d_with_subset")


class Kitti3DImporterTest(TestCase):
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([Kitti3dImporter.NAME], detected_formats)

    def test_can_load(self):
        """
        <b>Description:</b>
        Ensure that the dataset can be loaded correctly from the KITTI3D format.

        <b>Expected results:</b>
        The loaded dataset should have the same number of data items as the expected dataset.
        The data items in the loaded dataset should have the same attributes and values as the expected data items.
        The point clouds and images associated with the data items should be loaded correctly.

        <b>Steps:</b>
        1. Prepare an expected dataset with known data items, point clouds, images, and attributes.
        2. Load the dataset from the KITTI3D format.
        3. Compare the loaded dataset with the expected dataset.
        """

        image1 = Image.from_file(path=osp.join(DUMMY_DATASET_DIR, "image_2", "000001.png"))

        expected_label_cat = LabelCategories(
            attributes={"occluded", "truncated", "alpha", "dimensions", "location", "rotation_y"}
        )
        expected_label_list = [
            "DontCare",
            "Car",
            "Pedestrian",
            "Van",
            "Truck",
            "Cyclist",
            "Sitter",
            "Train",
            "Motorcycle",
            "Bus",
            "Misc",
        ]
        for label in expected_label_list:
            expected_label_cat.add(label)
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="000001",
                    annotations=[
                        Bbox(
                            600,  # x1
                            150,  # y1
                            30,  # x2-x1
                            40,  # y2-y1
                            label=4,
                            id=0,
                            attributes={
                                "truncated": 0.0,
                                "occluded": 0,
                                "alpha": -1.57,
                                "dimensions": [2.85, 2.63, 12.34],
                                "location": [0.47, 1.49, 69.44],
                                "rotation_y": -1.56,
                            },
                        ),
                        Bbox(
                            650,  # x1
                            160,  # y1
                            50,  # x2-x1
                            40,  # y2-y1
                            label=1,
                            id=1,
                            attributes={
                                "truncated": 0.0,
                                "occluded": 3,
                                "alpha": -1.65,
                                "dimensions": [1.86, 0.6, 2.02],
                                "location": [4.59, 1.32, 45.84],
                                "rotation_y": -1.55,
                            },
                        ),
                        Bbox(
                            500,  # x1
                            170,  # y1
                            90,  # x2-x1
                            20,  # y2-y1
                            label=0,
                            id=2,
                            attributes={
                                "truncated": -1.0,
                                "occluded": -1,
                                "alpha": -10.0,
                                "dimensions": [-1.0, -1.0, -1.0],
                                "location": [-1000.0, -1000.0, -1000.0],
                                "rotation_y": -10.0,
                            },
                        ),
                    ],
                    media=image1,
                    attributes={"calib_path": osp.join(DUMMY_DATASET_DIR, "calib", "000001.txt")},
                ),
            ],
            categories={AnnotationType.label: expected_label_cat},
        )

        parsed_dataset = Dataset.import_from(DUMMY_DATASET_DIR, "kitti3d")

        compare_datasets_3d(self, expected_dataset, parsed_dataset, require_point_cloud=True)

    def test_can_load_with_subset(self):
        """
        <b>Description:</b>
        Ensure that the dataset can be loaded correctly from the KITTI3D format with a specified subset of data items.

        <b>Expected results:</b>
        The loaded dataset should contain only the specified subset of data items from the original dataset.
        The data items in the loaded dataset should have the same attributes and values as the expected data items.

        <b>Steps:</b>
        1. Prepare an expected dataset with a subset of data items from the original dataset.
        2. Load the dataset from the KITTI3D format, specifying the subset of data items to load.
        3. Compare the loaded dataset with the expected dataset.
        """
        expected_label_cat = LabelCategories(
            attributes={"occluded", "truncated", "alpha", "dimensions", "location", "rotation_y"}
        )
        expected_label_list = [
            "DontCare",
            "Car",
            "Pedestrian",
            "Van",
            "Truck",
            "Cyclist",
            "Sitter",
            "Train",
            "Motorcycle",
            "Bus",
            "Misc",
        ]
        for label in expected_label_list:
            expected_label_cat.add(label)
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="000000",
                    subset="train",
                    annotations=[
                        Bbox(
                            700,  # x1
                            150,  # y1
                            100,  # x2-x1
                            150,  # y2-y1
                            label=2,
                            id=0,
                            attributes={
                                "truncated": 0.0,
                                "occluded": 0,
                                "alpha": -0.2,
                                "dimensions": [1.89, 0.48, 1.20],
                                "location": [1.84, 1.47, 8.41],
                                "rotation_y": 0.01,
                            },
                        ),
                    ],
                    media=Image.from_file(
                        path=osp.join(DUMMY_SUBSET_DATASET_DIR, "image_2", "train", "000000.png")
                    ),
                    attributes={
                        "calib_path": osp.join(
                            DUMMY_SUBSET_DATASET_DIR, "calib", "train", "000000.txt"
                        )
                    },
                ),
                DatasetItem(
                    id="000001",
                    subset="val",
                    annotations=[
                        Bbox(
                            330,  # x1
                            180,  # y1
                            30,  # x2-x1
                            60,  # y2-y1
                            label=2,
                            id=0,
                            attributes={
                                "truncated": 0.0,
                                "occluded": 0,
                                "alpha": 1.94,
                                "dimensions": [1.87, 0.96, 0.65],
                                "location": [-8.50, 2.07, 23.02],
                                "rotation_y": 1.59,
                            },
                        ),
                        Bbox(
                            600,  # x1
                            170,  # y1
                            20,  # x2-x1
                            15,  # y2-y1
                            label=0,
                            id=1,
                            attributes={
                                "truncated": -1,
                                "occluded": -1,
                                "alpha": -10,
                                "dimensions": [-1, -1, -1],
                                "location": [-1000, -1000, -1000],
                                "rotation_y": -10,
                            },
                        ),
                    ],
                    media=Image.from_file(
                        path=osp.join(DUMMY_SUBSET_DATASET_DIR, "image_2", "val", "000001.png")
                    ),
                    attributes={
                        "calib_path": osp.join(
                            DUMMY_SUBSET_DATASET_DIR, "calib", "val", "000001.txt"
                        )
                    },
                ),
                DatasetItem(
                    id="000002",
                    subset="test",
                    annotations=[
                        Bbox(
                            0,  # x1
                            190,  # y1
                            400,  # x2-x1
                            190,  # y2-y1
                            label=1,
                            id=0,
                            attributes={
                                "truncated": 0.88,
                                "occluded": 3,
                                "alpha": -0.69,
                                "dimensions": [1.60, 1.57, 3.23],
                                "location": [-2.70, 1.74, 3.68],
                                "rotation_y": -1.29,
                            },
                        ),
                        Bbox(
                            800,  # x1
                            160,  # y1
                            25,  # x2-x1
                            25,  # y2-y1
                            label=0,
                            id=1,
                            attributes={
                                "truncated": -1,
                                "occluded": -1,
                                "alpha": -10,
                                "dimensions": [-1, -1, -1],
                                "location": [-1000, -1000, -1000],
                                "rotation_y": -10,
                            },
                        ),
                    ],
                    media=Image.from_file(
                        path=osp.join(DUMMY_SUBSET_DATASET_DIR, "image_2", "test", "000002.png")
                    ),
                    attributes={
                        "calib_path": osp.join(
                            DUMMY_SUBSET_DATASET_DIR, "calib", "test", "000002.txt"
                        )
                    },
                ),
            ],
            categories={AnnotationType.label: expected_label_cat},
        )

        parsed_dataset = Dataset.import_from(DUMMY_SUBSET_DATASET_DIR, "kitti3d")

        compare_datasets_3d(self, expected_dataset, parsed_dataset, require_point_cloud=True)
