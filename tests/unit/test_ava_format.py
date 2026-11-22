import os.path as osp
import shutil
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.ava.ava import *
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets


class AvaFormatTest(TestCase):
    def test_can_save_and_load_video_detection(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="video0/000000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(x=0, y=0, w=4, h=4, label=0, attributes={"track_id": 0}),
                        Bbox(x=0, y=4, w=2, h=2, label=1, attributes={"track_id": 1}),
                    ],
                ),
                DatasetItem(
                    id="video0/000001",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(x=2, y=0, w=4, h=4, label=0, attributes={"track_id": 0}),
                        Bbox(x=2, y=4, w=2, h=2, label=1, attributes={"track_id": 1}),
                    ],
                ),
                DatasetItem(
                    id="video1/000000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(x=0, y=0, w=2, h=2, label=2, attributes={"track_id": 0}),
                        Bbox(x=2, y=2, w=3, h=3, label=3, attributes={"track_id": 1}),
                        Bbox(x=4, y=4, w=4, h=4, label=4, attributes={"track_id": 2}),
                    ],
                ),
                DatasetItem(
                    id="video1/000001",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(x=4, y=4, w=2, h=2, label=2, attributes={"track_id": 0}),
                        Bbox(x=4, y=2, w=3, h=3, label=3, attributes={"track_id": 1}),
                        Bbox(x=6, y=4, w=2, h=4, label=4, attributes={"track_id": 2}),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable("label_" + str(label) for label in range(5)),
            },
        )

        with TestDir() as test_dir:
            AvaExporter.convert(source_dataset, test_dir, save_media=True, save_dataset_meta=True)
            parsed_dataset = Dataset.import_from(test_dir, "ava")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)


DUMMY_DATASET_DIR = get_test_asset_path("ava_dataset")


class AvaImporterTest(TestCase):
    def test_can_import_video_detection(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="video0/000000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(x=0, y=0, w=4, h=4, label=0, attributes={"track_id": 0}),
                        Bbox(x=0, y=4, w=2, h=2, label=1, attributes={"track_id": 1}),
                    ],
                ),
                DatasetItem(
                    id="video0/000001",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(x=2, y=0, w=4, h=4, label=0, attributes={"track_id": 0}),
                        Bbox(x=2, y=4, w=2, h=2, label=1, attributes={"track_id": 1}),
                    ],
                ),
                DatasetItem(
                    id="video1/000000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(x=0, y=0, w=2, h=2, label=2, attributes={"track_id": 0}),
                        Bbox(x=2, y=2, w=3, h=3, label=3, attributes={"track_id": 1}),
                        Bbox(x=4, y=4, w=4, h=4, label=4, attributes={"track_id": 2}),
                    ],
                ),
                DatasetItem(
                    id="video1/000001",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(x=4, y=4, w=2, h=2, label=2, attributes={"track_id": 0}),
                        Bbox(x=4, y=2, w=3, h=3, label=3, attributes={"track_id": 1}),
                        Bbox(x=6, y=4, w=2, h=4, label=4, attributes={"track_id": 2}),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable("label_" + str(label) for label in range(5)),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "ava")
        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_import_label_list_with_non_ascii_names(self):
        # the label list is UTF-8, whatever the platform's default encoding
        label_name = "\u8d70\u308b"

        with TestDir() as test_dir:
            dataset_dir = osp.join(test_dir, "ava")
            shutil.copytree(DUMMY_DATASET_DIR, dataset_dir)

            label_list_path = osp.join(dataset_dir, AvaPath.ANNOTATION_DIR, AvaPath.LABEL_LIST)
            with open(label_list_path, encoding="utf-8") as f:
                label_list = f.read()
            with open(label_list_path, "w", encoding="utf-8") as f:
                f.write(label_list.replace("label_1", label_name))

            dataset = Dataset.import_from(dataset_dir, "ava")

            label_names = [label.name for label in dataset.categories()[AnnotationType.label]]
            self.assertEqual(label_name, label_names[1])

    def test_can_detect_ava(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([AvaImporter.NAME], detected_formats)
