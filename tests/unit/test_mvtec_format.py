from unittest import TestCase

import cv2
import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories, Mask
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.mvtec.exporter import MvtecExporter
from datumaro.plugins.data_formats.mvtec.importer import (
    MvtecClassificationImporter,
    MvtecDetectionImporter,
    MvtecSegmentationImporter,
)

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets


class MVTecFormatTest(TestCase):
    def test_can_save_and_load_classification(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="label_1/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(2)
                ),
            },
        )

        with TestDir() as test_dir:
            MvtecExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "mvtec_classification")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_segmentation(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="label_1/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=1)],
                ),
                DatasetItem(
                    id="label_2/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Mask(image=np.zeros((8, 8), dtype=np.uint8), label=2)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(3)
                ),
            },
        )

        with TestDir() as test_dir:
            MvtecExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "mvtec_segmentation")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_detection(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="label_1/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Bbox(x=0, y=0, w=3, h=3, label=1)],
                ),
                DatasetItem(
                    id="label_2/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Bbox(x=4, y=4, w=3, h=3, label=2)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(3)
                ),
            },
        )

        with TestDir() as test_dir:
            MvtecExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "mvtec_detection")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)


DUMMY_DATASET_DIR = get_test_asset_path("mvtec_dataset", "category_0")


class MvtecImporterTest(TestCase):
    def test_can_import_classification(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(label=0)],
                ),
                DatasetItem(
                    id="label_1/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(label=1)],
                ),
                DatasetItem(
                    id="label_2/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(label=2)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(3)
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "mvtec_classification")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_import_segmentation(self):
        mask_label_1 = np.pad(np.ones((4, 4), dtype=np.uint8), ((0, 4), (0, 4)), "constant")
        mask_label_2 = np.pad(np.ones((4, 4), dtype=np.uint8), ((4, 0), (4, 0)), "constant")

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(label=0)],
                ),
                DatasetItem(
                    id="label_1/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Mask(
                            image=mask_label_1,
                            label=1,
                        )
                    ],
                ),
                DatasetItem(
                    id="label_2/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Mask(image=mask_label_2, label=2)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(3)
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "mvtec_segmentation")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_import_detection(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(label=0)],
                ),
                DatasetItem(
                    id="label_1/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Bbox(x=0, y=0, w=3, h=3, label=1)],
                ),
                DatasetItem(
                    id="label_2/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Bbox(x=4, y=4, w=3, h=3, label=2)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(3)
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "mvtec_detection")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_detect_mvtec(self):
        env = Environment()
        path = DUMMY_DATASET_DIR

        detected_formats = env.detect_dataset(path)
        self.assertEqual(
            sorted(detected_formats),
            [
                MvtecClassificationImporter.NAME,
                MvtecDetectionImporter.NAME,
                MvtecSegmentationImporter.NAME,
            ],
        )


class MVTecExporterTest(TestCase):
    def test_segmentation_masks_saved_as_binary_image(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_1/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(2)
                ),
            },
        )

        with TestDir() as test_dir:
            MvtecExporter.convert(source_dataset, test_dir, save_media=True)
            assert cv2.imread(test_dir + "/ground_truth/label_1/000_mask.png").max() == 255

    def test_detection_masks_saved_as_binary_image(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_1/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Bbox(0, 0, 8, 8, label=1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(2)
                ),
            },
        )

        with TestDir() as test_dir:
            MvtecExporter.convert(source_dataset, test_dir, save_media=True)
            assert cv2.imread(test_dir + "/ground_truth/label_1/000_mask.png").max() == 255
