from unittest import TestCase

import cv2
import numpy as np
import pytest

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
from datumaro.util.image import load_image, save_image
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
                AnnotationType.label: LabelCategories.from_iterable("label_" + str(label) for label in range(2)),
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
                AnnotationType.label: LabelCategories.from_iterable("label_" + str(label) for label in range(3)),
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
                AnnotationType.label: LabelCategories.from_iterable("label_" + str(label) for label in range(3)),
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
                AnnotationType.label: LabelCategories.from_iterable("label_" + str(label) for label in range(3)),
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
                AnnotationType.label: LabelCategories.from_iterable("label_" + str(label) for label in range(3)),
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
                    annotations=[Bbox(x=0, y=0, w=4, h=4, label=1)],
                ),
                DatasetItem(
                    id="label_2/000",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Bbox(x=4, y=4, w=4, h=4, label=2)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable("label_" + str(label) for label in range(3)),
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
                AnnotationType.label: LabelCategories.from_iterable("label_" + str(label) for label in range(2)),
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
                AnnotationType.label: LabelCategories.from_iterable("label_" + str(label) for label in range(2)),
            },
        )

        with TestDir() as test_dir:
            MvtecExporter.convert(source_dataset, test_dir, save_media=True)
            assert cv2.imread(test_dir + "/ground_truth/label_1/000_mask.png").max() == 255


@pytest.mark.parametrize(
    "x, y, width, height", [(3, 2, 4, 3), (5, 4, 1, 1), (3, 2, 4, 1), (3, 2, 1, 4), (0, 0, 10, 10)]
)
def test_detection_boxes_cover_defect_pixels(tmp_path, x, y, width, height):
    image_dir = tmp_path / "test" / "defect"
    mask_dir = tmp_path / "ground_truth" / "defect"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[y : y + height, x : x + width] = 255
    save_image(str(image_dir / "sample.png"), np.zeros((10, 10, 3), dtype=np.uint8))
    save_image(str(mask_dir / "sample_mask.png"), mask)

    dataset = Dataset.import_from(str(tmp_path), "mvtec_detection")

    annotations = dataset.get("defect/sample", subset="test").annotations
    assert len(annotations) == 1
    assert annotations[0].get_bbox() == [x, y, width, height]

    export_dir = tmp_path / "exported"
    dataset.export(str(export_dir), "mvtec_detection", save_media=True)
    exported = load_image(str(export_dir / "ground_truth" / "defect" / "sample_mask.png"))
    np.testing.assert_array_equal(np.squeeze(exported) > 0, mask > 0)
