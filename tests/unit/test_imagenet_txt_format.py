import os.path as osp
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.imagenet_txt import ImagenetTxtExporter, ImagenetTxtImporter

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets


class ImagenetTxtFormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="train", annotations=[Label(0)]),
                DatasetItem(
                    id="2",
                    subset="train",
                    media=Image.from_numpy(data=np.zeros((8, 8, 3))),
                    annotations=[Label(1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(4)
                ),
            },
        )

        with TestDir() as test_dir:
            ImagenetTxtExporter.convert(source_dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "imagenet_txt")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_with_no_save_media(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="train", annotations=[Label(0)]),
            ],
            categories=["label_0"],
        )

        with TestDir() as test_dir:
            ImagenetTxtExporter.convert(source_dataset, test_dir, save_media=False)

            parsed_dataset = Dataset.import_from(test_dir, "imagenet_txt")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_dataset_with_save_dataset_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="train", annotations=[Label(0)]),
                DatasetItem(id="2", subset="train", annotations=[Label(1)]),
            ],
            categories=["label_0", "label_1"],
        )

        with TestDir() as test_dir:
            ImagenetTxtExporter.convert(source_dataset, test_dir, save_dataset_meta=True)

            parsed_dataset = Dataset.import_from(test_dir, "imagenet_txt")

            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))
            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_and_load_with_multiple_labels(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="train", annotations=[Label(1), Label(2), Label(3)]),
                DatasetItem(
                    id="2",
                    subset="train",
                    media=Image.from_numpy(data=np.zeros((2, 8, 3))),
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            ImagenetTxtExporter.convert(source_dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "imagenet_txt")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_dataset_with_no_subsets(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/b/c",
                    media=Image.from_numpy(data=np.zeros((8, 4, 3))),
                    annotations=[Label(1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            ImagenetTxtExporter.convert(source_dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "imagenet_txt")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    media=Image.from_numpy(data=np.zeros((8, 8, 3))),
                    annotations=[Label(0), Label(1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(2)
                ),
            },
        )

        with TestDir() as test_dir:
            ImagenetTxtExporter.convert(dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "imagenet_txt")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG")
                ),
                DatasetItem(
                    id="b/c/d/2", media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp")
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            ImagenetTxtExporter.convert(dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "imagenet_txt")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)


DUMMY_DATASET_DIR = get_test_asset_path("imagenet_txt_dataset", "basic")
DUMMY_DATASET_WITH_CUSTOM_LABELS_DIR = get_test_asset_path("imagenet_txt_dataset", "custom_labels")
DUMMY_DATASET_WITH_NO_LABELS_DIR = get_test_asset_path("imagenet_txt_dataset", "no_labels")


class ImagenetTxtImporterTest(TestCase):
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image.from_numpy(data=np.zeros((8, 6, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="2",
                    subset="train",
                    media=Image.from_numpy(data=np.zeros((2, 8, 3))),
                    annotations=[Label(5)],
                ),
                DatasetItem(id="3", subset="train", annotations=[Label(3)]),
                DatasetItem(id="4", subset="train", annotations=[Label(5)]),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_%s" % label for label in range(10)
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "imagenet_txt")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_import_with_custom_labels_file(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="train", annotations=[Label(0)]),
            ],
            categories=["alt_label_%s" % label for label in range(10)],
        )

        dataset = Dataset.import_from(
            DUMMY_DATASET_WITH_CUSTOM_LABELS_DIR, "imagenet_txt", labels_file="synsets-alt.txt"
        )

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_import_with_no_labels_file(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="train", annotations=[Label(4)]),
            ],
            categories=["class-%s" % label for label in range(5)],
        )

        dataset = Dataset.import_from(
            DUMMY_DATASET_WITH_NO_LABELS_DIR, "imagenet_txt", labels="generate"
        )

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_detect_imagenet(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([ImagenetTxtImporter.NAME], detected_formats)
