import os.path as osp
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories, Points
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.vgg_face2 import VggFace2Exporter, VggFace2Importer

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets


class VggFace2FormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=0),
                        Points([3.2, 3.12, 4.11, 3.2, 2.11, 2.5, 3.5, 2.11, 3.8, 2.13], label=0),
                    ],
                ),
                DatasetItem(
                    id="label_1/2",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Points(
                            [4.23, 4.32, 5.34, 4.45, 3.54, 3.56, 4.52, 3.51, 4.78, 3.34], label=1
                        ),
                    ],
                ),
                DatasetItem(
                    id="label_2/3",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(2)],
                ),
                DatasetItem(
                    id="label_3/4",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=3),
                        Points([3.2, 3.12, 4.11, 3.2, 2.11, 2.5, 3.5, 2.11, 3.8, 2.13], label=3),
                    ],
                ),
                DatasetItem(
                    id="no_label/a/5",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(2, 2, 2, 2),
                    ],
                ),
                DatasetItem(
                    id="no_label/label_0",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [("label_%s" % i, "class_%s" % i) for i in range(5)]
                ),
            },
        )

        with TestDir() as test_dir:
            VggFace2Exporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "vgg_face2")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_no_subsets(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/b/1",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=0),
                        Points(
                            [4.23, 4.32, 5.34, 4.45, 3.54, 3.56, 4.52, 3.51, 4.78, 3.34], label=0
                        ),
                    ],
                ),
            ],
            categories=["a"],
        )

        with TestDir() as test_dir:
            VggFace2Exporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "vgg_face2")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/кириллица с пробелом",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Points(
                            [4.23, 4.32, 5.34, 4.45, 3.54, 3.56, 4.52, 3.51, 4.78, 3.34], label=0
                        ),
                    ],
                ),
            ],
            categories=["a"],
        )

        with TestDir() as test_dir:
            VggFace2Exporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "vgg_face2")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    def test_can_save_dataset_with_no_save_media(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/1",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=0),
                        Points(
                            [4.23, 4.32, 5.34, 4.45, 3.54, 3.56, 4.52, 3.51, 4.78, 3.34], label=0
                        ),
                    ],
                ),
            ],
            categories=["label_0"],
        )

        with TestDir() as test_dir:
            VggFace2Exporter.convert(source_dataset, test_dir, save_media=False)
            parsed_dataset = Dataset.import_from(test_dir, "vgg_face2")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_no_labels(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="no_label/1",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2),
                        Points([4.23, 4.32, 5.34, 4.45, 3.54, 3.56, 4.52, 3.51, 4.78, 3.34]),
                    ],
                ),
                DatasetItem(
                    id="no_label/2",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(2, 2, 4, 2),
                    ],
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            VggFace2Exporter.convert(source_dataset, test_dir, save_media=False)
            parsed_dataset = Dataset.import_from(test_dir, "vgg_face2")

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_wrong_number_of_points(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="no_label/1",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Points([4.23, 4.32, 5.34, 3.51, 4.78, 3.34]),
                    ],
                ),
            ],
            categories=[],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="no_label/1", media=Image.from_numpy(data=np.ones((8, 8, 3))), annotations=[]
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            VggFace2Exporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "vgg_face2")

            compare_datasets(self, target_dataset, parsed_dataset)

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "no_label/q/1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG")
                ),
                DatasetItem(
                    "a/b/c/2",
                    media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp"),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=0),
                        Points(
                            [4.23, 4.32, 5.34, 4.45, 3.54, 3.56, 4.52, 3.51, 4.78, 3.34], label=0
                        ),
                    ],
                ),
            ],
            categories=["a"],
        )

        with TestDir() as test_dir:
            VggFace2Exporter.convert(dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "vgg_face2")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)

    def test_can_save_and_load_with_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="class_0/1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=0),
                        Points([3.2, 3.12, 4.11, 3.2, 2.11, 2.5, 3.5, 2.11, 3.8, 2.13], label=0),
                    ],
                ),
                DatasetItem(
                    id="class_1/2",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Points(
                            [4.23, 4.32, 5.34, 4.45, 3.54, 3.56, 4.52, 3.51, 4.78, 3.34], label=1
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [("class_%s" % i) for i in range(5)]
                ),
            },
        )

        with TestDir() as test_dir:
            VggFace2Exporter.convert(
                source_dataset, test_dir, save_media=True, save_dataset_meta=True
            )
            parsed_dataset = Dataset.import_from(test_dir, "vgg_face2")

            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))
            compare_datasets(self, source_dataset, parsed_dataset)


DUMMY_DATASET_DIR = get_test_asset_path("vgg_face2_dataset")


class VggFace2ImporterTest(TestCase):
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([VggFace2Importer.NAME], detected_formats)

    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="n000001/0001_01",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(2, 2, 1, 2, label=0),
                        Points(
                            [2.787, 2.898, 2.965, 2.79, 2.8, 2.456, 2.81, 2.32, 2.89, 2.3], label=0
                        ),
                    ],
                ),
                DatasetItem(
                    id="n000002/0001_01",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(2, 4, 2, 2, label=1),
                        Points([2.3, 4.9, 2.9, 4.93, 2.62, 4.745, 2.54, 4.45, 2.76, 4.43], label=1),
                    ],
                ),
                DatasetItem(
                    id="n000002/0002_01",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(1, 3, 1, 1, label=1),
                        Points([1.2, 3.8, 1.8, 3.82, 1.51, 3.634, 1.43, 3.34, 1.65, 3.32], label=1),
                    ],
                ),
                DatasetItem(
                    id="n000003/0003_01",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(1, 1, 1, 1, label=2),
                        Points([0.2, 2.8, 0.8, 2.9, 0.5, 2.6, 0.4, 2.3, 0.6, 2.3], label=2),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [("n000001", "Karl"), ("n000002", "Jay"), ("n000003", "Pol")]
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "vgg_face2")

        compare_datasets(self, expected_dataset, dataset)

    def test_can_import_specific_subset(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="n000003/0003_01",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(1, 1, 1, 1, label=2),
                        Points([0.2, 2.8, 0.8, 2.9, 0.5, 2.6, 0.4, 2.3, 0.6, 2.3], label=2),
                    ],
                )
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [("n000001", "Karl"), ("n000002", "Jay"), ("n000003", "Pol")]
                ),
            },
        )

        specific_subset = osp.join(DUMMY_DATASET_DIR, "bb_landmark", "loose_bb_test.csv")
        dataset = Dataset.import_from(specific_subset, "vgg_face2")

        compare_datasets(self, expected_dataset, dataset)
