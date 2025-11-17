import os.path as osp
from collections import OrderedDict
from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories, Mask
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.kitti.exporter import KittiExporter
from datumaro.plugins.data_formats.kitti.format import (
    KittiLabelMap,
    KittiPath,
    KittiTask,
    make_kitti_categories,
    parse_label_map,
    write_label_map,
)
from datumaro.plugins.data_formats.kitti.importer import (
    KittiDetectionImporter,
    KittiImporter,
    KittiSegmentationImporter,
)
from datumaro.util.meta_file_util import parse_meta_file
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, check_save_and_load, compare_datasets

DUMMY_DATASET_DIR = get_test_asset_path("kitti_dataset")


class KittiFormatTest(TestCase):
    def test_can_write_and_parse_labelmap(self):
        src_label_map = KittiLabelMap

        with TestDir() as test_dir:
            file_path = osp.join(test_dir, "label_colors.txt")

            write_label_map(file_path, src_label_map)
            dst_label_map = parse_label_map(file_path)

            self.assertEqual(src_label_map, dst_label_map)

    def test_can_write_and_parse_dataset_meta_file(self):
        src_label_map = KittiLabelMap

        with TestDir() as test_dir:
            source_dataset = Dataset.from_iterable([], categories=make_kitti_categories(src_label_map))

            KittiExporter.convert(source_dataset, test_dir, save_dataset_meta=True)
            dst_label_map = parse_meta_file(test_dir)

            self.assertEqual(src_label_map, dst_label_map)


class KittiImportTest(TestCase):
    def test_can_import_segmentation(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="000030_10",
                    subset="training",
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[1, 1, 0, 0, 0]]),
                            id=0,
                            label=3,
                            attributes={"is_crowd": True},
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 0, 0]]),
                            id=1,
                            label=27,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]]),
                            id=2,
                            label=27,
                            attributes={"is_crowd": False},
                        ),
                    ],
                ),
                DatasetItem(
                    id="000030_11",
                    subset="training",
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[1, 1, 0, 0, 0]]),
                            id=1,
                            label=31,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 0, 0]]),
                            id=1,
                            label=12,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]]),
                            id=0,
                            label=3,
                            attributes={"is_crowd": True},
                        ),
                    ],
                ),
            ],
            categories=make_kitti_categories(),
        )

        parsed_dataset = Dataset.import_from(osp.join(DUMMY_DATASET_DIR, "kitti_segmentation"), "kitti")

        compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_import_detection(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="000030_10",
                    subset="training",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(
                            0,
                            1,
                            2,
                            2,
                            label=0,
                            id=0,
                            attributes={"truncated": True, "occluded": False},
                        ),
                        Bbox(
                            0,
                            5,
                            1,
                            3,
                            label=1,
                            id=1,
                            attributes={"truncated": False, "occluded": False},
                        ),
                    ],
                ),
                DatasetItem(
                    id="000030_11",
                    subset="training",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(
                            0,
                            0,
                            2,
                            2,
                            label=1,
                            id=0,
                            attributes={"truncated": True, "occluded": True},
                        ),
                        Bbox(
                            4,
                            4,
                            2,
                            2,
                            label=1,
                            id=1,
                            attributes={"truncated": False, "occluded": False},
                        ),
                        Bbox(
                            6,
                            6,
                            1,
                            3,
                            label=1,
                            id=2,
                            attributes={"truncated": False, "occluded": True},
                        ),
                    ],
                ),
            ],
            categories=["Truck", "Van"],
        )

        parsed_dataset = Dataset.import_from(osp.join(DUMMY_DATASET_DIR, "kitti_detection"), "kitti")

        compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_detect_kitti(self):
        matrix = [
            # Whole dataset
            (DUMMY_DATASET_DIR, KittiImporter),
            # Subformats
            (DUMMY_DATASET_DIR, KittiSegmentationImporter),
            (DUMMY_DATASET_DIR, KittiDetectionImporter),
            # Subsets of subformats
            (osp.join(DUMMY_DATASET_DIR, "kitti_detection"), KittiDetectionImporter),
            (osp.join(DUMMY_DATASET_DIR, "kitti_detection", "training"), KittiDetectionImporter),
            (osp.join(DUMMY_DATASET_DIR, "kitti_segmentation"), KittiSegmentationImporter),
            (
                osp.join(DUMMY_DATASET_DIR, "kitti_segmentation", "training"),
                KittiSegmentationImporter,
            ),
        ]

        env = Environment()

        for path, subtask in matrix:
            with self.subTest(path=path, task=subtask):
                detected_formats = env.detect_dataset(path)
                self.assertIn(subtask.NAME, detected_formats)
                # Test false positives too
                for detected_format in detected_formats:
                    self.assertIn(detected_format, {"kitti", "kitti_segmentation", "kitti_detection"})


class TestExtractorBase(DatasetBase):
    def _label(self, kitti_label):
        return self.categories()[AnnotationType.label].find(kitti_label)[0]

    def categories(self):
        return make_kitti_categories()


class KittiExporterTest(TestCase):
    def _test_save_and_load(
        self, source_dataset, converter, test_dir, target_dataset=None, importer_args=None, **kwargs
    ):
        return check_save_and_load(
            self,
            source_dataset,
            converter,
            test_dir,
            importer="kitti",
            target_dataset=target_dataset,
            importer_args=importer_args,
            **kwargs,
        )

    def test_can_save_kitti_segm(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="1_2",
                            subset="test",
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    image=np.array([[0, 0, 0, 1, 0]]),
                                    label=3,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    image=np.array([[0, 1, 1, 0, 0]]),
                                    label=24,
                                    id=1,
                                    attributes={"is_crowd": False},
                                ),
                                Mask(
                                    image=np.array([[1, 0, 0, 0, 1]]),
                                    label=15,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                        DatasetItem(
                            id="3",
                            subset="val",
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    image=np.array([[1, 1, 0, 1, 1]]),
                                    label=3,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    image=np.array([[0, 0, 1, 0, 0]]),
                                    label=5,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(KittiExporter.convert, label_map="kitti", save_media=True),
                test_dir,
            )

    def test_can_save_kitti_detection(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1_2",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(
                            0,
                            1,
                            2,
                            2,
                            label=0,
                            id=0,
                            attributes={"truncated": False, "occluded": False, "score": 1.0},
                        ),
                    ],
                ),
                DatasetItem(
                    id="1_3",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(
                            0,
                            0,
                            2,
                            2,
                            label=1,
                            id=0,
                            attributes={"truncated": True, "occluded": False, "score": 1.0},
                        ),
                        Bbox(
                            6,
                            2,
                            3,
                            4,
                            label=1,
                            id=1,
                            attributes={"truncated": False, "occluded": True, "score": 1.0},
                        ),
                    ],
                ),
            ],
            categories=["label_0", "label_1"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(KittiExporter.convert, save_media=True, tasks=KittiTask.detection),
                test_dir,
            )

    def test_can_save_kitti_segm_unpainted(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="1_2",
                            subset="test",
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    image=np.array([[0, 0, 0, 1, 0]]),
                                    label=3,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    image=np.array([[0, 1, 1, 0, 0]]),
                                    label=24,
                                    id=1,
                                    attributes={"is_crowd": False},
                                ),
                                Mask(
                                    image=np.array([[1, 0, 0, 0, 1]]),
                                    label=15,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(KittiExporter.convert, label_map="kitti", save_media=True, apply_colormap=False),
                test_dir,
            )

    def test_can_save_kitti_dataset_with_no_subsets(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="1_2",
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    image=np.array([[1, 0, 0, 1, 0]]),
                                    label=0,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    image=np.array([[0, 1, 1, 0, 1]]),
                                    label=3,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                        DatasetItem(
                            id="1_3",
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    image=np.array([[1, 1, 0, 1, 0]]),
                                    label=1,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    image=np.array([[0, 0, 1, 0, 1]]),
                                    label=2,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(KittiExporter.convert, label_map="kitti", save_media=True),
                test_dir,
            )

    def test_can_save_kitti_dataset_without_frame_and_sequence(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="data",
                            subset="test",
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    image=np.array([[1, 0, 0, 1, 1]]),
                                    label=3,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    image=np.array([[0, 1, 1, 0, 0]]),
                                    label=24,
                                    id=1,
                                    attributes={"is_crowd": False},
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(KittiExporter.convert, label_map="kitti", save_media=True),
                test_dir,
            )

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="кириллица с пробелом",
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    image=np.array([[1, 0, 0, 1, 1]]),
                                    label=3,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    image=np.array([[0, 1, 1, 0, 0]]),
                                    label=24,
                                    id=1,
                                    attributes={"is_crowd": False},
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(KittiExporter.convert, label_map="kitti", save_media=True),
                test_dir,
            )

    def test_can_save_kitti_dataset_with_complex_id(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a/b/1",
                            subset="test",
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    image=np.array([[1, 0, 0, 1, 1]]),
                                    label=3,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    image=np.array([[0, 1, 1, 0, 0]]),
                                    label=24,
                                    id=1,
                                    attributes={"is_crowd": False},
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(KittiExporter.convert, label_map="kitti", save_media=True),
                test_dir,
            )

    def test_can_save_with_no_masks(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="city_1_2",
                            subset="test",
                            media=Image.from_numpy(data=np.ones((2, 5, 3))),
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(KittiExporter.convert, label_map="kitti", save_media=True),
                test_dir,
            )

    def test_dataset_with_source_labelmap_undefined(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[1, 0, 0, 1, 1]]),
                            label=1,
                            id=1,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 1, 1, 0, 0]]),
                            label=2,
                            id=2,
                            attributes={"is_crowd": False},
                        ),
                    ],
                )

            def categories(self):
                label_cat = LabelCategories()
                label_cat.add("background")
                label_cat.add("Label_1")
                label_cat.add("label_2")
                return {
                    AnnotationType.label: label_cat,
                }

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[1, 0, 0, 1, 1]]),
                            attributes={"is_crowd": False},
                            id=1,
                            label=self._label("Label_1"),
                        ),
                        Mask(
                            image=np.array([[0, 1, 1, 0, 0]]),
                            attributes={"is_crowd": False},
                            id=2,
                            label=self._label("label_2"),
                        ),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = None
                label_map["Label_1"] = None
                label_map["label_2"] = None
                return make_kitti_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(KittiExporter.convert, label_map="source", save_media=True),
                test_dir,
                target_dataset=DstExtractor(),
            )

    def test_dataset_with_source_labelmap_defined(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[1, 0, 0, 1, 1]]),
                            label=1,
                            id=1,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 1, 1, 0, 0]]),
                            label=2,
                            id=2,
                            attributes={"is_crowd": False},
                        ),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = (0, 0, 0)
                label_map["label_1"] = (1, 2, 3)
                label_map["label_2"] = (3, 2, 1)
                return make_kitti_categories(label_map)

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[1, 0, 0, 1, 1]]),
                            attributes={"is_crowd": False},
                            id=1,
                            label=self._label("label_1"),
                        ),
                        Mask(
                            image=np.array([[0, 1, 1, 0, 0]]),
                            attributes={"is_crowd": False},
                            id=2,
                            label=self._label("label_2"),
                        ),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = (0, 0, 0)
                label_map["label_1"] = (1, 2, 3)
                label_map["label_2"] = (3, 2, 1)
                return make_kitti_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(KittiExporter.convert, label_map="source", save_media=True),
                test_dir,
                target_dataset=DstExtractor(),
            )

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(id="q/1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".jpeg")),
                        DatasetItem(
                            id="a/b/c/2",
                            media=Image.from_numpy(data=np.ones((1, 5, 3)), ext=".bmp"),
                            annotations=[
                                Mask(
                                    image=np.array([[1, 0, 0, 1, 0]]),
                                    label=0,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    image=np.array([[0, 1, 1, 0, 1]]),
                                    label=1,
                                    id=0,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                    ]
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["a"] = None
                label_map["b"] = None
                return make_kitti_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(KittiExporter.convert, save_media=True),
                test_dir,
                require_media=True,
            )

            self.assertTrue(osp.isfile(osp.join(test_dir, "default", KittiPath.IMAGES_DIR, "a/b/c/2.bmp")))
            self.assertTrue(osp.isfile(osp.join(test_dir, "default", KittiPath.IMAGES_DIR, "q/1.jpeg")))

    def test_can_save_and_load_with_no_save_media_segmentation(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a",
                            media=Image.from_numpy(data=np.ones((5, 5, 3))),
                            annotations=[
                                Mask(
                                    image=np.array([[1, 0, 0, 0, 0]] * 5),
                                    label=0,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    image=np.array([[0, 1, 1, 1, 1]] * 5),
                                    label=1,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(KittiExporter.convert, save_media=False, label_map="kitti"),
                test_dir,
            )

    def test_can_save_and_load_with_no_save_media_detection(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="b",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Bbox(
                            0,
                            0,
                            3,
                            3,
                            label=0,
                            attributes={"truncated": True, "occluded": False, "score": 0.9},
                        )
                    ],
                )
            ],
            categories=["label_0"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(KittiExporter.convert, tasks=KittiTask.detection, save_media=False),
                test_dir,
            )

    def test_can_save_and_load_segmentation_with_unordered_labels(self):
        source_label_map = {
            "background": (0, 0, 0),
            "label_1": (10, 10, 10),
            "label_0": (20, 20, 20),
        }

        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[1, 0, 0, 0, 0]]),
                            attributes={"is_crowd": False},
                            label=0,
                            id=1,
                        ),
                        Mask(
                            image=np.array([[0, 1, 0, 0, 0]]),
                            attributes={"is_crowd": False},
                            label=1,
                            id=1,
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 1, 1]]),
                            attributes={"is_crowd": False},
                            label=2,
                            id=2,
                        ),
                    ],
                )
            ],
            categories=make_kitti_categories(source_label_map),
        )

        expected_label_map = {
            "background": (0, 0, 0),
            "label_0": (20, 20, 20),
            "label_1": (10, 10, 10),
        }

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[1, 0, 0, 0, 0]]),
                            attributes={"is_crowd": False},
                            label=0,
                            id=1,
                        ),
                        Mask(
                            image=np.array([[0, 1, 0, 0, 0]]),
                            attributes={"is_crowd": False},
                            label=2,
                            id=1,
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 1, 1]]),
                            attributes={"is_crowd": False},
                            label=1,
                            id=2,
                        ),
                    ],
                )
            ],
            categories=make_kitti_categories(expected_label_map),
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(KittiExporter.convert, tasks=KittiTask.segmentation, label_map=source_label_map),
                test_dir,
                target_dataset=expected_dataset,
            )

    def test_can_save_detection_with_score_attribute(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1_2",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(
                            0,
                            1,
                            2,
                            2,
                            label=0,
                            id=0,
                            attributes={"truncated": False, "occluded": False, "score": 0.78},
                        ),
                    ],
                ),
                DatasetItem(
                    id="1_3",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(
                            0,
                            0,
                            2,
                            2,
                            label=1,
                            id=0,
                            attributes={"truncated": True, "occluded": False, "score": 0.8},
                        ),
                        Bbox(
                            6,
                            2,
                            3,
                            4,
                            label=1,
                            id=1,
                            attributes={"truncated": False, "occluded": True, "score": 0.67},
                        ),
                    ],
                ),
            ],
            categories=["label_0", "label_1"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(KittiExporter.convert, save_media=True, tasks=KittiTask.detection),
                test_dir,
            )

    def test_can_save_detection_with_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1_2",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(
                            0,
                            1,
                            2,
                            2,
                            label=0,
                            id=0,
                            attributes={"truncated": False, "occluded": False, "score": 1.0},
                        ),
                    ],
                ),
                DatasetItem(
                    id="1_3",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(
                            0,
                            0,
                            2,
                            2,
                            label=1,
                            id=0,
                            attributes={"truncated": True, "occluded": False, "score": 1.0},
                        ),
                        Bbox(
                            6,
                            2,
                            3,
                            4,
                            label=1,
                            id=1,
                            attributes={"truncated": False, "occluded": True, "score": 1.0},
                        ),
                    ],
                ),
            ],
            categories=["label_0", "label_1"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(
                    KittiExporter.convert,
                    save_media=True,
                    save_dataset_meta=True,
                    tasks=KittiTask.detection,
                ),
                test_dir,
            )

    def test_can_save_segmentation_with_meta_file(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[1, 0, 0, 1, 1]]),
                            label=1,
                            id=1,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 1, 1, 0, 0]]),
                            label=2,
                            id=2,
                            attributes={"is_crowd": False},
                        ),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = (0, 0, 0)
                label_map["label_1"] = (1, 2, 3)
                label_map["label_2"] = (3, 2, 1)
                return make_kitti_categories(label_map)

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[1, 0, 0, 1, 1]]),
                            attributes={"is_crowd": False},
                            id=1,
                            label=self._label("label_1"),
                        ),
                        Mask(
                            image=np.array([[0, 1, 1, 0, 0]]),
                            attributes={"is_crowd": False},
                            id=2,
                            label=self._label("label_2"),
                        ),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = (0, 0, 0)
                label_map["label_1"] = (1, 2, 3)
                label_map["label_2"] = (3, 2, 1)
                return make_kitti_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(
                    KittiExporter.convert,
                    label_map="source",
                    save_media=True,
                    save_dataset_meta=True,
                ),
                test_dir,
                target_dataset=DstExtractor(),
            )
            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))
