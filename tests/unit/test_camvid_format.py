import os
import os.path as osp
from collections import OrderedDict
from functools import partial
from unittest import TestCase

import numpy as np

import datumaro.plugins.data_formats.camvid as Camvid
from datumaro.components.annotation import AnnotationType, LabelCategories, Mask, MaskCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.camvid import CamvidExporter, CamvidImporter
from datumaro.util.meta_file_util import parse_meta_file
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, check_save_and_load, compare_datasets


class CamvidFormatTest(TestCase):
    def test_can_write_and_parse_labelmap(self):
        src_label_map = Camvid.CamvidLabelMap

        with TestDir() as test_dir:
            file_path = osp.join(test_dir, "label_colors.txt")
            Camvid.write_label_map(file_path, src_label_map)
            dst_label_map = Camvid.parse_label_map(file_path)

            self.assertEqual(src_label_map, dst_label_map)

    def test_can_write_and_parse_meta_file(self):
        src_label_map = Camvid.CamvidLabelMap

        with TestDir() as test_dir:
            source_dataset = Dataset.from_iterable(
                [],
                categories=Camvid.make_camvid_categories(src_label_map),
            )

            CamvidExporter.convert(source_dataset, test_dir, save_dataset_meta=True)
            dst_label_map = parse_meta_file(test_dir)

            self.assertEqual(src_label_map, dst_label_map)


DUMMY_DATASET_DIR = get_test_asset_path("camvid_dataset")


class TestExtractorBase(DatasetBase):
    def _label(self, camvid_label):
        return self.categories()[AnnotationType.label].find(camvid_label)[0]

    def categories(self):
        return Camvid.make_camvid_categories()


class CamvidImportTest(TestCase):
    def test_can_import(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0001TP_008550",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 0, 0]]), label=1),
                        Mask(image=np.array([[0, 0, 1, 0, 0]]), label=18),
                        Mask(image=np.array([[0, 0, 0, 1, 1]]), label=22),
                    ],
                ),
                DatasetItem(
                    id="0001TP_008580",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 0, 0]]), label=2),
                        Mask(image=np.array([[0, 0, 1, 0, 0]]), label=4),
                        Mask(image=np.array([[0, 0, 0, 1, 1]]), label=27),
                    ],
                ),
                DatasetItem(
                    id="0001TP_006690",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 1]]), label=3),
                        Mask(image=np.array([[0, 0, 1, 0, 0]]), label=18),
                    ],
                ),
                DatasetItem(
                    id="0016E5_07959",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(image=np.array([[1, 1, 1, 0, 0]]), label=1),
                        Mask(image=np.array([[0, 0, 0, 1, 1]]), label=8),
                    ],
                ),
            ],
            categories=Camvid.make_camvid_categories(),
        )

        parsed_dataset = Dataset.import_from(DUMMY_DATASET_DIR, "camvid")

        compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_detect_camvid(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([CamvidImporter.NAME], detected_formats)


class CamvidExporterTest(TestCase):
    def _test_save_and_load(
        self, source_dataset, converter, test_dir, target_dataset=None, importer_args=None, **kwargs
    ):
        return check_save_and_load(
            self,
            source_dataset,
            converter,
            test_dir,
            importer="camvid",
            target_dataset=target_dataset,
            importer_args=importer_args,
            **kwargs,
        )

    def test_can_save_camvid_segm(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a/b/1",
                            subset="test",
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(image=np.array([[0, 0, 0, 1, 0]]), label=0),
                                Mask(image=np.array([[0, 1, 1, 0, 0]]), label=3),
                                Mask(image=np.array([[1, 0, 0, 0, 1]]), label=4),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(), partial(CamvidExporter.convert, label_map="camvid"), test_dir)

    def test_can_save_camvid_segm_unpainted(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id=1,
                            subset="a",
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(image=np.array([[0, 0, 0, 1, 0]]), label=0),
                                Mask(image=np.array([[0, 1, 1, 0, 0]]), label=3),
                                Mask(image=np.array([[1, 0, 0, 0, 1]]), label=4),
                            ],
                        ),
                    ]
                )

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id=1,
                            subset="a",
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(image=np.array([[0, 0, 0, 1, 0]]), label=0),
                                Mask(image=np.array([[0, 1, 1, 0, 0]]), label=3),
                                Mask(image=np.array([[1, 0, 0, 0, 1]]), label=4),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(CamvidExporter.convert, label_map="camvid", apply_colormap=False),
                test_dir,
                target_dataset=DstExtractor(),
            )

    def test_can_save_dataset_with_no_subsets(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id=1,
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(image=np.array([[1, 0, 0, 1, 0]]), label=0),
                                Mask(image=np.array([[0, 1, 1, 0, 1]]), label=3),
                            ],
                        ),
                        DatasetItem(
                            id=2,
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(image=np.array([[1, 1, 0, 1, 0]]), label=1),
                                Mask(image=np.array([[0, 0, 1, 0, 1]]), label=2),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(), partial(CamvidExporter.convert, label_map="camvid"), test_dir)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="кириллица с пробелом",
                            media=Image.from_numpy(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(image=np.array([[1, 0, 0, 1, 0]]), label=0),
                                Mask(image=np.array([[0, 1, 1, 0, 1]]), label=3),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(), partial(CamvidExporter.convert, label_map="camvid"), test_dir)

    def test_can_save_with_no_masks(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a/b/1",
                            subset="test",
                            media=Image.from_numpy(data=np.ones((2, 5, 3))),
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(), partial(CamvidExporter.convert, label_map="camvid"), test_dir)

    def test_dataset_with_source_labelmap_undefined(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 0]]), label=0),
                        Mask(image=np.array([[0, 0, 1, 0, 0]]), label=1),
                    ],
                )

            def categories(self):
                label_cat = LabelCategories()
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
                        Mask(image=np.array([[1, 1, 0, 1, 0]]), label=self._label("Label_1")),
                        Mask(image=np.array([[0, 0, 1, 0, 0]]), label=self._label("label_2")),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = None
                label_map["Label_1"] = None
                label_map["label_2"] = None
                return Camvid.make_camvid_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(CamvidExporter.convert, label_map="source"),
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
                        Mask(image=np.array([[1, 1, 0, 1, 0]]), label=1),
                        Mask(image=np.array([[0, 0, 1, 0, 1]]), label=2),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = (0, 0, 0)
                label_map["label_1"] = (1, 2, 3)
                label_map["label_2"] = (3, 2, 1)
                return Camvid.make_camvid_categories(label_map)

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 0]]), label=self._label("label_1")),
                        Mask(image=np.array([[0, 0, 1, 0, 1]]), label=self._label("label_2")),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = (0, 0, 0)
                label_map["label_1"] = (1, 2, 3)
                label_map["label_2"] = (3, 2, 1)
                return Camvid.make_camvid_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(CamvidExporter.convert, label_map="source"),
                test_dir,
                target_dataset=DstExtractor(),
            )

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(id="q/1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG")),
                        DatasetItem(
                            id="a/b/c/2",
                            media=Image.from_numpy(data=np.ones((1, 5, 3)), ext=".bmp"),
                            annotations=[
                                Mask(np.array([[0, 0, 0, 1, 0]]), label=self._label("a")),
                                Mask(np.array([[0, 1, 1, 0, 0]]), label=self._label("b")),
                            ],
                        ),
                    ]
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["a"] = None
                label_map["b"] = None
                return Camvid.make_camvid_categories(label_map)

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(id="q/1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG")),
                        DatasetItem(
                            id="a/b/c/2",
                            media=Image.from_numpy(data=np.ones((1, 5, 3)), ext=".bmp"),
                            annotations=[
                                Mask(np.array([[1, 0, 0, 0, 1]]), label=self._label("background")),
                                Mask(np.array([[0, 0, 0, 1, 0]]), label=self._label("a")),
                                Mask(np.array([[0, 1, 1, 0, 0]]), label=self._label("b")),
                            ],
                        ),
                    ]
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = None
                label_map["a"] = None
                label_map["b"] = None
                return Camvid.make_camvid_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(CamvidExporter.convert, save_media=True),
                test_dir,
                require_media=True,
                target_dataset=DstExtractor(),
            )

    def test_inplace_save_writes_only_updated_data(self):
        src_mask_cat = MaskCategories.generate(3, include_background=False)

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((2, 1, 3))),
                    annotations=[Mask(np.ones((2, 1)), label=2)],
                ),
                DatasetItem(2, subset="a", media=Image.from_numpy(data=np.ones((3, 2, 3)))),
                DatasetItem(2, subset="b"),
            ],
            categories=Camvid.make_camvid_categories(
                OrderedDict(
                    [
                        ("background", (0, 0, 0)),
                        ("a", src_mask_cat.colormap[0]),
                        ("b", src_mask_cat.colormap[1]),
                    ]
                )
            ),
        )

        with TestDir() as path:
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        1,
                        subset="a",
                        media=Image.from_numpy(data=np.ones((2, 1, 3))),
                        annotations=[Mask(np.ones((2, 1)), label=1)],
                    ),
                    DatasetItem(2, subset="b"),
                    DatasetItem(
                        3,
                        subset="c",
                        media=Image.from_numpy(data=np.ones((2, 2, 3))),
                        annotations=[Mask(np.ones((2, 2)), label=0)],
                    ),
                ],
                categories={
                    AnnotationType.label: LabelCategories.from_iterable(["a", "b"]),
                    AnnotationType.mask: src_mask_cat,
                },
                media_type=Image,
            )
            dataset.export(path, "camvid", save_media=True)

            dataset.put(DatasetItem(2, subset="a", media=Image.from_numpy(data=np.ones((3, 2, 3)))))
            dataset.remove(3, "c")
            dataset.save(save_media=True)

            self.assertEqual({"a", "aannot", "a.txt", "b.txt", "label_colors.txt"}, set(os.listdir(path)))
            self.assertEqual({"1.jpg", "2.jpg"}, set(os.listdir(osp.join(path, "a"))))
            compare_datasets(self, expected, Dataset.import_from(path, "camvid"), require_media=True)

    def test_dataset_with_meta_file(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 0]]), label=1),
                        Mask(image=np.array([[0, 0, 1, 0, 1]]), label=2),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = (0, 0, 0)
                label_map["label_1"] = (1, 2, 3)
                label_map["label_2"] = (3, 2, 1)
                return Camvid.make_camvid_categories(label_map)

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 0]]), label=self._label("label_1")),
                        Mask(image=np.array([[0, 0, 1, 0, 1]]), label=self._label("label_2")),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = (0, 0, 0)
                label_map["label_1"] = (1, 2, 3)
                label_map["label_2"] = (3, 2, 1)
                return Camvid.make_camvid_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(CamvidExporter.convert, label_map="source", save_dataset_meta=True),
                test_dir,
                target_dataset=DstExtractor(),
            )
            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))
