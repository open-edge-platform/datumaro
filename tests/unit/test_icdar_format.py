import os.path as osp
from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Bbox, Caption, Mask, Polygon
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.icdar.base import (
    IcdarTextLocalizationImporter,
    IcdarTextSegmentationImporter,
    IcdarWordRecognitionImporter,
)
from datumaro.plugins.data_formats.icdar.exporter import (
    IcdarTextLocalizationExporter,
    IcdarTextSegmentationExporter,
    IcdarWordRecognitionExporter,
)

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, check_save_and_load, compare_datasets

DUMMY_DATASET_DIR = get_test_asset_path("icdar_dataset")


class IcdarImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_word_recognition(self):
        detected_formats = Environment().detect_dataset(
            osp.join(DUMMY_DATASET_DIR, "word_recognition")
        )
        self.assertEqual([IcdarWordRecognitionImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_text_localization(self):
        detected_formats = Environment().detect_dataset(
            osp.join(DUMMY_DATASET_DIR, "text_localization")
        )
        self.assertEqual([IcdarTextLocalizationImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_text_segmentation(self):
        detected_formats = Environment().detect_dataset(
            osp.join(DUMMY_DATASET_DIR, "text_segmentation")
        )
        self.assertEqual([IcdarTextSegmentationImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_captions(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="word_1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Caption("PROPER"),
                    ],
                ),
                DatasetItem(
                    id="word_2",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Caption("Canon"),
                    ],
                ),
            ],
        )

        dataset = Dataset.import_from(
            osp.join(DUMMY_DATASET_DIR, "word_recognition"), "icdar_word_recognition"
        )

        compare_datasets(self, expected_dataset, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_bboxes(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="img_1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Polygon([0, 0, 3, 1, 4, 6, 1, 7], attributes={"text": "FOOD"}),
                    ],
                ),
                DatasetItem(
                    id="img_2",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(0, 0, 2, 3, attributes={"text": "RED"}),
                        Bbox(3, 3, 2, 3, attributes={"text": "LION"}),
                    ],
                ),
            ],
        )

        dataset = Dataset.import_from(
            osp.join(DUMMY_DATASET_DIR, "text_localization"), "icdar_text_localization"
        )

        compare_datasets(self, expected_dataset, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_masks(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((2, 5, 3))),
                    annotations=[
                        Mask(
                            group=0,
                            image=np.array([[0, 1, 1, 0, 0], [0, 0, 0, 0, 0]]),
                            attributes={
                                "index": 0,
                                "color": "108 225 132",
                                "text": "F",
                                "center": "0 1",
                            },
                        ),
                        Mask(
                            group=1,
                            image=np.array([[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]]),
                            attributes={
                                "index": 1,
                                "color": "82 174 214",
                                "text": "T",
                                "center": "1 3",
                            },
                        ),
                        Mask(
                            group=1,
                            image=np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]),
                            attributes={
                                "index": 2,
                                "color": "241 73 144",
                                "text": "h",
                                "center": "1 4",
                            },
                        ),
                    ],
                ),
            ],
        )

        dataset = Dataset.import_from(
            osp.join(DUMMY_DATASET_DIR, "text_segmentation"), "icdar_text_segmentation"
        )

        compare_datasets(self, expected_dataset, dataset)


class IcdarConverterTest(TestCase):
    def _test_save_and_load(
        self,
        source_dataset,
        converter,
        test_dir,
        importer,
        target_dataset=None,
        importer_args=None,
        **kwargs,
    ):
        return check_save_and_load(
            self,
            source_dataset,
            converter,
            test_dir,
            importer,
            target_dataset=target_dataset,
            importer_args=importer_args,
            **kwargs,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_captions(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/b/1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Caption("caption 0"),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Caption("caption_1"),
                    ],
                ),
            ],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected_dataset,
                partial(IcdarWordRecognitionExporter.convert, save_media=True),
                test_dir,
                "icdar_word_recognition",
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_captions_with_no_save_media(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/b/1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Caption("caption 0"),
                    ],
                )
            ],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected_dataset,
                partial(IcdarWordRecognitionExporter.convert, save_media=False),
                test_dir,
                "icdar_word_recognition",
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_bboxes(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/b/1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(1, 3, 6, 10),
                        Bbox(0, 1, 3, 5, attributes={"text": "word 0"}),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Polygon([0, 0, 3, 0, 4, 7, 1, 8], attributes={"text": "word 1"}),
                        Polygon([1, 2, 5, 3, 6, 8, 0, 7]),
                    ],
                ),
                DatasetItem(
                    id=3,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Polygon([2, 2, 8, 3, 7, 10, 2, 9], attributes={"text": "word_2"}),
                        Bbox(0, 2, 5, 9, attributes={"text": "word_3"}),
                    ],
                ),
            ],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected_dataset,
                partial(IcdarTextLocalizationExporter.convert, save_media=True),
                test_dir,
                "icdar_text_localization",
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_bboxes_with_no_save_media(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Polygon([2, 2, 8, 3, 7, 10, 2, 9], attributes={"text": "word_2"}),
                        Bbox(0, 2, 5, 9, attributes={"text": "word_3"}),
                    ],
                ),
            ],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected_dataset,
                partial(IcdarTextLocalizationExporter.convert, save_media=False),
                test_dir,
                "icdar_text_localization",
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_masks(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/b/1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]]),
                            group=1,
                            attributes={
                                "index": 1,
                                "color": "82 174 214",
                                "text": "j",
                                "center": "0 3",
                            },
                        ),
                        Mask(
                            image=np.array([[0, 1, 1, 0, 0]]),
                            group=1,
                            attributes={
                                "index": 0,
                                "color": "108 225 132",
                                "text": "F",
                                "center": "0 1",
                            },
                        ),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[0, 0, 0, 0, 0, 1]]),
                            group=0,
                            attributes={
                                "index": 3,
                                "color": "183 6 28",
                                "text": " ",
                                "center": "0 5",
                            },
                        ),
                        Mask(
                            image=np.array([[1, 0, 0, 0, 0, 0]]),
                            group=1,
                            attributes={
                                "index": 0,
                                "color": "108 225 132",
                                "text": "L",
                                "center": "0 0",
                            },
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1, 0]]),
                            group=1,
                            attributes={
                                "index": 1,
                                "color": "82 174 214",
                                "text": "o",
                                "center": "0 3",
                            },
                        ),
                        Mask(
                            image=np.array([[0, 1, 1, 0, 0, 0]]),
                            group=0,
                            attributes={
                                "index": 2,
                                "color": "241 73 144",
                                "text": "P",
                                "center": "0 1",
                            },
                        ),
                    ],
                ),
            ],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected_dataset,
                partial(IcdarTextSegmentationExporter.convert, save_media=True),
                test_dir,
                "icdar_text_segmentation",
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_masks_with_no_save_media(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/b/1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]]),
                            group=1,
                            attributes={
                                "index": 1,
                                "color": "82 174 214",
                                "text": "j",
                                "center": "0 3",
                            },
                        ),
                        Mask(
                            image=np.array([[0, 1, 1, 0, 0]]),
                            group=1,
                            attributes={
                                "index": 0,
                                "color": "108 225 132",
                                "text": "F",
                                "center": "0 1",
                            },
                        ),
                    ],
                ),
            ],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected_dataset,
                partial(IcdarTextSegmentationExporter.convert, save_media=False),
                test_dir,
                "icdar_text_segmentation",
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_no_subsets(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 3, 5),
                    ],
                ),
            ],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected_dataset,
                IcdarTextLocalizationExporter.convert,
                test_dir,
                "icdar_text_localization",
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом", media=Image.from_numpy(data=np.ones((8, 8, 3)))
                )
            ]
        )

        for importer, converter in [
            ("icdar_word_recognition", IcdarWordRecognitionExporter),
            ("icdar_text_localization", IcdarTextLocalizationExporter),
            ("icdar_text_segmentation", IcdarTextSegmentationExporter),
        ]:
            with self.subTest(subformat=converter), TestDir() as test_dir:
                self._test_save_and_load(
                    expected_dataset,
                    partial(converter.convert, save_media=True),
                    test_dir,
                    importer,
                    require_media=True,
                )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id="q/1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG")
                ),
                DatasetItem(
                    id="a/b/c/2", media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp")
                ),
            ],
        )

        for importer, converter in [
            ("icdar_word_recognition", IcdarWordRecognitionExporter),
            ("icdar_text_localization", IcdarTextLocalizationExporter),
            ("icdar_text_segmentation", IcdarTextSegmentationExporter),
        ]:
            with self.subTest(subformat=converter), TestDir() as test_dir:
                self._test_save_and_load(
                    expected,
                    partial(converter.convert, save_media=True),
                    test_dir,
                    importer,
                    require_media=True,
                )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_captions_with_quotes(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[Caption('caption"')],
                )
            ],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected_dataset,
                partial(IcdarWordRecognitionExporter.convert, save_media=True),
                test_dir,
                "icdar_word_recognition",
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_segm_wo_color_attribute(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]]),
                            group=1,
                            attributes={
                                "index": 1,
                                "text": "j",
                                "center": "0 3",
                                "color": "0 128 0",
                            },
                        ),
                        Mask(
                            image=np.array([[0, 1, 1, 0, 0]]),
                            group=1,
                            attributes={"index": 0, "text": "F", "center": "0 1"},
                        ),
                        Mask(
                            image=np.array([[1, 0, 0, 0, 0]]),
                            group=1,
                            attributes={"index": 2, "text": "i", "center": "0 2"},
                        ),
                    ],
                ),
            ],
        )

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]]),
                            group=1,
                            attributes={
                                "index": 1,
                                "text": "j",
                                "center": "0 3",
                                "color": "0 128 0",
                            },
                        ),
                        Mask(
                            image=np.array([[0, 1, 1, 0, 0]]),
                            group=1,
                            attributes={
                                "index": 0,
                                "text": "F",
                                "center": "0 1",
                                "color": "128 0 0",
                            },
                        ),
                        Mask(
                            image=np.array([[1, 0, 0, 0, 0]]),
                            group=1,
                            attributes={
                                "index": 2,
                                "text": "i",
                                "center": "0 2",
                                "color": "128 128 0",
                            },
                        ),
                    ],
                ),
            ],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(IcdarTextSegmentationExporter.convert, save_media=True),
                test_dir,
                "icdar_text_segmentation",
                expected_dataset,
            )
