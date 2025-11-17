# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
import os
import os.path as osp
import pickle
from typing import Any, Dict

import numpy as np
import pytest
from lxml import etree as ElementTree  # nosec

import datumaro.plugins.data_formats.voc.format as VOC
from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories, Mask, MaskCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.errors import (
    AnnotationImportError,
    DatasetImportError,
    InvalidAnnotationError,
    ItemImportError,
)
from datumaro.components.exporter import Exporter
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.plugins.data_formats.voc.exporter import (
    VocActionExporter,
    VocClassificationExporter,
    VocDetectionExporter,
    VocInstanceSegmentationExporter,
    VocLayoutExporter,
    VocSegmentationExporter,
)
from datumaro.plugins.data_formats.voc.format import VocTask
from datumaro.plugins.data_formats.voc.importer import (
    VocActionImporter,
    VocClassificationImporter,
    VocDetectionImporter,
    VocInstanceSegmentationImporter,
    VocLayoutImporter,
    VocSegmentationImporter,
)
from datumaro.util.image import save_image
from datumaro.util.mask_tools import load_mask
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

from .base import TestDataFormatBase

DUMMY_DATASET_DIR = get_test_asset_path("voc_dataset", "voc_dataset1")


@pytest.fixture
def fxt_classification_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="2007_000001",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[Label(label=l) for l in range(len(VOC.VocLabel)) if l % 2 == 1],
            ),
            DatasetItem(
                id="2007_000002",
                subset="test",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
            ),
        ],
        categories=VOC.make_voc_categories(task=VocTask.voc_classification),
    )


@pytest.fixture
def fxt_detection_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="2007_000001",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[
                    Bbox(
                        1.0,
                        2.0,
                        2.0,
                        2.0,
                        label=8,
                        id=0,
                        group=0,
                        attributes={
                            "difficult": False,
                            "truncated": True,
                            "occluded": False,
                            "pose": "Unspecified",
                        },
                    ),
                    Bbox(
                        4.0,
                        5.0,
                        2.0,
                        2.0,
                        label=15,
                        id=1,
                        group=1,
                        attributes={
                            "difficult": False,
                            "truncated": False,
                            "occluded": False,
                        },
                    ),
                ],
            ),
            DatasetItem(
                id="2007_000002",
                subset="test",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
            ),
        ],
        categories=VOC.make_voc_categories(task=VocTask.voc_detection),
    )


@pytest.fixture
def fxt_segmentation_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="2007_000001",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[Mask(image=np.ones([10, 20]), label=2, group=1)],
            ),
            DatasetItem(
                id="2007_000002",
                subset="test",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
            ),
        ],
        categories=VOC.make_voc_categories(task=VocTask.voc_segmentation),
    )


@pytest.fixture
def fxt_layout_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="2007_000001",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[
                    Bbox(
                        4.0,
                        5.0,
                        2.0,
                        2.0,
                        label=1,
                        id=0,
                        group=0,
                        attributes={
                            "difficult": False,
                            "truncated": False,
                            "occluded": False,
                        },
                    ),
                    Bbox(5.5, 6.0, 2.0, 2.0, label=2, group=0),
                ],
            ),
            DatasetItem(
                id="2007_000002",
                subset="test",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
            ),
        ],
        categories=VOC.make_voc_categories(task=VocTask.voc_layout),
    )


@pytest.fixture
def fxt_action_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="2007_000001",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[
                    Bbox(
                        4.0,
                        5.0,
                        2.0,
                        2.0,
                        label=1,
                        id=0,
                        group=0,
                        attributes={
                            "difficult": False,
                            "truncated": False,
                            "occluded": False,
                            **{a.name: a.value % 2 == 1 for a in VOC.VocAction},
                        },
                    )
                ],
            ),
            DatasetItem(
                id="2007_000002",
                subset="test",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
            ),
        ],
        categories=VOC.make_voc_categories(task=VocTask.voc_action),
    )


class VocFormatImportExportTest(TestDataFormatBase):
    @pytest.mark.parametrize(
        "fxt_dataset_dir, importer",
        [
            (DUMMY_DATASET_DIR, VocClassificationImporter),
            (DUMMY_DATASET_DIR, VocDetectionImporter),
            (DUMMY_DATASET_DIR, VocSegmentationImporter),
            (DUMMY_DATASET_DIR, VocLayoutImporter),
            (DUMMY_DATASET_DIR, VocActionImporter),
        ],
        ids=["cls", "det", "seg", "layout", "action"],
    )
    def test_can_detect(self, fxt_dataset_dir: str, importer: Importer):
        detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(fxt_dataset_dir)
        assert importer.NAME in detected_formats

    @pytest.mark.parametrize(
        [
            "fxt_dataset_dir",
            "fxt_expected_dataset",
            "importer",
            "fxt_import_kwargs",
        ],
        [
            (DUMMY_DATASET_DIR, "fxt_classification_dataset", VocClassificationImporter, {}),
            (DUMMY_DATASET_DIR, "fxt_detection_dataset", VocDetectionImporter, {}),
            (DUMMY_DATASET_DIR, "fxt_segmentation_dataset", VocSegmentationImporter, {}),
            (DUMMY_DATASET_DIR, "fxt_layout_dataset", VocLayoutImporter, {}),
            (DUMMY_DATASET_DIR, "fxt_action_dataset", VocActionImporter, {}),
        ],
        indirect=["fxt_expected_dataset"],
        ids=["cls", "det", "seg", "layout", "action"],
    )
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
        importer: Importer,
    ):
        return super().test_can_import(
            fxt_dataset_dir,
            fxt_expected_dataset,
            fxt_import_kwargs,
            request,
            importer=importer,
        )

    @pytest.mark.parametrize(
        "fxt_expected_dataset, exporter, fxt_export_kwargs, importer, fxt_import_kwargs",
        [
            (
                "fxt_classification_dataset",
                VocClassificationExporter,
                {"label_map": "voc_classification"},
                VocClassificationImporter,
                {},
            ),
            (
                "fxt_detection_dataset",
                VocDetectionExporter,
                {"label_map": "voc_detection"},
                VocDetectionImporter,
                {},
            ),
            (
                "fxt_segmentation_dataset",
                VocSegmentationExporter,
                {"label_map": "voc_segmentation"},
                VocSegmentationImporter,
                {},
            ),
            (
                "fxt_layout_dataset",
                VocLayoutExporter,
                {"label_map": "voc_layout"},
                VocLayoutImporter,
                {},
            ),
            (
                "fxt_action_dataset",
                VocActionExporter,
                {"label_map": "voc_action"},
                VocActionImporter,
                {},
            ),
        ],
        indirect=["fxt_expected_dataset"],
    )
    def test_can_export_and_import(
        self,
        fxt_expected_dataset: Dataset,
        test_dir: str,
        fxt_import_kwargs: Dict[str, Any],
        fxt_export_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
        exporter: Exporter,
        importer: Importer,
    ):
        return super().test_can_export_and_import(
            fxt_expected_dataset,
            test_dir,
            fxt_import_kwargs,
            fxt_export_kwargs,
            request,
            exporter=exporter,
            importer=importer,
        )

    @pytest.mark.parametrize(
        "fxt_dataset_dir,fxt_format",
        [
            (DUMMY_DATASET_DIR, "voc_classification"),
            (DUMMY_DATASET_DIR, "voc_detection"),
            (DUMMY_DATASET_DIR, "voc_action"),
            (DUMMY_DATASET_DIR, "voc_layout"),
            (DUMMY_DATASET_DIR, "voc_detection"),
        ],
    )
    def test_can_pickle(self, fxt_dataset_dir: str, fxt_format: str, request: pytest.FixtureRequest):
        source = Dataset.import_from(fxt_dataset_dir, format=fxt_format)

        parsed = pickle.loads(pickle.dumps(source))

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, source, parsed)

    def test_colormap_generator(self):
        reference = np.array(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
                [224, 224, 192],  # ignored
            ]
        )

        assert np.array_equal(reference, list(VOC.VocColormap.values()))

    def test_can_write_and_parse_labelmap(self, test_dir: str):
        src_label_map = VOC.make_voc_label_map()
        src_label_map["qq"] = [None, ["part1", "part2"], ["act1", "act2"]]
        src_label_map["ww"] = [(10, 20, 30), [], ["act3"]]

        file_path = osp.join(test_dir, "test.txt")

        VOC.write_label_map(file_path, src_label_map)
        dst_label_map = VOC.parse_label_map(file_path)

        assert src_label_map == dst_label_map

    def test_can_write_and_parse_dataset_meta_file(self, test_dir: str):
        src_label_map = VOC.make_voc_label_map()
        src_label_map["qq"] = [None, ["part1", "part2"], ["act1", "act2"]]
        src_label_map["ww"] = [(10, 20, 30), [], ["act3"]]

        VOC.write_meta_file(test_dir, src_label_map)
        dst_label_map = VOC.parse_meta_file(test_dir)

        assert src_label_map == dst_label_map

    def test_can_report_invalid_line_in_labelmap(self, test_dir: str):
        path = osp.join(test_dir, "labelmap.txt")
        with open(path, "w") as f:
            f.write("a\n")

        with pytest.raises(InvalidAnnotationError) as err_info:
            VOC.parse_label_map(path)
        assert (
            str(err_info.value) == "Label description has wrong number of fields '1'. Expected 4 ':'-separated fields."
        )

    def test_can_report_repeated_label_in_labelmap(self, test_dir: str):
        path = osp.join(test_dir, "labelmap.txt")
        with open(path, "w") as f:
            f.write("a:::\n")
            f.write("a:::\n")

        with pytest.raises(InvalidAnnotationError) as err_info:
            VOC.parse_label_map(path)
        assert str(err_info.value) == "Label 'a' is already defined in the label map"

    def test_can_report_invalid_color_in_labelmap(self, test_dir: str):
        path = osp.join(test_dir, "labelmap.txt")
        with open(path, "w") as f:
            f.write("a:10,20::\n")

        with pytest.raises(InvalidAnnotationError) as err_info:
            VOC.parse_label_map(path)
        assert str(err_info.value) == "Label 'a' has wrong color '['10', '20']'. Expected an 'r,g,b' triplet."


class VocFormatPracticeTest:
    def test_can_save_attributes(self, test_dir: str, request: pytest.FixtureRequest):
        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    annotations=[
                        Bbox(
                            2,
                            3,
                            4,
                            5,
                            label=15,
                            attributes={"occluded": True, "x": 1, "y": "2"},
                        ),
                    ],
                ),
            ],
            categories=VOC.make_voc_categories(task=VocTask.voc_detection),
        )

        dst_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    annotations=[
                        Bbox(
                            2,
                            3,
                            4,
                            5,
                            label=15,
                            id=0,
                            group=0,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": True,
                                "x": "1",
                                "y": "2",  # can only read strings
                            },
                        ),
                    ],
                ),
            ],
            categories=VOC.make_voc_categories(task=VocTask.voc_detection),
        )

        VocDetectionExporter.convert(
            src_dataset,
            save_dir=test_dir,
            save_media=True,
        )
        imported = Dataset.import_from(test_dir, VocDetectionImporter.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, dst_dataset, imported, require_media=True)

    def test_can_save_voc_with_custom_labelmap(self, test_dir: str, request: pytest.FixtureRequest):
        def src_categories():
            label_cat = LabelCategories()
            label_cat.add(VOC.VocLabel.cat.name)
            label_cat.add("non_voc_label")
            return {
                AnnotationType.label: label_cat,
                AnnotationType.mask: MaskCategories(colormap=VOC.generate_colormap(3)),
            }

        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(2, 3, 4, 5, label=0, id=1),
                        Bbox(1, 2, 3, 4, label=1, id=2),
                    ],
                )
            ],
            categories=src_categories(),
        )

        def dst_categories():
            label_cat = LabelCategories()
            label_cat.attributes.update(["difficult", "truncated", "occluded"])
            label_cat.add(VOC.VocLabel.cat.name)
            label_cat.add("non_voc_label")
            return {
                AnnotationType.label: label_cat,
            }

        dst_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        # drop non voc label
                        Bbox(
                            2,
                            3,
                            4,
                            5,
                            label=0,
                            id=0,
                            group=0,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=1,
                            id=1,
                            group=1,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                    ],
                )
            ],
            categories=dst_categories(),
        )

        VocDetectionExporter.convert(
            src_dataset,
            save_dir=test_dir,
            save_media=True,
        )
        imported_dataset = Dataset.import_from(test_dir, VocDetectionImporter.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, dst_dataset, imported_dataset, require_media=True)

    def test_can_save_voc_segm_unpainted(self, test_dir: str, request: pytest.FixtureRequest):
        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="a",
                    annotations=[
                        # overlapping masks, the first should be truncated
                        # the second and third are different instances
                        Mask(image=np.array([[0, 1, 1, 1, 0]]), label=4, z_order=1),
                        Mask(image=np.array([[1, 1, 0, 0, 0]]), label=3, z_order=2),
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3, z_order=3),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["background", "a", "b", "c", "d"]),
                AnnotationType.mask: MaskCategories(colormap=VOC.generate_colormap(5)),
            },
        )

        dst_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="a",
                    annotations=[
                        Mask(image=np.array([[0, 0, 1, 0, 0]]), label=4, group=1),
                        Mask(image=np.array([[1, 1, 0, 0, 0]]), label=3, group=2),
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3, group=3),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    ["background", "a", "b", "c", "d"],
                ),
                AnnotationType.mask: MaskCategories(colormap=VOC.generate_colormap(5)),
            },
        )

        VocSegmentationExporter.convert(
            src_dataset,
            save_dir=test_dir,
            save_media=True,
        )
        imported_dataset = Dataset.import_from(test_dir, VocSegmentationImporter.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, dst_dataset, imported_dataset, require_media=True)

    def test_can_save_voc_segm_with_many_instances(self, test_dir: str, request: pytest.FixtureRequest):
        def bit(x, y, shape):
            mask = np.zeros(shape)
            mask[y, x] = 1
            return mask

        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="a",
                    annotations=[
                        Mask(
                            image=bit(x, y, shape=[10, 10]),
                            label=3,
                            z_order=10 * y + x + 1,
                        )
                        for y in range(10)
                        for x in range(10)
                    ],
                ),
            ],
            categories=VOC.make_voc_categories(task=VocTask.voc_segmentation),
        )

        dst_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="a",
                    annotations=[
                        Mask(
                            image=bit(x, y, shape=[10, 10]),
                            label=3,
                            group=10 * y + x + 1,
                        )
                        for y in range(10)
                        for x in range(10)
                    ],
                ),
            ],
            categories=VOC.make_voc_categories(task=VocTask.voc_segmentation),
        )

        VocSegmentationExporter.convert(src_dataset, save_dir=test_dir, save_media=True, label_map="voc_segmentation")
        imported_dataset = Dataset.import_from(test_dir, VocSegmentationImporter.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, dst_dataset, imported_dataset, require_media=True)

    def test_dataset_with_source_labelmap_undefined(self, test_dir: str, request: pytest.FixtureRequest):
        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(2, 3, 4, 5, label=0, id=1),
                        Bbox(1, 2, 3, 4, label=1, id=2),
                    ],
                ),
            ],
            categories={AnnotationType.label: LabelCategories.from_iterable(["Label_1", "label_2"])},
        )

        dst_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(
                            2,
                            3,
                            4,
                            5,
                            label=1,
                            id=0,
                            group=0,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=2,
                            id=1,
                            group=1,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                    ],
                )
            ],
            categories={AnnotationType.label: LabelCategories.from_iterable(["background", "Label_1", "label_2"])},
        )

        VocDetectionExporter.convert(
            src_dataset,
            save_dir=test_dir,
            save_media=True,
        )
        imported_dataset = Dataset.import_from(test_dir, VocDetectionImporter.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, dst_dataset, imported_dataset, require_media=True)

    def test_dataset_with_source_labelmap_defined(self, test_dir: str, request: pytest.FixtureRequest):
        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(2, 3, 4, 5, label=0, id=1),
                        Bbox(1, 2, 3, 4, label=2, id=2),
                    ],
                )
            ],
            categories={AnnotationType.label: LabelCategories.from_iterable(["label_1", "background", "label_2"])},
        )

        dst_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(
                            2,
                            3,
                            4,
                            5,
                            label=1,
                            id=0,
                            group=0,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=2,
                            id=1,
                            group=1,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                    ],
                )
            ],
            categories={AnnotationType.label: LabelCategories.from_iterable(["background", "label_1", "label_2"])},
        )

        VocDetectionExporter.convert(
            src_dataset,
            save_dir=test_dir,
            save_media=True,
        )
        imported_dataset = Dataset.import_from(test_dir, VocDetectionImporter.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, dst_dataset, imported_dataset, require_media=True)

    def test_can_inplace_save_writes_only_updated_data_with_transforms(
        self, test_dir: str, request: pytest.FixtureRequest
    ):
        dst_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "3",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((2, 3, 3))),
                    annotations=[
                        Bbox(
                            0,
                            1,
                            0,
                            0,
                            label=3,
                            id=0,
                            group=0,
                        )
                    ],
                ),
                DatasetItem(
                    "4",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((2, 4, 3))),
                    annotations=[
                        Bbox(
                            1,
                            0,
                            0,
                            0,
                            label=3,
                            id=0,
                            group=0,
                        ),
                        Mask(np.ones((2, 2)), label=1, group=0),
                    ],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "1",
                    subset="a",
                    media=Image.from_numpy(data=np.ones((2, 1, 3))),
                    annotations=[Bbox(0, 0, 0, 1, label=1)],
                ),
                DatasetItem(
                    "2",
                    subset="b",
                    media=Image.from_numpy(data=np.ones((2, 2, 3))),
                    annotations=[
                        Bbox(0, 0, 1, 0, label=2),
                        Mask(np.ones((2, 2)), label=1),
                    ],
                ),
                DatasetItem(
                    "3",
                    subset="b",
                    media=Image.from_numpy(data=np.ones((2, 3, 3))),
                    annotations=[Bbox(0, 1, 0, 0, label=3)],
                ),
                DatasetItem(
                    "4",
                    subset="c",
                    media=Image.from_numpy(data=np.ones((2, 4, 3))),
                    annotations=[Bbox(1, 0, 0, 0, label=3), Mask(np.ones((2, 2)), label=1)],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        src_dataset.export(test_dir, "voc_instance_segmentation", save_media=True)

        src_dataset.filter("/item[id >= 3]")
        src_dataset.transform("random_split", splits=(("train", 0.5), ("test", 0.5)), seed=42)
        src_dataset.save(save_media=True)

        assert {"3.xml", "4.xml"} == set(os.listdir(osp.join(test_dir, "Annotations")))
        assert {"3.jpg", "4.jpg"} == set(os.listdir(osp.join(test_dir, "JPEGImages")))
        assert {"4.png"} == set(os.listdir(osp.join(test_dir, "SegmentationClass")))
        assert {"4.png"} == set(os.listdir(osp.join(test_dir, "SegmentationObject")))
        assert {"train.txt", "test.txt"} == set(os.listdir(osp.join(test_dir, "ImageSets", "Main")))

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, dst_dataset, src_dataset, require_media=True)

    def test_can_inplace_save_writes_only_updated_data_with_direct_changes(
        self, test_dir: str, request: pytest.FixtureRequest
    ):
        dst_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((1, 2, 3))),
                    annotations=[
                        # Bbox(0, 0, 0, 0, label=1) # won't find removed anns
                    ],
                ),
                DatasetItem(
                    2,
                    subset="b",
                    media=Image.from_numpy(data=np.ones((3, 2, 3))),
                    annotations=[
                        Bbox(
                            0,
                            0,
                            0,
                            0,
                            label=3,
                            id=0,
                            group=0,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        )
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["background", "a", "b", "c", "d"]),
            },
        )

        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    subset="a",
                    media=Image.from_numpy(data=np.ones((1, 2, 3))),
                    annotations=[Bbox(0, 0, 0, 0, label=1)],
                ),
                DatasetItem(2, subset="b", annotations=[Bbox(0, 0, 0, 0, label=2)]),
                DatasetItem(
                    3,
                    subset="c",
                    media=Image.from_numpy(data=np.ones((2, 2, 3))),
                    annotations=[Bbox(0, 0, 0, 0, label=3), Mask(np.ones((2, 2)), label=1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["background", "a", "b", "c", "d"]),
                AnnotationType.mask: MaskCategories(colormap=VOC.generate_colormap(5)),
            },
        )

        src_dataset.export(test_dir, "voc_detection", save_media=True)
        os.unlink(osp.join(test_dir, "Annotations", "1.xml"))
        os.unlink(osp.join(test_dir, "Annotations", "2.xml"))
        os.unlink(osp.join(test_dir, "Annotations", "3.xml"))

        src_dataset.put(
            DatasetItem(
                2,
                subset="b",
                media=Image.from_numpy(data=np.ones((3, 2, 3))),
                annotations=[Bbox(0, 0, 0, 0, label=3)],
            )
        )
        src_dataset.remove(3, "c")
        src_dataset.save(save_media=True)

        assert {"2.xml"} == set(os.listdir(osp.join(test_dir, "Annotations")))
        assert {"1.jpg", "2.jpg"} == set(os.listdir(osp.join(test_dir, "JPEGImages")))
        assert {"a.txt", "b.txt"} == set(os.listdir(osp.join(test_dir, "ImageSets", "Main")))

        imported_dataset = Dataset.import_from(test_dir, "voc_detection")

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, dst_dataset, imported_dataset, require_media=True)

    def test_can_save_dataset_with_no_data_images(self, test_dir: str, request: pytest.FixtureRequest):
        src_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="frame1",
                    subset="test",
                    media=Image.from_file(path="frame1.jpg"),
                    annotations=[
                        Bbox(
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            attributes={
                                "difficult": False,
                                "truncated": False,
                                "occluded": False,
                            },
                            id=0,
                            label=0,
                            group=0,
                        )
                    ],
                )
            ],
            categories=VOC.make_voc_categories(task=VocTask.voc_detection),
        )

        VocDetectionExporter.convert(
            src_dataset,
            save_dir=test_dir,
            save_media=True,
        )
        imported_dataset = Dataset.import_from(test_dir, VocDetectionImporter.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, src_dataset, imported_dataset, require_media=True)

    @pytest.mark.parametrize(
        "fxt_task,fxt_importer,fxt_exporter",
        [
            (VocTask.voc_classification, VocClassificationImporter, VocClassificationExporter),
            (VocTask.voc_detection, VocDetectionImporter, VocDetectionExporter),
            (VocTask.voc_segmentation, VocSegmentationImporter, VocSegmentationExporter),
            (
                VocTask.voc_instance_segmentation,
                VocInstanceSegmentationImporter,
                VocInstanceSegmentationExporter,
            ),
            (VocTask.voc_layout, VocLayoutImporter, VocLayoutExporter),
            (VocTask.voc_action, VocActionImporter, VocActionExporter),
        ],
    )
    def test_can_save_dataset_with_only_images(
        self, fxt_task, fxt_importer, fxt_exporter, test_dir: str, request: pytest.FixtureRequest
    ):
        def dataset_with_images(task):
            return Dataset.from_iterable(
                [
                    DatasetItem(id=1, subset="a", media=Image.from_numpy(data=np.ones([4, 5, 3]))),
                    DatasetItem(id=2, subset="a", media=Image.from_numpy(data=np.ones([4, 5, 3]))),
                    DatasetItem(id=3, subset="b", media=Image.from_numpy(data=np.ones([2, 6, 3]))),
                ],
                categories=VOC.make_voc_categories(task=task),
            )

        src_dataset = dataset_with_images(fxt_task)

        fxt_exporter.convert(src_dataset, save_dir=test_dir, save_media=True, label_map=fxt_importer.NAME)
        imported_dataset = Dataset.import_from(test_dir, fxt_importer.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, src_dataset, imported_dataset, require_media=True)

    @pytest.mark.parametrize(
        "fxt_task,fxt_importer,fxt_exporter",
        [
            (VocTask.voc_classification, VocClassificationImporter, VocClassificationExporter),
            (VocTask.voc_detection, VocDetectionImporter, VocDetectionExporter),
            (VocTask.voc_segmentation, VocSegmentationImporter, VocSegmentationExporter),
            (
                VocTask.voc_instance_segmentation,
                VocInstanceSegmentationImporter,
                VocInstanceSegmentationExporter,
            ),
            (VocTask.voc_layout, VocLayoutImporter, VocLayoutExporter),
            (VocTask.voc_action, VocActionImporter, VocActionExporter),
        ],
    )
    def test_can_save_dataset_with_no_subsets(
        self, fxt_task, fxt_importer, fxt_exporter, test_dir: str, request: pytest.FixtureRequest
    ):
        def dataset_with_no_subsets(task):
            return Dataset.from_iterable(
                [
                    DatasetItem(id=1),
                    DatasetItem(id=2),
                ],
                categories=VOC.make_voc_categories(task=task),
            )

        src_dataset = dataset_with_no_subsets(fxt_task)

        fxt_exporter.convert(src_dataset, save_dir=test_dir, save_media=True, label_map=fxt_importer.NAME)
        imported_dataset = Dataset.import_from(test_dir, fxt_importer.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, src_dataset, imported_dataset, require_media=True)

    @pytest.mark.parametrize(
        "fxt_task,fxt_importer,fxt_exporter",
        [
            (VocTask.voc_classification, VocClassificationImporter, VocClassificationExporter),
            (VocTask.voc_detection, VocDetectionImporter, VocDetectionExporter),
            (VocTask.voc_segmentation, VocSegmentationImporter, VocSegmentationExporter),
            (
                VocTask.voc_instance_segmentation,
                VocInstanceSegmentationImporter,
                VocInstanceSegmentationExporter,
            ),
            (VocTask.voc_layout, VocLayoutImporter, VocLayoutExporter),
            (VocTask.voc_action, VocActionImporter, VocActionExporter),
        ],
    )
    def test_can_save_dataset_with_spaces_in_filename(
        self, fxt_task, fxt_importer, fxt_exporter, test_dir: str, request: pytest.FixtureRequest
    ):
        def dataset_with_cyrillic_and_spaces_in_filename(task):
            return Dataset.from_iterable(
                [
                    DatasetItem(id="кириллица с пробелом 1"),
                    DatasetItem(
                        id="кириллица с пробелом 2",
                        media=Image.from_numpy(data=np.ones([4, 5, 3])),
                    ),
                ],
                categories=VOC.make_voc_categories(task=task),
            )

        src_dataset = dataset_with_cyrillic_and_spaces_in_filename(fxt_task)

        fxt_exporter.convert(src_dataset, save_dir=test_dir, save_media=True, label_map=fxt_importer.NAME)
        imported_dataset = Dataset.import_from(test_dir, fxt_importer.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, src_dataset, imported_dataset, require_media=True)

    def test_background_masks_dont_introduce_instances_but_cover_others(self, test_dir: str):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    media=Image.from_numpy(data=np.zeros((4, 1, 1))),
                    annotations=[
                        Mask([1, 1, 1, 1], label=1, attributes={"z_order": 1}),
                        Mask([0, 0, 1, 1], label=2, attributes={"z_order": 2}),
                        Mask([0, 0, 1, 1], label=0, attributes={"z_order": 3}),
                    ],
                )
            ],
            categories=["background", "a", "b"],
        )

        VocSegmentationExporter.convert(dataset, test_dir, apply_colormap=False)

        cls_mask = load_mask(osp.join(test_dir, "SegmentationClass", "1.png"))
        inst_mask = load_mask(osp.join(test_dir, "SegmentationObject", "1.png"))
        assert np.array_equal([0, 1], np.unique(cls_mask))
        assert np.array_equal([0, 1], np.unique(inst_mask))

    @pytest.mark.parametrize(
        "fxt_task,fxt_importer,fxt_exporter",
        [
            (VocTask.voc_classification, VocClassificationImporter, VocClassificationExporter),
            (VocTask.voc_detection, VocDetectionImporter, VocDetectionExporter),
            (VocTask.voc_segmentation, VocSegmentationImporter, VocSegmentationExporter),
            (
                VocTask.voc_instance_segmentation,
                VocInstanceSegmentationImporter,
                VocInstanceSegmentationExporter,
            ),
            (VocTask.voc_layout, VocLayoutImporter, VocLayoutExporter),
            (VocTask.voc_action, VocActionImporter, VocActionExporter),
        ],
    )
    def test_can_save_dataset_with_image_info(
        self, fxt_task, fxt_importer, fxt_exporter, test_dir: str, request: pytest.FixtureRequest
    ):
        def dataset_with_image_info(task):
            return Dataset.from_iterable(
                [
                    DatasetItem(id=1, media=Image.from_file(path="1.jpg", size=(10, 15))),
                ],
                categories=VOC.make_voc_categories(task=task),
            )

        src_dataset = dataset_with_image_info(fxt_task)

        fxt_exporter.convert(src_dataset, save_dir=test_dir, save_media=True, label_map=fxt_importer.NAME)
        imported_dataset = Dataset.import_from(test_dir, fxt_importer.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, src_dataset, imported_dataset, require_media=True)

    @pytest.mark.parametrize(
        "fxt_task,fxt_importer,fxt_exporter",
        [
            (VocTask.voc_classification, VocClassificationImporter, VocClassificationExporter),
            (VocTask.voc_detection, VocDetectionImporter, VocDetectionExporter),
            (VocTask.voc_segmentation, VocSegmentationImporter, VocSegmentationExporter),
            (
                VocTask.voc_instance_segmentation,
                VocInstanceSegmentationImporter,
                VocInstanceSegmentationExporter,
            ),
            (VocTask.voc_layout, VocLayoutImporter, VocLayoutExporter),
            (VocTask.voc_action, VocActionImporter, VocActionExporter),
        ],
    )
    def test_can_save_and_load_image_with_arbitrary_extension(
        self, fxt_task, fxt_importer, fxt_exporter, test_dir: str, request: pytest.FixtureRequest
    ):
        def dataset_with_arbitrary_extension(task):
            return Dataset.from_iterable(
                [
                    DatasetItem(id="q/1", media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG")),
                    DatasetItem(id="a/b/c/2", media=Image.from_numpy(data=np.zeros((3, 4, 3)))),
                ],
                categories=VOC.make_voc_categories(task=task),
            )

        src_dataset = dataset_with_arbitrary_extension(fxt_task)

        fxt_exporter.convert(src_dataset, save_dir=test_dir, save_media=True, label_map=fxt_importer.NAME)
        imported_dataset = Dataset.import_from(test_dir, fxt_importer.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, src_dataset, imported_dataset, require_media=True)

    @pytest.mark.parametrize(
        "fxt_task,fxt_importer,fxt_exporter",
        [
            (VocTask.voc_classification, VocClassificationImporter, VocClassificationExporter),
            (VocTask.voc_detection, VocDetectionImporter, VocDetectionExporter),
            (VocTask.voc_segmentation, VocSegmentationImporter, VocSegmentationExporter),
            (
                VocTask.voc_instance_segmentation,
                VocInstanceSegmentationImporter,
                VocInstanceSegmentationExporter,
            ),
            (VocTask.voc_layout, VocLayoutImporter, VocLayoutExporter),
            (VocTask.voc_action, VocActionImporter, VocActionExporter),
        ],
    )
    def test_save_dataset_with_relative_paths(
        self, fxt_task, fxt_importer, fxt_exporter, test_dir: str, request: pytest.FixtureRequest
    ):
        def dataset_with_relative_paths(task):
            return Dataset.from_iterable(
                [
                    DatasetItem(id="1", media=Image.from_numpy(data=np.ones((4, 2, 3)))),
                    DatasetItem(id="subdir1/1", media=Image.from_numpy(data=np.ones((2, 6, 3)))),
                    DatasetItem(id="subdir2/1", media=Image.from_numpy(data=np.ones((5, 4, 3)))),
                ],
                categories=VOC.make_voc_categories(task=task),
            )

        src_dataset = dataset_with_relative_paths(fxt_task)

        fxt_exporter.convert(src_dataset, save_dir=test_dir, save_media=True, label_map=fxt_importer.NAME)
        imported_dataset = Dataset.import_from(test_dir, fxt_importer.NAME)

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, src_dataset, imported_dataset, require_media=True)


class VocFormatErrorTest:
    # ?xml... must be in the file beginning
    XML_ANNOTATION_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<annotation>
<filename>a.jpg</filename>
<size><width>20</width><height>10</height><depth>3</depth></size>
<object>
    <name>person</name>
    <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox>
    <difficult>1</difficult>
    <truncated>1</truncated>
    <occluded>1</occluded>
    <point><x>1</x><y>1</y></point>
    <attributes><attribute><name>a</name><value>42</value></attribute></attributes>
    <actions><jumping>1</jumping></actions>
    <part>
        <name>head</name>
        <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox>
    </part>
</object>
</annotation>
    """

    @classmethod
    def _write_xml_dataset(cls, root_dir, format_dir="Main", mangle_xml=None):
        subset_file = osp.join(root_dir, "ImageSets", format_dir, "test.txt")
        if not osp.exists(subset_file):
            os.makedirs(osp.dirname(subset_file))
        with open(subset_file, "w") as f:
            f.write("a\n" if format_dir != "Layout" else "a 0\n")

        ann_file = osp.join(root_dir, "Annotations", "a.xml")
        if not osp.exists(ann_file):
            os.makedirs(osp.dirname(ann_file))
        with open(ann_file, "wb") as f:
            xml = ElementTree.fromstring(cls.XML_ANNOTATION_TEMPLATE.encode())
            if mangle_xml:
                mangle_xml(xml)
            f.write(ElementTree.tostring(xml))

    @pytest.mark.parametrize(
        "fxt_format,fxt_format_dir",
        [
            ("voc_detection", "Main"),
            ("voc_layout", "Layout"),
            ("voc_action", "Action"),
        ],
    )
    def test_can_parse_xml_without_errors(self, fxt_format: str, fxt_format_dir: str, test_dir: str):
        self._write_xml_dataset(test_dir, format_dir=fxt_format_dir)

        dataset = Dataset.import_from(test_dir, fxt_format)
        assert len(dataset) == 1

    def test_can_report_invalid_label_in_xml(self, test_dir: str):
        def mangle_xml(xml: ElementTree.ElementBase):
            xml.find("object/name").text = "test"

        self._write_xml_dataset(test_dir, format_dir="Main", mangle_xml=mangle_xml)

        with pytest.raises(AnnotationImportError) as err_info:
            Dataset.import_from(test_dir, format="voc_detection").init_cache()
        assert str(err_info.value) == "Failed to import item ('a', 'test') annotation: Undeclared label 'test'"

    @pytest.mark.parametrize(
        "fxt_format,fxt_format_dir",
        [
            ("voc_detection", "Main"),
            ("voc_layout", "Layout"),
            ("voc_action", "Action"),
        ],
    )
    @pytest.mark.parametrize(
        "fxt_key,fxt_value",
        [
            ("size/width", "a"),
            ("size/height", "a"),
            ("object/bndbox/xmin", "a"),
            ("object/bndbox/ymin", "a"),
            ("object/bndbox/xmax", "a"),
            ("object/bndbox/ymax", "a"),
            ("object/occluded", "a"),
            ("object/difficult", "a"),
            ("object/truncated", "a"),
            ("object/point/x", "a"),
            ("object/point/y", "a"),
        ],
    )
    def test_can_report_invalid_field_in_xml(
        self, fxt_format: str, fxt_format_dir: str, fxt_key: str, fxt_value: str, test_dir: str
    ):
        def mangle_xml(xml: ElementTree.ElementBase):
            xml.find(fxt_key).text = fxt_value

        self._write_xml_dataset(test_dir, format_dir=fxt_format_dir, mangle_xml=mangle_xml)

        with pytest.raises(ItemImportError) as err_info:
            Dataset.import_from(test_dir, format=fxt_format).init_cache()
        assert "Invalid annotation field" in str(err_info.value)

    @pytest.mark.parametrize(
        "fxt_format,fxt_format_dir",
        [
            ("voc_detection", "Main"),
            ("voc_layout", "Layout"),
            ("voc_action", "Action"),
        ],
    )
    @pytest.mark.parametrize(
        "fxt_key",
        [
            "object/name",
            "object/bndbox",
            "object/bndbox/xmin",
            "object/bndbox/ymin",
            "object/bndbox/xmax",
            "object/bndbox/ymax",
            "object/point/x",
            "object/point/y",
            "object/attributes/attribute/name",
            "object/attributes/attribute/value",
        ],
    )
    def test_can_report_missing_field_in_xml(self, fxt_format: str, fxt_format_dir: str, fxt_key: str, test_dir: str):
        def mangle_xml(xml: ElementTree.ElementBase):
            for elem in xml.findall(fxt_key):
                elem.getparent().remove(elem)

        self._write_xml_dataset(test_dir, format_dir=fxt_format_dir, mangle_xml=mangle_xml)

        with pytest.raises(AnnotationImportError) as err_info:
            Dataset.import_from(test_dir, format=fxt_format).init_cache()
        assert "Missing annotation field" in str(err_info.value)

    def test_can_parse_classification_without_errors(self, test_dir: str, request: pytest.FixtureRequest):
        subset_file = osp.join(test_dir, "ImageSets", "Main", "test.txt")
        os.makedirs(osp.dirname(subset_file))
        with open(subset_file, "w") as f:
            f.write("a\n")
            f.write("b\n")
            f.write("c\n")

        ann_file = osp.join(test_dir, "ImageSets", "Main", "cat_test.txt")
        with open(ann_file, "w") as f:
            f.write("a -1\n")
            f.write("b 0\n")
            f.write("c 1\n")

        parsed = Dataset.import_from(test_dir, format="voc_classification")

        expected = Dataset.from_iterable(
            [
                DatasetItem("a", subset="test"),
                DatasetItem("b", subset="test"),
                DatasetItem("c", subset="test", annotations=[Label(VOC.VocLabel.cat.value)]),
            ],
            categories=VOC.make_voc_categories(task=VocTask.voc_classification),
        )

        helper_tc = request.getfixturevalue("helper_tc")
        compare_datasets(helper_tc, expected, parsed)

    def test_can_report_missing_field_in_classification(self, test_dir: str):
        subset_file = osp.join(test_dir, "ImageSets", "Main", "test.txt")
        os.makedirs(osp.dirname(subset_file))
        with open(subset_file, "w") as f:
            f.write("a\n")

        ann_file = osp.join(test_dir, "ImageSets", "Main", "cat_test.txt")
        with open(ann_file, "w") as f:
            f.write("a\n")

        with pytest.raises(InvalidAnnotationError) as err_info:
            Dataset.import_from(test_dir, format="voc_classification").init_cache()
        assert str(err_info.value) == "cat_test.txt:1: invalid number of fields in line, expected 2"

    def test_can_report_invalid_annotation_value_in_classification(self, test_dir: str):
        subset_file = osp.join(test_dir, "ImageSets", "Main", "test.txt")
        os.makedirs(osp.dirname(subset_file))
        with open(subset_file, "w") as f:
            f.write("a\n")

        ann_file = osp.join(test_dir, "ImageSets", "Main", "cat_test.txt")
        with open(ann_file, "w") as f:
            f.write("a 3\n")

        with pytest.raises(InvalidAnnotationError) as err_info:
            Dataset.import_from(test_dir, format="voc_classification").init_cache()
        assert str(err_info.value) == "cat_test.txt:1: unexpected class existence value '3', expected -1, 0 or 1"

    def test_can_report_invalid_label_in_segmentation_cls_mask(self, test_dir: str):
        subset_file = osp.join(test_dir, "ImageSets", "Segmentation", "test.txt")
        os.makedirs(osp.dirname(subset_file))
        with open(subset_file, "w") as f:
            f.write("a\n")

        ann_file = osp.join(test_dir, "SegmentationClass", "a.png")
        os.makedirs(osp.dirname(ann_file))
        save_image(ann_file, np.array([[30]], dtype=np.uint8))

        with pytest.raises(AnnotationImportError) as err_info:
            Dataset.import_from(test_dir, format="voc_segmentation").init_cache()
        assert str(err_info.value) == "Failed to import item ('a', 'test') annotation: Undeclared label '30'"

    def test_can_report_invalid_label_in_segmentation_both_masks(self, test_dir: str):
        subset_file = osp.join(test_dir, "ImageSets", "Segmentation", "test.txt")
        os.makedirs(osp.dirname(subset_file))
        with open(subset_file, "w") as f:
            f.write("a\n")

        cls_file = osp.join(test_dir, "SegmentationClass", "a.png")
        os.makedirs(osp.dirname(cls_file))
        save_image(cls_file, np.array([[30]], dtype=np.uint8))

        inst_file = osp.join(test_dir, "SegmentationObject", "a.png")
        os.makedirs(osp.dirname(inst_file))
        save_image(inst_file, np.array([[1]], dtype=np.uint8))

        with pytest.raises(AnnotationImportError) as err_info:
            Dataset.import_from(test_dir, format="voc_segmentation").init_cache()
        assert str(err_info.value) == "Failed to import item ('a', 'test') annotation: Undeclared label '30'"

    def test_can_report_invalid_quotes_in_lists_of_layout_task(self, test_dir: str):
        subset_file = osp.join(test_dir, "ImageSets", "Layout", "test.txt")
        os.makedirs(osp.dirname(subset_file))
        with open(subset_file, "w") as f:
            f.write('"qwe 1\n')

        with pytest.raises(DatasetImportError) as err_info:
            Dataset.import_from(test_dir, format="voc_layout").init_cache()
        assert "Failed to import dataset" in str(err_info.value)
