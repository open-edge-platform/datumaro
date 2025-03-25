# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging
import os
import os.path as osp
import pickle  # nosec B403
import shutil
from copy import deepcopy
from functools import partial
from io import StringIO
from unittest import TestCase, skip

import numpy as np
import pytest

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    Ellipse,
    Label,
    LabelCategories,
    Mask,
    Points,
    PointsCategories,
    Polygon,
)
from datumaro.components.dataset import Dataset, StreamDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.errors import (
    AnnotationImportError,
    DatasetImportError,
    InvalidAnnotationError,
    InvalidFieldTypeError,
    ItemImportError,
    MissingFieldError,
    UndeclaredLabelError,
)
from datumaro.components.media import Image
from datumaro.plugins.data_formats.coco.base import CocoInstancesBase
from datumaro.plugins.data_formats.coco.exporter import (
    CocoCaptionsExporter,
    CocoExporter,
    CocoImageInfoExporter,
    CocoInstancesExporter,
    CocoLabelsExporter,
    CocoPanopticExporter,
    CocoPersonKeypointsExporter,
    CocoStuffExporter,
)
from datumaro.plugins.data_formats.coco.format import CocoPath
from datumaro.util import dump_json_file, parse_json_file

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import (
    TestDir,
    check_is_stream,
    check_save_and_load,
    compare_datasets,
    compare_datasets_strict,
)

DUMMY_DATASET_DIR = get_test_asset_path("coco_dataset")


class CocoImporterTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "format, path",
        [
            ("coco_captions", osp.join(DUMMY_DATASET_DIR, "coco_captions")),
            ("coco_image_info", osp.join(DUMMY_DATASET_DIR, "coco_image_info")),
            ("coco_instances", osp.join(DUMMY_DATASET_DIR, "coco_instances")),
            ("coco_labels", osp.join(DUMMY_DATASET_DIR, "coco_labels")),
            ("coco_person_keypoints", osp.join(DUMMY_DATASET_DIR, "coco_person_keypoints")),
            ("coco_stuff", osp.join(DUMMY_DATASET_DIR, "coco_stuff")),
        ],
    )
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_export_and_import_back(self, format, path, stream, test_dir, helper_tc):
        dataset = Dataset.import_from(path, format)
        dataset.export(test_dir, format)
        back_dataset = (
            Dataset.import_from(test_dir) if not stream else StreamDataset.import_from(test_dir)
        )
        check_is_stream(back_dataset, stream)
        compare_datasets(helper_tc, dataset, back_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_from_any_cwd(self, stream):
        class ChangeCWD:
            def __init__(self, path):
                self.cwd = os.getcwd()
                self.cd_path = path

            def __enter__(self):
                os.chdir(self.cd_path)

            def __exit__(self, exc_type, exc_val, exc_tb):
                os.chdir(self.cwd)

        format_paths = [
            ("coco_captions", osp.join(DUMMY_DATASET_DIR, "coco_captions")),
            ("coco_image_info", osp.join(DUMMY_DATASET_DIR, "coco_image_info")),
            ("coco_instances", osp.join(DUMMY_DATASET_DIR, "coco_instances")),
            ("coco_labels", osp.join(DUMMY_DATASET_DIR, "coco_labels")),
            ("coco_person_keypoints", osp.join(DUMMY_DATASET_DIR, "coco_person_keypoints")),
            ("coco_stuff", osp.join(DUMMY_DATASET_DIR, "coco_stuff")),
        ]
        for format, path in format_paths:
            # absolute path import
            Dataset.import_from(path, format)
            for anno_file in os.listdir(os.path.join(path, CocoPath.ANNOTATIONS_DIR)):
                Dataset.import_from(os.path.join(path, CocoPath.ANNOTATIONS_DIR, anno_file), format)

            # relative path import
            for rel_path in [
                path,
                os.path.join(path, CocoPath.ANNOTATIONS_DIR),
                os.path.join(path, CocoPath.IMAGES_DIR),
                os.getcwd(),
            ]:
                with ChangeCWD(rel_path):
                    Dataset.import_from(osp.relpath(path, rel_path), format)
                    for anno_file in os.listdir(os.path.join(path, CocoPath.ANNOTATIONS_DIR)):
                        dataset_dir = osp.relpath(
                            os.path.join(path, CocoPath.ANNOTATIONS_DIR, anno_file), rel_path
                        )
                        dataset = (
                            Dataset.import_from(dataset_dir, format)
                            if not stream
                            else StreamDataset.import_from(dataset_dir, format)
                        )
                        check_is_stream(dataset, stream)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("format", ["coco", "coco_instances"])
    @pytest.mark.parametrize(
        "subset, path",
        [
            ("", osp.join(DUMMY_DATASET_DIR, "coco_instances")),
            (
                "train",
                osp.join(
                    DUMMY_DATASET_DIR, "coco_instances", "annotations", "instances_train.json"
                ),
            ),
            (
                "val",
                osp.join(DUMMY_DATASET_DIR, "coco_instances", "annotations", "instances_val.json"),
            ),
        ],
    )
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_instances(self, format, subset, path, stream, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Bbox(2, 2, 3, 1, label=1, group=1, id=1, attributes={"is_crowd": False})
                    ],
                ),
                DatasetItem(
                    id="b",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((10, 5, 3))),
                    attributes={"id": 40},
                    annotations=[
                        Polygon(
                            [0, 0, 1, 0, 1, 2, 0, 2],
                            label=0,
                            id=1,
                            group=1,
                            attributes={"is_crowd": False, "x": 1, "y": "hello"},
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            1.0,
                            2.0,
                            id=1,
                            attributes={"x": 1, "y": "hello", "is_crowd": False},
                            group=1,
                            label=0,
                            z_order=0,
                        ),
                        Mask(
                            np.array([[1, 1, 0, 0, 0]] * 10),
                            label=1,
                            id=2,
                            group=2,
                            attributes={"is_crowd": True},
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            1.0,
                            9.0,
                            id=2,
                            attributes={"is_crowd": True},
                            group=2,
                            label=1,
                            z_order=0,
                        ),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        expected = expected_dataset.get_subset(subset) if subset else expected_dataset

        dataset = (
            Dataset.import_from(path, format)
            if not stream
            else StreamDataset.import_from(path, format)
        )
        check_is_stream(dataset, stream)
        compare_datasets(helper_tc, expected, dataset, require_media=True)

    @skip(
        "COCO format is required to specify the task in annotation file "
        " for resolving ambiguity problem."
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_instances_with_any_annotation_filename(self, stream, test_dir, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="default",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Bbox(2, 2, 3, 1, label=1, group=1, id=1, attributes={"is_crowd": False})
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        format = "coco_instances"

        dataset_dir = osp.join(test_dir, "dataset")
        expected_dataset.export(dataset_dir, format, save_media=True)
        os.rename(
            osp.join(dataset_dir, "annotations", "instances_default.json"),
            osp.join(dataset_dir, "annotations", "aa_bbbb_cccc.json"),
        )

        imported_dataset = (
            Dataset.import_from(dataset_dir, format)
            if not stream
            else StreamDataset.import_from(dataset_dir, format)
        )

        check_is_stream(imported_dataset, stream)
        compare_datasets(helper_tc, expected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("stream", [True, False])
    def test_warning_users_with_zero_category_id(self, stream, test_dir):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="default",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Bbox(2, 2, 3, 1, label=1, group=1, id=1, attributes={"is_crowd": False})
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        class CaptureLogger:
            def __init__(self, logger=None, level=None):
                self._logger = logger if logger else logging.getLogger()
                self._level = level
                self._origin_level = None
                self._string_io = StringIO()

            def __enter__(self):
                self._logger.addHandler(logging.StreamHandler(self._string_io))
                if self._level is not None:
                    self._origin_level = self._logger.level
                    self._logger.setLevel(self._level)
                return self._string_io

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._logger.handlers.pop()
                if self._origin_level is not None:
                    self._logger.setLevel(self._origin_level)

        format = "coco_instances"

        dataset_dir = osp.join(test_dir, "dataset")
        expected_dataset.export(dataset_dir, format, save_media=True)

        # modify annotation file to have zero category id
        anno_file = osp.join(dataset_dir, "annotations", "instances_default.json")
        anno = parse_json_file(anno_file)
        anno["categories"][0]["id"] = 0
        anno["annotations"][0]["category_id"] = 0
        dump_json_file(anno_file, anno)

        with CaptureLogger(level=logging.WARNING) as strio:
            imported_dataset = (
                Dataset.import_from(dataset_dir, format)
                if not stream
                else StreamDataset.import_from(dataset_dir, format)
            )
            check_is_stream(imported_dataset, stream)
            assert "Category id of '0' is reserved for no class" in strio.getvalue()

        item = next(iter(imported_dataset))
        assert item.annotations[0].label is None

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_instances_with_original_cat_ids(self, stream, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Bbox(2, 2, 3, 1, label=2, group=1, id=1, attributes={"is_crowd": False})
                    ],
                ),
            ],
            categories=["class-0", "a", "b", "class-3", "c"],
        )

        dataset_dir = osp.join(
            DUMMY_DATASET_DIR, "coco_instances", "annotations", "instances_train.json"
        )
        format = "coco_instances"
        actual_dataset = (
            Dataset.import_from(dataset_dir, format, keep_original_category_ids=True)
            if not stream
            else StreamDataset.import_from(dataset_dir, format, keep_original_category_ids=True)
        )
        check_is_stream(actual_dataset, stream)
        compare_datasets(helper_tc, expected_dataset, actual_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("format", ["coco", "coco_captions"])
    @pytest.mark.parametrize(
        "subset, path",
        [
            ("", osp.join(DUMMY_DATASET_DIR, "coco_captions")),
            (
                "train",
                osp.join(DUMMY_DATASET_DIR, "coco_captions", "annotations", "captions_train.json"),
            ),
            (
                "val",
                osp.join(DUMMY_DATASET_DIR, "coco_captions", "annotations", "captions_val.json"),
            ),
        ],
    )
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_captions(self, format, subset, path, stream, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Caption("hello", id=1, group=1),
                    ],
                ),
                DatasetItem(
                    id="b",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((10, 5, 3))),
                    attributes={"id": 40},
                    annotations=[
                        Caption("world", id=1, group=1),
                        Caption("text", id=2, group=2),
                    ],
                ),
            ],
        )

        expected = expected_dataset.get_subset(subset) if subset else expected_dataset

        dataset = (
            Dataset.import_from(path, format)
            if not stream
            else StreamDataset.import_from(path, format)
        )
        check_is_stream(dataset, stream)
        compare_datasets(helper_tc, expected, dataset, require_media=True)

    @skip(
        "COCO format is required to specify the task in annotation file "
        " for resolving ambiguity problem."
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_captions_with_any_annotation_filename(self, stream, test_dir, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="default",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Caption("hello", id=1, group=1),
                    ],
                ),
            ],
        )

        format = "coco_captions"

        dataset_dir = osp.join(test_dir, "dataset")
        expected_dataset.export(dataset_dir, format, save_media=True)
        os.rename(
            osp.join(dataset_dir, "annotations", "captions_default.json"),
            osp.join(dataset_dir, "annotations", "aa_bbbb_cccc.json"),
        )

        imported_dataset = (
            Dataset.import_from(dataset_dir, format)
            if not stream
            else StreamDataset.import_from(dataset_dir, format)
        )
        check_is_stream(imported_dataset, stream)
        compare_datasets(helper_tc, expected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("format", ["coco", "coco_labels"])
    @pytest.mark.parametrize(
        "subset, path",
        [
            ("", osp.join(DUMMY_DATASET_DIR, "coco_labels")),
            (
                "train",
                osp.join(DUMMY_DATASET_DIR, "coco_labels", "annotations", "labels_train.json"),
            ),
            ("val", osp.join(DUMMY_DATASET_DIR, "coco_labels", "annotations", "labels_val.json")),
        ],
    )
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_labels(self, format, subset, path, stream, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Label(1, id=1, group=1),
                    ],
                ),
                DatasetItem(
                    id="b",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((10, 5, 3))),
                    attributes={"id": 40},
                    annotations=[
                        Label(0, id=1, group=1),
                        Label(1, id=2, group=2),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        expected = expected_dataset.get_subset(subset) if subset else expected_dataset

        dataset = (
            Dataset.import_from(path, format)
            if not stream
            else StreamDataset.import_from(path, format)
        )
        check_is_stream(dataset, stream)
        compare_datasets(helper_tc, expected, dataset, require_media=True)

    @skip(
        "COCO format is required to specify the task in annotation file "
        " for resolving ambiguity problem."
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_labels_with_any_annotation_filename(self, stream, test_dir, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="default",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Label(1, id=1, group=1),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        format = "coco_labels"

        dataset_dir = osp.join(test_dir, "dataset")
        expected_dataset.export(dataset_dir, format, save_media=True)
        os.rename(
            osp.join(dataset_dir, "annotations", "labels_default.json"),
            osp.join(dataset_dir, "annotations", "aa_bbbb_cccc.json"),
        )

        imported_dataset = (
            Dataset.import_from(dataset_dir, format)
            if not stream
            else StreamDataset.import_from(dataset_dir, format)
        )
        check_is_stream(imported_dataset, stream)
        compare_datasets(helper_tc, expected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("format", ["coco", "coco_person_keypoints"])
    @pytest.mark.parametrize(
        "subset, path",
        [
            ("", osp.join(DUMMY_DATASET_DIR, "coco_person_keypoints")),
            (
                "train",
                osp.join(
                    DUMMY_DATASET_DIR,
                    "coco_person_keypoints",
                    "annotations",
                    "person_keypoints_train.json",
                ),
            ),
            (
                "val",
                osp.join(
                    DUMMY_DATASET_DIR,
                    "coco_person_keypoints",
                    "annotations",
                    "person_keypoints_val.json",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_keypoints(self, format, subset, path, stream, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Points(
                            [0, 0, 0, 2, 4, 1],
                            [0, 1, 2],
                            label=1,
                            id=1,
                            group=1,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(2, 2, 3, 1, label=1, id=1, group=1, attributes={"is_crowd": False}),
                    ],
                ),
                DatasetItem(
                    id="b",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((10, 5, 3))),
                    attributes={"id": 40},
                    annotations=[
                        Points(
                            [1, 2, 3, 4, 2, 3],
                            label=0,
                            id=1,
                            group=1,
                            attributes={"is_crowd": False, "x": 1, "y": "hello"},
                        ),
                        Polygon(
                            [0, 0, 1, 0, 1, 2, 0, 2],
                            label=0,
                            id=1,
                            group=1,
                            attributes={"is_crowd": False, "x": 1, "y": "hello"},
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            1.0,
                            2.0,
                            id=1,
                            attributes={"x": 1, "y": "hello", "is_crowd": False},
                            group=1,
                            label=0,
                            z_order=0,
                        ),
                        Points(
                            [2, 4, 4, 4, 4, 2],
                            label=1,
                            id=2,
                            group=2,
                            attributes={"is_crowd": True},
                        ),
                        Mask(
                            np.array([[1, 1, 0, 0, 0]] * 10),
                            label=1,
                            id=2,
                            group=2,
                            attributes={"is_crowd": True},
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            1.0,
                            9.0,
                            id=2,
                            attributes={"is_crowd": True},
                            group=2,
                            label=1,
                            z_order=0,
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a", "b"]),
                AnnotationType.points: PointsCategories.from_iterable(
                    (i, None, [[0, 1], [1, 2]]) for i in range(2)
                ),
            },
        )

        expected = expected_dataset.get_subset(subset) if subset else expected_dataset

        dataset = (
            Dataset.import_from(path, format)
            if not stream
            else StreamDataset.import_from(path, format)
        )
        check_is_stream(dataset, stream)
        compare_datasets(helper_tc, expected, dataset, require_media=True)

    @skip(
        "COCO format is required to specify the task in annotation file "
        " for resolving ambiguity problem."
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_keypoints_with_any_annotation_filename(self, stream, test_dir, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="default",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Points(
                            [0, 0, 0, 2, 4, 1],
                            [0, 1, 2],
                            label=1,
                            id=1,
                            group=1,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(2, 2, 3, 1, label=1, id=1, group=1, attributes={"is_crowd": False}),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a", "b"]),
                AnnotationType.points: PointsCategories.from_iterable(
                    (i, None, [[0, 1], [1, 2]]) for i in range(2)
                ),
            },
        )

        format = "coco_person_keypoints"

        dataset_dir = osp.join(test_dir, "dataset")
        expected_dataset.export(dataset_dir, format, save_media=True)
        os.rename(
            osp.join(dataset_dir, "annotations", "person_keypoints_default.json"),
            osp.join(dataset_dir, "annotations", "aa_bbbb_cccc.json"),
        )

        imported_dataset = (
            Dataset.import_from(dataset_dir, format)
            if not stream
            else StreamDataset.import_from(dataset_dir, format)
        )
        check_is_stream(imported_dataset, stream)
        compare_datasets(helper_tc, expected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_keypoints_with_original_cat_ids(self, stream, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Points(
                            [0, 0, 0, 2, 4, 1],
                            [0, 1, 2],
                            label=2,
                            id=1,
                            group=1,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(2, 2, 3, 1, label=2, id=1, group=1, attributes={"is_crowd": False}),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["class-0", "a", "b"]),
                AnnotationType.points: PointsCategories.from_iterable(
                    [(i, None, [[0, 1], [1, 2]]) for i in range(1, 3)],
                ),
            },
        )
        path = osp.join(
            DUMMY_DATASET_DIR,
            "coco_person_keypoints",
            "annotations",
            "person_keypoints_train.json",
        )
        format = "coco_person_keypoints"

        actual_dataset = (
            Dataset.import_from(path, format, keep_original_category_ids=True)
            if not stream
            else StreamDataset.import_from(path, format, keep_original_category_ids=True)
        )
        check_is_stream(actual_dataset, stream)

        compare_datasets(helper_tc, expected_dataset, actual_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("format", ["coco", "coco_image_info"])
    @pytest.mark.parametrize(
        "subset, path",
        [
            ("", osp.join(DUMMY_DATASET_DIR, "coco_image_info")),
            (
                "train",
                osp.join(
                    DUMMY_DATASET_DIR, "coco_image_info", "annotations", "image_info_train.json"
                ),
            ),
            (
                "val",
                osp.join(
                    DUMMY_DATASET_DIR, "coco_image_info", "annotations", "image_info_val.json"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_image_info(self, stream, format, subset, path, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                ),
                DatasetItem(
                    id="b",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((10, 5, 3))),
                    attributes={"id": 40},
                ),
            ],
        )

        expected = expected_dataset.get_subset(subset) if subset else expected_dataset
        dataset = (
            Dataset.import_from(path, format)
            if not stream
            else StreamDataset.import_from(path, format)
        )
        check_is_stream(dataset, stream)
        compare_datasets(helper_tc, expected, dataset, require_media=True)

    @skip(
        "COCO format is required to specify the task in annotation file "
        " for resolving ambiguity problem."
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_image_info_with_any_annotation_filename(self, stream, test_dir, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="default",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                ),
            ],
        )

        format = "coco_image_info"

        dataset_dir = osp.join(test_dir, "dataset")
        expected_dataset.export(dataset_dir, format, save_media=True)
        os.rename(
            osp.join(dataset_dir, "annotations", "image_info_default.json"),
            osp.join(dataset_dir, "annotations", "aa_bbbb_cccc.json"),
        )

        imported_dataset = (
            Dataset.import_from(dataset_dir, format)
            if not stream
            else StreamDataset.import_from(dataset_dir, format)
        )
        check_is_stream(imported_dataset, stream)

        compare_datasets(helper_tc, expected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("format", ["coco", "coco_panoptic"])
    @pytest.mark.parametrize(
        "subset, path",
        [
            ("", osp.join(DUMMY_DATASET_DIR, "coco_panoptic")),
            (
                "train",
                osp.join(DUMMY_DATASET_DIR, "coco_panoptic", "annotations", "panoptic_train.json"),
            ),
            (
                "val",
                osp.join(DUMMY_DATASET_DIR, "coco_panoptic", "annotations", "panoptic_val.json"),
            ),
        ],
    )
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_panoptic(self, format, subset, path, stream, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Mask(
                            np.ones((5, 5)),
                            label=0,
                            id=460551,
                            group=460551,
                            attributes={"is_crowd": False},
                        ),
                    ],
                ),
                DatasetItem(
                    id="b",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((10, 5, 3))),
                    attributes={"id": 40},
                    annotations=[
                        Mask(
                            np.array([[1, 1, 0, 0, 0]] * 10),
                            label=0,
                            id=7,
                            group=7,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            np.array([[0, 0, 1, 1, 0]] * 10),
                            label=1,
                            id=20,
                            group=20,
                            attributes={"is_crowd": True},
                        ),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        expected = expected_dataset.get_subset(subset) if subset else expected_dataset
        dataset = (
            Dataset.import_from(path, format)
            if not stream
            else StreamDataset.import_from(path, format)
        )
        compare_datasets(helper_tc, expected, dataset, require_media=True)

    @skip(
        "COCO format is required to specify the task in annotation file "
        " for resolving ambiguity problem."
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_panoptic_with_any_annotation_filename(self, stream, test_dir, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="default",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Mask(
                            np.ones((5, 5)),
                            label=0,
                            id=460551,
                            group=460551,
                            attributes={"is_crowd": False},
                        ),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        format = "coco_panoptic"

        dataset_dir = osp.join(test_dir, "dataset")
        expected_dataset.export(dataset_dir, format, save_media=True)
        os.rename(
            osp.join(dataset_dir, "annotations", "panoptic_default"),
            osp.join(dataset_dir, "annotations", "aa_bbbb_cccc"),
        )
        os.rename(
            osp.join(dataset_dir, "annotations", "panoptic_default.json"),
            osp.join(dataset_dir, "annotations", "aa_bbbb_cccc.json"),
        )

        imported_dataset = (
            Dataset.import_from(dataset_dir, format)
            if not stream
            else StreamDataset.import_from(dataset_dir, format)
        )
        check_is_stream(imported_dataset, stream)
        compare_datasets(helper_tc, expected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_panoptic_with_original_cat_ids(self, stream, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Mask(
                            np.ones((5, 5)),
                            label=1,
                            id=460551,
                            group=460551,
                            attributes={"is_crowd": False},
                        ),
                    ],
                ),
            ],
            categories=["class-0", "a", "b"],
        )
        dataset_dir = osp.join(
            DUMMY_DATASET_DIR, "coco_panoptic", "annotations", "panoptic_train.json"
        )
        format = "coco_panoptic"
        actual_dataset = (
            Dataset.import_from(dataset_dir, format, keep_original_category_ids=True)
            if not stream
            else StreamDataset.import_from(dataset_dir, format, keep_original_category_ids=True)
        )
        check_is_stream(actual_dataset, stream)

        compare_datasets(helper_tc, expected_dataset, actual_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("format", ["coco", "coco_stuff"])
    @pytest.mark.parametrize(
        "subset, path",
        [
            ("", osp.join(DUMMY_DATASET_DIR, "coco_stuff")),
            ("train", osp.join(DUMMY_DATASET_DIR, "coco_stuff", "annotations", "stuff_train.json")),
            ("val", osp.join(DUMMY_DATASET_DIR, "coco_stuff", "annotations", "stuff_val.json")),
        ],
    )
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_stuff(self, format, subset, path, stream, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Mask(
                            np.array([[0, 0, 1, 1, 0, 1, 1, 0, 0, 0]] * 5),
                            label=0,
                            id=7,
                            group=7,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(
                            2.0,
                            0.0,
                            4.0,
                            4.0,
                            id=7,
                            attributes={"is_crowd": True},
                            group=7,
                            label=0,
                            z_order=0,
                        ),
                    ],
                ),
                DatasetItem(
                    id="b",
                    subset="val",
                    media=Image.from_numpy(data=np.ones((10, 5, 3))),
                    attributes={"id": 40},
                    annotations=[
                        Mask(
                            np.array([[1, 1, 0, 0, 0]] * 10),
                            label=1,
                            id=2,
                            group=2,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            1.0,
                            9.0,
                            id=2,
                            attributes={"is_crowd": True},
                            group=2,
                            label=1,
                            z_order=0,
                        ),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        expected = expected_dataset.get_subset(subset) if subset else expected_dataset

        dataset = (
            Dataset.import_from(path, format)
            if not stream
            else StreamDataset.import_from(path, format)
        )
        check_is_stream(dataset, stream)
        compare_datasets(helper_tc, expected, dataset, require_media=True)

    @skip(
        "COCO format is required to specify the task in annotation file "
        " for resolving ambiguity problem."
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_import_stuff_with_any_annotation_filename(self, stream, test_dir, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="default",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    attributes={"id": 5},
                    annotations=[
                        Mask(
                            np.array([[0, 0, 1, 1, 0, 1, 1, 0, 0, 0]] * 5),
                            label=0,
                            id=7,
                            group=7,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(
                            2.0,
                            0.0,
                            4.0,
                            4.0,
                            id=7,
                            attributes={"is_crowd": True},
                            group=7,
                            label=0,
                            z_order=0,
                        ),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        format = "coco_stuff"

        dataset_dir = osp.join(test_dir, "dataset")
        expected_dataset.export(dataset_dir, format, save_media=True)
        os.rename(
            osp.join(dataset_dir, "annotations", "stuff_default.json"),
            osp.join(dataset_dir, "annotations", "aa_bbbb_cccc.json"),
        )

        imported_dataset = (
            Dataset.import_from(dataset_dir, format)
            if not stream
            else StreamDataset.import_from(dataset_dir, format)
        )
        check_is_stream(imported_dataset, stream)
        compare_datasets(helper_tc, expected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "subdir",
        [
            "coco",
            "coco_captions",
            "coco_image_info",
            "coco_instances",
            "coco_labels",
            "coco_panoptic",
            "coco_person_keypoints",
            "coco_stuff",
        ],
    )
    def test_can_detect(self, subdir):
        env = Environment()

        dataset_dir = osp.join(DUMMY_DATASET_DIR, subdir)

        detected_formats = env.detect_dataset(dataset_dir)
        assert [subdir] == detected_formats

    @mark_requirement(Requirements.DATUM_673)
    @pytest.mark.parametrize(
        "subdir",
        [
            # "coco",
            "coco_captions",
            "coco_image_info",
            "coco_instances",
            "coco_labels",
            "coco_panoptic",
            "coco_person_keypoints",
            "coco_stuff",
        ],
    )
    @pytest.mark.parametrize("stream", [True, False])
    def test_can_pickle(self, subdir, stream, helper_tc):
        dataset_dir = osp.join(DUMMY_DATASET_DIR, subdir)
        source = (
            Dataset.import_from(dataset_dir, format=subdir)
            if not stream
            else StreamDataset.import_from(dataset_dir, format=subdir)
        )

        parsed = pickle.loads(pickle.dumps(source))  # nosec

        compare_datasets_strict(helper_tc, source, parsed)

    @pytest.fixture
    def fxt_wrong_structure_1(self, test_dir):
        # Wrong structure: ./annotations -> ./labels
        dataset_dir = osp.join(test_dir, "coco")
        shutil.copytree(osp.join(DUMMY_DATASET_DIR, "coco"), dataset_dir)
        shutil.move(osp.join(dataset_dir, "annotations"), osp.join(dataset_dir, "labels"))
        yield dataset_dir

    @pytest.fixture
    def fxt_wrong_structure_2(self, test_dir):
        # Wrong structure: ./images -> ./imgs
        dataset_dir = osp.join(test_dir, "coco")
        shutil.copytree(osp.join(DUMMY_DATASET_DIR, "coco"), dataset_dir)
        shutil.move(osp.join(dataset_dir, "images"), osp.join(dataset_dir, "imgs"))
        yield dataset_dir

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("tc", ["fxt_wrong_structure_1", "fxt_wrong_structure_2"])
    @pytest.mark.parametrize("stream", [True, False])
    def test_import_error_on_wrong_directory_structure(
        self, tc, stream, request: pytest.FixtureRequest
    ):
        dataset_dir = request.getfixturevalue(tc)
        with pytest.raises(DatasetImportError):
            if not stream:
                Dataset.import_from(dataset_dir, format="coco")
            else:
                StreamDataset.import_from(dataset_dir, format="coco")


class CocoExtractorTests(TestCase):
    ANNOTATION_JSON_TEMPLATE = {
        "images": [
            {
                "id": 5,
                "width": 10,
                "height": 5,
                "file_name": "a.jpg",
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 5,
                "category_id": 1,
                "segmentation": [],
                "area": 3.0,
                "bbox": [2, 2, 3, 1],
                "iscrowd": 0,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "test",
            }
        ],
    }

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_unexpected_file(self):
        with TestDir() as test_dir:
            with self.assertRaisesRegex(FileNotFoundError, "JSON file"):
                CocoInstancesBase(test_dir)

    @staticmethod
    def _get_dummy_annotation_path(test_dir: str) -> str:
        ann_dir = osp.join(test_dir, "annotations")
        if not os.path.exists(ann_dir):
            os.makedirs(ann_dir)
        images_dir = osp.join(test_dir, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        return osp.join(test_dir, "annotations", "ann.json")

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_item_field(self):
        for field in ["id", "file_name"]:
            with self.subTest(field=field):
                with TestDir() as test_dir:
                    ann_path = self._get_dummy_annotation_path(test_dir)
                    anns = deepcopy(self.ANNOTATION_JSON_TEMPLATE)
                    anns["images"][0].pop(field)
                    dump_json_file(ann_path, anns)

                    with self.assertRaises(ItemImportError) as capture:
                        try:
                            Dataset.import_from(ann_path, "coco_instances")
                        except DatasetImportError as e:
                            if str(e).startswith("Failed to import dataset"):
                                raise e.__cause__
                            raise e
                    self.assertIsInstance(capture.exception.__cause__, MissingFieldError)
                    self.assertEqual(capture.exception.__cause__.name, field)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_ann_field(self):
        # https://github.com/open-edge-platform/datumaro/issues/1344 requires to make "segmentation" optional
        for field in ["id", "image_id", "iscrowd", "category_id", "bbox"]:
            with self.subTest(field=field):
                with TestDir() as test_dir:
                    ann_path = self._get_dummy_annotation_path(test_dir)
                    anns = deepcopy(self.ANNOTATION_JSON_TEMPLATE)
                    anns["annotations"][0].pop(field)
                    dump_json_file(ann_path, anns)

                    with self.assertRaises(AnnotationImportError) as capture:
                        try:
                            Dataset.import_from(ann_path, "coco_instances")
                        except DatasetImportError as e:
                            if str(e).startswith("Failed to import dataset"):
                                raise e.__cause__
                            raise e
                    self.assertIsInstance(capture.exception.__cause__, MissingFieldError)
                    self.assertEqual(capture.exception.__cause__.name, field)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_global_field(self):
        for field in ["images", "annotations", "categories"]:
            with self.subTest(field=field):
                with TestDir() as test_dir:
                    ann_path = self._get_dummy_annotation_path(test_dir)
                    anns = deepcopy(self.ANNOTATION_JSON_TEMPLATE)
                    anns.pop(field)
                    dump_json_file(ann_path, anns)

                    with self.assertRaises(MissingFieldError) as capture:
                        try:
                            Dataset.import_from(ann_path, "coco_instances")
                        except Exception as e:
                            if isinstance(e, DatasetImportError) and e.__cause__:
                                raise e.__cause__
                            raise
                    self.assertEqual(capture.exception.name, field)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_category_field(self):
        for field in ["id", "name"]:
            with self.subTest(field=field):
                with TestDir() as test_dir:
                    ann_path = self._get_dummy_annotation_path(test_dir)
                    anns = deepcopy(self.ANNOTATION_JSON_TEMPLATE)
                    anns["categories"][0].pop(field)
                    dump_json_file(ann_path, anns)

                    with self.assertRaises(MissingFieldError) as capture:
                        try:
                            Dataset.import_from(ann_path, "coco_instances")
                        except Exception as e:
                            if isinstance(e, DatasetImportError) and e.__cause__:
                                raise e.__cause__
                            raise
                    self.assertEqual(capture.exception.name, field)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_undeclared_label(self):
        with TestDir() as test_dir:
            ann_path = self._get_dummy_annotation_path(test_dir)
            anns = deepcopy(self.ANNOTATION_JSON_TEMPLATE)
            anns["annotations"][0]["category_id"] = 2
            dump_json_file(ann_path, anns)

            with self.assertRaises(AnnotationImportError) as capture:
                try:
                    Dataset.import_from(ann_path, "coco_instances")
                except DatasetImportError as e:
                    if str(e).startswith("Failed to import dataset"):
                        raise e.__cause__
                    raise e
            self.assertIsInstance(capture.exception.__cause__, UndeclaredLabelError)
            self.assertEqual(capture.exception.__cause__.id, "2")

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_bbox(self):
        with TestDir() as test_dir:
            ann_path = self._get_dummy_annotation_path(test_dir)
            anns = deepcopy(self.ANNOTATION_JSON_TEMPLATE)
            anns["annotations"][0]["bbox"] = [1, 2, 3, 4, 5]
            dump_json_file(ann_path, anns)

            with self.assertRaises(AnnotationImportError) as capture:
                try:
                    Dataset.import_from(ann_path, "coco_instances")
                except DatasetImportError as e:
                    if str(e).startswith("Failed to import dataset"):
                        raise e.__cause__
                    raise e
            self.assertIsInstance(capture.exception.__cause__, InvalidAnnotationError)
            self.assertIn("Bbox has wrong value count", str(capture.exception.__cause__))

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_polygon_odd_points(self):
        with TestDir() as test_dir:
            ann_path = self._get_dummy_annotation_path(test_dir)
            anns = deepcopy(self.ANNOTATION_JSON_TEMPLATE)
            anns["annotations"][0]["segmentation"] = [[1, 2, 3]]
            dump_json_file(ann_path, anns)

            with self.assertRaises(AnnotationImportError) as capture:
                try:
                    Dataset.import_from(ann_path, "coco_instances")
                except DatasetImportError as e:
                    if str(e).startswith("Failed to import dataset"):
                        raise e.__cause__
                    raise e
            self.assertIsInstance(capture.exception.__cause__, InvalidAnnotationError)
            self.assertIn("not divisible by 2", str(capture.exception.__cause__))

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_polygon_less_than_3_points(self):
        with TestDir() as test_dir:
            ann_path = self._get_dummy_annotation_path(test_dir)
            anns = deepcopy(self.ANNOTATION_JSON_TEMPLATE)
            anns["annotations"][0]["segmentation"] = [[1, 2, 3, 4]]
            dump_json_file(ann_path, anns)

            with self.assertRaises(AnnotationImportError) as capture:
                try:
                    Dataset.import_from(ann_path, "coco_instances")
                except DatasetImportError as e:
                    if str(e).startswith("Failed to import dataset"):
                        raise e.__cause__
                    raise e
            self.assertIsInstance(capture.exception.__cause__, InvalidAnnotationError)
            self.assertIn("at least 3 (x, y) pairs", str(capture.exception.__cause__))

    @skip(
        "CocoBase is changed to skip loading annotation "
        "if there is no image id reference rather than raising an error."
    )
    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_image_id(self):
        with TestDir() as test_dir:
            ann_path = self._get_dummy_annotation_path(test_dir)
            anns = deepcopy(self.ANNOTATION_JSON_TEMPLATE)
            anns["annotations"][0]["image_id"] = 10
            dump_json_file(ann_path, anns)

            with self.assertRaises(AnnotationImportError) as capture:
                try:
                    Dataset.import_from(ann_path, "coco_instances")
                except DatasetImportError as e:
                    if str(e).startswith("Failed to import dataset"):
                        raise e.__cause__
                    raise e
            self.assertIsInstance(capture.exception.__cause__, InvalidAnnotationError)
            self.assertIn("Unknown image id", str(capture.exception.__cause__))

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_item_field_type(self):
        with TestDir() as test_dir:
            for field, value in [("id", "q"), ("width", "q"), ("height", "q"), ("file_name", 0)]:
                with self.subTest(field=field, value=value):
                    ann_path = self._get_dummy_annotation_path(test_dir)
                    anns = deepcopy(self.ANNOTATION_JSON_TEMPLATE)
                    anns["images"][0][field] = value
                    dump_json_file(ann_path, anns)

                    with self.assertRaises(ItemImportError) as capture:
                        try:
                            Dataset.import_from(ann_path, "coco_instances")
                        except DatasetImportError as e:
                            if str(e).startswith("Failed to import dataset"):
                                raise e.__cause__
                            raise e
                    self.assertIsInstance(capture.exception.__cause__, InvalidFieldTypeError)
                    self.assertEqual(capture.exception.__cause__.name, field)
                    self.assertEqual(capture.exception.__cause__.actual, str(type(value)))

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_ann_field_type(self):
        with TestDir() as test_dir:
            for field, value in [
                ("id", "a"),
                ("image_id", "a"),
                ("segmentation", "a"),
                ("iscrowd", "a"),
                ("category_id", "a"),
                ("bbox", "a"),
                ("score", "a"),
            ]:
                with self.subTest(field=field):
                    ann_path = self._get_dummy_annotation_path(test_dir)
                    anns = deepcopy(self.ANNOTATION_JSON_TEMPLATE)
                    anns["annotations"][0][field] = value
                    dump_json_file(ann_path, anns)

                    with self.assertRaises(AnnotationImportError) as capture:
                        try:
                            Dataset.import_from(ann_path, "coco_instances")
                        except DatasetImportError as e:
                            if str(e).startswith("Failed to import dataset"):
                                raise e.__cause__
                            raise e
                    self.assertIsInstance(capture.exception.__cause__, InvalidFieldTypeError)
                    self.assertEqual(capture.exception.__cause__.name, field)
                    self.assertEqual(capture.exception.__cause__.actual, str(type(value)))


class CocoExporterTest:
    def _test_save_and_load(
        self,
        source_dataset,
        converter,
        test_dir,
        target_dataset=None,
        importer_args=None,
        stream: bool = False,
        **kwargs,
    ):
        return check_save_and_load(
            TestCase(),
            source_dataset,
            converter,
            test_dir,
            importer="coco",
            target_dataset=target_dataset,
            importer_args=importer_args,
            stream=stream,
            **kwargs,
        )

    @pytest.fixture(params=[True, False])
    def stream(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_captions(self, stream: bool, test_dir: str):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    annotations=[
                        Caption("hello", id=1, group=1),
                        Caption("world", id=2, group=2),
                    ],
                    attributes={"id": 1},
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    annotations=[
                        Caption("test", id=3, group=3),
                    ],
                    attributes={"id": 2},
                ),
                DatasetItem(
                    id=3,
                    subset="val",
                    annotations=[
                        Caption("word", id=1, group=1),
                    ],
                    attributes={"id": 1},
                ),
            ],
        )

        self._test_save_and_load(
            expected_dataset,
            CocoCaptionsExporter.convert,
            test_dir,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_instances(self, test_dir, stream: bool):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        # Bbox + single polygon
                        Bbox(0, 1, 2, 2, label=2, group=1, id=1, attributes={"is_crowd": False}),
                        Polygon(
                            [0, 1, 2, 1, 2, 3, 0, 3],
                            attributes={"is_crowd": False},
                            label=2,
                            group=1,
                            id=1,
                        ),
                    ],
                    attributes={"id": 1},
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        # Mask + bbox
                        Mask(
                            np.array(
                                [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]],
                            ),
                            attributes={"is_crowd": True},
                            label=4,
                            group=3,
                            id=3,
                        ),
                        Bbox(1, 0, 2, 2, label=4, group=3, id=3, attributes={"is_crowd": True}),
                    ],
                    attributes={"id": 2},
                ),
                DatasetItem(
                    id=3,
                    subset="val",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        # Bbox + mask
                        Bbox(0, 1, 2, 2, label=4, group=3, id=3, attributes={"is_crowd": True}),
                        Mask(
                            np.array(
                                [[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
                            ),
                            attributes={"is_crowd": True},
                            label=4,
                            group=3,
                            id=3,
                        ),
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Polygon(
                            [0, 1, 2, 1, 2, 3, 0, 3],
                            attributes={"is_crowd": False},
                            label=2,
                            group=1,
                            id=1,
                        ),
                        Bbox(
                            0.0,
                            1.0,
                            2.0,
                            2.0,
                            id=1,
                            attributes={"is_crowd": False},
                            group=1,
                            label=2,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 1},
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]],
                            ),
                            attributes={"is_crowd": True},
                            label=4,
                            group=3,
                            id=3,
                        ),
                        Bbox(
                            1.0,
                            0.0,
                            2.0,
                            2.0,
                            id=3,
                            attributes={"is_crowd": True},
                            group=3,
                            label=4,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 2},
                ),
                DatasetItem(
                    id=3,
                    subset="val",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
                            ),
                            attributes={"is_crowd": True},
                            label=4,
                            group=3,
                            id=3,
                        ),
                        Bbox(
                            0.0,
                            1.0,
                            2.0,
                            2.0,
                            id=3,
                            attributes={"is_crowd": True},
                            group=3,
                            label=4,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            source_dataset,
            CocoInstancesExporter.convert,
            test_dir,
            target_dataset=target_dataset,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_panoptic(self, test_dir, stream: bool):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Mask(
                            image=np.array(
                                [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]]
                            ),
                            attributes={"is_crowd": False},
                            label=4,
                            group=3,
                            id=3,
                        ),
                    ],
                    attributes={"id": 1},
                ),
                DatasetItem(
                    id=2,
                    subset="val",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0],
                                    [1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                            attributes={"is_crowd": False},
                            label=4,
                            group=3,
                            id=3,
                        ),
                        Mask(
                            image=np.array(
                                [
                                    [0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 1],
                                ]
                            ),
                            attributes={"is_crowd": False},
                            label=2,
                            group=2,
                            id=2,
                        ),
                    ],
                    attributes={"id": 2},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            dataset,
            partial(CocoPanopticExporter.convert, save_media=True),
            test_dir,
            require_media=True,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_stuff(self, test_dir, stream: bool):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]],
                            ),
                            attributes={"is_crowd": False},
                            label=4,
                            group=3,
                            id=3,
                        ),
                    ],
                    attributes={"id": 2},
                ),
                DatasetItem(
                    id=2,
                    subset="val",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
                            ),
                            attributes={"is_crowd": False},
                            label=4,
                            group=3,
                            id=3,
                        ),
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]],
                            ),
                            attributes={"is_crowd": False},
                            label=4,
                            group=3,
                            id=3,
                        ),
                        Bbox(
                            1.0,
                            0.0,
                            2.0,
                            2.0,
                            id=3,
                            attributes={"is_crowd": True},
                            group=3,
                            label=4,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 2},
                ),
                DatasetItem(
                    id=2,
                    subset="val",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
                            ),
                            attributes={"is_crowd": False},
                            label=4,
                            group=3,
                            id=3,
                        ),
                        Bbox(
                            0.0,
                            1.0,
                            2.0,
                            1.0,
                            id=3,
                            attributes={"is_crowd": True},
                            group=3,
                            label=4,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            source_dataset,
            CocoStuffExporter.convert,
            test_dir,
            target_dataset=target_dataset,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_polygons_on_loading(self, test_dir, stream: bool):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((6, 10, 3))),
                    annotations=[
                        Polygon([0, 0, 4, 0, 4, 4], label=3, id=4, group=4),
                        Polygon([5, 0, 9, 0, 5, 5], label=3, id=4, group=4),
                    ],
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((6, 10, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ],
                                # only internal fragment (without the border),
                                # but not everywhere...
                            ),
                            label=3,
                            id=4,
                            group=4,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            9.0,
                            5.0,
                            id=4,
                            attributes={"is_crowd": False},
                            group=4,
                            label=3,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            source_dataset,
            CocoInstancesExporter.convert,
            test_dir,
            importer_args={"merge_instance_polygons": True},
            target_dataset=target_dataset,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_crop_covered_segments(self, test_dir, stream: bool):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    [1, 1, 0, 1, 1],
                                    [1, 1, 1, 0, 0],
                                    [1, 1, 1, 0, 0],
                                ],
                            ),
                            label=2,
                            id=1,
                            z_order=0,
                        ),
                        Polygon([1, 1, 4, 1, 4, 4, 1, 4], label=1, id=2, z_order=1),
                    ],
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0],
                                ],
                            ),
                            attributes={"is_crowd": True},
                            label=2,
                            id=1,
                            group=1,
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            4.0,
                            4.0,
                            id=1,
                            attributes={"is_crowd": True},
                            group=1,
                            label=2,
                            z_order=0,
                        ),
                        Polygon(
                            [1, 1, 4, 1, 4, 4, 1, 4],
                            label=1,
                            id=2,
                            group=2,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(
                            1.0,
                            1.0,
                            3.0,
                            3.0,
                            id=2,
                            attributes={"is_crowd": False},
                            group=2,
                            label=1,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            source_dataset,
            partial(CocoInstancesExporter.convert, crop_covered=True),
            test_dir,
            target_dataset=target_dataset,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_convert_polygons_to_mask(self, test_dir, stream: bool):
        """
        <b>Description:</b>
        Ensure that the dataset polygon annotation can be properly converted into dataset segmentation mask.

        <b>Expected results:</b>
        Dataset segmentation mask converted from dataset polygon annotation is equal to expected mask.

        <b>Steps:</b>
        1. Prepare dataset with polygon annotation (source dataset)
        2. Prepare dataset with expected mask segmentation mode (target dataset)
        3. Convert source dataset to target, with conversion of annotation from polygon to mask. Verify that result
        segmentation mask is equal to expected mask.

        """

        # 1. Prepare dataset with polygon annotation (source dataset)
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((6, 10, 3))),
                    annotations=[
                        Polygon([0, 0, 4, 0, 4, 4], label=3, id=4, group=4),
                        Polygon([5, 0, 9, 0, 5, 5], label=3, id=4, group=4),
                    ],
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        # 2. Prepare dataset with expected mask segmentation mode (target dataset)
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((6, 10, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ],
                                # only internal fragment (without the border),
                                # but not everywhere...
                            ),
                            attributes={"is_crowd": True},
                            label=3,
                            id=4,
                            group=4,
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            9.0,
                            5.0,
                            id=4,
                            attributes={"is_crowd": True},
                            group=4,
                            label=3,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        # 3. Convert source dataset to target, with conversion of annotation from polygon to mask. Verify that result
        # segmentation mask is equal to expected mask.
        self._test_save_and_load(
            source_dataset,
            partial(CocoInstancesExporter.convert, segmentation_mode="mask"),
            test_dir,
            target_dataset=target_dataset,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_convert_masks_to_polygons(self, test_dir, stream: bool):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ),
                            label=3,
                            id=4,
                            group=4,
                        ),
                    ],
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Polygon(
                            [1, 0, 3, 2, 3, 0, 1, 0],
                            label=3,
                            id=4,
                            group=4,
                            attributes={"is_crowd": False},
                        ),
                        Polygon(
                            [5, 0, 5, 3, 8, 0, 5, 0],
                            label=3,
                            id=4,
                            group=4,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(
                            1.0,
                            0.0,
                            7.0,
                            3.0,
                            id=4,
                            attributes={"is_crowd": False},
                            group=4,
                            label=3,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            source_dataset,
            partial(CocoInstancesExporter.convert, segmentation_mode="polygons"),
            test_dir,
            target_dataset=target_dataset,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_images(self, test_dir, stream: bool):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, subset="train", attributes={"id": 1}),
                DatasetItem(id=2, subset="train", attributes={"id": 2}),
                DatasetItem(id=2, subset="val", attributes={"id": 2}),
                DatasetItem(id=3, subset="val", attributes={"id": 3}),
                DatasetItem(id=4, subset="val", attributes={"id": 4}),
                DatasetItem(id=5, subset="test", attributes={"id": 1}),
            ],
        )

        self._test_save_and_load(
            expected_dataset,
            CocoImageInfoExporter.convert,
            test_dir,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_231)
    def test_can_save_dataset_with_cjk_categories(self, test_dir, stream: bool):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Bbox(0, 1, 2, 2, label=0, group=1, id=1, attributes={"is_crowd": False}),
                    ],
                    attributes={"id": 1},
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Bbox(1, 0, 2, 2, label=1, group=2, id=2, attributes={"is_crowd": False}),
                    ],
                    attributes={"id": 2},
                ),
                DatasetItem(
                    id=3,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Bbox(0, 1, 2, 2, label=2, group=3, id=3, attributes={"is_crowd": False}),
                    ],
                    attributes={"id": 3},
                ),
            ],
            categories=["고양이", "ネコ", "猫"],
        )

        self._test_save_and_load(
            expected_dataset, CocoInstancesExporter.convert, test_dir, stream=stream
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self, test_dir, stream: bool):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="кириллица с пробелом", subset="train", attributes={"id": 1}),
            ],
        )

        self._test_save_and_load(
            expected_dataset,
            CocoImageInfoExporter.convert,
            test_dir,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_labels(self, test_dir, stream: bool):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    annotations=[
                        Label(4, id=1, group=1),
                        Label(9, id=2, group=2),
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            expected_dataset,
            CocoLabelsExporter.convert,
            test_dir,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_keypoints(self, test_dir, stream: bool):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        # Full instance annotations: polygon + keypoints
                        Points([0, 0, 0, 2, 4, 1], [0, 1, 2], label=3, group=1, id=1),
                        Polygon([0, 0, 4, 0, 4, 4], label=3, group=1, id=1),
                        # Full instance annotations: bbox + keypoints
                        Points([1, 2, 3, 4, 2, 3], group=2, id=2),
                        Bbox(1, 2, 2, 2, group=2, id=2),
                        # Solitary keypoints
                        Points([1, 2, 0, 2, 4, 1], label=5, id=3),
                        # Some other solitary annotations (bug #1387)
                        Polygon([0, 0, 4, 0, 4, 4], label=3, id=4),
                        # Solitary keypoints with no label
                        Points([0, 0, 1, 2, 3, 4], [0, 1, 2], id=5),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(str(i) for i in range(10)),
                AnnotationType.points: PointsCategories.from_iterable(
                    (i, None, [[0, 1], [1, 2]]) for i in range(10)
                ),
            },
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        Points(
                            [0, 0, 0, 2, 4, 1],
                            [0, 1, 2],
                            label=3,
                            group=1,
                            id=1,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            4.0,
                            4.0,
                            id=1,
                            attributes={"is_crowd": False},
                            group=1,
                            label=3,
                            z_order=0,
                        ),
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            label=3,
                            group=1,
                            id=1,
                            attributes={"is_crowd": False},
                        ),
                        Points([1, 2, 3, 4, 2, 3], group=2, id=2, attributes={"is_crowd": False}),
                        Bbox(1, 2, 2, 2, group=2, id=2, attributes={"is_crowd": False}),
                        Points(
                            [1, 2, 0, 2, 4, 1],
                            label=5,
                            group=3,
                            id=3,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(0, 1, 4, 1, label=5, group=3, id=3, attributes={"is_crowd": False}),
                        Points(
                            [0, 0, 1, 2, 3, 4],
                            [0, 1, 2],
                            group=5,
                            id=5,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(1, 2, 2, 2, group=5, id=5, attributes={"is_crowd": False}),
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(str(i) for i in range(10)),
                AnnotationType.points: PointsCategories.from_iterable(
                    (i, None, [[0, 1], [1, 2]]) for i in range(10)
                ),
            },
        )

        self._test_save_and_load(
            source_dataset,
            CocoPersonKeypointsExporter.convert,
            test_dir,
            target_dataset=target_dataset,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_no_subsets(self, test_dir, stream: bool):
        test_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, attributes={"id": 1}),
                DatasetItem(id=2, attributes={"id": 2}),
            ],
        )

        self._test_save_and_load(
            test_dataset,
            CocoExporter.convert,
            test_dir,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_image_info(self, test_dir, stream: bool):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1, media=Image.from_file(path="1.jpg", size=(10, 15)), attributes={"id": 1}
                ),
            ],
        )

        self._test_save_and_load(
            expected_dataset, CocoImageInfoExporter.convert, test_dir, stream=stream
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_relative_paths(self, test_dir, stream: bool):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1", media=Image.from_numpy(data=np.ones((4, 2, 3))), attributes={"id": 1}
                ),
                DatasetItem(
                    id="subdir1/1",
                    media=Image.from_numpy(data=np.ones((2, 6, 3))),
                    attributes={"id": 2},
                ),
                DatasetItem(
                    id="subdir2/1",
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    attributes={"id": 3},
                ),
            ],
        )

        self._test_save_and_load(
            expected_dataset,
            partial(CocoImageInfoExporter.convert, save_media=True),
            test_dir,
            require_media=True,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self, test_dir, stream: bool):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id="q/1",
                    media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG"),
                    attributes={"id": 1},
                ),
                DatasetItem(
                    id="a/b/c/2",
                    media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp"),
                    attributes={"id": 2},
                ),
            ],
        )

        self._test_save_and_load(
            expected,
            partial(CocoImageInfoExporter.convert, save_media=True),
            test_dir,
            require_media=True,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_preserve_coco_ids(self, test_dir, stream: bool):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="some/name1",
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    attributes={"id": 40},
                ),
            ],
        )

        self._test_save_and_load(
            expected_dataset,
            partial(CocoImageInfoExporter.convert, save_media=True),
            test_dir,
            require_media=True,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_annotation_attributes(self, test_dir, stream: bool):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            label=5,
                            group=1,
                            id=1,
                            attributes={"is_crowd": False, "x": 5, "y": "abc"},
                        ),
                    ],
                    attributes={"id": 1},
                )
            ],
            categories=[str(i) for i in range(10)],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            label=5,
                            group=1,
                            id=1,
                            attributes={"is_crowd": False, "x": 5, "y": "abc"},
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            4.0,
                            4.0,
                            id=1,
                            attributes={"x": 5, "y": "abc", "is_crowd": False},
                            group=1,
                            label=5,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 1},
                )
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            source_dataset,
            CocoExporter.convert,
            test_dir,
            target_dataset=target_dataset,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_auto_annotation_ids(self, test_dir, stream: bool):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=2,
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    annotations=[
                        Polygon([0, 0, 4, 0, 4, 4], label=0),
                    ],
                )
            ],
            categories=[str(i) for i in range(10)],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=2,
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            label=0,
                            id=1,
                            group=1,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            4.0,
                            4.0,
                            id=1,
                            attributes={"is_crowd": False},
                            group=1,
                            label=0,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 1},
                )
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            source_dataset,
            CocoExporter.convert,
            test_dir,
            target_dataset=target_dataset,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_subset_can_contain_underscore(self, test_dir, stream: bool):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=2,
                    subset="subset_1",
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            label=0,
                            id=1,
                            group=1,
                            attributes={"is_crowd": False},
                        ),
                    ],
                    attributes={"id": 1},
                )
            ],
            categories=[str(i) for i in range(10)],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=2,
                    subset="subset_1",
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            label=0,
                            id=1,
                            group=1,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            4.0,
                            4.0,
                            id=1,
                            attributes={"is_crowd": False},
                            group=1,
                            label=0,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 1},
                )
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            source_dataset,
            CocoExporter.convert,
            test_dir,
            target_dataset=target_dataset,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex(self, test_dir, stream: bool):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=2,
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    annotations=[
                        Polygon([0, 0, 4, 0, 4, 4], label=0, id=5),
                    ],
                    attributes={"id": 22},
                )
            ],
            categories=[str(i) for i in range(10)],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=2,
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            label=0,
                            id=1,
                            group=1,
                            attributes={"is_crowd": False},
                        ),
                        Bbox(
                            0.0,
                            0.0,
                            4.0,
                            4.0,
                            id=1,
                            attributes={"is_crowd": False},
                            group=1,
                            label=0,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 1},
                )
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            source_dataset,
            partial(CocoExporter.convert, reindex=True),
            test_dir,
            target_dataset=target_dataset,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_media_in_single_dir(self, test_dir, stream: bool):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((2, 4, 3))),
                    attributes={"id": 1},
                ),
            ],
        )

        self._test_save_and_load(
            dataset,
            partial(CocoImageInfoExporter.convert, save_media=True, merge_images=True),
            test_dir,
            require_media=True,
            stream=stream,
        )
        assert osp.isfile(osp.join(test_dir, "images", "1.jpg"))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_media_in_separate_dirs(self, test_dir, stream: bool):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((2, 4, 3))),
                    attributes={"id": 1},
                ),
            ],
        )

        self._test_save_and_load(
            dataset,
            partial(CocoImageInfoExporter.convert, save_media=True, merge_images=False),
            test_dir,
            require_media=True,
            stream=stream,
        )
        assert osp.isfile(osp.join(test_dir, "images", "train", "1.jpg"))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self, test_dir, stream: bool):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="a"),
                DatasetItem(2, subset="a", media=Image.from_numpy(data=np.ones((3, 2, 3)))),
                DatasetItem(2, subset="b"),
            ],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, subset="a"),
                DatasetItem(2, subset="b"),
                DatasetItem(3, subset="c", media=Image.from_numpy(data=np.ones((2, 2, 3)))),
            ],
        )
        dataset.export(test_dir, "coco", save_media=True)

        dataset.put(DatasetItem(2, subset="a", media=Image.from_numpy(data=np.ones((3, 2, 3)))))
        dataset.remove(3, "c")
        dataset.save(save_media=True, stream=stream)

        assert {"image_info_a.json", "image_info_b.json"} == set(
            os.listdir(osp.join(test_dir, "annotations"))
        )
        assert osp.isfile(osp.join(test_dir, "images", "a", "2.jpg"))
        assert osp.isfile(osp.join(test_dir, "images", "c", "3.jpg")) == False
        compare_datasets(
            TestCase(),
            expected,
            Dataset.import_from(test_dir, "coco"),
            require_media=True,
            ignored_attrs={"id"},
        )

    @mark_requirement(Requirements.DATUM_BUG_425)
    def test_can_save_and_load_grouped_masks_and_polygons(self, test_dir, stream: bool):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ],
                            ),
                            label=0,
                            id=0,
                            z_order=0,
                            group=1,
                        ),
                        Polygon([1, 1, 1, 3, 3, 3, 3, 1], label=0, id=1, z_order=0, group=1),
                    ],
                ),
            ],
            categories=["label_1"],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0],
                                ],
                            ),
                            attributes={"is_crowd": True},
                            label=0,
                            id=0,
                            group=1,
                        ),
                        Bbox(
                            1.0,
                            1.0,
                            2.0,
                            2.0,
                            id=1,
                            attributes={"is_crowd": True},
                            group=1,
                            label=0,
                            z_order=0,
                        ),
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories=["label_1"],
        )

        self._test_save_and_load(
            source_dataset,
            partial(CocoInstancesExporter.convert),
            test_dir,
            target_dataset=target_dataset,
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_panoptic_with_meta_file(self, test_dir, stream: bool):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Mask(
                            image=np.array(
                                [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]]
                            ),
                            attributes={"is_crowd": False},
                            label=4,
                            group=3,
                            id=3,
                        ),
                    ],
                    attributes={"id": 1},
                ),
                DatasetItem(
                    id=2,
                    subset="val",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Mask(
                            image=np.array(
                                [
                                    [0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 1],
                                ]
                            ),
                            attributes={"is_crowd": False},
                            label=2,
                            group=2,
                            id=2,
                        ),
                    ],
                    attributes={"id": 2},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            dataset,
            partial(CocoPanopticExporter.convert, save_media=True, save_dataset_meta=True),
            test_dir,
            require_media=True,
            stream=stream,
        )
        assert osp.isfile(osp.join(test_dir, "dataset_meta.json"))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_stuff_with_meta_file(self, test_dir, stream: bool):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]],
                            ),
                            attributes={"is_crowd": False},
                            label=4,
                            group=3,
                            id=3,
                        ),
                    ],
                    attributes={"id": 2},
                ),
                DatasetItem(
                    id=2,
                    subset="val",
                    media=Image.from_numpy(data=np.ones((4, 4, 3))),
                    annotations=[
                        Mask(
                            np.array(
                                [[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
                            ),
                            attributes={"is_crowd": False},
                            label=4,
                            group=3,
                            id=3,
                        ),
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            dataset,
            partial(CocoPanopticExporter.convert, save_media=True, save_dataset_meta=True),
            test_dir,
            require_media=True,
            stream=stream,
        )
        assert osp.isfile(osp.join(test_dir, "dataset_meta.json"))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_export_and_import_ellipse(self, test_dir, stream: bool):
        ellipses = [
            Ellipse(0, 0, 5, 5, id=1, label=1, group=1),
            Ellipse(5, 5, 10, 10, id=2, label=2, group=2),
        ]

        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=ellipses,
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Polygon(
                            ellipse.as_polygon(),
                            id=ellipse.id,
                            label=ellipse.label,
                            group=ellipse.group,
                        )
                        for ellipse in ellipses
                    ]
                    + [
                        Bbox(
                            *ellipse.get_bbox(),
                            id=ellipse.id,
                            label=ellipse.label,
                            group=ellipse.group,
                        )
                        for ellipse in ellipses
                    ],
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        self._test_save_and_load(
            source_dataset,
            CocoInstancesExporter.convert,
            test_dir,
            target_dataset=target_dataset,
            ignored_attrs={"is_crowd"},
            stream=stream,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "annotations, n_expected_anns",
        [
            (
                [
                    Bbox(0, 0, 20, 10, id=1, label=1, group=1),
                    Bbox(10, 10, 10, 20, id=2, label=2, group=2),
                ],
                2,
            ),
            (
                [
                    Ellipse(0, 0, 5, 5, id=1, label=1, group=1),
                    Ellipse(5, 5, 10, 10, id=2, label=2, group=2),
                ],
                4,
            ),
            (
                [
                    Polygon([0, 0, 4, 0, 4, 4], id=1, label=1, group=1),
                    Polygon([5, 0, 9, 0, 5, 5], id=2, label=2, group=2),
                ],
                4,
            ),
            (
                [
                    Mask(
                        np.array([[1, 1, 0, 0, 0]] * 10),
                        id=1,
                        label=1,
                        group=1,
                    ),
                    Mask(
                        np.array([[0, 0, 1, 1, 0]] * 10),
                        id=2,
                        label=2,
                        group=2,
                    ),
                ],
                0,
            ),
            (
                [
                    Polygon([0, 0, 4, 0, 4, 4], id=1, label=1, group=1),
                    Polygon([5, 0, 9, 0, 5, 5], id=2, label=2, group=2),
                    Mask(
                        np.array([[0, 0, 1, 1, 0]] * 10),
                        id=3,
                        label=2,
                        group=2,
                    ),
                ],
                0,
            ),
            (
                [
                    Polygon([0, 0, 4, 0, 4, 4], id=1, label=1, group=1),
                    Polygon([0, 5, 0, 0, 2, 0], id=2, label=1, group=1),
                    Polygon([5, 0, 9, 0, 5, 5], id=3, label=2, group=2),
                ],
                5,
            ),
        ],
    )
    def test_can_export_annotations_without_media(
        self, annotations, n_expected_anns, test_dir, stream: bool
    ):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=None,
                    annotations=annotations,
                    attributes={"id": 1},
                ),
            ],
            categories=[str(i) for i in range(10)],
        )

        dataset.export(test_dir, "coco")
        anno_file = osp.join(test_dir, "annotations", "instances_default.json")
        anno = parse_json_file(anno_file)
        assert len(anno["annotations"]) == 2

        if n_expected_anns > 0:  ## importable
            imported = Dataset.import_from(test_dir, "coco")
            imported_anns = []
            for item in imported:
                imported_anns.extend(item.annotations)
            assert len(imported_anns) == n_expected_anns
        else:
            with pytest.raises(AnnotationImportError) as capture:
                try:
                    Dataset.import_from(test_dir, "coco")
                except DatasetImportError as e:
                    if str(e).startswith("Failed to import dataset"):
                        raise e.__cause__
                    raise e
            assert "does not match image size" in str(capture.value)
