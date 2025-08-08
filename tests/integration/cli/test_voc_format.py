import os.path as osp
from collections import OrderedDict
from unittest import TestCase

import numpy as np

import datumaro.plugins.data_formats.voc.format as VOC
from datumaro.components.annotation import Bbox, Label, Mask
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.media import Image

from ...requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets
from tests.utils.test_utils import run_datum as run

DUMMY_DATASETS_DIR = get_test_asset_path("voc_dataset")


class VocIntegrationScenarios(TestCase):
    def _test_can_save_and_load(
        self,
        test_dir,
        source_path,
        expected_dataset,
        dataset_format,
        result_path="",
        label_map=None,
    ):
        # Convert the dataset to the target format
        result_dir = osp.join(test_dir, "result")
        extra_args = ["--", "--save-media"]
        if label_map:
            extra_args += ["--label-map", label_map]

        run(
            self,
            "convert",
            "-i",
            source_path,
            "-if",
            dataset_format,
            "-f",
            dataset_format,
            "-o",
            result_dir,
            *extra_args,
        )

        result_path = osp.join(result_dir, result_path)
        parsed_dataset = Dataset.import_from(result_path, dataset_format)
        compare_datasets(self, expected_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_preparing_dataset_for_train_model(self):
        """
        <b>Description:</b>
        Testing a particular example of working with VOC dataset.

        <b>Expected results:</b>
        A VOC dataset that matches the expected result.

        <b>Steps:</b>
        1. Get path to the source dataset from assets.
        2. Filter to leave only non-occluded annotations with `filter` command.
        3. Split the dataset into subsets with `transform` command.
        4. Export to a VOC dataset with `convert` command.
        5. Verify that the resulting dataset is equal to the expected result.
        """

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="c",
                    subset="train",
                    annotations=[
                        Bbox(
                            3.0,
                            1.0,
                            8.0,
                            5.0,
                            attributes={"truncated": False, "occluded": False, "difficult": False},
                            id=0,
                            label=2,
                            group=0,
                        )
                    ],
                ),
                DatasetItem(
                    id="d",
                    subset="test",
                    annotations=[
                        Bbox(
                            4.0,
                            4.0,
                            4.0,
                            4.0,
                            attributes={"truncated": False, "occluded": False, "difficult": False},
                            id=0,
                            label=3,
                            group=0,
                        )
                    ],
                ),
            ],
            categories=VOC.make_voc_categories(task=VOC.VocTask.voc_detection),
        )

        dataset_path = osp.join(DUMMY_DATASETS_DIR, "voc_dataset2")

        with TestDir() as test_dir:
            # First filter to keep only non-occluded annotations
            filtered_dir = osp.join(test_dir, "filtered")
            run(
                self,
                "filter",
                dataset_path + ":voc_detection",
                "-m",
                "i+a",
                "-e",
                "/item/annotation[occluded='False']",
                "-o",
                filtered_dir,
            )

            # Then split the dataset
            split_dir = osp.join(test_dir, "split")
            run(
                self,
                "transform",
                "-t",
                "random_split",
                "-o",
                split_dir,
                filtered_dir + ":voc_detection",  # Specify input format as part of path
                "--",
                "-s",
                "test:.5",
                "-s",
                "train:.5",
                "--seed",
                "1",
            )

            # Export to VOC format
            export_path = osp.join(test_dir, "dataset")
            run(
                self,
                "convert",
                "-i",
                split_dir,
                "-if",
                "voc_detection",
                "-f",
                "voc_detection",
                "-o",
                export_path,
                "--",
                "--label-map",
                "voc_detection",
            )

            parsed_dataset = Dataset.import_from(export_path, format="voc_detection")
            compare_datasets(self, expected_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_export_to_voc_format(self):
        label_map = OrderedDict(("label_%s" % i, [None, [], []]) for i in range(10))
        label_map["background"] = [None, [], []]
        label_map.move_to_end("background", last=False)

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(
                            0.0,
                            2.0,
                            4.0,
                            2.0,
                            attributes={"difficult": False, "truncated": False, "occluded": False},
                            id=0,
                            label=3,
                            group=0,
                        ),
                        Bbox(
                            3.0,
                            3.0,
                            2.0,
                            3.0,
                            attributes={"difficult": False, "truncated": False, "occluded": False},
                            id=1,
                            label=5,
                            group=1,
                        ),
                    ],
                )
            ],
            categories=VOC.make_voc_categories(label_map),
        )

        with TestDir() as test_dir:
            yolo_dir = get_test_asset_path("yolo_dataset")

            voc_export = osp.join(test_dir, "voc_export")
            run(
                self,
                "convert",
                "-if",
                "yolo",
                "-i",
                yolo_dir,
                "-f",
                "voc",
                "-o",
                voc_export,
                "--",
                "--save-media",
            )

            parsed_dataset = Dataset.import_from(voc_export, format="voc")
            compare_datasets(self, expected_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_283)
    def test_convert_to_voc_format(self):
        """
        <b>Description:</b>
        Ensure that the dataset can be converted to VOC format with
        command `datum convert`.

        <b>Expected results:</b>
        A VOC dataset that matches the expected dataset.

        <b>Steps:</b>
        1. Get path to the source dataset from assets.
        2. Convert source dataset to VOC format, using the `convert` command.
        3. Verify that resulting dataset is equal to the expected dataset.
        """

        label_map = OrderedDict(("label_" + str(i), [None, [], []]) for i in range(10))
        label_map["background"] = [None, [], []]
        label_map.move_to_end("background", last=False)

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="default",
                    media=Image.from_numpy(data=np.ones((16, 16, 3))),
                    annotations=[
                        Bbox(
                            0.0,
                            4.0,
                            4.0,
                            8.0,
                            attributes={
                                "difficult": False,
                                "truncated": False,
                                "occluded": False,
                                "visibility": "1.0",
                                "ignored": "False",
                            },
                            id=0,
                            label=3,
                            group=0,
                        )
                    ],
                )
            ],
            categories=VOC.make_voc_categories(label_map),
        )

        mot_dir = get_test_asset_path("mot_dataset")

        with TestDir() as test_dir:
            voc_dir = osp.join(test_dir, "voc")
            run(
                self,
                "convert",
                "-if",
                "mot_seq",
                "-i",
                mot_dir,
                "-f",
                "voc",
                "-o",
                voc_dir,
                "--",
                "--save-media",
            )

            target_dataset = Dataset.import_from(voc_dir, format="voc")
            compare_datasets(self, expected_dataset, target_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_283)
    def test_convert_from_voc_format(self):
        """
        <b>Description:</b>
        Ensure that the dataset can be converted from VOC format with
        command `datum convert`.

        <b>Expected results:</b>
        A ImageNet dataset that matches the expected dataset.

        <b>Steps:</b>
        1. Get path to the source dataset from assets.
        2. Convert source dataset to LabelMe format, using the `convert` command.
        3. Verify that resulting dataset is equal to the expected dataset.
        """

        labels = sorted([l.name for l in VOC.VocLabel if l.value % 2 == 1])

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=":".join([label, "2007_000001"]), subset="default", annotations=[Label(i)]
                )
                for i, label in enumerate(labels)
            ]
            + [
                DatasetItem(
                    id="no_label:2007_000002",
                    subset="default",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                )
            ],
            categories=labels,
        )

        voc_dir = osp.join(DUMMY_DATASETS_DIR, "voc_dataset1")
        with TestDir() as test_dir:
            imagenet_dir = osp.join(test_dir, "imagenet")
            run(
                self,
                "convert",
                "-if",
                "voc_classification",
                "-i",
                voc_dir,
                "-f",
                "imagenet",
                "-o",
                imagenet_dir,
                "--",
                "--save-media",
            )

            target_dataset = Dataset.import_from(imagenet_dir, format="imagenet")
            compare_datasets(self, expected_dataset, target_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_instance_segmentation_dataset(self):
        source_dataset = Dataset.from_iterable(
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
                            label=15,
                            id=1,
                            group=1,
                            attributes={
                                "difficult": False,
                                "truncated": False,
                                "occluded": False,
                            },
                        ),
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
                        Mask(image=np.ones([10, 20]), label=2, group=1),
                    ],
                ),
                DatasetItem(
                    id="2007_000002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                ),
            ],
            categories=VOC.make_voc_categories(task=VOC.VocTask.voc_instance_segmentation),
        )

        voc_dir = osp.join(DUMMY_DATASETS_DIR, "voc_dataset1")
        with TestDir() as test_dir:
            self._test_can_save_and_load(
                test_dir,
                voc_dir,
                source_dataset,
                "voc_instance_segmentation",
                label_map="voc_instance_segmentation",
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_layout_dataset(self):
        expected_dataset = Dataset.from_iterable(
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
                        Bbox(5.5, 6.0, 2.0, 2.0, label=2, id=0, group=0),
                    ],
                ),
                DatasetItem(
                    id="2007_000002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                ),
            ],
            categories=VOC.make_voc_categories(task=VOC.VocTask.voc_layout),
        )

        dataset_dir = osp.join(DUMMY_DATASETS_DIR, "voc_dataset1")
        with TestDir() as test_dir:
            self._test_can_save_and_load(
                test_dir, dataset_dir, expected_dataset, "voc_layout", label_map="voc_layout"
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_classification_dataset(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                    annotations=[Label(i) for i in range(22) if i % 2 == 1],
                ),
                DatasetItem(
                    id="2007_000002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                ),
            ],
            categories=VOC.make_voc_categories(task=VOC.VocTask.voc_classification),
        )

        dataset_dir = osp.join(DUMMY_DATASETS_DIR, "voc_dataset1")
        rpath = osp.join("ImageSets", "Main", "train.txt")
        matrix = [
            ("voc_classification", "", ""),
            ("voc_classification", "train", rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    expected = expected_dataset.get_subset(subset)
                else:
                    expected = expected_dataset

                with TestDir() as test_dir:
                    self._test_can_save_and_load(
                        test_dir,
                        dataset_dir,
                        expected,
                        format,
                        result_path=path,
                        label_map="voc_classification",
                    )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_detection_dataset(self):
        expected_dataset = Dataset.from_iterable(
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
                            label=15,
                            id=1,
                            group=1,
                            attributes={
                                "difficult": False,
                                "truncated": False,
                                "occluded": False,
                            },
                        ),
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
                    ],
                ),
                DatasetItem(
                    id="2007_000002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                ),
            ],
            categories=VOC.make_voc_categories(task=VOC.VocTask.voc_detection),
        )

        dataset_dir = osp.join(DUMMY_DATASETS_DIR, "voc_dataset1")
        rpath = osp.join("ImageSets", "Main", "train.txt")
        matrix = [
            ("voc_detection", "", ""),
            ("voc_detection", "train", rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    expected = expected_dataset.get_subset(subset)
                else:
                    expected = expected_dataset

                with TestDir() as test_dir:
                    self._test_can_save_and_load(
                        test_dir,
                        dataset_dir,
                        expected,
                        format,
                        result_path=path,
                        label_map="voc_detection",
                    )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_segmentation_dataset(self):
        expected_dataset = Dataset.from_iterable(
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
            categories=VOC.make_voc_categories(task=VOC.VocTask.voc_segmentation),
        )

        dataset_dir = osp.join(DUMMY_DATASETS_DIR, "voc_dataset1")
        rpath = osp.join("ImageSets", "Segmentation", "train.txt")
        matrix = [
            ("voc_segmentation", "", ""),
            ("voc_segmentation", "train", rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    expected = expected_dataset.get_subset(subset)
                else:
                    expected = expected_dataset

                with TestDir() as test_dir:
                    self._test_can_save_and_load(
                        test_dir,
                        dataset_dir,
                        expected,
                        format,
                        result_path=path,
                        label_map="voc_segmentation",
                    )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_action_dataset(self):
        expected_dataset = Dataset.from_iterable(
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
            categories=VOC.make_voc_categories(task=VOC.VocTask.voc_action),
        )

        dataset_dir = osp.join(DUMMY_DATASETS_DIR, "voc_dataset1")
        rpath = osp.join("ImageSets", "Action", "train.txt")
        matrix = [
            ("voc_action", "", ""),
            ("voc_action", "train", rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    expected = expected_dataset.get_subset(subset)
                else:
                    expected = expected_dataset

                with TestDir() as test_dir:
                    self._test_can_save_and_load(
                        test_dir,
                        dataset_dir,
                        expected,
                        format,
                        result_path=path,
                        label_map="voc_action",
                    )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_label_projection_with_masks(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                    annotations=[
                        Bbox(
                            1,
                            2,
                            2,
                            2,
                            label=3,
                            attributes={
                                "pose": VOC.VocPose(1).name,
                                "truncated": True,
                                "difficult": False,
                                "occluded": False,
                            },
                            id=0,
                            group=0,
                        ),
                    ],
                ),
                DatasetItem(
                    id="2007_000002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                ),
            ],
            categories=VOC.make_voc_categories(
                {
                    "background": [(0, 0, 0), [], []],  # Added on export
                    "a": [(128, 0, 0), [], []],  # Generated by the transform
                    "b": [(0, 128, 0), [], []],  # Generated by the transform
                    "cat": [(64, 0, 0), [], []],  # Original
                }
            ),
        )

        dataset_path = osp.join(DUMMY_DATASETS_DIR, "voc_dataset1")

        with TestDir() as test_dir:
            # Apply label projection transform and export to VOC format
            export_dir = osp.join(test_dir, "exported")
            run(
                self,
                "transform",
                "-t",
                "project_labels",
                "-o",
                export_dir,
                dataset_path + ":voc",
                "--",
                "-l",
                "a",
                "-l",
                "b",
                "-l",
                "cat",
            )

            parsed_dataset = Dataset.import_from(export_dir, "voc")
            compare_datasets(self, expected_dataset, parsed_dataset)
