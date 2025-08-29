# Copyright (C) 2023-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from unittest import TestCase

import numpy as np
import pytest

import datumaro.plugins.data_formats.voc.format as VOC
from datumaro.components.annotation import AnnotationType, Bbox
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets
from tests.utils.test_utils import run_datum as run


class YoloIntegrationScenarios(TestCase):
    def test_can_save_and_load_yolo_dataset(self):
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(0.0, 2.0, 4.0, 2.0, label=2, id=0, group=0),
                        Bbox(3.0, 3.0, 2.0, 3.0, label=4, id=1, group=1),
                    ],
                )
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            yolo_dir = get_test_asset_path("yolo_dataset")

            # Direct round-trip conversion: YOLO -> YOLO
            export_dir = osp.join(test_dir, "export_dir")
            run(
                self,
                "convert",
                "-if",
                "yolo",
                "-i",
                yolo_dir,
                "-f",
                "yolo",
                "-o",
                export_dir,
                "--",
                "--save-media",
            )

            parsed_dataset = Dataset.import_from(export_dir, format="yolo")
            compare_datasets(self, target_dataset, parsed_dataset)

    def test_can_export_mot_as_yolo(self):
        target_dataset = Dataset.from_iterable(
            [DatasetItem(id="1", subset="train", annotations=[Bbox(0.0, 4.0, 4.0, 8.0, label=2)])],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            mot_dir = get_test_asset_path("mot_dataset")

            yolo_dir = osp.join(test_dir, "yolo_dir")
            run(
                self,
                "convert",
                "-if",
                "mot_seq",
                "-i",
                mot_dir,
                "-f",
                "yolo",
                "-o",
                yolo_dir,
                "--",
                "--save-media",
            )

            parsed_dataset = Dataset.import_from(yolo_dir, format="yolo")
            compare_datasets(self, target_dataset, parsed_dataset)

    def test_can_convert_voc_to_yolo(self):
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                    annotations=[
                        Bbox(1.0, 2.0, 2.0, 2.0, label=8, id=0, group=0),
                        Bbox(5.5, 6.0, 2.0, 2.0, label=22, id=1, group=1),
                        Bbox(4.0, 5.0, 2.0, 2.0, label=15, id=2, group=2),
                    ],
                ),
                DatasetItem(
                    id="2007_000002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                ),
            ],
            categories=[
                label.name
                for label in VOC.make_voc_categories(task=VOC.VocTask.voc)[AnnotationType.label]
            ],
        )

        with TestDir() as test_dir:
            voc_dir = get_test_asset_path(
                "voc_dataset",
                "voc_dataset1",
            )
            yolo_dir = osp.join(test_dir, "yolo_dir")

            run(
                self,
                "convert",
                "-if",
                "voc",
                "-i",
                voc_dir,
                "-f",
                "yolo",
                "-o",
                yolo_dir,
                "--",
                "--save-media",
            )

            parsed_dataset = Dataset.import_from(yolo_dir, format="yolo")
            compare_datasets(self, target_dataset, parsed_dataset, require_media=True)

    def test_can_delete_labels_from_yolo_dataset(self):
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 15, 3))),
                    annotations=[Bbox(0.0, 2.0, 4.0, 2.0, label=0)],
                )
            ],
            categories=["label_2"],
        )

        with TestDir() as test_dir:
            yolo_dir = get_test_asset_path("yolo_dataset")

            # Apply filter and label remapping in one transform operation
            filtered_dir = osp.join(test_dir, "filtered")
            run(
                self,
                "filter",
                yolo_dir + ":yolo",
                "-m",
                "i+a",
                "-e",
                "/item/annotation[label='label_2']",
                "-o",
                filtered_dir,
            )

            # Apply label remapping to filtered dataset
            remapped_dir = osp.join(test_dir, "remapped")
            run(
                self,
                "transform",
                "-t",
                "remap_labels",
                "-o",
                remapped_dir,
                filtered_dir,
                "--",
                "-l",
                "label_2:label_2",
                "--default",
                "delete",
            )

            # Export directly to YOLO format
            export_dir = osp.join(test_dir, "export")
            run(
                self,
                "convert",
                "-i",
                remapped_dir,
                "-f",
                "yolo",
                "-o",
                export_dir,
                "--",
                "--save-media",
            )

            parsed_dataset = Dataset.import_from(export_dir, format="yolo")
            compare_datasets(self, target_dataset, parsed_dataset)


# TODO(vinnamki): Migrate above test cases to the pytest framework below
class YoloIntegrationScenariosTest:
    @pytest.fixture(params=["annotations", "labels", "strict"])
    def fxt_yolo_dir(self, request) -> str:
        return get_test_asset_path("yolo_dataset", request.param)

    def test_can_import_and_merge_nested_datasets(self, fxt_yolo_dir, test_dir, helper_tc):
        # Test merging multiple YOLO datasets without using project functionality
        num_total_items = 0
        dataset_dirs = []

        # Create reindexed copies of the dataset
        for i in range(1, 3):
            dataset_dir = osp.join(test_dir, f"dataset_{i}")

            # Transform the original dataset with reindexing to prevent overlapping IDs
            run(
                helper_tc,
                "transform",
                "-t",
                "reindex",
                "-o",
                dataset_dir,
                fxt_yolo_dir,
                "--",
                "--start",
                f"{i * 10}",  # Reindex starting from 10 or 20
            )

            dataset_dirs.append(dataset_dir)
            num_total_items += len(Dataset.import_from(dataset_dir, format="yolo"))

        # Merge the datasets
        merged_dir = osp.join(test_dir, "merged")
        run(
            helper_tc,
            "merge",
            "-o",
            merged_dir,
            "-f",
            "yolo",
            *dataset_dirs,
            "--",
            "--save-media",
        )

        # Verify the merged dataset has the expected number of items
        merged_dataset = Dataset.import_from(merged_dir, format="yolo")
        assert len(merged_dataset) == num_total_items
