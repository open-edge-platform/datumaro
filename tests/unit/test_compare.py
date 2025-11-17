# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
import os
import unittest
from unittest import TestCase
from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np
import pytest

from datumaro.cli.util.compare import DistanceCompareVisualizer
from datumaro.components.annotation import Bbox, Caption, Label, Mask, Points
from datumaro.components.comparator import DistanceComparator, EqualityComparator, TableComparator
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetItem
from datumaro.components.media import Image
from tests.utils.assets import get_test_asset_path


class DistanceComparatorTest(TestCase):
    def test_no_bbox_diff_with_same_item(self):
        detections = 3
        anns = [Bbox(i * 10, 10, 10, 10, label=i) for i in range(detections)]
        item = DatasetItem(id=0, annotations=anns)

        iou_thresh = 0.5
        comp = DistanceComparator(iou_threshold=iou_thresh)

        result = comp.match_boxes(item, item)

        matches, mispred, a_greater, b_greater = result
        self.assertEqual(0, len(mispred))
        self.assertEqual(0, len(a_greater))
        self.assertEqual(0, len(b_greater))
        self.assertEqual(len(item.annotations), len(matches))
        for a_bbox, b_bbox in matches:
            self.assertLess(iou_thresh, a_bbox.iou(b_bbox))
            self.assertEqual(a_bbox.label, b_bbox.label)

    def test_can_find_bbox_with_wrong_label(self):
        detections = 3
        class_count = 2
        item1 = DatasetItem(id=1, annotations=[Bbox(i * 10, 10, 10, 10, label=i) for i in range(detections)])
        item2 = DatasetItem(
            id=2,
            annotations=[Bbox(i * 10, 10, 10, 10, label=(i + 1) % class_count) for i in range(detections)],
        )

        iou_thresh = 0.5
        comp = DistanceComparator(iou_threshold=iou_thresh)

        result = comp.match_boxes(item1, item2)

        matches, mispred, a_greater, b_greater = result
        self.assertEqual(len(item1.annotations), len(mispred))
        self.assertEqual(0, len(a_greater))
        self.assertEqual(0, len(b_greater))
        self.assertEqual(0, len(matches))
        for a_bbox, b_bbox in mispred:
            self.assertLess(iou_thresh, a_bbox.iou(b_bbox))
            self.assertEqual((a_bbox.label + 1) % class_count, b_bbox.label)

    def test_can_find_missing_boxes(self):
        detections = 3
        class_count = 2
        item1 = DatasetItem(
            id=1,
            annotations=[Bbox(i * 10, 10, 10, 10, label=i) for i in range(detections) if i % 2 == 0],
        )
        item2 = DatasetItem(
            id=2,
            annotations=[
                Bbox(i * 10, 10, 10, 10, label=(i + 1) % class_count) for i in range(detections) if i % 2 == 1
            ],
        )

        iou_thresh = 0.5
        comp = DistanceComparator(iou_threshold=iou_thresh)

        result = comp.match_boxes(item1, item2)

        matches, mispred, a_greater, b_greater = result
        self.assertEqual(0, len(mispred))
        self.assertEqual(len(item1.annotations), len(a_greater))
        self.assertEqual(len(item2.annotations), len(b_greater))
        self.assertEqual(0, len(matches))

    def test_no_label_diff_with_same_item(self):
        detections = 3
        anns = [Label(i) for i in range(detections)]
        item = DatasetItem(id=1, annotations=anns)

        result = DistanceComparator().match_labels(item, item)

        matches, a_greater, b_greater = result
        self.assertEqual(0, len(a_greater))
        self.assertEqual(0, len(b_greater))
        self.assertEqual(len(item.annotations), len(matches))

    def test_can_find_wrong_label(self):
        item1 = DatasetItem(
            id=1,
            annotations=[
                Label(0),
                Label(1),
                Label(2),
            ],
        )
        item2 = DatasetItem(
            id=2,
            annotations=[
                Label(2),
                Label(3),
                Label(4),
            ],
        )

        result = DistanceComparator().match_labels(item1, item2)

        matches, a_greater, b_greater = result
        self.assertEqual(2, len(a_greater))
        self.assertEqual(2, len(b_greater))
        self.assertEqual(1, len(matches))

    def test_can_match_points(self):
        item1 = DatasetItem(
            id=1,
            annotations=[
                Points([1, 2, 2, 0, 1, 1], label=0),
                Points([3, 5, 5, 7, 5, 3], label=0),
            ],
        )
        item2 = DatasetItem(
            id=2,
            annotations=[
                Points([1.5, 2, 2, 0.5, 1, 1.5], label=0),
                Points([5, 7, 7, 7, 7, 5], label=0),
            ],
        )

        result = DistanceComparator().match_points(item1, item2)

        matches, mismatches, a_greater, b_greater = result
        self.assertEqual(1, len(a_greater))
        self.assertEqual(1, len(b_greater))
        self.assertEqual(1, len(matches))
        self.assertEqual(0, len(mismatches))


class ExactComparatorTest(TestCase):
    def test_class_comparison(self):
        a = Dataset.from_iterable([], categories=["a", "b", "c"])
        b = Dataset.from_iterable([], categories=["b", "c"])

        comp = EqualityComparator()
        output = comp.compare_datasets(a, b)
        errors = output["errors"]

        self.assertEqual(1, len(errors), errors)

    def test_item_comparison(self):
        a = Dataset.from_iterable(
            [
                DatasetItem(id=1, subset="train"),
                DatasetItem(id=2, subset="test", attributes={"x": 1}),
            ],
            categories=["a", "b", "c"],
        )

        b = Dataset.from_iterable(
            [
                DatasetItem(id=2, subset="test"),
                DatasetItem(id=3),
            ],
            categories=["a", "b", "c"],
        )

        comp = EqualityComparator()
        output = comp.compare_datasets(a, b)

        a_extra_items = output["a_extra_items"]
        b_extra_items = output["b_extra_items"]
        errors = output["errors"]

        self.assertEqual([("1", "train")], a_extra_items)
        self.assertEqual([("3", DEFAULT_SUBSET_NAME)], b_extra_items)
        self.assertEqual(1, len(errors), errors)

    def test_annotation_comparison(self):
        a = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Caption("hello"),  # unmatched
                        Caption("world", group=5),
                        Label(
                            2,
                            attributes={
                                "x": 1,
                                "y": "2",
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=4,
                            z_order=1,
                            attributes={
                                "score": 1.0,
                            },
                        ),
                        Bbox(5, 6, 7, 8, group=5),
                        Points([1, 2, 2, 0, 1, 1], label=0, z_order=4),
                        Mask(label=3, z_order=2, image=np.ones((2, 3))),
                    ],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        b = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    annotations=[
                        Caption("world", group=5),
                        Label(
                            2,
                            attributes={
                                "x": 1,
                                "y": "2",
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=4,
                            z_order=1,
                            attributes={
                                "score": 1.0,
                            },
                        ),
                        Bbox(5, 6, 7, 8, group=5),
                        Bbox(5, 6, 7, 8, group=5),  # unmatched
                        Points([1, 2, 2, 0, 1, 1], label=0, z_order=4),
                        Mask(label=3, z_order=2, image=np.ones((2, 3))),
                    ],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        comp = EqualityComparator(all=True)
        output = comp.compare_datasets(a, b)

        matched = output["matches"]
        unmatched = output["mismatches"]
        errors = output["errors"]
        self.assertEqual(6, len(matched), matched)
        self.assertEqual(2, len(unmatched), unmatched)
        self.assertEqual(0, len(errors), errors)

    def test_image_comparison(self):
        a = Dataset.from_iterable(
            [
                DatasetItem(
                    id=11,
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[
                        Bbox(5, 6, 7, 8),
                    ],
                ),
                DatasetItem(
                    id=12,
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4),
                        Bbox(5, 6, 7, 8),
                    ],
                ),
                DatasetItem(
                    id=13,
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[
                        Bbox(9, 10, 11, 12),  # mismatch
                    ],
                ),
                DatasetItem(
                    id=14,
                    media=Image.from_numpy(data=np.zeros((5, 4, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4),
                        Bbox(5, 6, 7, 8),
                    ],
                    attributes={"a": 1},
                ),
                DatasetItem(
                    id=15,
                    media=Image.from_numpy(data=np.zeros((5, 5, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4),
                        Bbox(5, 6, 7, 8),
                    ],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        b = Dataset.from_iterable(
            [
                DatasetItem(
                    id=21,
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[
                        Bbox(5, 6, 7, 8),
                    ],
                ),
                DatasetItem(
                    id=22,
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4),
                        Bbox(5, 6, 7, 8),
                    ],
                ),
                DatasetItem(
                    id=23,
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[
                        Bbox(10, 10, 11, 12),  # mismatch
                    ],
                ),
                DatasetItem(
                    id=24,
                    media=Image.from_numpy(data=np.zeros((5, 4, 3))),
                    annotations=[
                        Bbox(6, 6, 7, 8),  # 1 ann missing, mismatch
                    ],
                    attributes={"a": 2},
                ),
                DatasetItem(
                    id=25,
                    media=Image.from_numpy(data=np.zeros((4, 4, 3))),
                    annotations=[
                        Bbox(6, 6, 7, 8),
                    ],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        comp = EqualityComparator(match_images=True, all=True)
        output = comp.compare_datasets(a, b)

        matched_ann = output["matches"]
        unmatched_ann = output["mismatches"]
        a_unmatched = output["a_extra_items"]
        b_unmatched = output["b_extra_items"]
        errors = output["errors"]
        self.assertEqual(3, len(matched_ann), matched_ann)
        self.assertEqual(5, len(unmatched_ann), unmatched_ann)
        self.assertEqual(1, len(a_unmatched), a_unmatched)
        self.assertEqual(1, len(b_unmatched), b_unmatched)
        self.assertEqual(1, len(errors), errors)


class TableComparatorTest(unittest.TestCase):
    def test_compare_datasets(self):
        # Import datatset
        first_dataset = Dataset.import_from(get_test_asset_path("mnist_dataset"))
        second_dataset = Dataset.import_from(get_test_asset_path("cifar10_dataset"))

        # Create instance of TableComparator
        table_comparator = TableComparator()

        # Call the compare_datasets method
        (
            high_level_table,
            mid_level_table,
            comparison_dict,
        ) = table_comparator.compare_datasets(first_dataset, second_dataset)

        # Assert that the return values are not empty
        self.assertIsNotNone(high_level_table)
        self.assertIsNotNone(mid_level_table)
        self.assertIsNotNone(comparison_dict)

    # Mocking is used to replace parts of the system that are being tested with mock objects.
    @patch("os.makedirs")
    @patch("datumaro.components.comparator.generate_next_file_name")
    @patch("builtins.open", new_callable=mock_open)
    @patch("datumaro.components.comparator.dump_json_file")
    def test_save_compare_report(self, mock_dump_json_file, mock_file, mock_generate_next_file_name, mock_makedirs):
        # Define mock variables
        mock_high_level_table = "High-level table"
        mock_mid_level_table = "Mid-level table"
        mock_comparison_dict = {"comparison": "data"}
        mock_report_dir = "/mock/dir"
        mock_json_output_file = "/mock/dir/table_compare_1.json"
        mock_txt_output_file = "/mock/dir/table_compare_1.txt"

        mock_generate_next_file_name.side_effect = [mock_json_output_file, mock_txt_output_file]

        TableComparator.save_compare_report(
            mock_high_level_table,
            mock_mid_level_table,
            mock_comparison_dict,
            mock_report_dir,
        )

        # Check that the os.makedirs function is called once with the correct arguments.
        mock_makedirs.assert_called_once_with(mock_report_dir, exist_ok=True)

        # Check that the dump_json_file function is called with the correct arguments.
        calls = [call(mock_json_output_file, mock_comparison_dict, indent=True)]
        mock_dump_json_file.assert_has_calls(calls)

        # Check that the open and write function is called with the correct arguments.
        mock_file.assert_any_call(mock_txt_output_file, "w")
        mock_file().write.assert_has_calls(
            [
                call(f"High-level Comparison:\n{mock_high_level_table}\n\n"),
                call(f"Mid-level Comparison:\n{mock_mid_level_table}\n\n"),
            ]
        )


class DistanceCompareVisualizerTest:
    @pytest.fixture
    def fxt_dataset_pair(self):
        """Two datasets pair which has the same (id, subset) dataset item to be compared."""
        class_count = 3
        dataset_1 = Dataset.from_iterable(
            [DatasetItem(id=1, annotations=[Label(label=idx) for idx in range(class_count)])],
            categories=[f"label_{idx}" for idx in range(class_count)],
        )
        dataset_2 = Dataset.from_iterable(
            [DatasetItem(id=1, annotations=[Label(label=idx) for idx in range(class_count)])],
            categories=[f"label_{idx}" for idx in range(class_count)],
        )
        return dataset_1, dataset_2

    @pytest.mark.parametrize("output_format", ["simple", "tensorboard"])
    def test_save(self, fxt_dataset_pair, test_dir, output_format):
        mock_dist_comparator = MagicMock(spec=DistanceComparator)

        # matches, a_unmatched, b_unmatched = label_diff
        mock_dist_comparator.match_labels.return_value = ([0], [1], [2])
        mock_dist_comparator.match_boxes.return_value = ([], [], [], [])
        mock_dist_comparator.match_polygons.return_value = ([], [], [], [])
        mock_dist_comparator.match_masks.return_value = ([], [], [], [])

        with DistanceCompareVisualizer(
            save_dir=test_dir,
            comparator=mock_dist_comparator,
            output_format=output_format,
        ) as visualizer:
            first_dataset, second_dataset = fxt_dataset_pair
            visualizer.save(first_dataset, second_dataset)

            # Assert non-empty after save()
            assert os.listdir(test_dir)
