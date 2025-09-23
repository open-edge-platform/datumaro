# Copyright (C) 2023-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from textwrap import wrap
from typing import Dict, List, Set, Tuple
from unittest import TestCase

from attr import attrib, attrs

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.util.os_util import generate_next_name


def generate_next_file_name(basename, basedir=".", sep=".", ext=""):
    """Generate next file name using the core utility function."""
    return generate_next_name(os.listdir(basedir), basename, sep, ext)


from datumaro.components.annotations.matcher import LineMatcher, PointsMatcher, match_segments_pair
from datumaro.components.dataset import Dataset
from datumaro.components.operations import (
    compute_ann_statistics,
    compute_image_statistics,
    match_items_by_id,
    match_items_by_image_hash,
)
from datumaro.util import dump_json_file, filter_dict, find
from datumaro.util.annotation_util import find_instances, max_bbox
from datumaro.util.attrs_util import default_if_none


@attrs
class DistanceComparator:
    iou_threshold = attrib(converter=float, default=0.5)

    def match_annotations(self, item_a, item_b):
        return {t: self._match_ann_type(t, item_a, item_b) for t in AnnotationType}

    def _match_ann_type(self, t, *args):
        # pylint: disable=no-value-for-parameter
        if t == AnnotationType.label:
            return self.match_labels(*args)
        elif t == AnnotationType.bbox:
            return self.match_boxes(*args)
        elif t == AnnotationType.polygon:
            return self.match_polygons(*args)
        elif t == AnnotationType.mask:
            return self.match_masks(*args)
        elif t == AnnotationType.points:
            return self.match_points(*args)
        elif t == AnnotationType.polyline:
            return self.match_lines(*args)
        # pylint: enable=no-value-for-parameter
        else:
            raise NotImplementedError("Unexpected annotation type %s" % t)

    @staticmethod
    def _get_ann_type(t, item):
        return [a for a in item.annotations if a.type == t]

    def match_labels(self, item_a, item_b):
        a_labels = set(a.label for a in self._get_ann_type(AnnotationType.label, item_a))
        b_labels = set(a.label for a in self._get_ann_type(AnnotationType.label, item_b))

        matches = a_labels & b_labels
        a_unmatched = a_labels - b_labels
        b_unmatched = b_labels - a_labels
        return matches, a_unmatched, b_unmatched

    def _match_segments(self, t, item_a, item_b):
        a_boxes = self._get_ann_type(t, item_a)
        b_boxes = self._get_ann_type(t, item_b)
        return match_segments_pair(a_boxes, b_boxes, dist_thresh=self.iou_threshold)

    def match_polygons(self, item_a, item_b):
        return self._match_segments(AnnotationType.polygon, item_a, item_b)

    def match_masks(self, item_a, item_b):
        return self._match_segments(AnnotationType.mask, item_a, item_b)

    def match_boxes(self, item_a, item_b):
        return self._match_segments(AnnotationType.bbox, item_a, item_b)

    def match_points(self, item_a, item_b):
        a_points = self._get_ann_type(AnnotationType.points, item_a)
        b_points = self._get_ann_type(AnnotationType.points, item_b)

        instance_map = {}
        for s in [item_a.annotations, item_b.annotations]:
            s_instances = find_instances(s)
            for inst in s_instances:
                inst_bbox = max_bbox(inst)
                for ann in inst:
                    instance_map[id(ann)] = [inst, inst_bbox]
        matcher = PointsMatcher(instance_map=instance_map)

        return match_segments_pair(
            a_points, b_points, dist_thresh=self.iou_threshold, distance=matcher.distance
        )

    def match_lines(self, item_a, item_b):
        a_lines = self._get_ann_type(AnnotationType.polyline, item_a)
        b_lines = self._get_ann_type(AnnotationType.polyline, item_b)

        matcher = LineMatcher()

        return match_segments_pair(
            a_lines, b_lines, dist_thresh=self.iou_threshold, distance=matcher.distance
        )


@attrs
class EqualityComparator:
    match_images: bool = attrib(kw_only=True, default=False)
    ignored_fields = attrib(kw_only=True, factory=set, validator=default_if_none(set))
    ignored_attrs = attrib(kw_only=True, factory=set, validator=default_if_none(set))
    ignored_item_attrs = attrib(kw_only=True, factory=set, validator=default_if_none(set))
    all = attrib(kw_only=True, default=False)

    _test: TestCase = attrib(init=False)
    errors: list = attrib(init=False)

    def __attrs_post_init__(self):
        self._test = TestCase()
        self._test.maxDiff = None

    def _match_items(self, a, b):
        if self.match_images:
            return match_items_by_image_hash(a, b)
        else:
            return match_items_by_id(a, b)

    def _compare_categories(self, a, b):
        test = self._test
        errors = self.errors

        try:
            test.assertEqual(sorted(a, key=lambda t: t.value), sorted(b, key=lambda t: t.value))
        except AssertionError as e:
            errors.append({"type": "categories", "message": str(e)})

        if AnnotationType.label in a:
            try:
                test.assertEqual(
                    a[AnnotationType.label].items,
                    b[AnnotationType.label].items,
                )
            except AssertionError as e:
                errors.append({"type": "labels", "message": str(e)})
        if AnnotationType.mask in a:
            try:
                test.assertEqual(
                    a[AnnotationType.mask].colormap,
                    b[AnnotationType.mask].colormap,
                )
            except AssertionError as e:
                errors.append({"type": "colormap", "message": str(e)})
        if AnnotationType.points in a:
            try:
                test.assertEqual(
                    a[AnnotationType.points].items,
                    b[AnnotationType.points].items,
                )
            except AssertionError as e:
                errors.append({"type": "points", "message": str(e)})

    def _compare_annotations(self, a, b):
        ignored_fields = self.ignored_fields
        ignored_attrs = self.ignored_attrs

        a_fields = {k: None for k in a.as_dict() if k in ignored_fields}
        b_fields = {k: None for k in b.as_dict() if k in ignored_fields}
        if "attributes" not in ignored_fields:
            a_fields["attributes"] = filter_dict(a.attributes, ignored_attrs)
            b_fields["attributes"] = filter_dict(b.attributes, ignored_attrs)

        result = a.wrap(**a_fields) == b.wrap(**b_fields)

        return result

    def _compare_items(self, item_a, item_b):
        test = self._test

        a_id = (item_a.id, item_a.subset)
        b_id = (item_b.id, item_b.subset)

        matched = []
        unmatched = []
        errors = []

        try:
            test.assertEqual(
                filter_dict(item_a.attributes, self.ignored_item_attrs),
                filter_dict(item_b.attributes, self.ignored_item_attrs),
            )
        except AssertionError as e:
            errors.append({"type": "item_attr", "a_item": a_id, "b_item": b_id, "message": str(e)})

        b_annotations = item_b.annotations[:]
        for ann_a in item_a.annotations:
            ann_b_candidates = [x for x in item_b.annotations if x.type == ann_a.type]

            ann_b = find(
                enumerate(self._compare_annotations(ann_a, x) for x in ann_b_candidates),
                lambda x: x[1],
            )
            if ann_b is None:
                unmatched.append(
                    {
                        "item": a_id,
                        "source": "a",
                        "ann": str(ann_a),
                    }
                )
                continue
            else:
                ann_b = ann_b_candidates[ann_b[0]]

            b_annotations.remove(ann_b)  # avoid repeats
            matched.append({"a_item": a_id, "b_item": b_id, "a": str(ann_a), "b": str(ann_b)})

        for ann_b in b_annotations:
            unmatched.append({"item": b_id, "source": "b", "ann": str(ann_b)})

        return matched, unmatched, errors

    @staticmethod
    def _print_output(output: dict):
        print("Found:")
        print("The first project has %s unmatched items" % len(output.get("a_extra_items", [])))
        print("The second project has %s unmatched items" % len(output.get("b_extra_items", [])))
        print("%s item conflicts" % len(output.get("errors", [])))
        print("%s matching annotations" % len(output.get("matches", [])))
        print("%s mismatching annotations" % len(output.get("mismatches", [])))

    def compare_datasets(self, a, b):
        self.errors = []
        errors = self.errors

        self._compare_categories(a.categories(), b.categories())

        matched = []
        unmatched = []

        matches, a_unmatched, b_unmatched = self._match_items(a, b)

        if a.categories().get(AnnotationType.label) != b.categories().get(AnnotationType.label):
            output = {
                "mismatches": unmatched,
                "a_extra_items": sorted(a_unmatched),
                "b_extra_items": sorted(b_unmatched),
                "errors": errors,
            }
            if self.all:
                output["matches"] = matched

            self._print_output(output)
            return output

        _dist = lambda s: len(s[1]) + len(s[2])
        for a_ids, b_ids in matches:
            # build distance matrix
            match_status = {}  # (a_id, b_id): [matched, unmatched, errors]
            a_matches = {a_id: None for a_id in a_ids}
            b_matches = {b_id: None for b_id in b_ids}

            for a_id in a_ids:
                item_a = a.get(*a_id)
                candidates = {}

                for b_id in b_ids:
                    item_b = b.get(*b_id)

                    i_m, i_um, i_err = self._compare_items(item_a, item_b)
                    candidates[b_id] = [i_m, i_um, i_err]

                    if len(i_um) == 0:
                        a_matches[a_id] = b_id
                        b_matches[b_id] = a_id
                        matched.extend(i_m)
                        errors.extend(i_err)
                        break

                match_status[a_id] = candidates

            # assign
            for a_id in a_ids:
                if len(b_ids) == 0:
                    break

                # find the closest, ignore already assigned
                matched_b = a_matches[a_id]
                if matched_b is not None:
                    continue
                min_dist = -1
                for b_id in b_ids:
                    if b_matches[b_id] is not None:
                        continue
                    d = _dist(match_status[a_id][b_id])
                    if d < min_dist and 0 <= min_dist:
                        continue
                    min_dist = d
                    matched_b = b_id

                if matched_b is None:
                    continue
                a_matches[a_id] = matched_b
                b_matches[matched_b] = a_id

                m = match_status[a_id][matched_b]
                matched.extend(m[0])
                unmatched.extend(m[1])
                errors.extend(m[2])

            a_unmatched |= set(a_id for a_id, m in a_matches.items() if not m)
            b_unmatched |= set(b_id for b_id, m in b_matches.items() if not m)

        output = {
            "mismatches": unmatched,
            "a_extra_items": sorted(a_unmatched),
            "b_extra_items": sorted(b_unmatched),
            "errors": errors,
        }
        if self.all:
            output["matches"] = matched
        self._print_output(output)
        return output

    @staticmethod
    def save_compare_report(
        output: Dict,
        report_dir: str,
    ) -> None:
        """Saves the comparison report to JSON and text files.

        Args:
            output: A dictionary containing the comparison data.
            report_dir: A string representing the directory to save the report files.
        """
        os.makedirs(report_dir, exist_ok=True)
        output_file = osp.join(
            report_dir,
            generate_next_file_name("equality_compare", ext=".json", basedir=report_dir),
        )

        log.info(f"Saving compare json to {output_file}")
        dump_json_file(output_file, output, indent=True)


@attrs
class TableComparator:
    """
    Class for comparing datasets and generating comparison report table.
    """

    @staticmethod
    def _extract_labels(dataset: Dataset) -> Set[str]:
        """Extracts labels from the dataset.

        Args:
            dataset: An instance of a Dataset class.

        Returns:
            A set of labels present in the dataset.
        """
        label_cat = dataset.categories().get(AnnotationType.label, LabelCategories())
        return set(c.name for c in label_cat)

    @staticmethod
    def _compute_statistics(dataset: Dataset) -> Tuple[Dict, Dict]:
        """Computes image and annotation statistics of the dataset.

        Args:
            dataset: An instance of a Dataset class.

        Returns:
            A tuple containing image statistics and annotation statistics.
        """
        image_stats = compute_image_statistics(dataset)
        ann_stats = compute_ann_statistics(dataset)
        return image_stats, ann_stats

    def _analyze_dataset(self, dataset: Dataset) -> Tuple[str, Set[str], Dict, Dict]:
        """Analyzes the dataset to get labels, format, and statistics.

        Args:
            dataset: An instance of a Dataset class.

        Returns:
            A tuple containing Dataset format, set of label names, image statistics,
            and annotation statistics.
        """
        dataset_format = dataset.format
        dataset_labels = self._extract_labels(dataset)
        image_stats, ann_stats = self._compute_statistics(dataset)
        return dataset_format, dataset_labels, image_stats, ann_stats

    @staticmethod
    def _create_table(headers: List[str], rows: List[List[str]]) -> str:
        """Creates a table with the given headers and rows using the tabulate module.

        Args:
            headers: A list containing table headers.
            rows: A list containing table rows.

        Returns:
            A string representation of the table.
        """

        def wrapfunc(item):
            """Wrap a item consisted of text, returning a list of wrapped lines."""
            max_len = 35
            return "\n".join(wrap(item, max_len))

        wrapped_rows = []
        for row in rows:
            new_row = [wrapfunc(item) for item in row]
            wrapped_rows.append(new_row)

        from tabulate import tabulate

        return tabulate(wrapped_rows, headers, tablefmt="grid")

    @staticmethod
    def _create_dict(rows: List[List[str]]) -> Dict[str, List[str]]:
        """Creates a dictionary from the rows of the table.

        Args:
            rows: A list containing table rows.

        Returns:
            A dictionary where the key is the first element of a row and the value is
            the rest of the row.
        """
        data_dict = {row[0]: row[1:] for row in rows[1:]}
        return data_dict

    def _create_high_level_comparison_table(
        self, first_info: Tuple, second_info: Tuple
    ) -> Tuple[str, Dict]:
        """Generates a high-level comparison table.

        Args:
            first_info: A tuple containing information about the first dataset.
            second_info: A tuple containing information about the second dataset.

        Returns:
            A tuple containing the table as a string and a dictionary representing the data
            of the table.
        """
        first_format, first_labels, first_image_stats, first_ann_stats = first_info
        second_format, second_labels, second_image_stats, second_ann_stats = second_info

        headers = ["Field", "First", "Second"]

        rows = [
            ["Format", first_format, second_format],
            ["Number of classes", str(len(first_labels)), str(len(second_labels))],
            [
                "Common classes",
                ", ".join(sorted(list(first_labels.intersection(second_labels)))),
                ", ".join(sorted(list(second_labels.intersection(first_labels)))),
            ],
            ["Classes", ", ".join(sorted(first_labels)), ", ".join(sorted(second_labels))],
            [
                "Images count",
                str(first_image_stats["dataset"]["images count"]),
                str(second_image_stats["dataset"]["images count"]),
            ],
            [
                "Unique images count",
                str(first_image_stats["dataset"]["unique images count"]),
                str(second_image_stats["dataset"]["unique images count"]),
            ],
            [
                "Repeated images count",
                str(first_image_stats["dataset"]["repeated images count"]),
                str(second_image_stats["dataset"]["repeated images count"]),
            ],
            [
                "Annotations count",
                str(first_ann_stats["annotations count"]),
                str(second_ann_stats["annotations count"]),
            ],
            [
                "Unannotated images count",
                str(first_ann_stats["unannotated images count"]),
                str(second_ann_stats["unannotated images count"]),
            ],
        ]

        table = self._create_table(headers, rows)
        data_dict = self._create_dict(rows)

        return table, data_dict

    def _create_mid_level_comparison_table(
        self, first_info: Tuple, second_info: Tuple
    ) -> Tuple[str, Dict]:
        """Generates a mid-level comparison table.

        Args:
            first_info: A tuple containing information about the first dataset.
            second_info: A tuple containing information about the second dataset.

        Returns:
            A tuple containing the table as a string and a dictionary representing the data
            of the table.
        """
        _, _, first_image_stats, first_ann_stats = first_info
        _, _, second_image_stats, second_ann_stats = second_info

        headers = ["Field", "First", "Second"]

        rows = []

        first_subsets = sorted(list(first_image_stats["subsets"].keys()))
        second_subsets = sorted(list(second_image_stats["subsets"].keys()))

        subset_names = first_subsets.copy()
        subset_names.extend(item for item in second_subsets if item not in first_subsets)

        for subset_name in subset_names:
            first_subset_data = first_image_stats["subsets"].get(subset_name, {})
            second_subset_data = second_image_stats["subsets"].get(subset_name, {})

            image_mean_array_first = first_subset_data.get("image mean (RGB)", [])
            mean_str_first = (
                ", ".join(f"{val:<6.2f}" for val in image_mean_array_first)
                if image_mean_array_first != "n/a"
                and isinstance(image_mean_array_first, list)
                and len(image_mean_array_first) > 0
                else ""
            )

            image_std_array_first = first_subset_data.get("image std (RGB)", [])
            std_str_first = (
                ", ".join(f"{val:6.2f}" for val in image_std_array_first)
                if image_std_array_first != "n/a"
                and isinstance(image_std_array_first, list)
                and len(image_std_array_first) > 0
                else ""
            )
            image_mean_array_second = second_subset_data.get("image mean (RGB)", [])
            mean_str_second = (
                ", ".join(f"{val:6.2f}" for val in image_mean_array_second)
                if image_mean_array_second != "n/a"
                and isinstance(image_mean_array_second, list)
                and len(image_mean_array_second) > 0
                else ""
            )

            image_std_array_second = second_subset_data.get("image std (RGB)", [])
            std_str_second = (
                ", ".join(f"{val:6.2f}" for val in image_std_array_second)
                if image_std_array_second != "n/a"
                and isinstance(image_std_array_second, list)
                and len(image_std_array_second) > 0
                else ""
            )
            rows.append([f"{subset_name} - Image Mean (RGB)", mean_str_first, mean_str_second])
            rows.append([f"{subset_name} - Image Std (RGB)", std_str_first, std_str_second])

        first_labels = sorted(list(first_ann_stats["annotations"]["labels"]["distribution"].keys()))
        second_labels = sorted(
            list(second_ann_stats["annotations"]["labels"]["distribution"].keys())
        )

        label_names = first_labels.copy()
        label_names.extend(item for item in second_labels if item not in first_labels)

        for label_name in label_names:
            count_dist_first = first_ann_stats["annotations"]["labels"]["distribution"].get(
                label_name, [0, 0.0]
            )
            count_dist_second = second_ann_stats["annotations"]["labels"]["distribution"].get(
                label_name, [0, 0.0]
            )
            count_first, dist_first = count_dist_first if count_dist_first[0] != 0 else ["", ""]
            count_second, dist_second = count_dist_second if count_dist_second[0] != 0 else ["", ""]
            rows.append(
                [
                    f"Label - {label_name}",
                    f"imgs: {count_first}, percent: {dist_first:.4f}" if count_first != "" else "",
                    f"imgs: {count_second}, percent: {dist_second:.4f}"
                    if count_second != ""
                    else "",
                ]
            )

        table = self._create_table(headers, rows)
        data_dict = self._create_dict(rows)

        return table, data_dict

    def compare_datasets(
        self, first: Dataset, second: Dataset, mode: str = "all"
    ) -> Tuple[str, str, Dict]:
        """Compares two datasets and generates comparison reports.

        Args:
            first: The first dataset to compare.
            second: The second dataset to compare.

        Returns:
            A tuple containing high-level table, mid-level table, low-level table, and a
            dictionary representation of the comparison.
        """
        first_info = self._analyze_dataset(first)
        second_info = self._analyze_dataset(second)

        high_level_table, high_level_dict = None, {}
        mid_level_table, mid_level_dict = None, {}

        if mode in ["high", "all"]:
            high_level_table, high_level_dict = self._create_high_level_comparison_table(
                first_info, second_info
            )
        if mode in ["mid", "all"]:
            mid_level_table, mid_level_dict = self._create_mid_level_comparison_table(
                first_info, second_info
            )

        comparison_dict = dict(high_level=high_level_dict, mid_level=mid_level_dict)

        print(f"High-level comparison:\n{high_level_table}\n")
        print(f"Mid-level comparison:\n{mid_level_table}\n")

        return high_level_table, mid_level_table, comparison_dict

    @staticmethod
    def save_compare_report(
        high_level_table: str,
        mid_level_table: str,
        comparison_dict: Dict,
        report_dir: str,
    ) -> None:
        """Saves the comparison report to JSON and text files.

        Args:
            high_level_table: High-level comparison table as a string.
            mid_level_table: Mid-level comparison table as a string.
            comparison_dict: A dictionary containing the comparison data.
            report_dir: A string representing the directory to save the report files.
        """
        os.makedirs(report_dir, exist_ok=True)
        json_output_file = osp.join(
            report_dir, generate_next_file_name("table_compare", ext=".json", basedir=report_dir)
        )
        txt_output_file = osp.join(
            report_dir, generate_next_file_name("table_compare", ext=".txt", basedir=report_dir)
        )

        log.info(f"Saving compare json to {json_output_file}")
        log.info(f"Saving compare table to {txt_output_file}")

        dump_json_file(json_output_file, comparison_dict, indent=True)
        with open(txt_output_file, "w") as f:
            f.write(f"High-level Comparison:\n{high_level_table}\n\n")
            f.write(f"Mid-level Comparison:\n{mid_level_table}\n\n")
