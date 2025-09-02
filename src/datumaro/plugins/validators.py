# Copyright (C) 2021-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from copy import deepcopy

import numpy as np

from datumaro.components.annotation import (
    AnnotationType,
    GroupType,
    LabelCategories,
    TabularCategories,
)
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.errors import (
    AttributeDefinedButNotFound,
    BrokenAnnotation,
    EmptyCaption,
    EmptyLabel,
    FarFromAttrMean,
    FarFromCaptionMean,
    FarFromLabelMean,
    FewSamplesInAttribute,
    FewSamplesInCaption,
    FewSamplesInLabel,
    ImbalancedAttribute,
    ImbalancedCaptions,
    ImbalancedDistInAttribute,
    ImbalancedDistInCaption,
    ImbalancedDistInLabel,
    ImbalancedLabels,
    InvalidValue,
    LabelDefinedButNotFound,
    MissingAnnotation,
    MissingAttribute,
    MissingLabelCategories,
    MultiLabelAnnotations,
    NegativeLength,
    OnlyOneAttributeValue,
    OnlyOneLabel,
    OutlierInCaption,
    RedundanciesInCaption,
    UndefinedAttribute,
    UndefinedLabel,
)
from datumaro.components.validator import Severity, TaskType, Validator
from datumaro.util import parse_str_enum_value
from datumaro.util.tabular_util import emoji_pattern

DEFAULT_LABEL_GROUP = "default"


class _TaskValidator(Validator, CliPlugin):
    DEFAULT_FEW_SAMPLES_THR = 1
    DEFAULT_IMBALANCE_RATIO_THR = 50
    DEFAULT_FAR_FROM_MEAN_THR = 5
    DEFAULT_DOMINANCE_RATIO_THR = 0.8
    DEFAULT_TOPK_BINS = 0.1

    # statistics templates
    numerical_stat_template = {
        "items_far_from_mean": {},
        "mean": None,
        "stdev": None,
        "min": None,
        "max": None,
        "median": None,
        "histogram": {
            "bins": [],
            "counts": [],
        },
        "distribution": [],
    }

    """
    A base class for task-specific validators.

    Attributes
    ----------
    task_type : str or TaskType
        task type (ie. classification, detection, segmentation, tabular)
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-fs",
            "--few-samples-thr",
            default=cls.DEFAULT_FEW_SAMPLES_THR,
            type=int,
            help="Threshold for giving a warning for minimum number of "
            "samples per class (default: %(default)s)",
        )
        parser.add_argument(
            "-ir",
            "--imbalance-ratio-thr",
            default=cls.DEFAULT_IMBALANCE_RATIO_THR,
            type=int,
            help="Threshold for giving data imbalance warning. "
            "IR(imbalance ratio) = majority/minority "
            "(default: %(default)s)",
        )
        parser.add_argument(
            "-m",
            "--far-from-mean-thr",
            default=cls.DEFAULT_FAR_FROM_MEAN_THR,
            type=float,
            help="Threshold for giving a warning that data is far from mean. "
            "A constant used to define mean +/- k * standard deviation "
            "(default: %(default)s)",
        )
        parser.add_argument(
            "-dr",
            "--dominance-ratio-thr",
            default=cls.DEFAULT_DOMINANCE_RATIO_THR,
            type=float,
            help="Threshold for giving a warning for bounding box imbalance. "
            "Dominace_ratio = ratio of Top-k bin to total in histogram "
            "(default: %(default)s)",
        )
        parser.add_argument(
            "-k",
            "--topk-bins",
            default=cls.DEFAULT_TOPK_BINS,
            type=float,
            help="Ratio of bins with the highest number of data "
            "to total bins in the histogram. A value in the range [0, 1] "
            "(default: %(default)s)",
        )
        return parser

    def __init__(
        self,
        task_type,
        few_samples_thr=None,
        imbalance_ratio_thr=None,
        far_from_mean_thr=None,
        dominance_ratio_thr=None,
        topk_bins=None,
    ):
        """
        Validator

        Parameters
        ---------------
        few_samples_thr: int
            minimum number of samples per class
            warn user when samples per class is less than threshold
        imbalance_ratio_thr: int
            ratio of majority attribute to minority attribute
            warn user when annotations are unevenly distributed
        far_from_mean_thr: float
            constant used to define mean +/- m * stddev
            warn user when there are too big or small values
        dominance_ratio_thr: float
            ratio of Top-k bin to total
            warn user when dominance ratio is over threshold
        topk_bins: float
            ratio of selected bins with most item number to total bins
            warn user when values are not evenly distributed
        """
        self.task_type = parse_str_enum_value(task_type, TaskType, default=TaskType.classification)

        if self.task_type == TaskType.classification:
            self.ann_types = {AnnotationType.label}
            self.str_ann_type = "label"
        elif self.task_type == TaskType.detection:
            self.ann_types = {AnnotationType.bbox}
            self.str_ann_type = "bounding box"
        elif self.task_type == TaskType.segmentation:
            self.ann_types = {AnnotationType.mask, AnnotationType.polygon, AnnotationType.ellipse}
            self.str_ann_type = "mask or polygon or ellipse"
        elif self.task_type == TaskType.tabular:
            self.ann_types = {AnnotationType.label, AnnotationType.caption}
            self.str_ann_type = "label or caption"

        if few_samples_thr is None:
            few_samples_thr = self.DEFAULT_FEW_SAMPLES_THR

        if imbalance_ratio_thr is None:
            imbalance_ratio_thr = self.DEFAULT_IMBALANCE_RATIO_THR

        if far_from_mean_thr is None:
            far_from_mean_thr = self.DEFAULT_FAR_FROM_MEAN_THR

        if dominance_ratio_thr is None:
            dominance_ratio_thr = self.DEFAULT_DOMINANCE_RATIO_THR

        if topk_bins is None:
            topk_bins = self.DEFAULT_TOPK_BINS

        self.few_samples_thr = few_samples_thr
        self.imbalance_ratio_thr = imbalance_ratio_thr
        self.far_from_mean_thr = far_from_mean_thr
        self.dominance_thr = dominance_ratio_thr
        self.topk_bins_ratio = topk_bins

    def _compute_common_statistics(self, dataset):
        defined_attr_template = {"items_missing_attribute": [], "distribution": {}}
        undefined_attr_template = {"items_with_undefined_attr": [], "distribution": {}}
        undefined_label_template = {
            "count": 0,
            "items_with_undefined_label": [],
        }

        stats = {
            "label_distribution": {
                "defined_labels": {},
                "undefined_labels": {},
            },
            "attribute_distribution": {"defined_attributes": {}, "undefined_attributes": {}},
        }
        stats["total_ann_count"] = 0
        stats["items_missing_annotation"] = []

        label_dist = stats["label_distribution"]
        defined_label_dist = label_dist["defined_labels"]
        undefined_label_dist = label_dist["undefined_labels"]

        attr_dist = stats["attribute_distribution"]
        defined_attr_dist = attr_dist["defined_attributes"]
        undefined_attr_dist = attr_dist["undefined_attributes"]

        label_categories = dataset.categories().get(AnnotationType.label, LabelCategories())
        base_valid_attrs = label_categories.attributes

        for category in label_categories:
            defined_label_dist[category.name] = 0

        filtered_anns = []
        for item in dataset:
            item_key = (item.id, item.subset)
            annotations = []
            for ann in item.annotations:
                if ann.type in self.ann_types:
                    annotations.append(ann)
            ann_count = len(annotations)
            filtered_anns.append((item_key, annotations))

            if ann_count == 0:
                stats["items_missing_annotation"].append(item_key)
            stats["total_ann_count"] += ann_count

            for ann in annotations:
                if not 0 <= ann.label < len(label_categories):
                    label_name = ann.label

                    label_stats = undefined_label_dist.setdefault(
                        ann.label, deepcopy(undefined_label_template)
                    )
                    label_stats["items_with_undefined_label"].append(item_key)

                    label_stats["count"] += 1
                    valid_attrs = set()
                    missing_attrs = set()
                else:
                    label_name = label_categories[ann.label].name
                    defined_label_dist[label_name] += 1

                    defined_attr_stats = defined_attr_dist.setdefault(label_name, {})

                    valid_attrs = base_valid_attrs.union(label_categories[ann.label].attributes)
                    ann_attrs = getattr(ann, "attributes", {}).keys()
                    missing_attrs = valid_attrs.difference(ann_attrs)

                    for attr in valid_attrs:
                        defined_attr_stats.setdefault(attr, deepcopy(defined_attr_template))

                for attr in missing_attrs:
                    attr_dets = defined_attr_stats[attr]
                    attr_dets["items_missing_attribute"].append(item_key)

                for attr, value in ann.attributes.items():
                    if attr not in valid_attrs:
                        undefined_attr_stats = undefined_attr_dist.setdefault(label_name, {})
                        attr_dets = undefined_attr_stats.setdefault(
                            attr, deepcopy(undefined_attr_template)
                        )
                        attr_dets["items_with_undefined_attr"].append(item_key)
                    else:
                        attr_dets = defined_attr_stats[attr]

                    attr_dets["distribution"].setdefault(str(value), 0)
                    attr_dets["distribution"][str(value)] += 1

        return stats, filtered_anns

    def _generate_common_reports(self, stats):
        """
        Validates the dataset for classification tasks based on its statistics.

        Parameters
        ----------
        dataset : IDataset object
        stats: Dict object

        Returns
        -------
        reports (list): List of validation reports (DatasetValidationError).
        """

        reports = []

        # report for dataset
        reports += self._check_missing_label_categories(stats)

        # report for item
        reports += self._check_missing_annotation(stats)

        # report for label
        reports += self._check_undefined_label(stats)
        reports += self._check_label_defined_but_not_found(stats)
        reports += self._check_only_one_label(stats)
        reports += self._check_few_samples_in_label(stats)
        reports += self._check_imbalanced_labels(stats)

        # report for attributes
        attr_dist = stats["attribute_distribution"]
        defined_attr_dist = attr_dist["defined_attributes"]
        undefined_attr_dist = attr_dist["undefined_attributes"]

        defined_labels = defined_attr_dist.keys()
        for label_name in defined_labels:
            attr_stats = defined_attr_dist[label_name]

            reports += self._check_attribute_defined_but_not_found(label_name, attr_stats)

            for attr_name, attr_dets in attr_stats.items():
                reports += self._check_missing_attribute(label_name, attr_name, attr_dets)
                reports += self._check_only_one_attribute(label_name, attr_name, attr_dets)
                reports += self._check_few_samples_in_attribute(label_name, attr_name, attr_dets)
                reports += self._check_imbalanced_attribute(label_name, attr_name, attr_dets)

        for label_name, attr_stats in undefined_attr_dist.items():
            for attr_name, attr_dets in attr_stats.items():
                reports += self._check_undefined_attribute(label_name, attr_name, attr_dets)

        return reports

    def _generate_validation_report(self, error, *args, **kwargs):
        return [error(*args, **kwargs)]

    def _check_missing_label_categories(self, stats):
        validation_reports = []

        if len(stats["label_distribution"]["defined_labels"]) == 0:
            validation_reports += self._generate_validation_report(
                MissingLabelCategories, Severity.error
            )

        return validation_reports

    def _check_missing_annotation(self, stats):
        validation_reports = []

        items_missing = stats["items_missing_annotation"]
        for item_id, item_subset in items_missing:
            validation_reports += self._generate_validation_report(
                MissingAnnotation, Severity.warning, item_id, item_subset, self.str_ann_type
            )

        return validation_reports

    def _check_missing_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []

        items_missing_attr = attr_dets["items_missing_attribute"]
        for item_id, item_subset in items_missing_attr:
            details = (item_subset, label_name, attr_name)
            validation_reports += self._generate_validation_report(
                MissingAttribute, Severity.warning, item_id, *details
            )

        return validation_reports

    def _check_undefined_label(self, stats):
        validation_reports = []

        undefined_label_dist = stats["label_distribution"]["undefined_labels"]
        for label_name, label_stats in undefined_label_dist.items():
            for item_id, item_subset in label_stats["items_with_undefined_label"]:
                details = (item_subset, label_name)
                validation_reports += self._generate_validation_report(
                    UndefinedLabel, Severity.error, item_id, *details
                )

        return validation_reports

    def _check_undefined_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []

        items_with_undefined_attr = attr_dets["items_with_undefined_attr"]
        for item_id, item_subset in items_with_undefined_attr:
            details = (item_subset, label_name, attr_name)
            validation_reports += self._generate_validation_report(
                UndefinedAttribute, Severity.error, item_id, *details
            )

        return validation_reports

    def _check_label_defined_but_not_found(self, stats):
        validation_reports = []
        count_by_defined_labels = stats["label_distribution"]["defined_labels"]
        labels_not_found = [
            label_name for label_name, count in count_by_defined_labels.items() if count == 0
        ]

        for label_name in labels_not_found:
            validation_reports += self._generate_validation_report(
                LabelDefinedButNotFound, Severity.warning, label_name
            )

        return validation_reports

    def _check_attribute_defined_but_not_found(self, label_name, attr_stats):
        validation_reports = []
        attrs_not_found = [
            attr_name
            for attr_name, attr_dets in attr_stats.items()
            if len(attr_dets["distribution"]) == 0
        ]

        for attr_name in attrs_not_found:
            details = (label_name, attr_name)
            validation_reports += self._generate_validation_report(
                AttributeDefinedButNotFound, Severity.warning, *details
            )

        return validation_reports

    def _check_only_one_label(self, stats):
        validation_reports = []
        count_by_defined_labels = stats["label_distribution"]["defined_labels"]
        labels_found = [
            label_name for label_name, count in count_by_defined_labels.items() if count > 0
        ]

        if len(labels_found) == 1:
            validation_reports += self._generate_validation_report(
                OnlyOneLabel, Severity.info, labels_found[0]
            )

        return validation_reports

    def _check_only_one_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []
        values = list(attr_dets["distribution"].keys())

        if len(values) == 1:
            details = (label_name, attr_name, values[0])
            validation_reports += self._generate_validation_report(
                OnlyOneAttributeValue, Severity.info, *details
            )

        return validation_reports

    def _check_few_samples_in_label(self, stats):
        validation_reports = []
        thr = self.few_samples_thr

        defined_label_dist = stats["label_distribution"]["defined_labels"]
        labels_with_few_samples = [
            (label_name, count)
            for label_name, count in defined_label_dist.items()
            if 0 < count <= thr
        ]

        for label_name, count in labels_with_few_samples:
            validation_reports += self._generate_validation_report(
                FewSamplesInLabel, Severity.info, label_name, count
            )

        return validation_reports

    def _check_few_samples_in_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []
        thr = self.few_samples_thr

        attr_values_with_few_samples = [
            (attr_value, count)
            for attr_value, count in attr_dets["distribution"].items()
            if count <= thr
        ]

        for attr_value, count in attr_values_with_few_samples:
            details = (label_name, attr_name, attr_value, count)
            validation_reports += self._generate_validation_report(
                FewSamplesInAttribute, Severity.info, *details
            )

        return validation_reports

    def _check_imbalanced_labels(self, stats):
        validation_reports = []
        thr = self.imbalance_ratio_thr

        defined_label_dist = stats["label_distribution"]["defined_labels"]
        count_by_defined_labels = [count for label, count in defined_label_dist.items()]

        if len(count_by_defined_labels) == 0:
            return validation_reports

        count_max = np.max(count_by_defined_labels)
        count_min = np.min(count_by_defined_labels)
        balance = count_max / count_min if count_min > 0 else float("inf")
        if balance >= thr:
            validation_reports += self._generate_validation_report(ImbalancedLabels, Severity.info)

        return validation_reports

    def _check_imbalanced_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []
        thr = self.imbalance_ratio_thr

        count_by_defined_attr = list(attr_dets["distribution"].values())
        if len(count_by_defined_attr) == 0:
            return validation_reports

        count_max = np.max(count_by_defined_attr)
        count_min = np.min(count_by_defined_attr)
        balance = count_max / count_min if count_min > 0 else float("inf")
        if balance >= thr:
            validation_reports += self._generate_validation_report(
                ImbalancedAttribute, Severity.info, label_name, attr_name
            )

        return validation_reports


class ClassificationValidator(_TaskValidator):
    """
    A specific validator class for classification task.
    """

    def __init__(
        self,
        task_type=TaskType.classification,
        few_samples_thr=None,
        imbalance_ratio_thr=None,
        far_from_mean_thr=None,
        dominance_ratio_thr=None,
        topk_bins=None,
    ):
        super().__init__(
            task_type=task_type,
            few_samples_thr=few_samples_thr,
            imbalance_ratio_thr=imbalance_ratio_thr,
            far_from_mean_thr=far_from_mean_thr,
            dominance_ratio_thr=dominance_ratio_thr,
            topk_bins=topk_bins,
        )

    def compute_statistics(self, dataset):
        """
        Computes statistics of the dataset for the classification task.

        Parameters
        ----------
        dataset : IDataset object

        Returns
        -------
        stats (dict): A dict object containing statistics of the dataset.
        """

        stats, filtered_anns = self._compute_common_statistics(dataset)

        label_cat = dataset.categories()[AnnotationType.label]
        label_groups = label_cat.label_groups

        label_name_to_group = {}
        for label_group in label_groups:
            for idx, label_name in enumerate(label_group.labels):
                if label_group.group_type == GroupType.EXCLUSIVE:
                    label_name_to_group[label_name] = label_group.name
                else:
                    label_name_to_group[label_name] = label_group.name + f"_{idx}"

        undefined_label_name = list(stats["label_distribution"]["undefined_labels"].keys())

        stats["items_with_multiple_labels"] = []
        for item_key, anns in filtered_anns:
            occupied_groups = set()
            for ann in anns:
                if ann.label in undefined_label_name:
                    continue
                label_name = label_cat[ann.label].name
                label_group = label_name_to_group.get(label_name, DEFAULT_LABEL_GROUP)
                if label_group in occupied_groups:
                    stats["items_with_multiple_labels"].append(item_key)
                    break
                occupied_groups.add(label_group)

        return stats

    def generate_reports(self, stats):
        """
        Validates the dataset for classification tasks based on its statistics.

        Parameters
        ----------
        dataset : IDataset object
        stats: Dict object

        Returns
        -------
        reports (list): List of validation reports (DatasetValidationError).
        """

        reports = self._generate_common_reports(stats)
        reports += self._check_multi_label_annotations(stats)

        return reports

    def _check_multi_label_annotations(self, stats):
        validation_reports = []

        items_with_multiple_labels = stats["items_with_multiple_labels"]
        for item_id, item_subset in items_with_multiple_labels:
            validation_reports += self._generate_validation_report(
                MultiLabelAnnotations, Severity.error, item_id, item_subset
            )

        return validation_reports


class DetectionValidator(_TaskValidator):
    """
    A specific validator class for detection task.
    """

    def __init__(
        self,
        task_type=TaskType.detection,
        few_samples_thr=None,
        imbalance_ratio_thr=None,
        far_from_mean_thr=None,
        dominance_ratio_thr=None,
        topk_bins=None,
    ):
        super().__init__(
            task_type=task_type,
            few_samples_thr=few_samples_thr,
            imbalance_ratio_thr=imbalance_ratio_thr,
            far_from_mean_thr=far_from_mean_thr,
            dominance_ratio_thr=dominance_ratio_thr,
            topk_bins=topk_bins,
        )

        self.point_template = {
            "width": deepcopy(self.numerical_stat_template),
            "height": deepcopy(self.numerical_stat_template),
            "area(wxh)": deepcopy(self.numerical_stat_template),
            "ratio(w/h)": deepcopy(self.numerical_stat_template),
            "short": deepcopy(self.numerical_stat_template),
            "long": deepcopy(self.numerical_stat_template),
        }

    def compute_statistics(self, dataset):
        """
        Computes statistics of the dataset for the detection task.

        Parameters
        ----------
        dataset : IDataset object

        Returns
        -------
        stats (dict): A dict object containing statistics of the dataset.
        """

        stats, filtered_items = self._compute_common_statistics(dataset)

        stats["items_with_negative_length"] = {}
        stats["items_with_invalid_value"] = {}
        stats["point_distribution_in_label"] = {}
        stats["point_distribution_in_attribute"] = {}
        stats["point_distribution_in_dataset_item"] = {}

        self.items = filtered_items

        def _generate_ann_bbox_info(_x, _y, _w, _h, area, ratio, _short, _long):
            return {
                "x": _x,
                "y": _y,
                "width": _w,
                "height": _h,
                "area(wxh)": area,
                "ratio(w/h)": ratio,
                "short": _short,
                "long": _long,
            }

        def _update_bbox_stats_by_label(item_key, ann, bbox_label_stats):
            bbox_has_error = False

            _x, _y, _w, _h = ann.get_bbox()
            area = ann.get_area()

            if _h != 0 and _h != float("inf"):
                ratio = _w / _h
            else:
                ratio = float("nan")

            _short = _w if _w < _h else _h
            _long = _w if _w > _h else _h

            ann_bbox_info = _generate_ann_bbox_info(_x, _y, _w, _h, area, ratio, _short, _long)

            items_w_invalid_val = stats["items_with_invalid_value"]
            for prop, val in ann_bbox_info.items():
                if val == float("inf") or np.isnan(val):
                    bbox_has_error = True
                    anns_w_invalid_val = items_w_invalid_val.setdefault(item_key, {})
                    invalid_props = anns_w_invalid_val.setdefault(ann.id, [])
                    invalid_props.append(prop)

            items_w_neg_len = stats["items_with_negative_length"]
            for prop in ["width", "height"]:
                val = ann_bbox_info[prop]
                if val < 1:
                    bbox_has_error = True
                    anns_w_neg_len = items_w_neg_len.setdefault(item_key, {})
                    neg_props = anns_w_neg_len.setdefault(ann.id, {})
                    neg_props[prop] = val

            if not bbox_has_error:
                ann_bbox_info.pop("x")
                ann_bbox_info.pop("y")
                self._update_prop_distributions(ann_bbox_info, bbox_label_stats)

            return ann_bbox_info, bbox_has_error

        # Collect property distribution
        label_categories = dataset.categories().get(AnnotationType.label, LabelCategories())
        self._compute_prop_dist(label_categories, stats, _update_bbox_stats_by_label)

        # Compute property statistics from distribution
        dist_by_label = stats["point_distribution_in_label"]
        dist_by_attr = stats["point_distribution_in_attribute"]
        self._compute_prop_stats_from_dist(dist_by_label, dist_by_attr)

        def _is_valid_bbox(item_key, ann):
            has_defined_label = 0 <= ann.label < len(label_categories)
            if not has_defined_label:
                return False

            bbox_has_neg_len = ann.id in stats["items_with_negative_length"].get(item_key, {})
            bbox_has_invalid_val = ann.id in stats["items_with_invalid_value"].get(item_key, {})
            return not (bbox_has_neg_len or bbox_has_invalid_val)

        def _update_bbox_props_far_from_mean(item_key, ann):
            base_valid_attrs = label_categories.attributes
            valid_attrs = base_valid_attrs.union(label_categories[ann.label].attributes)
            label_name = label_categories[ann.label].name
            bbox_label_stats = dist_by_label[label_name]

            _x, _y, _w, _h = ann.get_bbox()
            area = ann.get_area()
            ratio = _w / _h
            _short = _w if _w < _h else _h
            _long = _w if _w > _h else _h

            ann_bbox_info = _generate_ann_bbox_info(_x, _y, _w, _h, area, ratio, _short, _long)
            ann_bbox_info.pop("x")
            ann_bbox_info.pop("y")

            for prop, val in ann_bbox_info.items():
                prop_stats = bbox_label_stats[prop]
                self._compute_far_from_mean(prop_stats, val, item_key, ann)

            for attr, value in ann.attributes.items():
                if attr in valid_attrs:
                    bbox_attr_stats = dist_by_attr[label_name][attr]
                    bbox_val_stats = bbox_attr_stats[str(value)]

                    for prop, val in ann_bbox_info.items():
                        prop_stats = bbox_val_stats[prop]
                        self._compute_far_from_mean(prop_stats, val, item_key, ann)

        # Compute far_from_mean from property
        for item_key, annotations in self.items:
            for ann in annotations:
                if _is_valid_bbox(item_key, ann):
                    _update_bbox_props_far_from_mean(item_key, ann)

        return stats

    def generate_reports(self, stats):
        """
        Validates the dataset for detection tasks based on its statistics.

        Parameters
        ----------
        dataset : IDataset object
        stats : Dict object

        Returns
        -------
        reports (list): List of validation reports (DatasetValidationError).
        """

        reports = self._generate_common_reports(stats)
        reports += self._check_negative_length(stats)
        reports += self._check_invalid_value(stats)

        defined_attr_dist = stats["attribute_distribution"]["defined_attributes"]

        dist_by_label = stats["point_distribution_in_label"]
        dist_by_attr = stats["point_distribution_in_attribute"]

        defined_labels = defined_attr_dist.keys()
        for label_name in defined_labels:
            bbox_label_stats = dist_by_label[label_name]
            bbox_attr_label = dist_by_attr.get(label_name, {})

            reports += self._check_far_from_label_mean(label_name, bbox_label_stats)
            reports += self._check_imbalanced_dist_in_label(label_name, bbox_label_stats)

            for attr_name, bbox_attr_stats in bbox_attr_label.items():
                reports += self._check_far_from_attr_mean(label_name, attr_name, bbox_attr_stats)
                reports += self._check_imbalanced_dist_in_attr(
                    label_name, attr_name, bbox_attr_stats
                )

        return reports

    def _update_prop_distributions(self, curr_stats, target_stats):
        for prop, val in curr_stats.items():
            prop_stats = target_stats[prop]
            prop_stats["distribution"].append(val)

    def _compute_prop_dist(self, label_categories, stats, update_stats_by_label):
        dist_by_label = stats["point_distribution_in_label"]
        dist_by_attr = stats["point_distribution_in_attribute"]
        point_dist_in_item = stats["point_distribution_in_dataset_item"]

        base_valid_attrs = label_categories.attributes

        for item_key, annotations in self.items:
            ann_count = len(annotations)
            point_dist_in_item[item_key] = ann_count

            for ann in annotations:
                if not 0 <= ann.label < len(label_categories):
                    label_name = ann.label
                    valid_attrs = set()
                else:
                    label_name = label_categories[ann.label].name
                    valid_attrs = base_valid_attrs.union(label_categories[ann.label].attributes)

                    point_label_stats = dist_by_label.setdefault(
                        label_name, deepcopy(self.point_template)
                    )
                    ann_point_info, _has_error = update_stats_by_label(
                        item_key, ann, point_label_stats
                    )

                for attr, value in ann.attributes.items():
                    if attr in valid_attrs:
                        point_attr_label = dist_by_attr.setdefault(label_name, {})
                        point_attr_stats = point_attr_label.setdefault(attr, {})
                        point_val_stats = point_attr_stats.setdefault(
                            str(value), deepcopy(self.point_template)
                        )

                        if not _has_error:
                            self._update_prop_distributions(ann_point_info, point_val_stats)

    def _compute_prop_stats_from_dist(self, dist_by_label, dist_by_attr):
        for label_name, stats in dist_by_label.items():
            prop_stats_list = list(stats.values())
            attr_label = dist_by_attr.get(label_name, {})
            for vals in attr_label.values():
                for val_stats in vals.values():
                    prop_stats_list += list(val_stats.values())

            for prop_stats in prop_stats_list:
                prop_dist = prop_stats.pop("distribution", [])
                if len(prop_dist) > 0:
                    prop_stats["mean"] = np.mean(prop_dist)
                    prop_stats["stdev"] = np.std(prop_dist)
                    prop_stats["min"] = np.min(prop_dist)
                    prop_stats["max"] = np.max(prop_dist)
                    prop_stats["median"] = np.median(prop_dist)

                    counts, bins = np.histogram(prop_dist)
                    prop_stats["histogram"]["bins"] = bins.tolist()
                    prop_stats["histogram"]["counts"] = counts.tolist()

    def _compute_far_from_mean(self, prop_stats, val, item_key, ann):
        def _far_from_mean(val, mean, stdev):
            thr = self.far_from_mean_thr
            return val > mean + (thr * stdev) or val < mean - (thr * stdev)

        mean = prop_stats["mean"]
        stdev = prop_stats["stdev"]

        if _far_from_mean(val, mean, stdev):
            items_far_from_mean = prop_stats["items_far_from_mean"]
            far_from_mean = items_far_from_mean.setdefault(item_key, {})
            far_from_mean[ann.id] = val

    def _check_negative_length(self, stats):
        validation_reports = []

        items_w_neg_len = stats["items_with_negative_length"]
        for item_dets, anns_w_neg_len in items_w_neg_len.items():
            item_id, item_subset = item_dets
            for ann_id, props in anns_w_neg_len.items():
                for prop, val in props.items():
                    val = round(val, 2)
                    details = (item_subset, ann_id, f"{self.str_ann_type} {prop}", val)
                    validation_reports += self._generate_validation_report(
                        NegativeLength, Severity.error, item_id, *details
                    )

        return validation_reports

    def _check_invalid_value(self, stats):
        validation_reports = []

        items_w_invalid_val = stats["items_with_invalid_value"]
        for item_dets, anns_w_invalid_val in items_w_invalid_val.items():
            item_id, item_subset = item_dets
            for ann_id, props in anns_w_invalid_val.items():
                for prop in props:
                    details = (item_subset, ann_id, f"{self.str_ann_type} {prop}")
                    validation_reports += self._generate_validation_report(
                        InvalidValue, Severity.error, item_id, *details
                    )

        return validation_reports

    def _check_imbalanced_dist_in_label(self, label_name, label_stats):
        validation_reports = []
        thr = self.dominance_thr
        topk_ratio = self.topk_bins_ratio

        for prop, prop_stats in label_stats.items():
            value_counts = prop_stats["histogram"]["counts"]
            n_bucket = len(value_counts)
            if n_bucket < 2:
                continue
            topk = max(1, int(np.around(n_bucket * topk_ratio)))

            if topk > 0:
                topk_values = np.sort(value_counts)[-topk:]
                ratio = np.sum(topk_values) / np.sum(value_counts)
                if ratio >= thr:
                    details = (label_name, f"{self.str_ann_type} {prop}")
                    validation_reports += self._generate_validation_report(
                        ImbalancedDistInLabel, Severity.info, *details
                    )

        return validation_reports

    def _check_imbalanced_dist_in_attr(self, label_name, attr_name, attr_stats):
        validation_reports = []
        thr = self.dominance_thr
        topk_ratio = self.topk_bins_ratio

        for attr_value, value_stats in attr_stats.items():
            for prop, prop_stats in value_stats.items():
                value_counts = prop_stats["histogram"]["counts"]
                n_bucket = len(value_counts)
                if n_bucket < 2:
                    continue
                topk = max(1, int(np.around(n_bucket * topk_ratio)))

                if topk > 0:
                    topk_values = np.sort(value_counts)[-topk:]
                    ratio = np.sum(topk_values) / np.sum(value_counts)
                    if ratio >= thr:
                        details = (label_name, attr_name, attr_value, f"{self.str_ann_type} {prop}")
                        validation_reports += self._generate_validation_report(
                            ImbalancedDistInAttribute, Severity.info, *details
                        )

        return validation_reports

    def _check_far_from_label_mean(self, label_name, label_stats):
        validation_reports = []

        for prop, prop_stats in label_stats.items():
            items_far_from_mean = prop_stats["items_far_from_mean"]
            if prop_stats["mean"] is not None:
                mean = round(prop_stats["mean"], 2)

            for item_dets, anns_far in items_far_from_mean.items():
                item_id, item_subset = item_dets
                for ann_id, val in anns_far.items():
                    val = round(val, 2)
                    details = (
                        item_subset,
                        label_name,
                        ann_id,
                        f"{self.str_ann_type} {prop}",
                        mean,
                        val,
                    )
                    validation_reports += self._generate_validation_report(
                        FarFromLabelMean, Severity.warning, item_id, *details
                    )

        return validation_reports

    def _check_far_from_attr_mean(self, label_name, attr_name, attr_stats):
        validation_reports = []

        for attr_value, value_stats in attr_stats.items():
            for prop, prop_stats in value_stats.items():
                items_far_from_mean = prop_stats["items_far_from_mean"]
                if prop_stats["mean"] is not None:
                    mean = round(prop_stats["mean"], 2)

                for item_dets, anns_far in items_far_from_mean.items():
                    item_id, item_subset = item_dets
                    for ann_id, val in anns_far.items():
                        val = round(val, 2)
                        details = (
                            item_subset,
                            label_name,
                            ann_id,
                            attr_name,
                            attr_value,
                            f"{self.str_ann_type} {prop}",
                            mean,
                            val,
                        )
                        validation_reports += self._generate_validation_report(
                            FarFromAttrMean, Severity.warning, item_id, *details
                        )

        return validation_reports


class SegmentationValidator(DetectionValidator):
    """
    A specific validator class for (instance) segmentation task.
    """

    def __init__(
        self,
        task_type=TaskType.segmentation,
        few_samples_thr=None,
        imbalance_ratio_thr=None,
        far_from_mean_thr=None,
        dominance_ratio_thr=None,
        topk_bins=None,
    ):
        super().__init__(
            task_type=task_type,
            few_samples_thr=few_samples_thr,
            imbalance_ratio_thr=imbalance_ratio_thr,
            far_from_mean_thr=far_from_mean_thr,
            dominance_ratio_thr=dominance_ratio_thr,
            topk_bins=topk_bins,
        )

        self.point_template = {
            "area": deepcopy(self.numerical_stat_template),
            "width": deepcopy(self.numerical_stat_template),
            "height": deepcopy(self.numerical_stat_template),
        }

    def compute_statistics(self, dataset):
        """
        Computes statistics of the dataset for the segmentation task.

        Parameters
        ----------
        dataset : IDataset object

        Returns
        -------
        stats (dict): A dict object containing statistics of the dataset.
        """

        stats, filtered_items = self._compute_common_statistics(dataset)

        stats["items_with_invalid_value"] = {}
        stats["point_distribution_in_label"] = {}
        stats["point_distribution_in_attribute"] = {}
        stats["point_distribution_in_dataset_item"] = {}

        self.items = filtered_items

        def _generate_ann_mask_info(area, _w, _h):
            return {
                "area": area,
                "width": _w,
                "height": _h,
            }

        def _update_mask_stats_by_label(item_key, ann, mask_label_stats):
            mask_has_error = False

            _, _, _w, _h = ann.get_bbox()

            # Detete the following block when #226 is resolved
            # https://github.com/open-edge-platform/datumaro/issues/226
            if ann.type == AnnotationType.mask:
                _w += 1
                _h += 1

            area = ann.get_area()

            ann_mask_info = _generate_ann_mask_info(area, _w, _h)

            items_w_invalid_val = stats["items_with_invalid_value"]
            for prop, val in ann_mask_info.items():
                if val == float("inf") or np.isnan(val):
                    mask_has_error = True
                    anns_w_invalid_val = items_w_invalid_val.setdefault(item_key, {})
                    invalid_props = anns_w_invalid_val.setdefault(ann.id, [])
                    invalid_props.append(prop)

            if not mask_has_error:
                self._update_prop_distributions(ann_mask_info, mask_label_stats)

            return ann_mask_info, mask_has_error

        # Collect property distribution
        label_categories = dataset.categories().get(AnnotationType.label, LabelCategories())
        self._compute_prop_dist(label_categories, stats, _update_mask_stats_by_label)

        # Compute property statistics from distribution
        dist_by_label = stats["point_distribution_in_label"]
        dist_by_attr = stats["point_distribution_in_attribute"]
        self._compute_prop_stats_from_dist(dist_by_label, dist_by_attr)

        def _is_valid_mask(item_key, ann):
            has_defined_label = 0 <= ann.label < len(label_categories)
            if not has_defined_label:
                return False

            mask_has_invalid_val = ann.id in stats["items_with_invalid_value"].get(item_key, {})
            return not mask_has_invalid_val

        def _update_mask_props_far_from_mean(item_key, ann):
            base_valid_attrs = label_categories.attributes
            valid_attrs = base_valid_attrs.union(label_categories[ann.label].attributes)
            label_name = label_categories[ann.label].name
            mask_label_stats = dist_by_label[label_name]

            _, _, _w, _h = ann.get_bbox()

            # Detete the following block when #226 is resolved
            # https://github.com/open-edge-platform/datumaro/issues/226
            if ann.type == AnnotationType.mask:
                _w += 1
                _h += 1
            area = ann.get_area()

            ann_mask_info = _generate_ann_mask_info(area, _w, _h)

            for prop, val in ann_mask_info.items():
                prop_stats = mask_label_stats[prop]
                self._compute_far_from_mean(prop_stats, val, item_key, ann)

            for attr, value in ann.attributes.items():
                if attr in valid_attrs:
                    mask_attr_stats = dist_by_attr[label_name][attr]
                    mask_val_stats = mask_attr_stats[str(value)]

                    for prop, val in ann_mask_info.items():
                        prop_stats = mask_val_stats[prop]
                        self._compute_far_from_mean(prop_stats, val, item_key, ann)

        for item_key, annotations in self.items:
            for ann in annotations:
                if _is_valid_mask(item_key, ann):
                    _update_mask_props_far_from_mean(item_key, ann)

        return stats

    def generate_reports(self, stats):
        """
        Validates the dataset for segmentation tasks based on its statistics.

        Parameters
        ----------
        dataset : IDataset object
        stats : Dict object

        Returns
        -------
        reports (list): List of validation reports (DatasetValidationError).
        """

        reports = self._generate_common_reports(stats)
        reports += self._check_invalid_value(stats)

        defined_attr_dist = stats["attribute_distribution"]["defined_attributes"]

        dist_by_label = stats["point_distribution_in_label"]
        dist_by_attr = stats["point_distribution_in_attribute"]

        defined_labels = defined_attr_dist.keys()
        for label_name in defined_labels:
            mask_label_stats = dist_by_label[label_name]
            mask_attr_label = dist_by_attr.get(label_name, {})

            reports += self._check_far_from_label_mean(label_name, mask_label_stats)
            reports += self._check_imbalanced_dist_in_label(label_name, mask_label_stats)

            for attr_name, mask_attr_stats in mask_attr_label.items():
                reports += self._check_far_from_attr_mean(label_name, attr_name, mask_attr_stats)
                reports += self._check_imbalanced_dist_in_attr(
                    label_name, attr_name, mask_attr_stats
                )

        return reports


from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TabularValidationStats:
    total_ann_count: int = field(default=0)
    items_missing_annotation: List[Any] = field(default_factory=list)

    @classmethod
    def create_with_dataset(cls, dataset):
        instance = cls()
        instance.__post_init__(dataset)
        return instance

    def __post_init__(self, dataset: Optional[Any] = None):
        if dataset:
            self.label_categories = dataset.categories().get(
                AnnotationType.label, LabelCategories()
            )
            self.tabular_categories = dataset.categories().get(
                AnnotationType.caption, TabularCategories()
            )
            self.label_columns = list({item.parent for item in self.label_categories.items})
            self.caption_columns = [cat.name for cat in self.tabular_categories]

            self.defined_labels = {cat.name: 0 for cat in self.label_categories}
            self.empty_labels = {cat: [] for cat in self.label_columns}

            self.defined_captions = {cat: 0 for cat in self.caption_columns}
            self.empty_captions = {cat: [] for cat in self.caption_columns}
            self.redundancies = {
                cat: {"stopword": [], "url": [], "html": [], "emoji": []}
                for cat in self.caption_columns
            }

    def to_dict(self):
        empty_labels = self._build_empty_labels_dict(self.empty_labels, "items_with_empty_label")
        empty_captions = self._build_empty_labels_dict(
            self.empty_captions, "items_with_empty_caption"
        )
        redundancies = self._build_redundancies_dict(self.redundancies)

        return {
            "total_ann_count": self.total_ann_count,
            "items_missing_annotation": self.items_missing_annotation,
            "items_broken_annotation": self._collect_broken_annotations(),
            "label_distribution": {
                "defined_labels": self.defined_labels,
                "empty_labels": empty_labels,
            },
            "caption_distribution": {
                "defined_captions": self.defined_captions,
                "empty_captions": empty_captions,
                "redundancies": redundancies,
            },
        }

    def _build_empty_labels_dict(self, empty_dict, key_name):
        result = defaultdict(dict)
        for label, items in empty_dict.items():
            result[label]["count"] = len(items)
            result[label][key_name] = list(items)
        return result

    def _build_redundancies_dict(self, redundancies):
        result = defaultdict(lambda: defaultdict(dict))
        for caption, items in redundancies.items():
            for key in ["stopword", "url", "html", "emoji"]:
                result[caption][key]["count"] = len(items[key])
                result[caption][key]["items_with_redundancies"] = list(items[key])
        return result

    def _collect_broken_annotations(self):
        broken_annotations = set()
        for items in [self.empty_labels, self.empty_captions]:
            for key, value in items.items():
                broken_annotations.update(value)
        return list(broken_annotations)


class TabularValidator(_TaskValidator):
    """
    A specific validator class for tabular dataset.
    """

    def __init__(
        self,
        task_type=TaskType.tabular,
        few_samples_thr=None,
        imbalance_ratio_thr=None,
        far_from_mean_thr=None,
        dominance_ratio_thr=None,
        topk_bins=None,
    ):
        super().__init__(
            task_type=task_type,
            few_samples_thr=few_samples_thr,
            imbalance_ratio_thr=imbalance_ratio_thr,
            far_from_mean_thr=far_from_mean_thr,
            dominance_ratio_thr=dominance_ratio_thr,
            topk_bins=topk_bins,
        )

        self.numerical_stat_template = {
            "items_far_from_mean": {},
            "mean": None,
            "stdev": None,
            "min": None,
            "max": None,
            "median": None,
            "histogram": {
                "bins": [],
                "counts": [],
            },
            "distribution": [],
            "items_outlier": {},
            "outlier": None,
        }
        self.value_template = {"value": deepcopy(self.numerical_stat_template)}

    def _compute_common_statistics(self, dataset):
        stats = TabularValidationStats.create_with_dataset(dataset=dataset)

        filtered_items = []
        for item in dataset:
            item_key = (item.id, item.subset)
            label_check = {cat: 0 for cat in stats.label_columns}
            annotations = [ann for ann in item.annotations if ann.type in self.ann_types]
            ann_count = len(annotations)
            filtered_items.append((item_key, annotations))
            if ann_count == 0:
                stats.items_missing_annotation.append(item_key)
            stats.total_ann_count += ann_count

            caption_check = deepcopy(stats.caption_columns)
            for ann in annotations:
                if ann.type == AnnotationType.caption:
                    caption_ = ann.caption
                    for cat in stats.caption_columns:
                        if caption_.startswith(cat):
                            stats.defined_captions[cat] += 1
                            caption_ = caption_.split(cat + ":")[-1]
                            caption_check.remove(cat)
                            self._check_contain_redundancies(caption_, stats, cat, item_key)
                else:
                    label_name = stats.label_categories[ann.label].name
                    stats.defined_labels[label_name] += 1
                    label_name = label_name.split(":")[0]
                    label_check[label_name] += 1

            for cap in caption_check:
                stats.empty_captions[cap].append(item_key)

            for label_col, v in label_check.items():
                if v == 0:
                    stats.empty_labels[label_col].append(item_key)

        return stats.to_dict(), filtered_items

    def _check_contain_redundancies(self, text, stats, column, item_key):
        if column not in stats.redundancies.keys():
            return

        import re

        from nltk.corpus import stopwords

        try:
            stop_words = set(stopwords.words("english"))  # TODO
        except LookupError:
            import nltk

            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))  # TODO

        def contains_emoji(text):
            return bool(emoji_pattern.search(text))

        def contains_html_tags(text):
            html_pattern = re.compile(r"<.*?>")
            return bool(html_pattern.search(text))

        def contains_url(text):
            url_pattern = re.compile(r"http\S+|www\S+|https\S+")
            return bool(url_pattern.search(text))

        def contains_stopword(text):
            return any(c in stop_words for c in str(text).lower())

        redun_stats = stats.redundancies[column]
        if contains_emoji(text):
            redun_stats["emoji"].append(item_key)
        if contains_html_tags(text):
            redun_stats["html"].append(item_key)
        if contains_url(text):
            redun_stats["url"].append(item_key)
        if contains_stopword(text):
            redun_stats["stopword"].append(item_key)

    def _compute_prop_dist(self, caption_columns, stats):
        dist_by_caption = stats["distribution_in_caption"]
        dist_in_item = stats["distribution_in_dataset_item"]

        for item_key, annotations in self.items:
            ann_count = len(annotations)
            dist_in_item[item_key] = ann_count

            for ann in annotations:
                if ann.type == AnnotationType.caption:
                    caption_ = ann.caption
                    for cat_name, type_ in caption_columns:
                        if caption_.startswith(cat_name):
                            caption_value = type_(caption_.split(f"{cat_name}:")[-1])
                            dist_by_caption[cat_name]["value"]["distribution"].append(caption_value)

    def _compute_prop_stats_from_dist(self, dist_by_caption):
        for stats in dist_by_caption.values():
            prop_stats = list(stats.values())[0]

            prop_dist = prop_stats.pop("distribution", [])
            if prop_dist:
                prop_stats["mean"] = np.mean(prop_dist)
                prop_stats["stdev"] = np.std(prop_dist)
                prop_stats["min"] = np.min(prop_dist)
                prop_stats["max"] = np.max(prop_dist)
                prop_stats["median"] = np.median(prop_dist)

                counts, bins = np.histogram(prop_dist)
                prop_stats["histogram"]["bins"] = bins.tolist()
                prop_stats["histogram"]["counts"] = counts.tolist()

                # Calculate Q1 (25th percentile) and Q3 (75th percentile)
                Q1, Q3 = np.percentile(prop_dist, [25, 75])
                IQR = Q3 - Q1

                # Calculate the acceptable range
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                prop_stats["outlier"] = (lower_bound, upper_bound)

    def _compute_far_from_mean(self, prop_stats, val, item_key):
        def _far_from_mean(val, mean, stdev):
            thr = self.far_from_mean_thr
            return val > mean + (thr * stdev) or val < mean - (thr * stdev)

        mean = prop_stats["mean"]
        stdev = prop_stats["stdev"]

        if _far_from_mean(val, mean, stdev):
            prop_stats["items_far_from_mean"][item_key] = val

    def _compute_outlier(self, prop_stats, val, item_key):
        lower_bound, upper_bound = prop_stats["outlier"]

        if (val < lower_bound) | (val > upper_bound):
            prop_stats["items_outlier"][item_key] = val

    def compute_statistics(self, dataset):
        """
        Computes statistics of the tabular dataset.

        Parameters
        ----------
        dataset : IDataset object

        Returns
        -------
        stats (dict): A dict object containing statistics of the dataset.
        """

        stats, filtered_items = self._compute_common_statistics(dataset)

        self.items = filtered_items

        num_caption_columns = [
            (cat.name, cat.dtype)
            for cat in dataset.categories().get(AnnotationType.caption, TabularCategories())
            if cat.dtype in [int, float]
        ]

        stats["distribution_in_caption"] = {
            cap: deepcopy(self.value_template) for cap, _ in num_caption_columns
        }
        stats["distribution_in_dataset_item"] = {}

        # Collect property distribution
        self._compute_prop_dist(num_caption_columns, stats)

        # Compute property statistics from distribution
        dist_by_caption = stats["distribution_in_caption"]
        self._compute_prop_stats_from_dist(dist_by_caption)

        def _update_captions_far_from_mean_outlier(caption_columns, item_key, ann):
            for col, type_ in caption_columns:
                prop_stats = dist_by_caption[col]["value"]
                if ann.caption.startswith(col):
                    val = type_(ann.caption.split(f"{col}:")[-1])
                    self._compute_far_from_mean(prop_stats, val, item_key)
                    self._compute_outlier(prop_stats, val, item_key)

        # Compute far_from_mean and outlier from property
        for item_key, annotations in self.items:
            for ann in annotations:
                if ann.type == AnnotationType.caption:
                    _update_captions_far_from_mean_outlier(num_caption_columns, item_key, ann)

        return stats

    def generate_reports(self, stats):
        """
        Validates the dataset for classification tasks based on its statistics.

        Parameters
        ----------
        dataset : IDataset object
        stats: Dict object

        Returns
        -------
        reports (list): List of validation reports (DatasetValidationError).
        """
        reports = []

        # report for dataset
        reports += self._check_missing_label_categories(stats)

        # report for item
        reports += self._check_missing_annotation(stats)

        # report for label
        reports += self._check_few_samples_in_label(stats)
        reports += self._check_imbalanced_labels(stats)

        # report for caption
        reports += self._check_few_samples_in_caption(stats)
        reports += self._check_redundancies_in_caption(stats)
        reports += self._check_imbalanced_captions(stats)

        # report for missing value
        reports += self._check_broken_annotation(stats)
        reports += self._check_empty_label(stats)
        reports += self._check_empty_caption(stats)

        dist_by_caption = stats["distribution_in_caption"]
        for caption, caption_stats in dist_by_caption.items():
            reports += self._check_far_from_caption_mean(caption, caption_stats)
            reports += self._check_caption_outliers(caption, caption_stats)
            reports += self._check_imbalanced_dist_in_caption(caption, caption_stats)

        return reports

    def _check_broken_annotation(self, stats):
        validation_reports = []

        items_broken = stats["items_broken_annotation"]
        for item_id, item_subset in items_broken:
            validation_reports += self._generate_validation_report(
                BrokenAnnotation, Severity.warning, item_id, item_subset, self.str_ann_type
            )

        return validation_reports

    def _check_empty_label(self, stats):
        validation_reports = []

        empty_label_dist = stats["label_distribution"]["empty_labels"]
        for label_name, label_stats in empty_label_dist.items():
            for item_id, item_subset in label_stats["items_with_empty_label"]:
                details = (item_subset, label_name)
                validation_reports += self._generate_validation_report(
                    EmptyLabel, Severity.warning, item_id, *details
                )

        return validation_reports

    def _check_empty_caption(self, stats):
        validation_reports = []

        empty_caption_dist = stats["caption_distribution"]["empty_captions"]
        for caption_name, caption_stats in empty_caption_dist.items():
            for item_id, item_subset in caption_stats["items_with_empty_caption"]:
                details = (item_subset, caption_name)
                validation_reports += self._generate_validation_report(
                    EmptyCaption, Severity.warning, item_id, *details
                )

        return validation_reports

    def _check_few_samples_in_caption(self, stats):
        validation_reports = []
        thr = self.few_samples_thr

        defined_caption_dist = stats["caption_distribution"]["defined_captions"]
        captions_with_few_samples = [
            (caption_name, count)
            for caption_name, count in defined_caption_dist.items()
            if 0 < count <= thr
        ]

        for caption_name, count in captions_with_few_samples:
            validation_reports += self._generate_validation_report(
                FewSamplesInCaption, Severity.info, caption_name, count
            )

        return validation_reports

    def _check_far_from_caption_mean(self, caption_name, caption_stats):
        prop_stats = list(caption_stats.values())[0]

        if prop_stats["mean"] is not None:
            mean = round(prop_stats["mean"], 2)
        stdev = prop_stats["stdev"]
        upper_bound = mean + (self.far_from_mean_thr * stdev)
        lower_bound = mean - (self.far_from_mean_thr * stdev)

        validation_reports = []
        items_far_from_mean = prop_stats["items_far_from_mean"]
        for item_dets, val in items_far_from_mean.items():
            item_id, item_subset = item_dets
            val = round(val, 2)
            details = (
                item_subset,
                caption_name,
                mean,
                upper_bound,
                lower_bound,
                val,
            )
            validation_reports += self._generate_validation_report(
                FarFromCaptionMean, Severity.info, item_id, *details
            )

        return validation_reports

    def _check_caption_outliers(self, caption_name, caption_stats):
        prop_stats = list(caption_stats.values())[0]
        lower_bound, upper_bound = prop_stats["outlier"]
        items_outlier = prop_stats["items_outlier"]

        validation_reports = []
        for item_dets, val in items_outlier.items():
            item_id, item_subset = item_dets
            val = round(val, 2)
            details = (
                item_subset,
                caption_name,
                lower_bound,
                upper_bound,
                val,
            )
            validation_reports += self._generate_validation_report(
                OutlierInCaption, Severity.info, item_id, *details
            )

        return validation_reports

    def _check_redundancies_in_caption(self, stats):
        validation_reports = []

        redundancies_in_caption_dist = stats["caption_distribution"]["redundancies"]
        captions_with_redundancies = []
        for cap_column, cap_stats in redundancies_in_caption_dist.items():
            for redundancy_type, items in cap_stats.items():
                if 0 < items["count"]:
                    captions_with_redundancies.append((cap_column, redundancy_type, items["count"]))

        for cap_column, redundancy_type, count in captions_with_redundancies:
            validation_reports += self._generate_validation_report(
                RedundanciesInCaption, Severity.info, cap_column, redundancy_type, count
            )

        return validation_reports

    def _check_imbalanced_captions(self, stats):
        validation_reports = []
        thr = self.imbalance_ratio_thr

        defined_caption_dist = stats["caption_distribution"]["defined_captions"]
        count_by_caption_labels = [count for _, count in defined_caption_dist.items()]

        if len(defined_caption_dist) == 0:
            return validation_reports

        count_max = np.max(count_by_caption_labels)
        count_min = np.min(count_by_caption_labels)
        balance = count_max / count_min if count_min > 0 else float("inf")
        if balance >= thr:
            validation_reports += self._generate_validation_report(
                ImbalancedCaptions, Severity.info
            )

        return validation_reports

    def _check_imbalanced_dist_in_caption(self, caption_name, caption_stats):
        validation_reports = []
        thr = self.dominance_thr
        topk_ratio = self.topk_bins_ratio

        for prop, prop_stats in caption_stats.items():
            value_counts = prop_stats["histogram"]["counts"]
            n_bucket = len(value_counts)
            if n_bucket < 2:
                continue
            topk = max(1, int(np.around(n_bucket * topk_ratio)))

            if topk > 0:
                topk_values = np.sort(value_counts)[-topk:]
                ratio = np.sum(topk_values) / np.sum(value_counts)
                if ratio >= thr:
                    validation_reports += self._generate_validation_report(
                        ImbalancedDistInCaption, Severity.info, caption_name
                    )

        return validation_reports
