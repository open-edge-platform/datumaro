# Copyright (C) 2023-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

from datumaro.components.annotation import Annotation, AnnotationType, LabelCategories
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import (
    AttributeDefinedButNotFound,
    BrokenAnnotation,
    DatasetValidationError,
    EmptyCaption,
    EmptyLabel,
    FarFromAttrMean,
    FarFromCaptionMean,
    FarFromLabelMean,
    FewSamplesInAttribute,
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
    UndefinedAttribute,
    UndefinedLabel,
)
from datumaro.components.validator import Severity, TaskType, Validator


@dataclass
class StatsData:
    categories: Dict[str, Dict[str, Set[str]]]
    undefined_attribute: Set[Tuple[str, str, str]]  # item_id, item_subset, label_name, attr_name
    undefined_label: Set[Tuple[str, str, str]]  # item_id, item_subset, label_name
    missing_attribute: Set[Tuple[str, str, str, str]]  # item_id, item_subset, label_name, attr_name
    missing_label: Set[Tuple[str, str]]  # item_id, item_subset


@dataclass
class ClsStatsData(StatsData):
    multiple_label: Set[Tuple[str, str]]  # item_id, item_subset


@dataclass
class DetStatsData(StatsData):
    invalid_value: Set[Tuple[str, str]]  # item_id, item_subset
    negative_length: Set[Tuple[str, str]]  # item_id, item_subset


@dataclass
class SegStatsData(StatsData):
    invalid_value: Set[Tuple[str, str]]  # item_id, item_subset


@dataclass
class TblStatsData:
    categories: Dict[str, Dict[str, Set[str]]]
    empty_label: Set[Tuple[str, str]]  # item_id, item_subset
    empty_caption: Set[Tuple[str, str]]  # item_id, item_subset
    missing_annotations: Set[Tuple[str, str]]  # item_id, item_subset
    broken_annotations: Set[Tuple[str, str]]  # item_id, item_subset


class _BaseAnnStats:
    ATTR_WARNINGS = {
        AttributeDefinedButNotFound,
        FewSamplesInAttribute,
        ImbalancedAttribute,
        MissingAttribute,
        OnlyOneAttributeValue,
        UndefinedAttribute,
    }

    def __init__(
        self,
        label_categories: LabelCategories,
        warnings: List[DatasetValidationError],
        few_samples_thr: None,
        imbalance_ratio_thr: None,
        far_from_mean_thr: None,
        dominance_thr: None,
        topk_bins_ratio: None,
    ):
        self.label_categories = label_categories
        self.warnings = set(warnings)

        self.few_samples_thr = few_samples_thr
        self.imbalance_ratio_thr = imbalance_ratio_thr
        self.far_from_mean_thr = far_from_mean_thr
        self.dominance_thr = dominance_thr
        self.topk_bins_ratio = topk_bins_ratio

        self.stats = StatsData(
            categories={},
            undefined_attribute=set(),
            undefined_label=set(),
            missing_attribute=set(),
            missing_label=set(),
        )

        for label_cat in self.label_categories.items:
            label_name = label_cat.name
            self.stats.categories[label_name] = {
                "cnt": 0,
                "type": set(),
                "attributes": {attr: {} for attr in label_cat.attributes},
            }

        # Create a dictionary that maps warning types to their corresponding update functions
        self.update_item_functions = set()
        self.update_ann_functions = set()
        self.update_report_functions = set()

        if LabelDefinedButNotFound in self.warnings:
            self.update_ann_functions.add(self._update_label_stats)
            self.update_report_functions.add(self._check_label_defined_but_not_found)
        if FewSamplesInLabel in self.warnings:
            self.update_ann_functions.add(self._update_label_stats)
            self.update_report_functions.add(self._check_few_samples_in_label)
        if ImbalancedLabels in self.warnings:
            self.update_ann_functions.add(self._update_label_stats)
            self.update_report_functions.add(self._check_imbalanced_labels)
        if OnlyOneLabel in self.warnings:
            self.update_ann_functions.add(self._update_label_stats)
            self.update_report_functions.add(self._check_only_one_label)
        if AttributeDefinedButNotFound in self.warnings:
            self.update_ann_functions.add(self._update_attribute_stats)
            self.update_report_functions.add(self._check_attribute_defined_but_not_found)
        if FewSamplesInAttribute in self.warnings:
            self.update_ann_functions.add(self._update_attribute_stats)
            self.update_report_functions.add(self._check_few_samples_in_attribute)
        if ImbalancedAttribute in self.warnings:
            self.update_ann_functions.add(self._update_attribute_stats)
            self.update_report_functions.add(self._check_imbalanced_attribute)
        if OnlyOneAttributeValue in self.warnings:
            self.update_ann_functions.add(self._update_attribute_stats)
            self.update_report_functions.add(self._check_only_one_attribute)
        if MissingAnnotation in self.warnings:
            self.update_item_functions.add(self._update_missing_annotation)
            self.update_report_functions.add(self._check_missing_label)
        if UndefinedLabel in self.warnings:
            self.update_ann_functions.add(self._update_undefined_label)
            self.update_report_functions.add(self._check_undefined_label)
        if MissingAttribute in self.warnings:
            self.update_ann_functions.add(self._update_missing_attribute)
            self.update_report_functions.add(self._check_missing_attribute)
        if UndefinedAttribute in self.warnings:
            self.update_ann_functions.add(self._update_undefined_attribute)
            self.update_report_functions.add(self._check_undefined_attribute)

    def _update_label_stats(self, item_key, annotation: Annotation):
        if annotation.label in self.label_categories:
            label_name = self.label_categories[annotation.label].name
            self.stats.categories[label_name]["cnt"] += 1
            self.stats.categories[label_name]["type"].add(annotation.type)

    def _update_attribute_stats(self, item_key, annotation):
        if annotation.label in self.label_categories:
            label_name = self.label_categories[annotation.label].name
            for attr, value in annotation.attributes.items():
                if attr in self.stats.categories[label_name]["attributes"]:
                    attr_cnt = self.stats.categories[label_name]["attributes"][attr].get(str(value), 0)
                    attr_cnt += 1

    def _update_missing_annotation(self, item):
        item_key = (item.id, item.subset)
        if sum([1 for ann in item.annotations]) == 0:
            self.stats.missing_label.add(item_key)

    def _update_undefined_label(self, item_key, annotation):
        if annotation.label not in self.label_categories:
            self.stats.undefined_label.add(item_key + (str(annotation.label),))

    def _update_missing_attribute(self, item_key, annotation):
        if annotation.label in self.label_categories:
            label_name = self.label_categories[annotation.label].name
            for attr in self.stats.categories[label_name]["attributes"]:
                if attr not in annotation.attributes:
                    self.stats.missing_attribute.add(item_key + (label_name, attr))

    def _update_undefined_attribute(self, item_key, annotation):
        if annotation.label in self.label_categories:
            label_name = self.label_categories[annotation.label].name
            for attr in annotation.attributes:
                if self.stats.categories[label_name]["attributes"].get(attr, None):
                    continue
                self.stats.undefined_attribute.add(item_key + (label_name, attr))
        else:
            for attr in annotation.attributes:
                self.stats.undefined_attribute.add(item_key + (str(annotation.label), attr))

    def _generate_validation_report(self, error, *args, **kwargs):
        return [error(*args, **kwargs)]

    def _check_missing_label_categories(self, stats: StatsData):
        validation_reports = []

        if len(stats.categories) == 0:
            validation_reports += self._generate_validation_report(MissingLabelCategories, Severity.error)

        return validation_reports

    def _check_missing_label(self, stats: StatsData):
        validation_reports = []

        items_missing_label = stats.missing_label
        for item_id, item_subset in items_missing_label:
            validation_reports += self._generate_validation_report(
                MissingAnnotation, Severity.warning, item_id, item_subset, "label"
            )

        return validation_reports

    def _check_missing_attribute(self, stats: StatsData):
        validation_reports = []

        items_missing_attr = stats.missing_attribute
        for item_id, item_subset, label_name, attr_name in items_missing_attr:
            details = (item_subset, label_name, attr_name)
            validation_reports += self._generate_validation_report(
                MissingAttribute, Severity.warning, item_id, *details
            )

        return validation_reports

    def _check_undefined_label(self, stats: StatsData):
        validation_reports = []

        items_undefined_label = stats.undefined_label
        for item_id, item_subset, label_name in items_undefined_label:
            details = (item_subset, label_name)
            validation_reports += self._generate_validation_report(UndefinedLabel, Severity.error, item_id, *details)

        return validation_reports

    def _check_undefined_attribute(self, stats: StatsData):
        validation_reports = []

        items_undefined_attr = stats.undefined_attribute
        for item_id, item_subset, label_name, attr_name in items_undefined_attr:
            details = (item_subset, label_name, attr_name)
            validation_reports += self._generate_validation_report(
                UndefinedAttribute, Severity.error, item_id, *details
            )

        return validation_reports

    def _check_label_defined_but_not_found(self, stats: StatsData):
        validation_reports = []

        for label_name in stats.categories:
            if stats.categories[label_name]["cnt"] == 0:
                validation_reports += self._generate_validation_report(
                    LabelDefinedButNotFound, Severity.warning, label_name
                )

        return validation_reports

    def _check_attribute_defined_but_not_found(self, stats: StatsData):
        validation_reports = []

        for label_name in stats.categories:
            for attr_name, dist in stats.categories[label_name]["attributes"].items():
                if dist:
                    continue
                details = (label_name, attr_name)
                validation_reports += self._generate_validation_report(
                    AttributeDefinedButNotFound, Severity.warning, *details
                )

        return validation_reports

    def _check_only_one_label(self, stats: StatsData):
        labels_found = []
        for label_name in stats.categories:
            if stats.categories[label_name]["cnt"] > 0:
                labels_found.append(label_name)

        validation_reports = []
        if len(labels_found) == 1:
            validation_reports += self._generate_validation_report(OnlyOneLabel, Severity.info, labels_found[0])

        return validation_reports

    def _check_only_one_attribute(self, stats: StatsData):
        validation_reports = []
        for label_name in stats.categories:
            for attr_name, values in stats.categories[label_name]["attributes"].items():
                for attr_value, cnt in values.items():
                    if cnt != 1:
                        continue
                    details = (label_name, attr_name, attr_value)
                    validation_reports += self._generate_validation_report(
                        OnlyOneAttributeValue, Severity.info, *details
                    )

        return validation_reports

    def _check_few_samples_in_label(self, stats: StatsData):
        validation_reports = []
        for label_name in stats.categories:
            if stats.categories[label_name]["cnt"] < self.few_samples_thr:
                validation_reports += self._generate_validation_report(
                    FewSamplesInLabel,
                    Severity.info,
                    label_name,
                    stats.categories[label_name]["cnt"],
                )

        return validation_reports

    def _check_few_samples_in_attribute(self, stats: StatsData):
        validation_reports = []
        for label_name in stats.categories:
            for attr_name, values in stats.categories[label_name]["attributes"].items():
                for attr_value, cnt in values.items():
                    if cnt >= self.few_samples_thr:
                        continue
                    details = (label_name, attr_name, attr_value, cnt)
                    validation_reports += self._generate_validation_report(
                        FewSamplesInAttribute, Severity.info, *details
                    )

        return validation_reports

    def _check_imbalanced_labels(self, stats: StatsData):
        validation_reports = []

        counts = [stats.categories[label_name]["cnt"] for label_name in stats.categories]

        if len(counts) == 0:
            return validation_reports

        count_max = np.max(counts)
        count_min = np.min(counts)
        balance = count_max / count_min if count_min > 0 else float("inf")
        if balance >= self.imbalance_ratio_thr:
            validation_reports += self._generate_validation_report(ImbalancedLabels, Severity.info)

        return validation_reports

    def _check_imbalanced_attribute(self, stats: StatsData):
        validation_reports = []

        for label_name in stats.categories:
            for attr_name, attr_vals in stats.categories[label_name]["attributes"].items():
                counts = [cnt for cnt in attr_vals.values()]

                if len(counts) == 0:
                    continue

                count_max = np.max(counts)
                count_min = np.min(counts)
                balance = count_max / count_min if count_min > 0 else float("inf")
                if balance >= self.imbalance_ratio_thr:
                    validation_reports += self._generate_validation_report(
                        ImbalancedAttribute, Severity.info, label_name, attr_name
                    )

        return validation_reports

    def update_item(self, item: DatasetItem):
        for func in self.update_item_functions:
            func(item)

    def update_ann(self, item_key, annotation: Annotation):
        for func in self.update_ann_functions:
            func(item_key, annotation)

    def generate_report(self):
        reports = []
        for func in self.update_report_functions:
            report = func(self.stats)
            if report:
                reports += report
        return reports


class ClsStats(_BaseAnnStats):
    def __init__(
        self,
        label_categories: LabelCategories,
        warnings: set,
        few_samples_thr: None,
        imbalance_ratio_thr: None,
        far_from_mean_thr: None,
        dominance_thr: None,
        topk_bins_ratio: None,
    ):
        super().__init__(
            label_categories=label_categories,
            warnings=warnings,
            few_samples_thr=few_samples_thr,
            imbalance_ratio_thr=imbalance_ratio_thr,
            far_from_mean_thr=far_from_mean_thr,
            dominance_thr=dominance_thr,
            topk_bins_ratio=topk_bins_ratio,
        )

        self.stats = ClsStatsData(
            categories={},
            undefined_attribute=set(),
            undefined_label=set(),
            missing_attribute=set(),
            missing_label=set(),
            multiple_label=set(),
        )

        for label_cat in self.label_categories.items:
            label_name = label_cat.name
            self.stats.categories[label_name] = {
                "cnt": 0,
                "type": set(),
                "attributes": {attr: {} for attr in label_cat.attributes},
            }

        if MultiLabelAnnotations in self.warnings:
            self.update_item_functions.add(self._update_multi_label)
            self.update_report_functions.add(self._check_multiple_label)

    def _update_multi_label(self, item: DatasetItem):
        item_key = (item.id, item.subset)
        if sum([1 for ann in item.annotations if ann.type == AnnotationType.label]) > 1:
            self.stats.multiple_label.add(item_key)

    def _check_multiple_label(self, stats: ClsStatsData):
        validation_reports = []

        items_multiple_label = stats.multiple_label
        for item_id, item_subset in items_multiple_label:
            validation_reports += self._generate_validation_report(
                MultiLabelAnnotations, Severity.error, item_id, item_subset
            )

        return validation_reports


class DetStats(_BaseAnnStats):
    def __init__(
        self,
        label_categories: LabelCategories,
        warnings: set,
        few_samples_thr: None,
        imbalance_ratio_thr: None,
        far_from_mean_thr: None,
        dominance_thr: None,
        topk_bins_ratio: None,
    ):
        super().__init__(
            label_categories=label_categories,
            warnings=warnings,
            few_samples_thr=few_samples_thr,
            imbalance_ratio_thr=imbalance_ratio_thr,
            far_from_mean_thr=far_from_mean_thr,
            dominance_thr=dominance_thr,
            topk_bins_ratio=topk_bins_ratio,
        )

        self.stats = DetStatsData(
            categories={},
            undefined_attribute=set(),
            undefined_label=set(),
            missing_attribute=set(),
            missing_label=set(),
            invalid_value=set(),
            negative_length=set(),
        )

        for label_cat in self.label_categories.items:
            label_name = label_cat.name
            self.stats.categories[label_name] = {
                "cnt": 0,
                "type": set(),
                "attributes": {attr: {} for attr in label_cat.attributes},
                "pts": [],  # (id, subset, ann_id, w, h, a, r)
            }

        if FarFromLabelMean in self.warnings:
            self.update_ann_functions.add(self._update_bbox_stats)
            self.update_report_functions.add(self._check_far_from_mean)
        if ImbalancedDistInLabel in self.warnings:
            self.update_ann_functions.add(self._update_bbox_stats)
            self.update_report_functions.add(self._check_imbalanced_dist_in_label)
        if FarFromAttrMean in self.warnings:
            self.update_ann_functions.add(self._update_bbox_stats)
            self.update_report_functions.add(self._check_far_from_mean)
        if ImbalancedDistInAttribute in self.warnings:
            self.update_ann_functions.add(self._update_bbox_stats)
            self.update_report_functions.add(self._check_imbalanced_dist_in_label)
        if InvalidValue in self.warnings:
            self.update_ann_functions.add(self._update_invalid_value)
            self.update_report_functions.add(self._check_invalid_value)
        if NegativeLength in self.warnings:
            self.update_ann_functions.add(self._update_negative_length)
            self.update_report_functions.add(self._check_negative_length)

    def _update_invalid_value(self, item_key: tuple, annotation: Annotation):
        _, _, w, h = annotation.get_bbox()

        if w == float("inf") or np.isnan(w):
            self.stats.invalid_value.add(item_key + (annotation.id, "width", w))

        if h == float("inf") or np.isnan(h):
            self.stats.invalid_value.add(item_key + (annotation.id, "height", h))

    def _update_negative_length(self, item_key: tuple, annotation: Annotation):
        _, _, w, h = annotation.get_bbox()

        if w < 1:
            self.stats.negative_length.add(item_key + (annotation.id, "width", w))

        if h < 1:
            self.stats.negative_length.add(item_key + (annotation.id, "height", h))

    def _update_bbox_stats(self, item_key: tuple, annotation: Annotation):
        if annotation.label not in self.label_categories:
            return

        _, _, w, h = annotation.get_bbox()
        area = annotation.get_area()
        if h != 0 and h != float("inf"):
            ratio = w / h
        else:
            ratio = float("nan")

        if not (
            w == float("inf") or np.isnan(w) or w < 1 or h == float("inf") or np.isnan(h) or h < 1 or np.isnan(ratio)
        ):
            label_name = self.label_categories[annotation.label].name
            self.stats.categories[label_name]["pts"].append(item_key + (annotation.id, w, h, area, ratio))

    def _check_far_from_mean(self, stats: StatsData):
        def _far_from_mean(val, mean, stdev):
            thr = self.far_from_mean_thr
            return val > mean + (thr * stdev) or val < mean - (thr * stdev)

        validation_reports = []

        for label_name in stats.categories:
            prop = {"width": [], "height": [], "ratio": [], "area": []}
            for _, _, _, w, h, a, r in stats.categories[label_name]["pts"]:
                prop["width"].append(w)
                prop["height"].append(h)
                prop["ratio"].append(r)
                prop["area"].append(a)

            prop_stats = {}
            for p, vals in prop.items():
                prop_stats[p] = {}
                prop_stats[p]["mean"] = np.mean(vals)
                prop_stats[p]["stdev"] = np.std(vals)

            for item_id, item_subset, ann_id, w, h, a, r in stats.categories[label_name]["pts"]:
                item_prop = {"width": w, "height": h, "ratio": r, "area": a}

                for p in prop_stats:
                    if _far_from_mean(item_prop[p], prop_stats[p]["mean"], prop_stats[p]["stdev"]):
                        details = (
                            item_subset,
                            label_name,
                            ann_id,
                            f"bbox {p}",
                            prop_stats[p]["mean"],
                            item_prop[p],
                        )
                        validation_reports += self._generate_validation_report(
                            FarFromLabelMean, Severity.warning, item_id, *details
                        )

        return validation_reports

    def _check_imbalanced_dist_in_label(self, stats: StatsData):
        validation_reports = []

        for label_name in stats.categories:
            prop = {"width": [], "height": [], "ratio": [], "area": []}
            for _, _, _, w, h, a, r in stats.categories[label_name]["pts"]:
                prop["width"].append(w)
                prop["height"].append(h)
                prop["area"].append(a)
                prop["ratio"].append(r)

            for p, vals in prop.items():
                counts, _ = np.histogram(vals)

                n_bucket = len(counts)
                if n_bucket < 2:
                    continue

                topk = max(1, int(np.around(n_bucket * self.topk_bins_ratio)))

                topk_values = np.sort(counts)[-topk:]
                ratio = np.sum(topk_values) / np.sum(counts)
                if ratio >= self.dominance_thr:
                    details = (label_name, f"bbox {p}")
                    validation_reports += self._generate_validation_report(
                        ImbalancedDistInLabel, Severity.info, *details
                    )

        return validation_reports

    def _check_negative_length(self, stats: DetStatsData):
        validation_reports = []

        items_neg_len = stats.negative_length
        for item_id, item_subset, ann_id, prop, val in items_neg_len:
            details = (item_subset, ann_id, f"bbox {prop}", val)
            validation_reports += self._generate_validation_report(NegativeLength, Severity.error, item_id, *details)

        return validation_reports

    def _check_invalid_value(self, stats: DetStatsData):
        validation_reports = []

        items_invalid_val = stats.invalid_value
        for item_id, item_subset, ann_id, prop, val in items_invalid_val:
            details = (item_subset, ann_id, f"bbox {prop}")
            validation_reports += self._generate_validation_report(InvalidValue, Severity.error, item_id, *details)

        return validation_reports


class SegStats(_BaseAnnStats):
    def __init__(
        self,
        label_categories: LabelCategories,
        warnings: set,
        few_samples_thr: None,
        imbalance_ratio_thr: None,
        far_from_mean_thr: None,
        dominance_thr: None,
        topk_bins_ratio: None,
    ):
        super().__init__(
            label_categories=label_categories,
            warnings=warnings,
            few_samples_thr=few_samples_thr,
            imbalance_ratio_thr=imbalance_ratio_thr,
            far_from_mean_thr=far_from_mean_thr,
            dominance_thr=dominance_thr,
            topk_bins_ratio=topk_bins_ratio,
        )

        self.stats = SegStatsData(
            categories={},
            undefined_attribute=set(),
            undefined_label=set(),
            missing_attribute=set(),
            missing_label=set(),
            invalid_value=set(),
        )

        for label_cat in self.label_categories.items:
            label_name = label_cat.name
            self.stats.categories[label_name] = {
                "cnt": 0,
                "type": set(),
                "attributes": {attr: {} for attr in label_cat.attributes},
                "pts": [],  # (id, subset, ann_id, w, h, a, r)
            }

        if FarFromLabelMean in self.warnings:
            self.update_ann_functions.add(self._update_mask_stats)
            self.update_report_functions.add(self._check_far_from_mean)
        if ImbalancedDistInLabel in self.warnings:
            self.update_ann_functions.add(self._update_mask_stats)
            self.update_report_functions.add(self._check_imbalanced_dist_in_label)
        if FarFromAttrMean in self.warnings:
            self.update_ann_functions.add(self._update_mask_stats)
            self.update_report_functions.add(self._check_far_from_mean)
        if ImbalancedDistInAttribute in self.warnings:
            self.update_ann_functions.add(self._update_mask_stats)
            self.update_report_functions.add(self._check_imbalanced_dist_in_label)
        if InvalidValue in self.warnings:
            self.update_ann_functions.add(self._update_invalid_value)
            self.update_report_functions.add(self._check_invalid_value)

    def _update_invalid_value(self, item_key: tuple, annotation: Annotation):
        _, _, w, h = annotation.get_bbox()

        if w == float("inf") or np.isnan(w):
            self.stats.invalid_value.add(item_key + (annotation.id, "width", w))

        if h == float("inf") or np.isnan(h):
            self.stats.invalid_value.add(item_key + (annotation.id, "height", h))

    def _update_mask_stats(self, item_key: tuple, annotation: Annotation):
        if annotation.label not in self.label_categories:
            return

        _, _, w, h = annotation.get_bbox()
        area = annotation.get_area()

        if not (w == float("inf") or np.isnan(w) or h == float("inf") or np.isnan(h)):
            label_name = self.label_categories[annotation.label].name
            self.stats.categories[label_name]["pts"].append(item_key + (annotation.id, w, h, area, 1))

    def _check_far_from_mean(self, stats: StatsData):
        def _far_from_mean(val, mean, stdev):
            thr = self.far_from_mean_thr
            return val > mean + (thr * stdev) or val < mean - (thr * stdev)

        validation_reports = []

        for label_name in stats.categories:
            prop = {"width": [], "height": [], "area": []}
            for _, _, _, w, h, a, _ in stats.categories[label_name]["pts"]:
                prop["width"].append(w)
                prop["height"].append(h)
                prop["area"].append(a)

            prop_stats = {}
            for p, vals in prop.items():
                prop_stats[p] = {}
                prop_stats[p]["mean"] = np.mean(vals)
                prop_stats[p]["stdev"] = np.std(vals)

            for item_id, item_subset, ann_id, w, h, a, _ in stats.categories[label_name]["pts"]:
                item_prop = {"width": w, "height": h, "area": a}

                for p in prop_stats:
                    if _far_from_mean(item_prop[p], prop_stats[p]["mean"], prop_stats[p]["stdev"]):
                        details = (
                            item_subset,
                            label_name,
                            ann_id,
                            f"polygon/mask/ellipse {p}",
                            prop_stats[p]["mean"],
                            item_prop[p],
                        )
                        validation_reports += self._generate_validation_report(
                            FarFromLabelMean, Severity.warning, item_id, *details
                        )

        return validation_reports

    def _check_imbalanced_dist_in_label(self, stats: StatsData):
        validation_reports = []

        for label_name in stats.categories:
            prop = {"width": [], "height": [], "area": []}
            for _, _, _, w, h, a, _ in stats.categories[label_name]["pts"]:
                prop["width"].append(w)
                prop["height"].append(h)
                prop["area"].append(a)

            for p, vals in prop.items():
                counts, _ = np.histogram(vals)

                n_bucket = len(counts)
                if n_bucket < 2:
                    continue

                topk = max(1, int(np.around(n_bucket * self.topk_bins_ratio)))

                topk_values = np.sort(counts)[-topk:]
                ratio = np.sum(topk_values) / np.sum(counts)
                if ratio >= self.dominance_thr:
                    details = (label_name, f"polygon/mask/ellipse {p}")
                    validation_reports += self._generate_validation_report(
                        ImbalancedDistInLabel, Severity.info, *details
                    )

        return validation_reports

    def _check_invalid_value(self, stats: DetStatsData):
        validation_reports = []

        items_invalid_val = stats.invalid_value
        for item_id, item_subset, ann_id, prop, val in items_invalid_val:
            details = (item_subset, ann_id, f"polygon/mask/ellipse {prop}")
            validation_reports += self._generate_validation_report(InvalidValue, Severity.error, item_id, *details)

        return validation_reports


from pandas.api.types import CategoricalDtype


class TblStats(_BaseAnnStats):
    def __init__(
        self,
        categories: dict,
        warnings: set,
        few_samples_thr: None,
        imbalance_ratio_thr: None,
        far_from_mean_thr: None,
        dominance_thr: None,
        topk_bins_ratio: None,
    ):
        self.caption_categories = categories.get(AnnotationType.caption, [])
        self.label_categories = categories.get(AnnotationType.label, [])

        self.warnings = set(warnings)

        self.few_samples_thr = few_samples_thr
        self.imbalance_ratio_thr = imbalance_ratio_thr
        self.far_from_mean_thr = far_from_mean_thr
        self.dominance_thr = dominance_thr
        self.topk_bins_ratio = topk_bins_ratio

        self.stats = TblStatsData(
            categories={},
            empty_label=set(),
            empty_caption=set(),
            missing_annotations=set(),
            broken_annotations=set(),
        )

        all_categories = [(cat.name, cat.dtype) for cat in self.caption_categories] + [
            (label_group.name, CategoricalDtype) for label_group in self.label_categories.label_groups
        ]
        for name, dtype in all_categories:
            self.stats.categories[name] = {
                "cnt": 0,
                "type": dtype,
                "ann_type": set(),
                "caption": [],
            }
        self.categories_len = len(all_categories)

        self.caption_columns = [cat.name for cat in self.caption_categories]
        self.label_columns = [label_group.name for label_group in self.label_categories.label_groups]

        # Create a dictionary that maps warning types to their corresponding update functions
        self.update_item_functions = set()
        self.update_ann_functions = set()
        self.update_report_functions = set()

        if MissingAnnotation in self.warnings:
            self.update_ann_functions.add(self._update_annotation_stats)
            self.update_item_functions.add(self._update_missing_annotation)
            self.update_report_functions.add(self._check_missing_annotation)
        if BrokenAnnotation in self.warnings:
            self.update_ann_functions.add(self._update_annotation_stats)
            self.update_item_functions.add(self._update_broken_annotation)
            self.update_report_functions.add(self._check_broken_annotation)

        if FewSamplesInLabel in self.warnings:
            self.update_ann_functions.add(self._update_annotation_stats)
            self.update_report_functions.add(self._check_few_samples_in_label)
        if ImbalancedLabels in self.warnings:
            self.update_ann_functions.add(self._update_annotation_stats)
            self.update_report_functions.add(self._check_imbalanced_labels)
        if EmptyLabel in self.warnings:
            self.update_item_functions.add(self._update_empty_label)
            self.update_report_functions.add(self._check_empty_label)

        if EmptyCaption in self.warnings:
            self.update_item_functions.add(self._update_empty_caption)
            self.update_report_functions.add(self._check_empty_caption)
        if ImbalancedCaptions in self.warnings:
            self.update_ann_functions.add(self._update_annotation_stats)
            self.update_report_functions.add(self._check_imbalanced_captions)
        if ImbalancedDistInCaption in self.warnings:
            self.update_ann_functions.add(self._update_annotation_stats)
            self.update_report_functions.add(self._check_imbalanced_dist_in_caption)
        if FarFromCaptionMean in self.warnings:
            self.update_ann_functions.add(self._update_annotation_stats)
            self.update_report_functions.add(self._check_far_from_caption_mean)
        if OutlierInCaption in self.warnings:
            self.update_ann_functions.add(self._update_annotation_stats)
            self.update_report_functions.add(self._check_outlier_in_caption)

    def _update_missing_annotation(self, item):
        item_key = (item.id, item.subset)
        if len(item.annotations) == 0:
            self.stats.missing_annotations.add(item_key)

    def _update_broken_annotation(self, item):
        item_key = (item.id, item.subset)
        if len(item.annotations) < self.categories_len:
            self.stats.broken_annotations.add(item_key)

    def _update_empty_label(self, item):
        if len(item.annotations) < self.categories_len:
            annotation_check = deepcopy(self.label_columns)
            for ann in item.annotations:
                if ann.type == AnnotationType.label:
                    label = self.label_categories[ann.label].name.split(":")[0]
                    annotation_check.remove(label)
            for ann in annotation_check:
                item_key = (item.id, item.subset, ann)
                self.stats.empty_label.add(item_key)

    def _update_empty_caption(self, item):
        if len(item.annotations) < self.categories_len:
            annotation_check = deepcopy(self.caption_columns)
            for ann in item.annotations:
                if ann.type == AnnotationType.caption:
                    label = next((cat for cat in self.caption_columns if ann.caption.startswith(cat)), None)
                    annotation_check.remove(label)
            for ann in annotation_check:
                item_key = (item.id, item.subset, ann)
                self.stats.empty_caption.add(item_key)

    def _update_annotation_stats(self, item_key, annotation: Annotation):
        if annotation.type == AnnotationType.label:
            label_name = self.label_categories[annotation.label].name.split(":")[0]
            self.stats.categories[label_name]["cnt"] += 1
            self.stats.categories[label_name]["ann_type"].add(annotation.type)
        elif annotation.type == AnnotationType.caption:
            for cat in self.caption_columns:
                if annotation.caption.startswith(cat):
                    self.stats.categories[cat]["cnt"] += 1
                    self.stats.categories[cat]["ann_type"].add(annotation.type)
                    caption = annotation.caption.split(cat + ":")[-1]
                    self.stats.categories[cat]["caption"].append(item_key + (annotation.id, caption))
                    break

    def _check_missing_annotation(self, stats: StatsData):
        validation_reports = []

        items_missing_annotations = stats.missing_annotations
        for item_id, item_subset in items_missing_annotations:
            validation_reports += self._generate_validation_report(
                MissingAnnotation, Severity.warning, item_id, item_subset, "label or caption"
            )

        return validation_reports

    def _check_broken_annotation(self, stats: StatsData):
        validation_reports = []

        items_broken_annotations = stats.broken_annotations
        for item_id, item_subset in items_broken_annotations:
            validation_reports += self._generate_validation_report(
                BrokenAnnotation, Severity.warning, item_id, item_subset, "label or caption"
            )

        return validation_reports

    def _check_empty_label(self, stats: StatsData):
        validation_reports = []

        items_empty_label = stats.empty_label
        for item_id, item_subset, col in items_empty_label:
            validation_reports += self._generate_validation_report(
                EmptyLabel, Severity.warning, item_id, item_subset, col
            )

        return validation_reports

    def _check_empty_caption(self, stats: StatsData):
        validation_reports = []

        items_empty_caption = stats.empty_caption
        for item_id, item_subset, col in items_empty_caption:
            validation_reports += self._generate_validation_report(
                EmptyCaption, Severity.warning, item_id, item_subset, col
            )

        return validation_reports

    def _check_imbalanced_labels(self, stats: StatsData):
        validation_reports = []

        counts = [
            stats.categories[label_name]["cnt"]
            for label_name in stats.categories.keys()
            if list(stats.categories[label_name]["ann_type"])[0] == AnnotationType.label
        ]

        if len(counts) == 0:
            return validation_reports

        count_max = np.max(counts)
        count_min = np.min(counts)
        balance = count_max / count_min if count_min > 0 else float("inf")
        if balance >= self.imbalance_ratio_thr:
            validation_reports += self._generate_validation_report(ImbalancedLabels, Severity.info)

        return validation_reports

    def _check_imbalanced_captions(self, stats: StatsData):
        validation_reports = []

        counts = [
            stats.categories[label_name]["cnt"]
            for label_name in stats.categories.keys()
            if list(stats.categories[label_name]["ann_type"])[0] == AnnotationType.caption
        ]

        if len(counts) == 0:
            return validation_reports

        count_max = np.max(counts)
        count_min = np.min(counts)
        balance = count_max / count_min if count_min > 0 else float("inf")
        if balance >= self.imbalance_ratio_thr:
            validation_reports += self._generate_validation_report(ImbalancedCaptions, Severity.info)

        return validation_reports

    def _check_far_from_caption_mean(self, stats: StatsData):
        def _far_from_mean(val, mean, stdev):
            thr = self.far_from_mean_thr
            return val > mean + (thr * stdev) or val < mean - (thr * stdev)

        validation_reports = []

        for label_name in stats.categories:
            type_ = stats.categories[label_name]["type"]
            if type_ in [float, int]:
                captions = [type_(caption[3]) for caption in stats.categories[label_name]["caption"]]
                prop_stats = {}
                prop_stats["mean"] = np.mean(captions)
                prop_stats["stdev"] = np.std(captions)

                upper_bound = prop_stats["mean"] + (self.far_from_mean_thr * prop_stats["stdev"])
                lower_bound = prop_stats["mean"] - (self.far_from_mean_thr * prop_stats["stdev"])
                for item_id, item_subset, ann_id, caption in stats.categories[label_name]["caption"]:
                    if _far_from_mean(type_(caption), prop_stats["mean"], prop_stats["stdev"]):
                        details = (
                            item_subset,
                            label_name,
                            prop_stats["mean"],
                            upper_bound,
                            lower_bound,
                            type_(caption),
                        )
                        validation_reports += self._generate_validation_report(
                            FarFromCaptionMean, Severity.info, item_id, *details
                        )

        return validation_reports

    def _check_imbalanced_dist_in_caption(self, stats: StatsData):
        validation_reports = []

        for label_name in stats.categories:
            type_ = stats.categories[label_name]["type"]
            if type_ in [float, int]:
                captions = [type_(caption[3]) for caption in stats.categories[label_name]["caption"]]
                counts, _ = np.histogram(captions)

                n_bucket = len(counts)
                if n_bucket < 2:
                    continue

                topk = max(1, int(np.around(n_bucket * self.topk_bins_ratio)))

                topk_values = np.sort(counts)[-topk:]
                ratio = np.sum(topk_values) / np.sum(counts)
                if ratio >= self.dominance_thr:
                    validation_reports += self._generate_validation_report(
                        ImbalancedDistInCaption, Severity.info, label_name
                    )
        return validation_reports

    def _check_outlier_in_caption(self, stats: StatsData):
        validation_reports = []

        for label_name, category in stats.categories.items():
            type_ = category["type"]
            if type_ in [float, int]:
                captions = np.array([type_(caption[3]) for caption in category["caption"]])

                if captions.size == 0:
                    continue

                # Calculate Q1 (25th percentile) and Q3 (75th percentile)
                Q1 = np.quantile(captions, 0.25)
                Q3 = np.quantile(captions, 0.75)
                IQR = Q3 - Q1

                # Calculate the acceptable rangepsrc/datumaro/plugins/validators.py
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                for item_id, item_subset, ann_id, caption in category["caption"]:
                    val = type_(caption)
                    if (val < lower_bound) | (val > upper_bound):
                        details = (
                            item_subset,
                            label_name,
                            upper_bound,
                            lower_bound,
                            val,
                        )
                        validation_reports += self._generate_validation_report(
                            OutlierInCaption, Severity.info, item_id, *details
                        )

        return validation_reports


class ConfigurableValidator(Validator, CliPlugin):
    DEFAULT_FEW_SAMPLES_THR = 1
    DEFAULT_IMBALANCE_RATIO_THR = 50
    DEFAULT_FAR_FROM_MEAN_THR = 5
    DEFAULT_DOMINANCE_RATIO_THR = 0.8
    DEFAULT_TOPK_BINS = 0.1

    ALL_WARNINGS = {
        AttributeDefinedButNotFound,  # annotation level
        FarFromAttrMean,  # annotation level
        FarFromLabelMean,  # annotation level
        FewSamplesInAttribute,  # annotation level
        FewSamplesInLabel,  # annotation level
        ImbalancedAttribute,  # annotation level
        ImbalancedDistInAttribute,  # annotation level: bbox
        ImbalancedDistInLabel,  # annotation level: bbox
        ImbalancedLabels,  # annotation level
        InvalidValue,  # annotation level
        LabelDefinedButNotFound,  # item level
        MissingAnnotation,  # item level
        MissingAttribute,  # annotation level
        MissingLabelCategories,  # dataset level
        MultiLabelAnnotations,  # item level
        NegativeLength,  # annotation level
        OnlyOneAttributeValue,  # annotation level
        OnlyOneLabel,  # annotation level
        UndefinedAttribute,  # annotation level
        UndefinedLabel,  # annotation level
    }

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-fs",
            "--few-samples-thr",
            default=cls.DEFAULT_FEW_SAMPLES_THR,
            type=int,
            help="Threshold for giving a warning for minimum number of samples per class (default: %(default)s)",
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
        tasks: List[TaskType] = [
            TaskType.classification,
            TaskType.detection,
            TaskType.segmentation,
            TaskType.tabular,
        ],
        warnings: Set[DatasetValidationError] = ALL_WARNINGS,
        few_samples_thr=None,
        imbalance_ratio_thr=None,
        far_from_mean_thr=None,
        dominance_ratio_thr=None,
        topk_bins=None,
    ):
        self.tasks = tasks
        self.warnings = warnings

        self.all_stats = {task: None for task in tasks}

        self.few_samples_thr = few_samples_thr if few_samples_thr else self.DEFAULT_FEW_SAMPLES_THR
        self.imbalance_ratio_thr = imbalance_ratio_thr if imbalance_ratio_thr else self.DEFAULT_IMBALANCE_RATIO_THR
        self.far_from_mean_thr = far_from_mean_thr if far_from_mean_thr else self.DEFAULT_FAR_FROM_MEAN_THR
        self.dominance_thr = dominance_ratio_thr if dominance_ratio_thr else self.DEFAULT_DOMINANCE_RATIO_THR
        self.topk_bins_ratio = topk_bins if topk_bins else self.DEFAULT_TOPK_BINS

    def _init_stats_collector(self, categories):
        for task in self.tasks:
            kwargs = {
                "few_samples_thr": self.few_samples_thr,
                "imbalance_ratio_thr": self.imbalance_ratio_thr,
                "far_from_mean_thr": self.far_from_mean_thr,
                "dominance_thr": self.dominance_thr,
                "topk_bins_ratio": self.topk_bins_ratio,
            }
            if task == TaskType.classification:
                self.all_stats[task] = ClsStats(label_categories=categories, warnings=self.warnings, **kwargs)
            elif task == TaskType.detection:
                self.all_stats[task] = DetStats(label_categories=categories, warnings=self.warnings, **kwargs)
            elif task == TaskType.segmentation:
                self.all_stats[task] = SegStats(label_categories=categories, warnings=self.warnings, **kwargs)
            elif task == TaskType.tabular:
                self.all_stats[task] = TblStats(categories=categories, warnings=self.warnings, **kwargs)

    def _get_stats_collector(self, ann_type, task_type):
        if task_type == [TaskType.tabular]:
            return self.all_stats.get(TaskType.tabular, None)
        if ann_type == AnnotationType.label:
            return self.all_stats.get(TaskType.classification, None)
        if ann_type == AnnotationType.bbox:
            return self.all_stats.get(TaskType.detection, None)
        if ann_type in [
            AnnotationType.mask,
            AnnotationType.polygon,
            AnnotationType.ellipse,
        ]:
            return self.all_stats.get(TaskType.segmentation, None)
        return None

    def compute_statistics(self, dataset):
        if self.tasks == [TaskType.tabular]:
            categories_input = dataset.categories()
        else:
            categories_input = dataset.categories()[AnnotationType.label]
        self._init_stats_collector(categories=categories_input)

        for item in dataset:
            for stats_collector in self.all_stats.values():
                stats_collector.update_item(item)

            item_key = (item.id, item.subset)
            for annotation in item.annotations:
                stats_collector = self._get_stats_collector(ann_type=annotation.type, task_type=self.tasks)
                if stats_collector:
                    stats_collector.update_ann(item_key, annotation)

        return {task: stats_collector for task, stats_collector in self.all_stats.items()}

    def generate_reports(self, task_stats):
        reports = {}
        for task, stats_collector in task_stats.items():
            reports[task] = stats_collector.generate_report()

        return reports
