# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Optional, Set

from datumaro.components.abstracts.merger import IMatcherContext
from datumaro.components.annotation import Annotation, AnnotationType, LabelCategories
from datumaro.components.annotations.matcher import BboxMatcher, match_segments_more_than_pair
from datumaro.components.dataset_base import DatasetItem, IDataset
from datumaro.components.launcher import Launcher
from datumaro.components.transformer import ModelTransform
from datumaro.util.deprecation import deprecated


@deprecated(deprecated_version="1.11", removed_version="1.12")
class MissingAnnotationDetection(ModelTransform):
    """This class is used to find annotations that are missing from the ground truth annotations.

    To accomplish this, it generates model predictions for the dataset using the given launcher.
    However, not all of these model predictions can be considered missing annotations since the dataset
    already contains ground truth annotations and some of the model predictions can be duplicated with them.
    To identify the missing annotations from the model predictions, this class filters out the predictions
    that spatially overlap with the ground truth annotations.

    For example, the follwing example will produce ``[Bbox(1, 1, 1, 1, label=1, attributes={"score": 0.5})]`` as
    the missing annotations since ``Bbox(0, 0, 1, 1, label=1, attributes={"score": 1.0})`` is overlapped with the
    ground-truth annotation, ``Bbox(0, 0, 1, 1, label=0)`` (``label_agnostic_matching=True``)

    .. code-block:: python

        ground_truth_annotations = [
            Bbox(0, 0, 1, 1, label=0),
            Bbox(1, 0, 1, 1, label=1),
            Bbox(0, 1, 1, 1, label=2),
        ]
        model_predictions = [
            Bbox(0, 0, 1, 1, label=1, attributes={"score": 1.0}),
            Bbox(1, 1, 1, 1, label=1, attributes={"score": 0.5}),
        ]

    Args:
        extractor: The dataset used to find missing labeled annotations.
        launcher: The launcher used to generate model predictions from the dataset.
        batch_size: The size of the batches used during processing.
        pairwise_dist: The distance metric used to measure the distance between two annotations.
            Typically, the distance metric is Intersection over Union (IoU), which is bounded between 0 and 1.
        score_threshold: The minimum score required for an annotation to be considered
            a candidate for missing annotations.
        label_agnostic_matching: If set to false, annotations with different labels are not matched
            to determine their spatial overlap. In the above example, ``label_agnostic_matching=False``
            will produce ``model_predictions`` as is since ``Bbox(0, 0, 1, 1, label=1, attributes={"score": 1.0})``
            has different label with ``Bbox(0, 0, 1, 1, label=0)``.
    """

    def __init__(
        self,
        extractor: IDataset,
        launcher: Launcher,
        batch_size: int = 1,
        pairwise_dist: float = 0.75,
        score_threshold: Optional[float] = None,
        label_agnostic_matching: bool = True,
    ):
        super().__init__(extractor, launcher, batch_size, append_annotation=False)
        self._score_threshold = score_threshold

        class LabelAgnosticMatcherContext(IMatcherContext):
            def get_any_label_name(self, ann: Annotation, label_id: int) -> str:
                return ""

        label_categories: LabelCategories = self.categories()[AnnotationType.label]

        class LabelSpecificMatcherContext(IMatcherContext):
            def get_any_label_name(self, ann: Annotation, label_id: int) -> str:
                return label_categories[label_id]

        self._support_matchers = {
            AnnotationType.bbox: BboxMatcher(
                pairwise_dist=pairwise_dist,
                context=LabelAgnosticMatcherContext()
                if label_agnostic_matching
                else LabelSpecificMatcherContext(),
                match_segments=match_segments_more_than_pair,
            ),
        }

    def _process_batch(
        self,
        batch: List[DatasetItem],
    ) -> List[DatasetItem]:
        inference = self._launcher.launch(
            batch=[item for item in batch if self._launcher.type_check(item)]
        )

        for annotations in inference:
            self._check_annotations(annotations)

        return [
            self.wrap_item(
                item,
                annotations=self._find_missing_anns(
                    gt_anns=item.annotations,
                    pseudo_anns=self._apply_score_threshold(annotations),
                ),
            )
            for item, annotations in zip(batch, inference)
        ]

    def _apply_score_threshold(self, annotations: List[Annotation]) -> List[Annotation]:
        if self._score_threshold is None:
            return annotations

        return [
            ann for ann in annotations if ann.attributes.get("score", 1.0) > self._score_threshold
        ]

    def _find_missing_anns(
        self, gt_anns: List[Annotation], pseudo_anns: List[Annotation]
    ) -> List[Annotation]:
        ids_of_pseudo_anns = set(id(ann) for ann in pseudo_anns)

        missing_labeled_anns = []
        for ann_type, matcher in self._support_matchers.items():
            clusters = matcher.match_annotations(
                [
                    [ann for ann in gt_anns if ann.type == ann_type],
                    [ann for ann in pseudo_anns if ann.type == ann_type],
                ]
            )
            for cluster in clusters:
                ann = self._pick_missing_ann_from_cluster(cluster, ids_of_pseudo_anns)
                if ann is not None:
                    missing_labeled_anns.append(ann)

        return missing_labeled_anns

    @staticmethod
    def _pick_missing_ann_from_cluster(
        cluster: List[Annotation], ids_of_pseudo_anns: Set[int]
    ) -> Optional[Annotation]:
        assert len(cluster) > 0, "cluster should not be empty."
        pseudo_label_anns = []
        gt_label_anns = []

        for ann in cluster:
            if id(ann) in ids_of_pseudo_anns:
                pseudo_label_anns.append(ann)
            else:
                gt_label_anns.append(ann)

        if len(gt_label_anns) > 0:
            return None

        return max(pseudo_label_anns, key=lambda ann: ann.attributes.get("score", -1))
