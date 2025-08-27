# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from attr import attrib, attrs

from datumaro.components.annotation import Bbox, Label
from datumaro.components.errors import FailedLabelVotingError
from datumaro.util.annotation_util import mean_bbox, segment_iou

from .matcher import (
    AnnotationMatcher,
    BboxMatcher,
    CaptionsMatcher,
    Cuboid2DMatcher,
    Cuboid3dMatcher,
    ImageAnnotationMatcher,
    LabelMatcher,
    LineMatcher,
    MaskMatcher,
    PointsMatcher,
    PolygonMatcher,
    RotatedBboxMatcher,
    ShapeMatcher,
    TabularMatcher,
)

__all__ = [
    "AnnotationMerger",
    "LabelMerger",
    "BboxMerger",
    "RotatedBboxMerger",
    "PolygonMerger",
    "MaskMerger",
    "PointsMerger",
    "LineMerger",
    "CaptionsMerger",
    "Cuboid3dMerger",
    "ImageAnnotationMerger",
    "EllipseMerger",
]


@attrs(kw_only=True)
class AnnotationMerger(AnnotationMatcher):
    def merge_clusters(self, clusters):
        raise NotImplementedError()


@attrs(kw_only=True)
class LabelMerger(AnnotationMerger, LabelMatcher):
    quorum = attrib(converter=int, default=0)

    def merge_clusters(self, clusters):
        assert len(clusters) <= 1
        if len(clusters) == 0:
            return []

        votes = {}  # label -> score
        for ann in clusters[0]:
            label = self._context._get_src_label_name(ann, ann.label)
            votes[label] = 1 + votes.get(label, 0)

        merged = []
        for label, count in votes.items():
            if count < self.quorum:
                sources = set(
                    self.get_ann_source(id(a))
                    for a in clusters[0]
                    if label not in [self._context._get_src_label_name(l, l.label) for l in a]
                )
                sources = [self._context._dataset_map[s][1] for s in sources]
                self._context.add_item_error(FailedLabelVotingError, votes, sources=sources)
                continue

            merged.append(
                Label(
                    self._context._get_label_id(label),
                    attributes={"score": count / len(self._context._dataset_map)},
                )
            )

        return merged


@attrs(kw_only=True)
class _ShapeMerger(AnnotationMerger, ShapeMatcher):
    quorum = attrib(converter=int, default=0)

    def merge_clusters(self, clusters):
        return list(map(self.merge_cluster, clusters))

    def find_cluster_label(self, cluster):
        votes = {}
        for s in cluster:
            label = self._context._get_src_label_name(s, s.label)
            state = votes.setdefault(label, [0, 0])
            state[0] += s.attributes.get("score", 1.0)
            state[1] += 1

        label, (score, count) = max(votes.items(), key=lambda e: e[1][0])
        if count < self.quorum:
            self._context.add_item_error(FailedLabelVotingError, votes)
            label = None
        score = score / len(self._context._dataset_map)
        label = self._context._get_label_id(label)
        return label, score

    @staticmethod
    def _merge_cluster_shape_mean_box_nearest(cluster):
        mbbox = Bbox(*mean_bbox(cluster))
        dist = (segment_iou(mbbox, s) for s in cluster)
        nearest_pos, _ = max(enumerate(dist), key=lambda e: e[1])
        return cluster[nearest_pos]

    def merge_cluster_shape(self, cluster):
        shape = self._merge_cluster_shape_mean_box_nearest(cluster)
        shape_score = sum(max(0, self.distance(shape, s)) for s in cluster) / len(cluster)
        return shape, shape_score

    def merge_cluster(self, cluster):
        label, label_score = self.find_cluster_label(cluster)
        shape, shape_score = self.merge_cluster_shape(cluster)

        shape.z_order = max(cluster, key=lambda a: a.z_order).z_order
        shape.label = label
        shape.attributes["score"] = label_score * shape_score if label is not None else shape_score

        return shape


@attrs
class BboxMerger(_ShapeMerger, BboxMatcher):
    pass


@attrs
class PolygonMerger(_ShapeMerger, PolygonMatcher):
    pass


@attrs
class MaskMerger(_ShapeMerger, MaskMatcher):
    pass


@attrs
class PointsMerger(_ShapeMerger, PointsMatcher):
    pass


@attrs
class LineMerger(_ShapeMerger, LineMatcher):
    pass


@attrs
class CaptionsMerger(AnnotationMerger, CaptionsMatcher):
    pass


@attrs
class Cuboid3dMerger(_ShapeMerger, Cuboid3dMatcher):
    @staticmethod
    def _merge_cluster_shape_mean_box_nearest(cluster):
        raise NotImplementedError()
        # mbbox = Bbox(*mean_cuboid(cluster))
        # dist = (segment_iou(mbbox, s) for s in cluster)
        # nearest_pos, _ = max(enumerate(dist), key=lambda e: e[1])
        # return cluster[nearest_pos]

    def merge_cluster(self, cluster):
        label, label_score = self.find_cluster_label(cluster)
        shape, shape_score = self.merge_cluster_shape(cluster)

        shape.label = label
        shape.attributes["score"] = label_score * shape_score if label is not None else shape_score

        return shape


@attrs
class ImageAnnotationMerger(AnnotationMerger, ImageAnnotationMatcher):
    pass


@attrs
class EllipseMerger(_ShapeMerger, ShapeMatcher):
    pass


@attrs
class TabularMerger(AnnotationMerger, TabularMatcher):
    pass


@attrs
class RotatedBboxMerger(_ShapeMerger, RotatedBboxMatcher):
    pass


@attrs
class Cuboid2DMerger(_ShapeMerger, Cuboid2DMatcher):
    pass
