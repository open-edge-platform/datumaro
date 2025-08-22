# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Optional, Union

import numpy as np
from attr import attrib, attrs

from datumaro.components.abstracts import IMergerContext
from datumaro.components.abstracts.merger import IMatcherContext
from datumaro.components.annotation import Annotation, Points
from datumaro.util.annotation_util import (
    OKS,
    approximate_line,
    bbox_iou,
    max_bbox,
    mean_bbox,
    segment_iou,
)

__all__ = [
    "match_segments_pair",
    "match_segments_more_than_pair",
    "AnnotationMatcher",
    "LabelMatcher",
    "ShapeMatcher",
    "BboxMatcher",
    "PolygonMatcher",
    "MaskMatcher",
    "PointsMatcher",
    "LineMatcher",
    "CaptionsMatcher",
    "Cuboid3dMatcher",
    "ImageAnnotationMatcher",
    "Cuboid2DMatcher",
]


def match_segments_pair(
    a_segms,
    b_segms,
    distance=segment_iou,
    dist_thresh=1.0,
    label_matcher=lambda a, b: a.label == b.label,
):
    """Match segments and return pairs of the two matched segments"""

    assert callable(distance), distance
    assert callable(label_matcher), label_matcher

    a_segms.sort(key=lambda ann: 1 - ann.attributes.get("score", 1))
    b_segms.sort(key=lambda ann: 1 - ann.attributes.get("score", 1))

    # a_matches: indices of b_segms matched to a bboxes
    # b_matches: indices of a_segms matched to b bboxes
    a_matches = -np.ones(len(a_segms), dtype=int)
    b_matches = -np.ones(len(b_segms), dtype=int)

    distances = np.array([[distance(a, b) for b in b_segms] for a in a_segms])

    # matches: boxes we succeeded to match completely
    # mispred: boxes we succeeded to match, having label mismatch
    matches = []
    mispred = []

    # It needs len(a_segms) > 0 and len(b_segms) > 0
    if len(b_segms) > 0:
        for a_idx, a_segm in enumerate(a_segms):
            matched_b = -1
            max_dist = -1
            b_indices = np.argsort(
                [not label_matcher(a_segm, b_segm) for b_segm in b_segms], kind="stable"
            )  # prioritize those with same label, keep score order
            for b_idx in b_indices:
                if 0 <= b_matches[b_idx]:  # assign a_segm with max conf
                    continue
                d = distances[a_idx, b_idx]
                if d < dist_thresh or d <= max_dist:
                    continue
                max_dist = d
                matched_b = b_idx

            if matched_b < 0:
                continue
            a_matches[a_idx] = matched_b
            b_matches[matched_b] = a_idx

            b_segm = b_segms[matched_b]

            if label_matcher(a_segm, b_segm):
                matches.append((a_segm, b_segm))
            else:
                mispred.append((a_segm, b_segm))

    # *_umatched: boxes of (*) we failed to match
    a_unmatched = [a_segms[i] for i, m in enumerate(a_matches) if m < 0]
    b_unmatched = [b_segms[i] for i, m in enumerate(b_matches) if m < 0]

    return matches, mispred, a_unmatched, b_unmatched


def match_segments_more_than_pair(
    a_segms,
    b_segms,
    distance=segment_iou,
    dist_thresh=1.0,
    label_matcher=lambda a, b: a.label == b.label,
):
    """Match segments and return sets of the matched segments which can be more than two"""

    assert callable(distance), distance
    assert callable(label_matcher), label_matcher

    # a_matches: indices of b_segms matched to a bboxes
    # b_matches: indices of a_segms matched to b bboxes
    a_matches = -np.ones(len(a_segms), dtype=int)
    b_matches = -np.ones(len(b_segms), dtype=int)

    distances = np.array([[distance(a, b) for b in b_segms] for a in a_segms])

    # matches: boxes we succeeded to match completely
    # mispred: boxes we succeeded to match, having label mismatch
    matches = []
    mispred = []

    # It needs len(a_segms) > 0 and len(b_segms) > 0
    if len(b_segms) > 0:
        for a_idx, a_segm in enumerate(a_segms):
            b_indices = np.argsort(
                [not label_matcher(a_segm, b_segm) for b_segm in b_segms], kind="stable"
            )  # prioritize those with same label, keep score order
            for b_idx in b_indices:
                d = distances[a_idx, b_idx]
                if d < dist_thresh:
                    continue

                a_matches[a_idx] = b_idx
                b_matches[b_idx] = a_idx

                b_segm = b_segms[b_idx]

                if label_matcher(a_segm, b_segm):
                    matches.append((a_segm, b_segm))
                else:
                    mispred.append((a_segm, b_segm))

    # *_umatched: boxes of (*) we failed to match
    a_unmatched = [a_segms[i] for i, m in enumerate(a_matches) if m < 0]
    b_unmatched = [b_segms[i] for i, m in enumerate(b_matches) if m < 0]

    return matches, mispred, a_unmatched, b_unmatched


@attrs(kw_only=True)
class AnnotationMatcher:
    _context: Optional[Union[IMatcherContext, IMergerContext]] = attrib(default=None)

    def match_annotations(self, sources):
        raise NotImplementedError()


@attrs
class LabelMatcher(AnnotationMatcher):
    def distance(self, a, b):
        a_label = self._context.get_any_label_name(a, a.label)
        b_label = self._context.get_any_label_name(b, b.label)
        return a_label == b_label

    def match_annotations(self, sources):
        return [sum(sources, [])]


@attrs(kw_only=True)
class ShapeMatcher(AnnotationMatcher):
    pairwise_dist = attrib(converter=float, default=0.9)
    cluster_dist = attrib(converter=float, default=-1.0)
    _match_segments = attrib(default=match_segments_pair)

    def match_annotations(self, sources: List[List[Annotation]]) -> List[List[Annotation]]:
        distance = self.distance
        label_matcher = self.label_matcher
        pairwise_dist = self.pairwise_dist
        cluster_dist = self.cluster_dist

        if cluster_dist < 0:
            cluster_dist = pairwise_dist

        id_segm = {id(a): (a, id(s)) for s in sources for a in s}

        def _is_close_enough(cluster, extra_id):
            # check if whole cluster IoU will not be broken
            # when this segment is added
            b = id_segm[extra_id][0]
            for a_id in cluster:
                a = id_segm[a_id][0]
                if distance(a, b) < cluster_dist:
                    return False
            return True

        def _has_same_source(cluster, extra_id):
            b = id_segm[extra_id][1]
            for a_id in cluster:
                a = id_segm[a_id][1]
                if a == b:
                    return True
            return False

        # match segments in sources, pairwise
        adjacent = {i: [] for i in id_segm}  # id(sgm) -> [id(adj_sgm1), ...]
        for a_idx, src_a in enumerate(sources):
            for src_b in sources[a_idx + 1 :]:
                matches, _, _, _ = self._match_segments(
                    src_a,
                    src_b,
                    dist_thresh=pairwise_dist,
                    distance=distance,
                    label_matcher=label_matcher,
                )
                for a, b in matches:
                    adjacent[id(a)].append(id(b))

        # join all segments into matching clusters
        clusters = []
        visited = set()
        for cluster_idx in adjacent:
            if cluster_idx in visited:
                continue

            cluster = set()
            to_visit = {cluster_idx}
            while to_visit:
                c = to_visit.pop()
                cluster.add(c)
                visited.add(c)

                for i in adjacent[c]:
                    if i in visited:
                        continue
                    if 0 < cluster_dist and not _is_close_enough(cluster, i):
                        continue
                    if _has_same_source(cluster, i):
                        continue

                    to_visit.add(i)

            clusters.append([id_segm[i][0] for i in cluster])

        return clusters

    def distance(self, a, b):
        return segment_iou(a, b)

    def label_matcher(self, a, b):
        a_label = self._context.get_any_label_name(a, a.label)
        b_label = self._context.get_any_label_name(b, b.label)
        return a_label == b_label


@attrs
class BboxMatcher(ShapeMatcher):
    pass


@attrs
class PolygonMatcher(ShapeMatcher):
    pass


@attrs
class MaskMatcher(ShapeMatcher):
    pass


@attrs(kw_only=True)
class PointsMatcher(ShapeMatcher):
    sigma: Optional[list] = attrib(default=None)
    instance_map = attrib(converter=dict)

    def distance(self, a, b):
        a_bbox = self.instance_map[id(a)][1]
        b_bbox = self.instance_map[id(b)][1]
        if bbox_iou(a_bbox, b_bbox) <= 0:
            return 0
        bbox = mean_bbox([a_bbox, b_bbox])
        return OKS(a, b, sigma=self.sigma, bbox=bbox)


@attrs
class LineMatcher(ShapeMatcher):
    def distance(self, a, b):
        # Compute inter-line area by using the Trapezoid formulae
        # https://en.wikipedia.org/wiki/Trapezoidal_rule
        # Normalize by common bbox and get the bbox fill ratio
        # Call this ratio the "distance"

        # The box area is an early-exit filter for non-intersected figures
        bbox = max_bbox([a, b])
        box_area = bbox[2] * bbox[3]
        if not box_area:
            return 1

        def _approx(line, segments):
            if len(line) // 2 != segments + 1:
                line = approximate_line(line, segments=segments)
            return np.reshape(line, (-1, 2))

        segments = max(len(a.points) // 2, len(b.points) // 2, 5) - 1

        a = _approx(a.points, segments)
        b = _approx(b.points, segments)
        dists = np.linalg.norm(a - b, axis=1)
        dists = dists[:-1] + dists[1:]
        a_steps = np.linalg.norm(a[1:] - a[:-1], axis=1)
        b_steps = np.linalg.norm(b[1:] - b[:-1], axis=1)

        # For the common bbox we can't use
        # - the AABB (axis-alinged bbox) of a point set
        # - the exterior of a point set
        # - the convex hull of a point set
        # because these soultions won't be correctly normalized.
        # The lines can have multiple self-intersections, which can give
        # the inter-line area more than internal area of the options above,
        # producing the value of the distance outside of the [0; 1] range.
        #
        # Instead, we can compute the upper boundary for the inter-line
        # area based on the maximum point distance and line length.
        max_area = np.max(dists) * max(np.sum(a_steps), np.sum(b_steps))

        area = np.dot(dists, a_steps + b_steps) * 0.5 * 0.5 / max(max_area, 1.0)

        return abs(1 - area)


@attrs
class CaptionsMatcher(AnnotationMatcher):
    def match_annotations(self, sources):
        raise NotImplementedError()


@attrs
class Cuboid3dMatcher(ShapeMatcher):
    def distance(self, a, b):
        raise NotImplementedError()


@attrs
class ImageAnnotationMatcher(AnnotationMatcher):
    def match_annotations(self, sources):
        raise NotImplementedError()


@attrs
class TabularMatcher(AnnotationMatcher):
    def match_annotations(self, sources):
        raise NotImplementedError()


@attrs
class RotatedBboxMatcher(ShapeMatcher):
    sigma: Optional[list] = attrib(default=None)

    def distance(self, a, b):
        a = Points([p for pt in a.as_polygon() for p in pt])
        b = Points([p for pt in b.as_polygon() for p in pt])

        return OKS(a, b, sigma=self.sigma)


@attrs
class Cuboid2DMatcher(ShapeMatcher):
    pass
