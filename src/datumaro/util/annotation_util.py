# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from itertools import groupby
from typing import Callable, Dict, Iterable, NewType, Optional, Sequence, Tuple, Union

import numpy as np
from typing_extensions import Literal

from datumaro.components.annotation import AnnotationType, LabelCategories, Mask, RleMask, Shape
from datumaro.util.mask_tools import mask_to_rle


def find_instances(instance_anns):
    instance_anns = sorted(instance_anns, key=lambda a: a.group)
    ann_groups = []
    for g_id, group in groupby(instance_anns, lambda a: a.group):
        if not g_id:
            ann_groups.extend(([a] for a in group))
        else:
            ann_groups.append(list(group))

    return ann_groups


def find_group_leader(group):
    return max(group, key=lambda x: x.get_area())


BboxCoords = Tuple[float, float, float, float]
_Shape = NewType("_Shape", Shape)
SpatialAnnotation = Union[_Shape, Mask]


def _get_bbox(ann: Union[Sequence, SpatialAnnotation]) -> BboxCoords:
    if isinstance(ann, (Shape, Mask)):
        return ann.get_bbox()
    elif hasattr(ann, "__len__") and len(ann) == 4:
        return ann
    elif hasattr(ann, "__len__") and len(ann) == 0:
        return [0, 0, 0, 0]
    else:
        raise ValueError("The value of type '%s' can't be treated as a " "bounding box" % type(ann))


def max_bbox(annotations: Iterable[Union[BboxCoords, SpatialAnnotation]]) -> BboxCoords:
    """
    Computes the maximum bbox for the set of spatial annotations and boxes.

    Returns:
      bbox (tuple): (x, y, w, h)
    """

    boxes = [_get_bbox(ann) for ann in annotations]
    x0 = min((b[0] for b in boxes), default=0)
    y0 = min((b[1] for b in boxes), default=0)
    x1 = max((b[0] + b[2] for b in boxes), default=0)
    y1 = max((b[1] + b[3] for b in boxes), default=0)
    return [x0, y0, x1 - x0, y1 - y0]


def mean_bbox(annotations: Iterable[Union[BboxCoords, SpatialAnnotation]]) -> BboxCoords:
    """
    Computes the mean bbox for the set of spatial annotations and boxes.

    Returns:
      bbox (tuple): (x, y, w, h)
    """

    le = len(annotations)
    boxes = [_get_bbox(ann) for ann in annotations]
    mlb = sum(b[0] for b in boxes) / le
    mtb = sum(b[1] for b in boxes) / le
    mrb = sum(b[0] + b[2] for b in boxes) / le
    mbb = sum(b[1] + b[3] for b in boxes) / le
    return [mlb, mtb, mrb - mlb, mbb - mtb]


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def nms(segments, iou_thresh=0.5):
    """
    Non-maxima suppression algorithm.
    """

    indices = np.argsort([b.attributes["score"] for b in segments])
    ious = np.array([[segment_iou(a, b) for b in segments] for a in segments])

    predictions = []
    while len(indices) != 0:
        i = len(indices) - 1
        pred_idx = indices[i]
        to_remove = [i]
        predictions.append(segments[pred_idx])
        for i, box_idx in enumerate(indices[:i]):
            if iou_thresh < ious[pred_idx, box_idx]:
                to_remove.append(i)
        indices = np.delete(indices, to_remove)

    return predictions


def bbox_iou(a, b) -> Union[Literal[-1], float]:
    """
    IoU computations for simple cases with bounding boxes
    """
    bbox_a = _get_bbox(a)
    bbox_b = _get_bbox(b)

    aX, aY, aW, aH = bbox_a
    bX, bY, bW, bH = bbox_b
    in_right = min(aX + aW, bX + bW)
    in_left = max(aX, bX)
    in_top = max(aY, bY)
    in_bottom = min(aY + aH, bY + bH)

    in_w = max(0, in_right - in_left)
    in_h = max(0, in_bottom - in_top)
    intersection = in_w * in_h
    if not intersection:
        return -1

    a_area = aW * aH
    b_area = bW * bH
    union = a_area + b_area - intersection
    return intersection / union


def segment_iou(a, b):
    """
    Generic IoU computation with masks, polygons, and boxes.
    Returns -1 if no intersection, [0; 1] otherwise
    """
    from pycocotools import mask as mask_utils

    a_bbox = list(a.get_bbox())
    b_bbox = list(b.get_bbox())

    types = {a.type, b.type}
    is_bbox = AnnotationType.bbox in types or AnnotationType.ellipse in types
    if is_bbox:
        a = [a_bbox]
        b = [b_bbox]
    else:
        w = max(a_bbox[0] + a_bbox[2], b_bbox[0] + b_bbox[2])
        h = max(a_bbox[1] + a_bbox[3], b_bbox[1] + b_bbox[3])

        def _to_rle(ann):
            if ann.type == AnnotationType.polygon:
                return mask_utils.frPyObjects([ann.points], h, w)
            elif isinstance(ann, RleMask):
                return [ann.rle]
            elif ann.type == AnnotationType.mask:
                return mask_utils.frPyObjects([mask_to_rle(ann.image)], h, w)
            else:
                raise TypeError("Unexpected arguments: %s, %s" % (a, b))

        a = _to_rle(a)
        b = _to_rle(b)
    iou = mask_utils.iou(a, b, [not is_bbox])
    if isinstance(iou, np.ndarray):
        iou = iou[0, 0]
    return float(iou)


def PDJ(a, b, eps=None, ratio=0.05, bbox=None):
    """
    Percentage of Detected Joints metric.
    Counts the number of matching points.
    """

    assert eps is not None or ratio is not None

    p1 = np.array(a.points).reshape((-1, 2))
    p2 = np.array(b.points).reshape((-1, 2))
    if len(p1) != len(p2):
        return 0

    if not eps:
        if bbox is None:
            bbox = mean_bbox([a, b])

        diag = (bbox[2] ** 2 + bbox[3] ** 2) ** 0.5
        eps = ratio * diag

    dists = np.linalg.norm(p1 - p2, axis=1)
    return np.sum(dists < eps) / len(p1)


def OKS(a, b, sigma=None, bbox=None, scale=None):
    """
    Object Keypoint Similarity metric.
    https://cocodataset.org/#keypoints-eval
    """

    p1 = np.array(a.points).reshape((-1, 2))
    p2 = np.array(b.points).reshape((-1, 2))
    if len(p1) != len(p2):
        return 0

    if not sigma:
        sigma = 0.1
    else:
        assert len(sigma) == len(p1)

    if not scale:
        if bbox is None:
            bbox = mean_bbox([a, b])
        scale = bbox[2] * bbox[3]

    dists = np.linalg.norm(p1 - p2, axis=1)
    return np.sum(np.exp(-(dists**2) / (2 * scale * (2 * sigma) ** 2)))


def approximate_line(points: Sequence[float], segments: int) -> np.ndarray:
    """
    Approximates a 2d line to the required number of segments. The new points
    are distributed uniformly across the input line.

    Args:
      points (Sequence): an array of line point coordinates.
        The size is [points * 2], the layout is [x0, y0, x1, y1, ...].
      segments (int): the required numebr of segments in the resulting line.

    Returns:
        new_points (numpy.ndarray): an array of new line point coordinates.
    Returns:
        The size is [(segments + 1) * 2], the layout is [x0, y0, x1, y1, ...].
    """

    assert 2 <= len(points) // 2 and len(points) % 2 == 0
    assert 0 < segments

    points = list(points)
    if len(points) == 2:
        points.extend(points)
    points = np.array(points).reshape((-1, 2))

    lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    dists = [0]
    for l in lengths:
        dists.append(dists[-1] + l)

    step = dists[-1] / segments

    new_points = np.zeros((segments + 1, 2))
    new_points[0] = points[0]

    old_segment = 0
    for new_segment in range(1, segments + 1):
        pos = new_segment * step
        while dists[old_segment + 1] < pos and old_segment + 2 < len(dists):
            old_segment += 1

        segment_start = dists[old_segment]
        segment_len = lengths[old_segment]
        prev_p = points[old_segment]
        next_p = points[old_segment + 1]
        r = (pos - segment_start) / segment_len

        new_points[new_segment] = prev_p * (1 - r) + next_p * r

    return np.reshape(new_points, (-1,))


def make_label_id_mapping(
    src_labels: LabelCategories, dst_labels: LabelCategories, fallback: int = 0
) -> Tuple[Callable[[int], Optional[int]], Dict[int, int], Dict[int, str], Dict[int, str]]:
    """
    Maps label ids from source to destination. Fallback id is used for missing
    labels.

    Returns:

    |   map_id (callable): src id -> dst id
    |   id_mapping (dict): src id -> dst i
    |   src_labels (dict): src id -> src label
    |   dst_labels (dict): dst id -> dst label
    """

    source_labels = {id: label.name for id, label in enumerate(src_labels or ())}
    target_labels = {label.name: id for id, label in enumerate(dst_labels or ())}
    id_mapping = {
        src_id: target_labels.get(src_label, fallback)
        for src_id, src_label in source_labels.items()
    }

    def map_id(src_id):
        return id_mapping.get(src_id, fallback)

    return map_id, id_mapping, source_labels, target_labels
