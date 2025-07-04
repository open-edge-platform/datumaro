# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import math
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import shapely.geometry as sg
import shapely.ops as so

from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    Caption,
    DepthAnnotation,
    Label,
    Mask,
    Points,
    Polygon,
    PolyLine,
)
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatumaroError
from datumaro.components.media import BboxIntCoords, MosaicImage
from datumaro.components.transformer import Transform
from datumaro.plugins.tiling.util import x1y1x2y2_to_xywh, xywh_to_x1y1x2y2

AnnotationsForMerge = List[Tuple[Annotation, BboxIntCoords, sg.Polygon]]


def _apply_offset(geom: sg.base.BaseGeometry, roi_box: sg.Polygon) -> sg.base.BaseGeometry:
    offset_x, offset_y = roi_box.bounds[:2]
    return so.transform(lambda x, y: (x + offset_x, y + offset_y), geom)


def _merge_mask(
    anns: AnnotationsForMerge, img_size: Tuple[int, int], *args, **kwargs
) -> List[Mask]:
    merged_masks = []
    group_by_label = defaultdict(list)

    for ann, roi_int, _ in anns:
        group_by_label[ann.label] += [(ann, roi_int)]

    for grouped_anns in group_by_label.values():
        tiled_mask = np.zeros(shape=(img_size[0], img_size[1]), dtype=np.uint8)

        for ann, roi_int in grouped_anns:
            x, y, w, h = roi_int
            tiled_mask[y : y + h, x : x + w] = ann.image

        merged_masks += [
            ann.wrap(
                image=tiled_mask,
                attributes=deepcopy(ann.attributes),
            )
        ]

    return merged_masks


def _merge_points(anns: AnnotationsForMerge, *args, **kwargs) -> List[Points]:
    merged_points = []

    for ann, _, roi_box in anns:
        points = sg.MultiPoint(ann.get_points())

        points = _apply_offset(points, roi_box)

        merged_points += [
            ann.wrap(
                points=[v for point in points.geoms for v in (point.x, point.y)],
                attributes=deepcopy(ann.attributes),
                visibility=deepcopy(ann.visibility),
            )
        ]

    return merged_points


def _merge_polygon(anns: AnnotationsForMerge, *args, **kwargs) -> List[Polygon]:
    merged_polygons = []

    group_by_id = defaultdict(list)

    for ann, _, roi_box in anns:
        group_by_id[ann.id] += [(ann, roi_box)]

    for grouped_anns in group_by_id.values():
        polygon = sg.Polygon()
        for ann, roi_box in grouped_anns:
            polygon = polygon.union(_apply_offset(sg.Polygon(ann.get_points()), roi_box))

        merged_polygons += [
            ann.wrap(
                points=[p for xy in polygon.exterior.coords for p in xy],
                attributes=deepcopy(ann.attributes),
            )
        ]

    return merged_polygons


def _merge_polyline(anns: AnnotationsForMerge, *args, **kwargs) -> List[PolyLine]:
    merged_polylines = []

    for ann, _, roi_box in anns:
        lines = sg.LineString(ann.get_points())

        lines = _apply_offset(lines, roi_box)

        merged_polylines += [
            ann.wrap(
                points=[v for point in lines.coords for v in (point[0], point[1])],
                attributes=deepcopy(ann.attributes),
            )
        ]

    return merged_polylines


def _merge_bbox(anns: AnnotationsForMerge, *args, **kwargs) -> List[Bbox]:
    merged_bboxes = []

    group_by_id = defaultdict(list)

    for ann, _, roi_box in anns:
        group_by_id[ann.id] += [(ann, roi_box)]

    for grouped_anns in group_by_id.values():
        minx, miny, maxx, maxy = math.inf, math.inf, -math.inf, -math.inf

        for ann, roi_box in grouped_anns:
            bbox: sg.Polygon = sg.box(*xywh_to_x1y1x2y2(*ann.get_bbox()))
            bbox = _apply_offset(bbox, roi_box)
            c_minx, c_miny, c_maxx, c_maxy = bbox.bounds
            minx = min(minx, c_minx)
            miny = min(miny, c_miny)
            maxx = max(maxx, c_maxx)
            maxy = max(maxy, c_maxy)

        x, y, w, h = x1y1x2y2_to_xywh(minx, miny, maxx, maxy)

        merged_bboxes += [
            ann.wrap(
                x=x,
                y=y,
                w=w,
                h=h,
                attributes=deepcopy(ann.attributes),
            )
        ]

    return merged_bboxes


def _merge_depth_annotation(
    anns: AnnotationsForMerge, img_size: Tuple[int, int], *args, **kwargs
) -> List[DepthAnnotation]:
    depth_img = np.zeros(shape=(img_size[0], img_size[1]))

    for ann, roi_int, _ in anns:
        x, y, w, h = roi_int
        depth_img[y : y + h, x : x + w] = ann.image

    return [ann.wrap(image=depth_img, attributes=deepcopy(ann.attributes))]


def _merge_by_copy(
    anns: AnnotationsForMerge, img_size: Tuple[int, int], *args, **kwargs
) -> Union[Label, Caption]:
    new_anns = {}
    for ann, _, _ in anns:
        label = getattr(ann, "label", None)
        caption = getattr(ann, "caption", None)

        if label is not None:
            new_anns[label] = ann
        elif caption is not None:
            new_anns[caption] = ann
        else:
            raise DatumaroError("The annotation should be Label or Caption.")

    return [ann.wrap(attributes=deepcopy(ann.attributes)) for ann in new_anns.values()]


def _merge_not_support(ann_type: AnnotationType, *args, **kwargs) -> None:
    raise DatumaroError(f"type(ann)={ann_type} is not support tiling.")


class MergeTile(Transform, CliPlugin):
    """
    Transformation to merge the previously tiled dataset. It can generally
    be understood as the inverse transform of TileTransform. However,
    A sequence of Tile -> MergeTile is a lossy transformation.
    It means that annotation information may be lost if some annotations
    are exists on the edge of tiled images. Therefore, it is generally
    better to revert TileTransform when you need to merge them. But,
    this will be helpful when you have another transformation between
    Tile and MergeTile. For example, Tile -> (an arbitrary Transform) -> MergeTile.
    """

    _merge_anns_func_map: Dict[AnnotationType, Callable[..., List[Annotation]]] = {
        AnnotationType.label: _merge_by_copy,
        AnnotationType.mask: _merge_mask,
        AnnotationType.points: _merge_points,
        AnnotationType.polygon: _merge_polygon,
        AnnotationType.polyline: _merge_polyline,
        AnnotationType.bbox: _merge_bbox,
        AnnotationType.caption: _merge_by_copy,
        AnnotationType.cuboid_3d: _merge_not_support,
        AnnotationType.super_resolution_annotation: _merge_not_support,
        AnnotationType.depth_annotation: _merge_depth_annotation,
    }

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        return parser

    def __init__(self, extractor):
        super().__init__(extractor)

    def __iter__(self):
        items_to_merge = defaultdict(list)

        for item in self._extractor:
            item_id = item.attributes.get("tile_id")
            roi = item.attributes.get("roi")

            if item_id is not None and roi is not None:
                items_to_merge[item_id] += [item]

        for item_id, items in items_to_merge.items():
            yield self._merge_items(item_id, items)

    def _merge_items(self, item_id: str, items: List[DatasetItem]) -> DatasetItem:
        assert len(items) > 0

        max_h = 0
        max_w = 0
        for item in items:
            roi = item.attributes.get("roi")
            x, y, w, h = roi
            max_w = max(max_w, x + w)
            max_h = max(max_h, y + h)
        img_size = (max_h, max_w)

        merged_item = self.wrap_item(
            items[0],
            id=item_id,
            media=MosaicImage.from_image_roi_pairs(
                [
                    (
                        item.media,
                        item.attributes.get("roi"),
                    )
                    for item in items
                ],
                img_size,
            ),
            attributes=self._merge_tiled_attributes(items),
            annotations=self._merge_tiled_annotations(items, img_size),
        )

        return merged_item

    @staticmethod
    def _merge_tiled_attributes(items: List[DatasetItem]) -> Dict[str, Any]:
        attrs = {}
        for item in items:
            attrs.update(item.attributes)

        del attrs["tile_idx"]
        del attrs["tile_id"]
        del attrs["roi"]
        return attrs

    def _merge_tiled_annotations(
        self, items: List[DatasetItem], img_size: Tuple[int, int]
    ) -> List[Annotation]:
        anns_to_merge: Dict[AnnotationType, AnnotationsForMerge] = defaultdict(list)

        for item in items:
            roi = item.attributes.get("roi")
            roi_box: sg.Polygon = sg.box(*xywh_to_x1y1x2y2(*roi))

            for ann in item.annotations:
                anns_to_merge[ann.type] += [(ann, roi, roi_box)]

        merged_anns = []

        for ann_type, anns in anns_to_merge.items():
            merged_anns += self._merge_anns_func_map[ann_type](
                anns=anns, img_size=img_size, ann_type=ann_type
            )

        return merged_anns
