# Copyright (C) 2020-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import logging as log
import os.path as osp
import random
import re
import string
from collections import Counter, defaultdict
from copy import deepcopy
from enum import Enum, auto
from itertools import chain
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils
from pandas.api.types import CategoricalDtype

import datumaro.util.mask_tools as mask_tools
from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    Ellipse,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    RleMask,
    Tabular,
    TabularCategories,
)
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetInfo, DatasetItem, IDataset
from datumaro.components.errors import (
    AnnotationTypeError,
    DatumaroError,
    EmptyCaption,
    EmptyLabel,
    FarFromAttrMean,
    FarFromCaptionMean,
    FarFromLabelMean,
    InvalidValue,
    MissingAnnotation,
    MissingAttribute,
    MissingLabelCategories,
    MultiLabelAnnotations,
    NegativeLength,
    OutlierInCaption,
    RedundanciesInCaption,
    UndefinedAttribute,
    UndefinedLabel,
)
from datumaro.components.media import Image, TableRow, VideoFrame
from datumaro.components.transformer import ItemTransform, TabularTransform, Transform
from datumaro.util import NOTSET, filter_dict, parse_json_file, parse_str_enum_value, take_by
from datumaro.util.annotation_util import find_group_leader, find_instances
from datumaro.util.tabular_util import emoji_pattern


class CropCoveredSegments(ItemTransform, CliPlugin):
    """
    Sorts polygons and masks ("segments") according to `z_order`,
    crops covered areas of underlying segments. If a segment is split
    into several independent parts by the segments above, produces
    the corresponding number of separate annotations joined into a group.
    """

    def transform_item(self, item):
        annotations = []
        segments = []
        for ann in item.annotations:
            if ann.type in {AnnotationType.polygon, AnnotationType.mask}:
                segments.append(ann)
            else:
                annotations.append(ann)
        if not segments:
            return item

        if not isinstance(item.media, Image):
            raise Exception("Image info is required for this transform")
        h, w = item.media.size
        segments = self.crop_segments(segments, w, h)

        annotations += segments
        return self.wrap_item(item, annotations=annotations)

    @classmethod
    def crop_segments(cls, segment_anns, img_width, img_height):
        segment_anns = sorted(segment_anns, key=lambda x: x.z_order)

        segments = []
        for s in segment_anns:
            if s.type == AnnotationType.polygon:
                segments.append(s.points)
            elif s.type == AnnotationType.mask:
                if isinstance(s, RleMask):
                    rle = s.rle
                else:
                    rle = mask_tools.mask_to_rle(s.image)
                segments.append(rle)

        segments = mask_tools.crop_covered_segments(segments, img_width, img_height)

        new_anns = []
        for ann, new_segment in zip(segment_anns, segments):
            fields = {
                "z_order": ann.z_order,
                "label": ann.label,
                "id": ann.id,
                "group": ann.group,
                "attributes": ann.attributes,
            }
            if ann.type == AnnotationType.polygon:
                if fields["group"] is None:
                    fields["group"] = cls._make_group_id(segment_anns + new_anns, fields["id"])
                for polygon in new_segment:
                    new_anns.append(Polygon(points=polygon, **fields))
            else:
                rle = mask_tools.mask_to_rle(new_segment)
                rle = mask_utils.frPyObjects(rle, *rle["size"])
                new_anns.append(RleMask(rle=rle, **fields))

        return new_anns

    @staticmethod
    def _make_group_id(anns, ann_id):
        if ann_id:
            return ann_id
        max_gid = max(anns, default=0, key=lambda x: x.group)
        return max_gid + 1


class MergeInstanceSegments(ItemTransform, CliPlugin):
    """
    Replaces instance masks and, optionally, polygons with a single mask.
    A group of annotations with the same group id is considered an "instance".
    The largest annotation in the group is considered the group "head", so the
    resulting mask takes properties from that annotation.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("--include-polygons", action="store_true", help="Include polygons")
        return parser

    def __init__(self, extractor, include_polygons=False):
        super().__init__(extractor)

        self._include_polygons = include_polygons

    def transform_item(self, item):
        annotations = []
        segments = []
        for ann in item.annotations:
            if ann.type in {AnnotationType.polygon, AnnotationType.mask}:
                segments.append(ann)
            else:
                annotations.append(ann)
        if not segments:
            return item

        if not isinstance(item.media, Image):
            raise Exception("Image info is required for this transform")
        h, w = item.media.size
        instances = self.find_instances(segments)
        segments = [self.merge_segments(i, w, h, self._include_polygons) for i in instances]
        segments = sum(segments, [])

        annotations += segments
        return self.wrap_item(item, annotations=annotations)

    @classmethod
    def merge_segments(cls, instance, img_width, img_height, include_polygons=False):
        polygons = [a for a in instance if a.type == AnnotationType.polygon]
        masks = [a for a in instance if a.type == AnnotationType.mask]
        if not polygons and not masks:
            return []
        if not polygons and len(masks) == 1:
            return masks

        leader = find_group_leader(polygons + masks)
        instance = []

        # Build the resulting mask
        mask = None

        if include_polygons and polygons:
            polygons = [p.points for p in polygons]
            mask = mask_tools.rles_to_mask(polygons, img_width, img_height)
        else:
            instance += polygons  # keep unused polygons

        if masks:
            masks = (m.image for m in masks)
            if mask is not None:
                masks = chain(masks, [mask])
            mask = mask_tools.merge_masks(masks)

        if mask is None:
            return instance

        mask = mask_tools.mask_to_rle(mask)
        mask = mask_utils.frPyObjects(mask, *mask["size"])
        instance.append(
            RleMask(
                rle=mask,
                label=leader.label,
                z_order=leader.z_order,
                id=leader.id,
                attributes=leader.attributes,
                group=leader.group,
            )
        )
        return instance

    @staticmethod
    def find_instances(annotations):
        return find_instances(
            a for a in annotations if a.type in {AnnotationType.polygon, AnnotationType.mask}
        )


class PolygonsToMasks(ItemTransform, CliPlugin):
    _allowed_types = {AnnotationType.polygon, AnnotationType.ellipse}

    def transform_item(self, item):
        annotations = []
        for ann in item.annotations:
            if ann.type in self._allowed_types:
                if not isinstance(item.media, Image):
                    raise Exception("Image info is required for this transform")
                h, w = item.media.size
                annotations.append(self.convert_polygon(ann, h, w))
            else:
                annotations.append(ann)

        return self.wrap_item(item, annotations=annotations)

    @staticmethod
    def convert_polygon(polygon: Union[Polygon, Ellipse], img_h, img_w):
        rle = mask_utils.frPyObjects([polygon.as_polygon()], img_h, img_w)[0]

        return RleMask(
            rle=rle,
            label=polygon.label,
            z_order=polygon.z_order,
            id=polygon.id,
            attributes=polygon.attributes,
            group=polygon.group,
        )


class BoxesToMasks(ItemTransform, CliPlugin):
    def transform_item(self, item):
        annotations = []
        for ann in item.annotations:
            if ann.type == AnnotationType.bbox:
                if not isinstance(item.media, Image):
                    raise Exception("Image info is required for this transform")
                h, w = item.media.size
                annotations.append(self.convert_bbox(ann, h, w))
            else:
                annotations.append(ann)

        return self.wrap_item(item, annotations=annotations)

    @staticmethod
    def convert_bbox(bbox, img_h, img_w):
        rle = mask_utils.frPyObjects([bbox.as_polygon()], img_h, img_w)[0]

        return RleMask(
            rle=rle,
            label=bbox.label,
            z_order=bbox.z_order,
            id=bbox.id,
            attributes=bbox.attributes,
            group=bbox.group,
        )


class BoxesToPolygons(ItemTransform, CliPlugin):
    def transform_item(self, item):
        annotations = [
            self.convert_bbox(ann) if ann.type == AnnotationType.bbox else ann
            for ann in item.annotations
        ]

        return self.wrap_item(item, annotations=annotations)

    @staticmethod
    def convert_bbox(bbox: Bbox):
        return Polygon(
            points=bbox.as_polygon(),
            id=bbox.id,
            attributes=bbox.attributes,
            group=bbox.group,
            label=bbox.label,
            z_order=bbox.z_order,
        )


class MasksToPolygons(ItemTransform, CliPlugin):
    def transform_item(self, item):
        annotations = []
        for ann in item.annotations:
            if ann.type == AnnotationType.mask:
                polygons = self.convert_mask(ann)
                if not polygons:
                    log.debug(
                        "[%s]: item %s: "
                        "Mask conversion to polygons resulted in too "
                        "small polygons, which were discarded" % (self.NAME, item.id)
                    )
                annotations.extend(polygons)
            else:
                annotations.append(ann)

        return self.wrap_item(item, annotations=annotations)

    @staticmethod
    def convert_mask(mask):
        polygons = mask_tools.mask_to_polygons(mask.image)

        return [
            Polygon(
                points=p,
                label=mask.label,
                z_order=mask.z_order,
                id=mask.id,
                attributes=mask.attributes,
                group=mask.group,
            )
            for p in polygons
        ]


class ShapesToBoxes(ItemTransform, CliPlugin):
    def transform_item(self, item):
        annotations = []
        for ann in item.annotations:
            if ann.type in {
                AnnotationType.mask,
                AnnotationType.polygon,
                AnnotationType.polyline,
                AnnotationType.points,
                AnnotationType.ellipse,
            }:
                annotations.append(self.convert_shape(ann))
            else:
                annotations.append(ann)

        return self.wrap_item(item, annotations=annotations)

    @staticmethod
    def convert_shape(shape):
        bbox = shape.get_bbox()
        return Bbox(
            *bbox,
            label=shape.label,
            z_order=shape.z_order,
            id=shape.id,
            attributes=shape.attributes,
            group=shape.group,
        )


class Reindex(Transform, CliPlugin):
    """
    Replaces dataset item IDs with sequential indices.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("-s", "--start", type=int, default=1, help="Start value for item ids")
        return parser

    def __init__(self, extractor, start: int = 1):
        super().__init__(extractor)
        self._length = "parent"
        self._start = start

    def __iter__(self):
        for i, item in enumerate(self._extractor):
            yield self.wrap_item(item, id=i + self._start)


class ReindexAnnotations(ItemTransform, CliPlugin):
    """
    Replaces dataset items' annotations with sequential indices.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("-s", "--start", type=int, default=1, help="Start value for item ids")
        parser.add_argument(
            "-r",
            "--reindex-each-item",
            action="store_true",
            help="If true, reindex for each item every timeReindex for each item. For example, "
            "item_1 will have ann(id=1), ann(id=2), ..., ann(id=5) and item_2 will have ann_1(id=1), ..., "
            "if reindex_each_item=true and start=1, otherwise, item_2 will have ann_1(id=6), ..., "
            "because item_1 has ann_5(id=5).",
        )
        return parser

    def __init__(self, extractor, start: int = 1, reindex_each_item: bool = False):
        super().__init__(extractor)
        self._length = "parent"
        self._start = start
        self._cur_idx = start
        self._reindex_each_item = reindex_each_item

    def transform_item(self, item: DatasetItem) -> DatasetItem:
        annotations = [
            ann.wrap(id=idx) for idx, ann in enumerate(item.annotations, start=self._cur_idx)
        ]

        self._cur_idx = self._start if self._reindex_each_item else self._cur_idx + len(annotations)

        return item.wrap(annotations=annotations)


class Sort(Transform, CliPlugin):
    """
    Sorts dataset items.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("-k", "--key", type=str, default=None, help="key functions to sort.")
        return parser

    def __init__(self, extractor, key=None):
        super().__init__(extractor)
        if key:
            if isinstance(key, str):
                key = eval(key)
            if not callable(key):
                raise Exception("key must be a function with one argument.")
        else:
            key = lambda item: item.id
        self._key = key

    def __iter__(self):
        items = sorted(list(iter(self._extractor)), key=lambda item: self._key(item))
        for item in items:
            yield item


class MapSubsets(ItemTransform, CliPlugin):
    """
    Renames subsets in the dataset.
    """

    @staticmethod
    def _mapping_arg(s):
        parts = s.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError()
        return parts

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-s",
            "--subset",
            action="append",
            type=cls._mapping_arg,
            dest="mapping",
            help="Subset mapping of the form: 'src:dst' (repeatable)",
        )
        return parser

    def __init__(self, extractor, mapping=None):
        super().__init__(extractor)

        if mapping is None:
            mapping = {}
        elif not isinstance(mapping, dict):
            mapping = dict(tuple(m) for m in mapping)
        self._mapping = mapping

        if extractor.subsets():
            counts = Counter(mapping.get(s, s) or DEFAULT_SUBSET_NAME for s in extractor.subsets())
            if all(c == 1 for c in counts.values()):
                self._length = "parent"
            self._subsets = set(counts)

    def transform_item(self, item):
        return self.wrap_item(item, subset=self._mapping.get(item.subset, item.subset))


class RandomSplit(Transform, CliPlugin):
    """
    Joins all subsets into one and splits the result into few parts.
    It is expected that item ids are unique and subset ratios sum up to 1.|n
    |n
    Example:|n

    .. code-block::

    |s|s|s|s%(prog)s --subset train:.67 --subset test:.33
    """

    # avoid https://bugs.python.org/issue16399
    _default_split = [("train", 0.67), ("test", 0.33)]

    @staticmethod
    def _split_arg(s):
        parts = s.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError()
        return (parts[0], float(parts[1]))

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-s",
            "--subset",
            action="append",
            type=cls._split_arg,
            dest="splits",
            help="Subsets in the form: '<subset>:<ratio>' "
            "(repeatable, default: %s)" % dict(cls._default_split),
        )
        parser.add_argument("--seed", type=int, help="Random seed")
        return parser

    def __init__(self, extractor, splits, seed=None):
        super().__init__(extractor)

        if splits is None:
            splits = self._default_split

        assert 0 < len(splits), "Expected at least one split"
        assert all(0.0 <= r and r <= 1.0 for _, r in splits), (
            "Ratios are expected to be in the range [0; 1], but got %s" % splits
        )

        total_ratio = sum(s[1] for s in splits)
        if not abs(total_ratio - 1.0) <= 1e-7:
            raise Exception(
                "Sum of ratios is expected to be 1, got %s, which is %s" % (splits, total_ratio)
            )

        dataset_size = len(extractor)
        indices = list(range(dataset_size))
        random.seed(seed)
        random.shuffle(indices)
        parts = []
        s = 0
        lower_boundary = 0
        for split_idx, (subset, ratio) in enumerate(splits):
            s += ratio
            upper_boundary = int(s * dataset_size)
            if split_idx == len(splits) - 1:
                upper_boundary = dataset_size
            subset_indices = set(indices[lower_boundary:upper_boundary])
            parts.append((subset_indices, subset))
            lower_boundary = upper_boundary
        self._parts = parts

        self._subsets = set(s[0] for s in splits)
        self._length = "parent"

    def _find_split(self, index):
        for subset_indices, subset in self._parts:
            if index in subset_indices:
                return subset
        return subset  # all the possible remainder goes to the last split

    def __iter__(self):
        for i, item in enumerate(self._extractor):
            yield self.wrap_item(item, subset=self._find_split(i))


class IdFromImageName(ItemTransform, CliPlugin):
    """
    Renames items in the dataset based on the image file name, excluding the extension.|n
    When 'ensure_unique' is enabled, a random suffix is appened to ensure each identifier is unique
    in cases where the image name is not distinct. By default, the random suffix is three characters long,
    but this can be adjusted with the 'suffix_length' parameter.|n
    |n
    Examples:|n
    |n
    |s|s- Renames items without duplication check:|n

      .. code-block::

    |s|s|s|s%(prog)s|n
    |n
    |s|s- Renames items with duplication check:|n

    .. code-block::

    |s|s|s|s%(prog)s --ensure_unique|n
    |n
    |s|s- Renames items with duplication check and alters the suffix length(default: 3):|n

    .. code-block::

    |s|s|s|s%(prog)s --ensure_unique --suffix_length 2
    """

    DEFAULT_RETRY = 1000
    SUFFIX_LETTERS = string.ascii_lowercase + string.digits

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-u",
            "--ensure_unique",
            action="store_true",
            help="Appends a random suffix to ensure each identifier is unique if the image name is duplicated.",
        )
        parser.add_argument(
            "-l",
            "--suffix_length",
            type=int,
            default=3,
            help="Alters the length of the random suffix if the 'ensure_unique' is enabled.",
        )

        return parser

    def __init__(self, extractor, ensure_unique: bool = False, suffix_length: int = 3):
        super().__init__(extractor)
        self._length = "parent"
        self._ensure_unique = ensure_unique
        self._names: set[str] = set()
        self._suffix_length = suffix_length
        if suffix_length <= 0:
            raise ValueError(
                f"The 'suffix_length' must be greater than 0. Received: {suffix_length}."
            )
        self._max_retry = min(
            self.DEFAULT_RETRY, pow(len(self.SUFFIX_LETTERS), self._suffix_length)
        )

    def _add_unique_suffix(self, name):
        count = 0
        while name in self._names:
            suffix = "".join(
                random.choices(self.SUFFIX_LETTERS, k=self._suffix_length)  # nosec B311
            )
            new_name = f"{name}__{suffix}"
            if new_name not in self._names:
                name = new_name
                break
            count += 1
            if count == self._max_retry:
                raise Exception(
                    f"Too many duplicate names. Failed to generate a unique suffix after {self._max_retry} attempts."
                )

        self._names.add(name)
        return name

    def transform_item(self, item):
        if isinstance(item.media, Image) and hasattr(item.media, "path"):
            name = osp.splitext(osp.basename(item.media.path))[0]
            if isinstance(item.media, VideoFrame):
                name += f"_frame-{item.media.index}"
            if self._ensure_unique:
                name = self._add_unique_suffix(name)
            return self.wrap_item(item, id=name)
        else:
            log.debug("Can't change item id for item '%s': " "item has no path info" % item.id)
            return item


class Rename(ItemTransform, CliPlugin):
    r"""
    Renames items in the dataset. Supports regular expressions.
    The first character in the expression is a delimiter for
    the pattern and replacement parts. Replacement part can also
    contain `str.format` replacement fields with the `item`
    (of type `DatasetItem`) object available.|n
    Please use doulbe quotes to represent regex.|n
    |n
    Examples:|n
    |s|s- Replace 'pattern' with 'replacement':|n

      .. code-block::

    |s|s|s|srename -e "|pattern|replacement|"|n
    |n
    |s|s- Remove 'frame_' from item ids:|n

      .. code-block::

    |s|s|s|srename -e "|^frame_||"|n
    |n
    |s|s- Rename by regex:|n

      .. code-block::

    |s|s|s|srename -e "|frame_(\d+)_extra|{item.subset}_id_\1|"
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-e",
            "--regex",
            help="Regex for renaming in the form " "'<sep><search><sep><replacement><sep>'",
        )
        return parser

    def __init__(self, extractor, regex):
        super().__init__(extractor)

        assert regex and isinstance(regex, str)
        parts = regex.split(regex[0], maxsplit=3)
        regex, sub = parts[1:3]
        self._re = re.compile(regex)
        self._sub = sub

    def transform_item(self, item):
        return self.wrap_item(item, id=self._re.sub(self._sub, item.id).format(item=item))


class RemapLabels(ItemTransform, CliPlugin):
    """
    Changes labels in the dataset.|n
    |n
    A label can be:|n
    |s|s- renamed (and joined with existing) -|n
    |s|s|s|swhen '--label <old_name>:<new_name>' is specified|n
    |s|s- deleted - when '--label <name>:' is specified, or default action |n
    |s|s|s|sis 'delete' and the label is not mentioned in the list. |n
    |s|s|s|sWhen a label is deleted, all the associated annotations are removed|n
    |s|s- kept unchanged - when specified '--label <name>:<name>'|n
    |s|s|s|sor default action is 'keep' and the label is not mentioned in the list.|n
    |n
    Annotations with no label are managed by the default action policy.|n
    |n
    Examples:|n
    |n
    |s|s- Remove the 'person' label (and corresponding annotations):|n

    |s|s.. code-block::

    |s|s|s|s%(prog)s -l person: --default keep|n
    |n
    |s|s- Rename 'person' to 'pedestrian' and 'human' to 'pedestrian', join:|n

    |s|s.. code-block::

    |s|s|s|s%(prog)s -l person:pedestrian -l human:pedestrian --default keep|n
    |n
    |s|s- Rename 'person' to 'car' and 'cat' to 'dog', keep 'bus', remove others:|n

    |s|s.. code-block::

    |s|s|s|s%(prog)s -l person:car -l bus:bus -l cat:dog --default delete
    """

    class DefaultAction(Enum):
        keep = auto()
        delete = auto()

    @staticmethod
    def _split_arg(s):
        parts = s.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError()
        return (parts[0], parts[1])

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-l",
            "--label",
            action="append",
            type=cls._split_arg,
            dest="mapping",
            help="Label in the form of: '<src>:<dst>' (repeatable)",
        )
        parser.add_argument(
            "--default",
            choices=[a.name for a in cls.DefaultAction],
            default=cls.DefaultAction.keep.name,
            help="Action for unspecified labels (default: %(default)s)",
        )
        return parser

    def __init__(
        self,
        extractor: IDataset,
        mapping: Union[Dict[str, str], List[Tuple[str, str]]],
        default: Union[None, str, DefaultAction] = None,
    ):
        super().__init__(extractor)

        default = parse_str_enum_value(default, self.DefaultAction, self.DefaultAction.keep)
        self._default_action = default

        assert isinstance(mapping, (dict, list))
        if isinstance(mapping, list):
            mapping = dict(mapping)

        self._categories = {}

        src_categories = self._extractor.categories()

        src_label_cat = src_categories.get(AnnotationType.label)
        if src_label_cat is not None:
            self._make_label_id_map(src_label_cat, mapping, default)

        src_mask_cat = src_categories.get(AnnotationType.mask)
        if src_mask_cat is not None:
            assert src_label_cat is not None
            dst_mask_cat = MaskCategories(attributes=deepcopy(src_mask_cat.attributes))
            for old_id, old_color in src_mask_cat.colormap.items():
                new_id = self._map_id(old_id)
                if new_id is not None and new_id not in dst_mask_cat:
                    dst_mask_cat.colormap[new_id] = deepcopy(old_color)

            self._categories[AnnotationType.mask] = dst_mask_cat

        src_point_cat = src_categories.get(AnnotationType.points)
        if src_point_cat is not None:
            assert src_label_cat is not None
            dst_point_cat = PointsCategories(attributes=deepcopy(src_point_cat.attributes))
            for old_id, old_cat in src_point_cat.items.items():
                new_id = self._map_id(old_id)
                if new_id is not None and new_id not in dst_point_cat:
                    dst_point_cat.items[new_id] = deepcopy(old_cat)

            self._categories[AnnotationType.points] = dst_point_cat

        assert len(self._categories) == len(src_categories)

    def _make_label_id_map(self, src_label_cat, label_mapping, default_action):
        dst_label_cat = LabelCategories(attributes=deepcopy(src_label_cat.attributes))

        id_mapping = {}
        for src_index, src_label in enumerate(src_label_cat.items):
            dst_label = label_mapping.get(src_label.name, NOTSET)
            if dst_label is NOTSET and default_action == self.DefaultAction.keep:
                dst_label = src_label.name  # keep unspecified as is
            elif not dst_label or dst_label is NOTSET:
                continue

            dst_index = dst_label_cat.find(dst_label)[0]
            if dst_index is None:
                dst_index = dst_label_cat.add(
                    dst_label, src_label.parent, deepcopy(src_label.attributes)
                )
            id_mapping[src_index] = dst_index

        if log.getLogger().isEnabledFor(log.DEBUG):
            log.debug("Label mapping:")
            for src_id, src_label in enumerate(src_label_cat.items):
                if id_mapping.get(src_id) is not None:
                    log.debug(
                        "#%s '%s' -> #%s '%s'",
                        src_id,
                        src_label.name,
                        id_mapping[src_id],
                        dst_label_cat.items[id_mapping[src_id]].name,
                    )
                else:
                    log.debug("#%s '%s' -> <deleted>", src_id, src_label.name)

        self._map_id = lambda src_id: id_mapping.get(src_id, None)

        for label in dst_label_cat:
            if label.parent not in dst_label_cat:
                label.parent = ""
        self._categories[AnnotationType.label] = dst_label_cat

    def categories(self):
        return self._categories

    def transform_item(self, item):
        annotations = []
        for ann in item.annotations:
            if getattr(ann, "label", None) is not None:
                conv_label = self._map_id(ann.label)
                if conv_label is not None:
                    annotations.append(ann.wrap(label=conv_label))
            elif self._default_action is self.DefaultAction.keep:
                annotations.append(ann.wrap())
        return item.wrap(annotations=annotations)


class ProjectInfos(Transform, CliPlugin):
    """
    Changes the content of infos.
    A user can add meta-data of dataset such as author, comments, or related papers.
    Infos values are not affect on the dataset structure.
    We thus can add any meta-data freely.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-i",
            "--infos",
            action="append",
            dest="dst_infos",
            help="A dictionary of the dataset meta-information",
        )
        parser.add_argument(
            "-o",
            "--overwrite",
            action="store_true",
            dest="overwrite",
            help="Overwrite the infos of src if True",
        )
        return parser

    def __init__(self, extractor: IDataset, dst_infos: DatasetInfo, overwrite: bool = False):
        super().__init__(extractor)

        if overwrite:
            self._infos = dst_infos
        else:
            self._infos = deepcopy(extractor.infos())
            for k, v in dst_infos.items():
                self._infos[k] = v

    def __iter__(self):
        for item in self._extractor:
            if item is not None:
                yield item

    def infos(self):
        return self._infos


class ProjectLabels(ItemTransform):
    """
    Changes the order of labels in the dataset from the existing
    to the desired one, removes unknown labels and adds new labels.
    Updates or removes the corresponding annotations.|n
    |n
    Labels are matched by names (case dependent). Parent labels are only kept
    if they are present in the resulting set of labels. If new labels are
    added, and the dataset has mask colors defined, new labels will obtain
    generated colors.|n
    |n
    Useful for merging similar datasets, whose labels need to be aligned.|n
    |n
    Examples:|n
    |s|s- Align the source dataset labels to [person, cat, dog]:|n

    |s|s.. code-block::

    |s|s|s|s%(prog)s -l person -l cat -l dog
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-l",
            "--label",
            action="append",
            dest="dst_labels",
            help="Label name (repeatable, ordered)",
        )
        return parser

    def __init__(self, extractor: IDataset, dst_labels: Union[Iterable[str], LabelCategories]):
        super().__init__(extractor)

        self._categories = {}

        src_categories = self._extractor.categories()

        src_label_cat = src_categories.get(AnnotationType.label)

        if isinstance(dst_labels, LabelCategories):
            dst_label_cat = deepcopy(dst_labels)
        else:
            dst_labels = list(dst_labels)

            if src_label_cat:
                dst_label_cat = LabelCategories(attributes=deepcopy(src_label_cat.attributes))

                for dst_label in dst_labels:
                    assert isinstance(dst_label, str)
                    src_label = src_label_cat.find(dst_label)[1]
                    if src_label is not None:
                        dst_label_cat.add(
                            dst_label, src_label.parent, deepcopy(src_label.attributes)
                        )
                    else:
                        dst_label_cat.add(dst_label)
            else:
                dst_label_cat = LabelCategories.from_iterable(dst_labels)

        for label in dst_label_cat:
            if label.parent not in dst_label_cat:
                label.parent = ""
        self._categories[AnnotationType.label] = dst_label_cat

        self._make_label_id_map(src_label_cat, dst_label_cat)

        src_mask_cat = src_categories.get(AnnotationType.mask)
        if src_mask_cat is not None:
            assert src_label_cat is not None
            dst_mask_cat = MaskCategories(attributes=deepcopy(src_mask_cat.attributes))
            for old_id, old_color in src_mask_cat.colormap.items():
                new_id = self._map_id(old_id)
                if new_id is not None and new_id not in dst_mask_cat:
                    dst_mask_cat.colormap[new_id] = deepcopy(old_color)

            # Generate new colors for new labels, keep old untouched
            existing_colors = set(dst_mask_cat.colormap.values())
            color_bank = iter(
                mask_tools.generate_colormap(len(dst_label_cat), include_background=False).values()
            )
            for new_id, new_label in enumerate(dst_label_cat):
                if new_label.name in src_label_cat:
                    continue
                if new_id in dst_mask_cat:
                    continue

                color = next(color_bank)
                while color in existing_colors:
                    color = next(color_bank)

                dst_mask_cat.colormap[new_id] = color

            self._categories[AnnotationType.mask] = dst_mask_cat

        src_point_cat = src_categories.get(AnnotationType.points)
        if src_point_cat is not None:
            assert src_label_cat is not None
            dst_point_cat = PointsCategories(attributes=deepcopy(src_point_cat.attributes))
            for old_id, old_cat in src_point_cat.items.items():
                new_id = self._map_id(old_id)
                if new_id is not None and new_id not in dst_point_cat:
                    dst_point_cat.items[new_id] = deepcopy(old_cat)

            self._categories[AnnotationType.points] = dst_point_cat

    def _make_label_id_map(self, src_label_cat, dst_label_cat):
        id_mapping = {
            src_id: dst_label_cat.find(src_label_cat[src_id].name)[0]
            for src_id in range(len(src_label_cat or ()))
        }
        self._map_id = lambda src_id: id_mapping.get(src_id, None)

    def categories(self):
        return self._categories

    def transform_item(self, item):
        annotations = []
        for ann in item.annotations:
            if getattr(ann, "label", None) is not None:
                conv_label = self._map_id(ann.label)
                if conv_label is not None:
                    annotations.append(ann.wrap(label=conv_label))
            else:
                annotations.append(ann.wrap())
        return item.wrap(annotations=annotations)


class AnnsToLabels(ItemTransform, CliPlugin):
    """
    Collects all labels from annotations (of all types) and
    transforms them into a set of annotations of type Label
    """

    def transform_item(self, item):
        labels = set(p.label for p in item.annotations if getattr(p, "label") is not None)
        annotations = []
        for label in labels:
            annotations.append(Label(label=label))

        return item.wrap(annotations=annotations)


class BboxValuesDecrement(ItemTransform, CliPlugin):
    """
    Subtracts one from the coordinates of bounding boxes
    """

    def transform_item(self, item):
        annotations = [p for p in item.annotations if p.type != AnnotationType.bbox]
        bboxes = [p for p in item.annotations if p.type == AnnotationType.bbox]
        for bbox in bboxes:
            annotations.append(
                Bbox(
                    bbox.x - 1,
                    bbox.y - 1,
                    bbox.w,
                    bbox.h,
                    label=bbox.label,
                    attributes=bbox.attributes,
                )
            )

        return item.wrap(annotations=annotations)


class ResizeTransform(ItemTransform):
    """
    Resizes images and annotations in the dataset to the specified size.
    Supports upscaling, downscaling and mixed variants.|n
    |n
    Examples:|n
        - Resize all images to 256x256 size|n

        .. code-block::

        |s|s%(prog)s -dw 256 -dh 256
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("-dw", "--width", type=int, help="Destination image width")
        parser.add_argument("-dh", "--height", type=int, help="Destination image height")
        return parser

    def __init__(self, extractor: IDataset, width: int, height: int) -> None:
        super().__init__(extractor)

        assert width > 0 and height > 0
        self._width = width
        self._height = height

    @staticmethod
    def _lazy_resize_image(image, new_size):
        def _resize_image():
            h, w = image.size
            yscale = new_size[0] / float(h)
            xscale = new_size[1] / float(w)

            # LANCZOS4 is preferable for upscaling, but it works quite slow
            method = cv2.INTER_AREA if (xscale * yscale) < 1 else cv2.INTER_CUBIC

            resized_image = cv2.resize(image.data / 255.0, new_size[::-1], interpolation=method)
            resized_image *= 255.0
            return resized_image

        return Image.from_numpy(_resize_image, ext=image.ext, size=new_size)

    @staticmethod
    def _lazy_resize_mask(mask, new_size):
        def _resize_image():
            # Can use only NEAREST for masks,
            # because we can't have interpolated values
            rescaled_mask = cv2.resize(
                mask.image.astype(np.float32), new_size[::-1], interpolation=cv2.INTER_NEAREST
            )
            return rescaled_mask.astype(np.uint8)

        return _resize_image

    @staticmethod
    def _lazy_resize_rlemask(mask, new_size):
        def _resize_image():
            # Can use only NEAREST for masks,
            # because we can't have interpolated values
            rescaled_mask = cv2.resize(mask.image, new_size[::-1], interpolation=cv2.INTER_NEAREST)
            return mask_utils.encode(np.asfortranarray(rescaled_mask.astype(np.uint8)))

        return _resize_image

    def transform_item(self, item):
        if not isinstance(item.media, Image):
            raise DatumaroError(
                "Item %s: image info is required for this " "transform" % (item.id,)
            )

        h, w = item.media.size
        xscale = self._width / float(w)
        yscale = self._height / float(h)

        new_size = (self._height, self._width)

        resized_image = None
        if item.media.has_data:
            resized_image = self._lazy_resize_image(item.media, new_size)

        resized_annotations = []
        for ann in item.annotations:
            if isinstance(ann, Bbox):
                resized_annotations.append(
                    ann.wrap(
                        x=ann.x * xscale,
                        y=ann.y * yscale,
                        w=ann.w * xscale,
                        h=ann.h * yscale,
                    )
                )
            elif isinstance(ann, (Polygon, Points, PolyLine)):
                resized_annotations.append(
                    ann.wrap(
                        points=[
                            p
                            for t in ((x * xscale, y * yscale) for x, y in take_by(ann.points, 2))
                            for p in t
                        ]
                    )
                )
            elif isinstance(ann, RleMask):
                rescaled_mask = self._lazy_resize_rlemask(ann, new_size)
                resized_annotations.append(ann.wrap(rle=rescaled_mask))
            elif isinstance(ann, Mask):
                rescaled_mask = self._lazy_resize_mask(ann, new_size)
                resized_annotations.append(ann.wrap(image=rescaled_mask))
            elif isinstance(ann, (Caption, Label)):
                resized_annotations.append(ann)
            else:
                assert False, f"Unexpected annotation type {type(ann)}"

        return self.wrap_item(item, media=resized_image, annotations=resized_annotations)


class RemoveItems(ItemTransform):
    """
    Allows to remove specific dataset items from dataset by their ids.|n
    |n
    Can be useful to clean the dataset from broken or unnecessary samples.|n
    |n
    Examples:|n
        - Remove specific items from the dataset|n

        .. code-block::

        |s|s%(prog)s --id 'image1:train' --id 'image2:test'
    """

    @staticmethod
    def _parse_id(s):
        full_id = s.split(":")
        if len(full_id) != 2:
            raise argparse.ArgumentTypeError(
                None, message="Invalid id format of '%s'. " "Expected a 'name:subset' pair." % s
            )
        return full_id

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--id",
            dest="ids",
            type=cls._parse_id,
            action="append",
            required=True,
            help="Item id to remove. Id is 'name:subset' pair (repeatable)",
        )
        return parser

    def __init__(self, extractor: IDataset, ids: Iterable[Tuple[str, str]]):
        super().__init__(extractor)
        self._ids = set(tuple(v) for v in (ids or []))

    def transform_item(self, item):
        if (item.id, item.subset) in self._ids:
            return None
        return item


class RemoveAnnotations(ItemTransform):
    """
    Allows to remove annotations on specific dataset items.|n
    |n
    Can be useful to clean the dataset from broken or unnecessary annotations.|n
    |n
    Examples:|n
        - Remove annotations from specific items in the dataset|n

        .. code-block::

        |s|s%(prog)s --id 'image1:train' --id 'image2:test'
    """

    @staticmethod
    def _parse_id(s):
        full_id = s.split(":")
        if len(full_id) == 3:
            full_id[-1] = int(full_id[-1])
        if len(full_id) != 2 or len(full_id) != 3:
            raise argparse.ArgumentTypeError(
                None,
                message="Invalid id format of '%s'. "
                "Expected 'name:subset:ann_id' or 'name:subset' pair." % s,
            )
        return full_id

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--id",
            dest="ids",
            type=cls._parse_id,
            action="append",
            help="Image id to clean from annotations. "
            "Id is 'name:subset:(optional)ann_id' pair. If ann_id is not specified, "
            "removes all annotations (repeatable) in the item 'name:subset'",
        )
        return parser

    def __init__(self, extractor: IDataset, *, ids: Iterable[Tuple[str, str, Optional[int]]]):
        super().__init__(extractor)

        self._ids = defaultdict(list)
        for v in ids:
            key = tuple(v[:2])
            val = v[2] if len(v) == 3 else None
            if val is not None:
                self._ids[key].append(val)
            else:
                self._ids[key] = []

    def transform_item(self, item: DatasetItem):
        if not self._ids:
            return item

        for item_id, ann_ids in self._ids.items():
            if (item.id, item.subset) == item_id:
                updated_anns = (
                    [ann for ann in item.annotations if ann.id not in ann_ids] if ann_ids else []
                )
                return item.wrap(annotations=updated_anns)

        return item


class RemoveAttributes(ItemTransform):
    """
    Allows to remove item and annotation attributes in a dataset.|n
    |n
    Can be useful to clean the dataset from broken or unnecessary attributes.|n
    |n
    Examples:|n
        - Remove the `is_crowd` attribute from dataset|n

        .. code-block::

        |s|s%(prog)s --attr 'is_crowd'|n
        |n
        - Remove the `occluded` attribute from annotations of|n
        |s|sthe `2010_001705` item in the `train` subset|n

        .. code-block::

        |s|s%(prog)s --id '2010_001705:train' --attr 'occluded'
    """

    @staticmethod
    def _parse_id(s):
        full_id = s.split(":")
        if len(full_id) != 2:
            raise argparse.ArgumentTypeError(
                None, message="Invalid id format of '%s'. " "Expected a 'name:subset' pair." % s
            )
        return full_id

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--id",
            dest="ids",
            type=cls._parse_id,
            action="append",
            help="Image id to clean from annotations. "
            "Id is 'name:subset' pair. If not specified, "
            "affects all images and annotations (repeatable)",
        )
        parser.add_argument(
            "-a",
            "--attr",
            action="append",
            dest="attributes",
            help="Attribute name to be removed. If not specified, "
            "removes all attributes (repeatable)",
        )
        return parser

    def __init__(
        self,
        extractor: IDataset,
        ids: Optional[Iterable[Tuple[str, str]]] = None,
        attributes: Optional[Iterable[str]] = None,
    ):
        super().__init__(extractor)
        self._ids = set(tuple(v) for v in (ids or []))
        self._attributes = set(attributes or [])

    def _filter_attrs(self, attrs):
        if not self._attributes:
            return None
        else:
            return filter_dict(attrs, exclude_keys=self._attributes)

    def transform_item(self, item: DatasetItem):
        if not self._ids or (item.id, item.subset) in self._ids:
            filtered_annotations = []
            for ann in item.annotations:
                filtered_annotations.append(ann.wrap(attributes=self._filter_attrs(ann.attributes)))

            return item.wrap(
                attributes=self._filter_attrs(item.attributes), annotations=filtered_annotations
            )
        return item


class Correct(Transform, CliPlugin):
    """
    This class provides functionality to correct and refine a dataset based on a validation report.|n
    It processes a validation report (typically in JSON format) to identify and rectify various |n
    dataset issues, such as undefined labels, missing annotations, outliers, empty labels/captions,|n
    and unnecessary characters in captions. The correction process includes:|n
    |n
    - Adding missing labels and attributes.|n
    - Removing or adjusting annotations with invalid or anomalous values.|n
    - Filling in missing labels and captions with appropriate values.|n
    - Removing unnecessary characters from text-based annotations like captions.|n
    - Handling outliers by capping values within specified bounds.|n
    - Updating dataset categories and annotations according to the corrections.|n
    |n
    The class is designed to be used as part of a command-line interface (CLI) and can be |n
    configured with different validation reports. It integrates with the dataset extraction |n
    process, ensuring that corrections are applied consistently across the dataset.|n
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-r",
            "--reports",
            type=str,
            default="validation_reports.json",
            help="A validation report from a 'validate' CLI",
        )
        return parser

    def __init__(
        self,
        extractor: IDataset,
        reports: Union[str, Dict],
    ):
        super().__init__(extractor)

        if isinstance(reports, str):
            reports = parse_json_file(reports)

        self._reports = reports["validation_reports"]

        self._categories = self._extractor.categories()

        self._remove_items = set()
        self._remove_anns = defaultdict(list)
        self._add_attrs = defaultdict(list)

        self._empty_labels = defaultdict(list)
        self._empty_captions = defaultdict(list)
        self._unnecessary_char_captions = list()

        self._outlier_captions = defaultdict(list)
        self._outlier_value = {}

        self._far_from_mean_caption = defaultdict(list)
        self._far_from_mean_value = {}

        self._far_from_mean_caption = defaultdict(list)
        self._far_from_mean_value = {}

        self._analyze_reports(report=self._reports)

        self._table = None
        self._label_value = {}
        self._caption_value = {}

        self.caption_type = {
            cat.name: cat.dtype
            for cat in self._extractor.categories().get(AnnotationType.caption, TabularCategories())
        }

    def categories(self):
        return self._categories

    def _parse_ann_ids(self, desc: str):
        return [int(s) for s in str.split(desc, "'") if s.isdigit()][0]

    def _analyze_reports(self, report):
        for rep in report:
            if rep["anomaly_type"] == MissingLabelCategories.__name__:
                unique_labels = sorted(
                    list({ann.label for item in self._extractor for ann in item.annotations})
                )
                label_categories = LabelCategories().from_iterable(
                    [str(label) for label in unique_labels]
                )
                for item in self._extractor:
                    for ann in item.annotations:
                        attrs = {attr for attr in ann.attributes}
                        label_categories[ann.label].attributes.update(attrs)
                self._categories[AnnotationType.label] = label_categories

            if rep["anomaly_type"] == UndefinedLabel.__name__:
                label_categories = self._categories[AnnotationType.label]
                desc = [s for s in rep["description"].split("'")]
                add_label_name = desc[1]
                label_id, _ = label_categories.find(add_label_name)
                if label_id is None:
                    label_categories.add(name=add_label_name)

            if rep["anomaly_type"] == UndefinedAttribute.__name__:
                label_categories = self._categories[AnnotationType.label]
                desc = [s for s in rep["description"].split("'")]
                attr_name, label_name = desc[1], desc[3]
                label_id = label_categories.find(label_name)[0]
                if label_id is not None:
                    label_categories[label_id].attributes.add(attr_name)

            # [TODO] Correct LabeleDefinedButNotFound: removing a label, reindexing, remapping others
            # if rep["anomaly_type"] == "LabelDefinedButNotFound":
            #     remove_label_name = self._parse_label_cat(rep["description"])
            #     label_cat = self._extractor.categories()[AnnotationType.label]
            #     if remove_label_name in [labels.name for labels in label_cat.items]:
            #         label_cat.remove(remove_label_name)

            if rep["anomaly_type"] in [MissingAnnotation.__name__, MultiLabelAnnotations.__name__]:
                self._remove_items.add((rep["item_id"], rep["subset"]))

            if rep["anomaly_type"] in [
                NegativeLength.__name__,
                InvalidValue.__name__,
                FarFromLabelMean.__name__,
                FarFromAttrMean.__name__,
            ]:
                ann_id = None or self._parse_ann_ids(rep["description"])
                self._remove_anns[(rep["item_id"], rep["subset"])].append(ann_id)

            if rep["anomaly_type"] == MissingAttribute.__name__:
                desc = [s for s in str.split(rep["description"], "'")]
                attr_name, label_name = desc[1], desc[3]
                label_id = self._extractor.categories()[AnnotationType.label].find(label_name)[0]
                self._add_attrs[(rep["item_id"], rep["subset"])].append((label_id, attr_name))

            if rep["anomaly_type"] == RedundanciesInCaption.__name__:
                desc = [s for s in str.split(rep["description"], "'")]
                attr_name, label_name = desc[1], desc[3]
                self._unnecessary_char_captions.append((label_name, attr_name))

            if rep["anomaly_type"] == EmptyLabel.__name__:
                label = rep["description"].split("'")[1]
                self._empty_labels[(rep["item_id"], rep["subset"])].append(label)

            if rep["anomaly_type"] == EmptyCaption.__name__:
                caption = rep["description"].split("'")[1]
                self._empty_captions[(rep["item_id"], rep["subset"])].append(caption)

            if rep["anomaly_type"] == OutlierInCaption.__name__:
                desc = rep["description"].split("'")
                caption = desc[1]
                lower_bound = float(desc[5])
                upper_bound = float(desc[7])
                self._outlier_captions[(rep["item_id"], rep["subset"])].append(caption)
                self._outlier_value[caption] = (lower_bound, upper_bound)

            if rep["anomaly_type"] == FarFromCaptionMean.__name__:
                desc = rep["description"].split("'")
                caption = desc[1]
                lower_bound = float(desc[9])
                upper_bound = float(desc[11])
                self._far_from_mean_caption[(rep["item_id"], rep["subset"])].append(caption)
                self._far_from_mean_value[caption] = (lower_bound, upper_bound)

    def remove_unnecessary_char(self, annotations, item_id):
        if self._table is None:
            items = [item_.media.data() for item_ in self._extractor]
            self._table = pd.DataFrame(items)

        import re

        from nltk.corpus import stopwords

        try:
            stop_words = set(stopwords.words("english"))  # TODO
        except LookupError:
            import nltk

            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))  # TODO

        def remove_stopwords(text):
            return " ".join([word for word in text.split() if word not in stop_words])

        def remove_urls(text):
            return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        def remove_html_tags(text):
            return re.sub(r"<.*?>", "", text)

        def remove_special_characters(text):
            return re.sub(r"[^A-Za-z\s]+", "", text)

        def remove_extra_whitespace(text):
            return re.sub(r"\s+", " ", text).strip()

        def remove_emojis(text):
            return emoji_pattern.sub(r"", text)

        col_redun_dict = {}
        for key, value in self._unnecessary_char_captions:
            if key in col_redun_dict:
                if isinstance(col_redun_dict[key], list):
                    col_redun_dict[key].append(value)
                else:
                    col_redun_dict[key] = [col_redun_dict[key], value]
            else:
                col_redun_dict[key] = [value]

        for col, attr in col_redun_dict.items():
            table = self._table[col].dropna()

            # Remove HTML
            if "html" in attr:
                table = table.apply(remove_html_tags)

            # Remove urls
            if "url" in attr:
                table = table.apply(remove_urls)

            # Remove emojis
            if "emoji" in attr:
                table = table.apply(remove_emojis)

            # Convert to lowercase
            table = table.apply(lambda x: x.lower())

            # Remove Special Characters and Punctuations
            table = table.apply(remove_special_characters)

            # Remove Extra Whitespace
            table = table.apply(remove_extra_whitespace)

            # Remove stopwords
            if "stopword" in attr:
                table = table.apply(remove_stopwords)

        for ann in annotations:
            if ann.type == AnnotationType.caption and ann.caption[: len(col)] == col:
                new_ann = Caption(f"{col}:{table.loc[int(item_id.split('@')[0])]}")
                annotations.remove(ann)
                annotations.append(new_ann)

        return annotations

    def update_caption_value(self):
        if self._table is None:
            items = [item_.media.data() for item_ in self._extractor]
            self._table = pd.DataFrame(items)

        empty_cap_cols = set([v for value in self._empty_captions.values() for v in value])

        for caption in empty_cap_cols:
            table = self._table[caption].dropna()
            caption_type = self.caption_type[caption]

            if caption_type in [int, float]:
                from scipy.stats import skew

                skewness = skew(table)
                if abs(skewness) > 1:  # TODO
                    self._caption_value[caption] = table.median()
                else:
                    self._caption_value[caption] = table.mean()
            elif caption_type is str:
                self._remove_items.update(
                    item_key
                    for item_key, captions in self._empty_captions.items()
                    if caption in captions
                )

    def update_label_value(self):
        if self._table is None:
            items = [item_.media.data() for item_ in self._extractor]
            self._table = pd.DataFrame(items)

        empty_lbl_cols = set([v for value in self._empty_labels.values() for v in value])

        for label in empty_lbl_cols:
            table = self._table[label].dropna()
            # "most_frequency"
            self._label_value[label] = table.mode().iloc[0]

    def fill_missing_value(self, annotations, labels, captions):
        if labels:
            sep_token = self._extractor._sep_token
            id_mapping = self._extractor.categories().get(AnnotationType.label)._indices
            label_value = self._label_value
            annotations.extend(
                Label(id_mapping[f"{label}{sep_token}{label_value[label]}"]) for label in labels
            )
        if captions:
            caption_value = self._caption_value
            annotations.extend(
                Caption(f"{caption}:{caption_value[caption]}") for caption in captions
            )
        return annotations

    def cap_far_from_mean(self, annotations, far_from_mean_captions):
        for ann in annotations:
            if ann.type != AnnotationType.caption:
                continue

            for col in far_from_mean_captions:
                if not ann.caption.startswith(col):
                    continue

                value_str = ann.caption[len(col) + 1 :]
                value = self.caption_type[col](value_str)

                lower_bound, upper_bound = self._far_from_mean_value[col]
                capped_value = max(min(value, upper_bound), lower_bound)

                new_ann = Caption(f"{col}:{capped_value}")
                annotations.remove(ann)
                annotations.append(new_ann)
                break

        return annotations

    def cap_outliers(self, annotations, outliers):
        for ann in annotations:
            if ann.type != AnnotationType.caption:
                continue

            for col in outliers:
                if not ann.caption.startswith(col):
                    continue

                value_str = ann.caption[len(col) + 1 :]
                value = self.caption_type[col](value_str)

                lower_bound, upper_bound = self._outlier_value[col]
                capped_value = max(min(value, upper_bound), lower_bound)

                new_ann = Caption(f"{col}:{capped_value}")
                annotations.remove(ann)
                annotations.append(new_ann)
                break

        return annotations

    def find_outliers(self, annotations, outliers):
        for ann in annotations:
            for col in outliers:
                if ann.type == AnnotationType.caption and ann.caption[: len(col)] == col:
                    value = self._extractor._tabular_cat_types[col](ann.caption[len(col) + 1 :])
                    # Cap outliers
                    lower_bound, upper_bound = self._outlier_value[col]
                    cap_outlier_val = np.where(
                        value > upper_bound,
                        upper_bound,
                        np.where(value < lower_bound, lower_bound, value),
                    ).item()
                    new_ann = Caption(f"{col}:{cap_outlier_val}")
                    annotations.remove(ann)
                    annotations.append(new_ann)
        return annotations

    def __iter__(self):
        if self._empty_captions:
            self.update_caption_value()
        if self._empty_labels:
            self.update_label_value()

        for item in self._extractor:
            item_key = (item.id, item.subset)
            if item_key in self._remove_items:
                continue

            ann_ids = self._remove_anns.get(item_key, None)
            empty_labels = self._empty_labels.get(item_key, None)
            empty_captions = self._empty_captions.get(item_key, None)
            outlier_captions = self._outlier_captions.get(item_key, None)
            far_from_mean_captions = self._far_from_mean_caption.get(item_key, None)

            if ann_ids:
                updated_anns = [ann for ann in item.annotations if ann.id not in ann_ids]
                yield item.wrap(annotations=updated_anns)

            if self._unnecessary_char_captions:
                item.annotations = self.remove_unnecessary_char(item.annotations, item_id=item.id)

            if outlier_captions:
                item.annotations = self.cap_outliers(item.annotations, outlier_captions)

            if far_from_mean_captions:
                item.annotations = self.cap_far_from_mean(item.annotations, far_from_mean_captions)

            if empty_labels or empty_captions:
                item.annotations = self.fill_missing_value(
                    item.annotations, empty_labels, empty_captions
                )

            if not ann_ids:
                updated_attrs = defaultdict(list)
                for label_id, attr_name in self._add_attrs.get((item.id, item.subset), []):
                    updated_attrs[label_id].append(attr_name)

                updated_anns = []
                for ann in item.annotations:
                    new_ann = ann.wrap(attributes=deepcopy(ann.attributes))
                    if ann.type == AnnotationType.label and ann.label in updated_attrs:
                        new_ann.attributes.update(
                            {attr_name: "" for attr_name in updated_attrs[ann.label]}
                        )
                    updated_anns.append(new_ann)
                yield item.wrap(annotations=updated_anns)


class AstypeAnnotations(ItemTransform):
    """
    Converts the types of annotations within a dataset based on a specified mapping.|n
    |n
    This transform changes annotations to 'Label' if they are categorical, and to 'Caption'
    if they are of type string, float, or integer. This is particularly useful when working
    with tabular data that needs to be converted into a format suitable for specific machine
    learning tasks.|n
    |n
    Examples:|n
        - Converts the type of a `title` annotation:|n

        .. code-block::

        |s|s%(prog)s --mapping 'title:text'
    """

    @staticmethod
    def _split_arg(s):
        columns = s.split(",")
        results = []
        for column in columns:
            parts = column.split(":")
            if len(parts) != 2:
                raise argparse.ArgumentTypeError()
            results.append((parts[0], parts[1]))
        return results

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--mapping",
            action="append",
            type=cls._split_arg,
            dest="mapping",
            help="Annotations type in the form of: '<src>:<dst>' (repeatable)",
        )
        return parser

    def __init__(
        self,
        extractor: IDataset,
        mapping: Optional[Union[Dict[str, str], List[Tuple[str, str]]]] = None,
    ):
        super().__init__(extractor)

        self._sep_token = ":"

        if extractor.ann_types() and AnnotationType.tabular not in extractor.ann_types():
            raise AnnotationTypeError(
                "Annotation type is not Tabular. This transform only support tabular annotation"
            )

        # Turn off for default setting
        assert mapping is None or isinstance(mapping, (dict, list)), "Mapping must be dict, or list"
        if isinstance(mapping, list):
            mapping = dict(mapping)

        self._categories = {}

        src_categories = self._extractor.categories()
        src_tabular_cat = src_categories.get(AnnotationType.tabular)
        dst_label_cat = LabelCategories()

        if src_tabular_cat is None:
            return

        for src_cat in src_tabular_cat:
            if src_cat.dtype == CategoricalDtype():
                dst_parent = src_cat.name
                dst_labels = sorted(src_cat.labels)
                for dst_label in dst_labels:
                    dst_label = dst_parent + self._sep_token + str(dst_label)
                    dst_label_cat.add(dst_label, parent=dst_parent, attributes={})
                dst_label_cat.add_label_group(src_cat.name, src_cat.labels, group_type=0)
            else:
                self._categories[AnnotationType.caption] = self._categories.get(
                    AnnotationType.caption, []
                ) + [src_cat]
        self._categories[AnnotationType.label] = dst_label_cat

    def categories(self):
        return self._categories

    def transform_item(self, item: DatasetItem):
        if AnnotationType.tabular not in [ann.type for ann in item.annotations]:
            return self.wrap_item(item, annotations=item.annotations)

        label_categories = self._categories.get(AnnotationType.label, LabelCategories())
        labels_set = {item.parent for item in label_categories.items}
        sep_token = self._sep_token
        label_indices = label_categories._indices

        annotations = [
            Label(label=label_indices[f"{name}{sep_token}{value}"])
            if name in labels_set and value is not None
            else Caption(f"{name}{sep_token}{value}")
            for name, value in item.annotations[0].values.items()
            if pd.notna(value)
        ]

        return self.wrap_item(item, annotations=annotations)


class Clean(TabularTransform):
    """
    A class used to refine the media items in a dataset.|n
    |n
    This class provides methods to clean and preprocess media data within a dataset.
    The media data can be of various types such as strings, numeric values, or categorical values.
    The cleaning process for each type of data is handled differently:|n
    |n
    - **String Media**: For string data, the class employs natural language processing (NLP)
    techniques to remove unnecessary characters. This involves cleaning tasks such as removing special
    characters, punctuation, and other irrelevant elements to refine the textual data.|n
    - **Numeric Media**: For numeric data, the class identifies and handles outliers and missing values.
    Outliers are either removed or replaced based on a defined strategy,
    and missing values are filled using appropriate methods such as mean, median, or a predefined value.|n
    """

    def __init__(
        self,
        extractor: IDataset,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super().__init__(extractor, batch_size, num_workers)

        self._outlier_value = {}
        self._missing_value = {}
        self._sep_token = ":"

    @staticmethod
    def remove_unnecessary_char(text):
        if pd.isna(text):
            return text
        try:
            from nltk.corpus import stopwords

            stop_words = set(stopwords.words("english"))  # TODO
        except LookupError:
            import nltk

            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))  # TODO

        text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # remove URLs
        text = emoji_pattern.sub(r"", text)  # Remove emojis
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"[^A-Za-z\s]+", "", text)  # Remove special characters and punctuation
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespaces
        text = " ".join(
            [word for word in text.split() if word not in stop_words]
        )  # Remove stopwords
        return text

    def check_outlier(self, table, numeric_cols):
        for col in numeric_cols:
            col_data = table[col].dropna()

            Q1 = np.quantile(col_data, 0.25)
            Q3 = np.quantile(col_data, 0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if table[col].dtype == int:
                lower_bound = self.find_closest_value(col_data[col], lower_bound)
                upper_bound = self.find_closest_value(col_data[col], upper_bound)
            self._outlier_value[col] = (lower_bound, upper_bound)

    def check_missing_value(self, table, float_cols, countable_cols):
        from scipy.stats import skew

        for col in table.columns:
            col_data = table[col].dropna()
            if col in float_cols:
                skewness = skew(col_data)
                if abs(skewness) > 1:  # TODO
                    self._missing_value[col] = col_data.median()
                else:
                    self._missing_value[col] = col_data.mean()
            elif col in countable_cols:
                self._missing_value[col] = col_data.mode().iloc[0]

    @staticmethod
    def find_closest_value(series, target_value):
        abs_diff = np.abs(series - target_value)
        closest_index = abs_diff.idxmin()
        return series.iloc[closest_index]

    def cap_outliers(self, table):
        lower, upper = self._outlier_value[table.name]
        val = table.iloc[0]
        if (val < lower) | (val > upper):
            capped_value = max(min(val, upper), lower)
            return table.replace(val, capped_value)
        return val

    def fill_missing_value(self, series):
        return series.fillna(self._missing_value[series.name])

    def refine_tabular_media(self, item):
        media = item.media
        df = pd.DataFrame(media.data(), index=[media.index])
        str_cols = [col for col in media.data().keys() if media.table.dtype(col) is str]
        float_cols = [col for col in media.data().keys() if media.table.dtype(col) is float]
        int_cols = [col for col in media.data().keys() if media.table.dtype(col) is int]
        countable_cols = [
            col
            for col in media.data().keys()
            if isinstance(item.media.table.dtype(col), CategoricalDtype)
            or item.media.table.dtype(col) is int
        ]

        df[str_cols] = df[str_cols].map(lambda x: self.remove_unnecessary_char(x))

        if not (self._outlier_value):
            self.check_outlier(media.table.data[float_cols + int_cols], float_cols + int_cols)
        df[float_cols + int_cols] = df[float_cols + int_cols].apply(lambda x: self.cap_outliers(x))

        if not (self._missing_value):
            self.check_missing_value(
                media.table.data[float_cols + countable_cols], float_cols, countable_cols
            )
        df[float_cols + countable_cols] = df[float_cols + countable_cols].apply(
            lambda x: self.fill_missing_value(x)
        )

        return TableRow.from_data(
            df.iloc[0].to_dict(), table=item.media.table, index=item.media.index
        )

    def transform_item(self, item):
        if not isinstance(item.media, TableRow):
            raise DatumaroError(
                "Item %s: TableRow info is required for this " "transform" % (item.id,)
            )

        sep_token = self._sep_token
        refined_media = self.refine_tabular_media(item) if item.media.has_data else None
        refined_annotations = []
        for ann in item.annotations:
            if isinstance(ann, Tabular):
                if len(item.annotations) != 1:
                    raise ValueError(
                        "If the item has a tabular annotation, it should have one annotation."
                    )
                annotation_values = {
                    key: refined_media.data[key] for key in item.annotations[0].values.keys()
                }  # only for tabular
                ann = ann.wrap(values=annotation_values)
            elif isinstance(ann, Caption):
                value = [
                    f"{key}{sep_token}{refined_media.data[key]}"
                    for key in refined_media.data.keys()
                    if ann.caption.startswith(key)
                ]
                ann = ann.wrap(caption=value[0])
            refined_annotations.append(ann)

        return self.wrap_item(item, media=refined_media, annotations=refined_annotations)
