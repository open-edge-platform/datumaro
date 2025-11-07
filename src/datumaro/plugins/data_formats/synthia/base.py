# Copyright (C) 2021-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import os.path as osp
from collections import OrderedDict
from glob import glob
from typing import Optional

import numpy as np

from datumaro.components.annotation import AnnotationType, ExtractedMask, LabelCategories, MaskCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import InvalidAnnotationError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image
from datumaro.util.image import find_images
from datumaro.util.mask_tools import generate_colormap, load_mask
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file

from .format import (
    SynthiaAlLabelMap,
    SynthiaAlPath,
    SynthiaRandLabelMap,
    SynthiaRandPath,
    SynthiaSfLabelMap,
    SynthiaSfPath,
)


def make_categories(label_map):
    categories = {}
    label_categories = LabelCategories()
    for label in label_map:
        label_categories.add(label)
    categories[AnnotationType.label] = label_categories

    has_colors = any(v is not None for v in label_map.values())
    if not has_colors:  # generate new colors
        colormap = generate_colormap(len(label_map))
    else:  # only copy defined colors
        colormap = {label_id: (desc[0], desc[1], desc[2]) for label_id, desc in enumerate(label_map.values())}
    mask_categories = MaskCategories(colormap)
    mask_categories.inverse_colormap  # pylint: disable=pointless-statement
    categories[AnnotationType.mask] = mask_categories
    return categories


def parse_label_map(path):
    label_map = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # skip empty and commented lines
            line = line.strip()
            if not line or line[0] == "#":
                continue

            # color, name
            label_desc = line.split()

            if len(label_desc) > 2:
                name = label_desc[3]
                color = tuple([int(c) for c in label_desc[:3]])
            else:
                name = label_desc[0]
                color = None

            if name in label_map:
                raise InvalidAnnotationError("Label '%s' is already defined" % name)

            label_map[name] = color
    return label_map


class _SynthiaBase(SubsetBase):
    def __init__(
        self,
        path: str,
        path_formats,
        label_map,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isdir(path):
            raise NotADirectoryError(errno.ENOTDIR, "Can't find dataset directory", path)

        super().__init__(subset=subset, ctx=ctx)

        self._path_formats = path_formats
        self._label_map = label_map
        self._ann_types = set()

        self._img_dir = None
        self._inst_dir = None
        self._seg_dir = None
        for path_format in vars(path_formats).keys():
            if path_format == "IMAGES_DIR":
                self._img_dir = osp.join(path, path_formats.IMAGES_DIR)
            elif path_format == "LABELS_SEGM_DIR":
                self._inst_dir = osp.join(path, path_formats.LABELS_SEGM_DIR)
            elif path_format == "SEMANTIC_SEGM_DIR":
                self._seg_dir = osp.join(path, path_formats.SEMANTIC_SEGM_DIR)

        self._categories = self._load_categories(path)
        self._items = list(self._load_items().values())

    def _load_categories(self, path):
        if has_meta_file(path):
            return make_categories(parse_meta_file(path))

        label_map_path = osp.join(path, "label_colors.txt")
        if osp.isfile(label_map_path):
            label_map = parse_label_map(label_map_path)
        else:
            label_map = self._label_map

        return make_categories(label_map)

    def _load_items(self):
        if self._img_dir and osp.isdir(self._img_dir):
            images = {
                osp.splitext(osp.relpath(p, self._img_dir))[0].replace("\\", "/"): p
                for p in find_images(self._img_dir, recursive=True)
            }
        else:
            images = {}

        items = {}
        if self._inst_dir and osp.isdir(self._inst_dir):
            gt_labels = glob(self._inst_dir + "/*.txt")
            for gt_label in gt_labels:
                item_id = osp.splitext(osp.relpath(gt_label, self._inst_dir))[0].replace("\\", "/")

                anno = []
                labels_mask = np.loadtxt(gt_label)
                classes = np.unique(labels_mask)
                for label_id in classes:
                    anno.append(ExtractedMask(index_mask=labels_mask, index=label_id, label=label_id))
                    self._ann_types.add(AnnotationType.mask)

                image = images.get(item_id)
                if image:
                    image = Image.from_file(path=image)

                items[item_id] = DatasetItem(id=item_id, media=image, annotations=anno)
        elif self._seg_dir and osp.isdir(self._seg_dir):
            for seg_img_path in find_images(self._seg_dir, recursive=True):
                item_id = osp.splitext(osp.relpath(seg_img_path, self._seg_dir))[0].replace("\\", "/")

                inverse_cls_colormap = self._categories[AnnotationType.mask].inverse_colormap

                color_mask = load_mask(seg_img_path, inverse_cls_colormap, default_id=0)

                anno = []
                classes = np.unique(color_mask)
                for label_id in classes:
                    anno.append(ExtractedMask(index_mask=color_mask, index=label_id, label=label_id))
                    self._ann_types.add(AnnotationType.mask)

                image = images.get(item_id)
                if image:
                    image = Image.from_file(path=image)

                items[item_id] = DatasetItem(id=item_id, media=image, annotations=anno)

        return items


class SynthiaRandBase(_SynthiaBase):
    def __init__(self, path: str, **kwargs):
        super().__init__(path=path, path_formats=SynthiaRandPath, label_map=SynthiaRandLabelMap, **kwargs)


class SynthiaSfBase(_SynthiaBase):
    def __init__(self, path, **kwargs):
        super().__init__(path=path, path_formats=SynthiaSfPath, label_map=SynthiaSfLabelMap, **kwargs)


class SynthiaAlBase(_SynthiaBase):
    def __init__(self, path, **kwargs):
        super().__init__(path=path, path_formats=SynthiaAlPath, label_map=SynthiaAlLabelMap, **kwargs)
