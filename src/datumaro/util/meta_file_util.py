# Copyright (C) 2022-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from collections import OrderedDict

from datumaro.components.annotation import AnnotationType
from datumaro.util import dump_json_file, find, parse_json_file

DATASET_META_FILE = "dataset_meta.json"


def is_meta_file(path):
    return osp.splitext(osp.basename(path))[1] == ".json"


def has_meta_file(path):
    return osp.isfile(get_meta_file(path))


def get_meta_file(path):
    return osp.join(path, DATASET_META_FILE)


def parse_meta_file(path):
    meta_file = path
    if osp.isdir(path):
        meta_file = get_meta_file(path)

    dataset_meta = parse_json_file(meta_file)

    label_map = OrderedDict()

    for label in dataset_meta.get("labels", []):
        label_map[label] = None

    colors = dataset_meta.get("segmentation_colors", [])
    for i, label in dataset_meta.get("label_map", {}).items():
        label_map[label] = None

        if any(colors) and colors[int(i)] is not None:
            label_map[label] = tuple(colors[int(i)])

    return label_map


def save_meta_file(path, categories):
    dataset_meta = {}

    labels = [label.name for label in categories[AnnotationType.label]]
    dataset_meta["labels"] = labels

    if categories.get(AnnotationType.mask):
        label_map = {}
        segmentation_colors = []
        for i, color in categories[AnnotationType.mask].colormap.items():
            if color:
                segmentation_colors.append([int(color[0]), int(color[1]), int(color[2])])
                label_map[str(i)] = labels[i]
        dataset_meta["label_map"] = label_map
        dataset_meta["segmentation_colors"] = segmentation_colors

        bg_label = find(
            categories[AnnotationType.mask].colormap.items(), lambda x: x[1] == (0, 0, 0)
        )
        if bg_label is not None:
            dataset_meta["background_label"] = str(bg_label[0])

    meta_file = path
    if osp.isdir(path):
        meta_file = get_meta_file(path)

    dump_json_file(meta_file, dataset_meta, indent=True)
