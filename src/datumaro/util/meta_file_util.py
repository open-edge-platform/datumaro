# Copyright (C) 2022-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from collections import OrderedDict

import numpy as np

from datumaro.components.annotation import AnnotationType, HashKey
from datumaro.util import dump_json_file, find, parse_json_file
from datumaro.util.deprecation import deprecated

DATASET_META_FILE = "dataset_meta.json"
DATASET_HASHKEY_FILE = "hash_keys.json"
DATASET_HASHKEY_FOLDER = "hash_key_meta"


def is_meta_file(path):
    return osp.splitext(osp.basename(path))[1] == ".json"


def has_meta_file(path):
    return osp.isfile(get_meta_file(path))


def has_hashkey_file(path):
    return osp.isfile(get_hashkey_file(path))


def get_meta_file(path):
    return osp.join(path, DATASET_META_FILE)


def get_hashkey_file(path):
    hashkey_folder_path = osp.join(path, DATASET_HASHKEY_FOLDER)
    return osp.join(hashkey_folder_path, DATASET_HASHKEY_FILE)


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


def parse_hashkey_file(path):
    meta_file = path
    if osp.isdir(path):
        meta_file = get_hashkey_file(path)

    if not osp.exists(meta_file):
        return None

    dataset_meta = parse_json_file(meta_file)

    hashkey_dict = OrderedDict()
    for id_, hashkey in dataset_meta.get("hashkey", {}).items():
        hashkey_dict[id_] = hashkey

    return hashkey_dict


@deprecated(deprecated_version="1.11", removed_version="1.12")
def save_hashkey_file(path, item_list):
    dataset_hashkey = {}

    if osp.isdir(path):
        meta_file = get_hashkey_file(path)
    hashkey_folder_path = osp.join(path, DATASET_HASHKEY_FOLDER)
    if not osp.exists(hashkey_folder_path):
        os.makedirs(hashkey_folder_path)

    hashkey_dict = parse_hashkey_file(path)
    if not hashkey_dict:
        hashkey_dict = {}

    for item in item_list:
        item_id = item.id
        item_subset = item.subset
        for annotation in item.annotations:
            if isinstance(annotation, HashKey):
                hashkey = annotation.hash_key
                break
        hashkey_dict.update({item_subset + "/" + item_id: hashkey.tolist()})

    dataset_hashkey["hashkey"] = hashkey_dict

    dump_json_file(meta_file, dataset_hashkey, indent=True)


@deprecated(deprecated_version="1.11", removed_version="1.12")
def _load_hash_key(path, dataset):
    hashkey_dict = parse_hashkey_file(path)
    for item in dataset:
        hash_key = hashkey_dict[item.subset + "/" + item.id]
        item.annotations.append(HashKey(hash_key=np.asarray(hash_key, dtype=np.uint8)))
    return dataset


def load_hash_key(path, dataset):
    if not os.path.isdir(path) or not has_hashkey_file(path):
        return dataset

    return _load_hash_key(path, dataset)
