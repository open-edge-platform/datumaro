# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from collections import OrderedDict
from enum import Enum, auto
from itertools import chain
from typing import Dict, List, Optional, Tuple

import numpy as np

from datumaro.components.annotation import AnnotationType, Colormap, LabelCategories, MaskCategories, RgbColor
from datumaro.components.dataset_base import CategoriesInfo
from datumaro.components.errors import InvalidAnnotationError
from datumaro.util import dump_json_file, find, parse_json_file
from datumaro.util.meta_file_util import get_meta_file


class VocTask(Enum):
    voc = auto()
    voc_classification = auto()
    voc_detection = auto()
    voc_segmentation = auto()
    voc_instance_segmentation = auto()
    voc_action = auto()
    voc_layout = auto()


class VocLabel(Enum):
    background = 0
    aeroplane = 1
    bicycle = 2
    bird = 3
    boat = 4
    bottle = 5
    bus = 6
    car = 7
    cat = 8
    chair = 9
    cow = 10
    diningtable = 11
    dog = 12
    horse = 13
    motorbike = 14
    person = 15
    pottedplant = 16
    sheep = 17
    sofa = 18
    train = 19
    tvmonitor = 20
    ignored = 255


class VocPose(Enum):
    Unspecified = auto()
    Left = auto()
    Right = auto()
    Frontal = auto()
    Rear = auto()


class VocBodyPart(Enum):
    head = auto()
    hand = auto()
    foot = auto()


class VocAction(Enum):
    other = auto()
    jumping = auto()
    phoning = auto()
    playinginstrument = auto()
    reading = auto()
    ridingbike = auto()
    ridinghorse = auto()
    running = auto()
    takingphoto = auto()
    usingcomputer = auto()
    walking = auto()


def generate_colormap(length: int = 256) -> Colormap:
    def get_bit(number, index):
        return (number >> index) & 1

    colormap = np.zeros((length, 3), dtype=int)
    indices = np.arange(length, dtype=int)

    for j in range(7, -1, -1):
        for c in range(3):
            colormap[:, c] |= get_bit(indices, c) << j
        indices >>= 3

    return OrderedDict((id, tuple(color)) for id, color in enumerate(colormap))


VocColormap: Colormap = {id: color for id, color in generate_colormap(256).items() if id in {l.value for l in VocLabel}}
VocInstColormap = generate_colormap(256)


class VocImporterType(Enum):
    default = auto()
    roboflow = auto()


class VocPath:
    IMAGES_DIR = "JPEGImages"
    ANNOTATIONS_DIR = "Annotations"
    SEGMENTATION_DIR = "SegmentationClass"
    INSTANCES_DIR = "SegmentationObject"
    SUBSETS_DIR = "ImageSets"
    IMAGE_EXT = ".jpg"
    SEGM_EXT = ".png"
    LABELMAP_FILE = "labelmap.txt"

    TASK_DIR = {
        VocTask.voc: "Main",
        VocTask.voc_classification: "Main",
        VocTask.voc_detection: "Main",
        VocTask.voc_segmentation: "Segmentation",
        VocTask.voc_instance_segmentation: "Segmentation",
        VocTask.voc_action: "Action",
        VocTask.voc_layout: "Layout",
    }


LabelMapConfig = Dict[str, Tuple[Optional[RgbColor], List[str], List[str]]]
"""A type representing a label map config"""
# Not totally type-correct, tuple elements are supposed to support modification.
# Therefore, the tuple is typically a list
# TODO: refactor, make type annotations conform with actual usage


def make_voc_label_map(task: VocTask = None) -> LabelMapConfig:
    if task == VocTask.voc_action:
        label_map = OrderedDict(
            {
                VocLabel.background.name: [VocColormap[VocLabel.background.value], [], []],
                VocLabel.person.name: [
                    VocColormap[VocLabel.person.value],
                    [],
                    [a.name for a in VocAction],
                ],
            }
        )
    elif task == VocTask.voc_layout:
        label_map = OrderedDict(
            {
                VocLabel.background.name: [VocColormap[VocLabel.background.value], [], []],
                VocLabel.person.name: [
                    VocColormap[VocLabel.person.value],
                    [p.name for p in VocBodyPart],
                    [],
                ],
            }
        )
    elif task in [
        VocTask.voc_classification,
        VocTask.voc_detection,
        VocTask.voc_segmentation,
        VocTask.voc_instance_segmentation,
    ]:
        labels = sorted(VocLabel, key=lambda l: l.value)
        label_map = OrderedDict((label.name, [VocColormap[label.value], [], []]) for label in labels)
    else:
        labels = sorted(VocLabel, key=lambda l: l.value)
        label_map = OrderedDict((label.name, [VocColormap[label.value], [], []]) for label in labels)
        label_map[VocLabel.person.name][1] = [p.name for p in VocBodyPart]
        label_map[VocLabel.person.name][2] = [a.name for a in VocAction]

    return label_map


def parse_label_map(path: str) -> LabelMapConfig:
    """
    Parses a label map file in the format:
    'name : color (r, g, b) : parts (hand, feet, ...) : actions (siting, standing, ...)'

    Parameters:
        path: File path

    Returns:
        A dictionary: label -> (color, parts, actions)
    """

    label_map = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # skip empty and commented lines
            line = line.strip()
            if not line or line[0] == "#":
                continue

            # name : color : parts : actions
            label_desc = line.strip().split(":")
            if len(label_desc) != 4:
                raise InvalidAnnotationError(
                    f"Label description has wrong number of fields '{len(label_desc)}'. "
                    "Expected 4 ':'-separated fields."
                )
            name = label_desc[0]

            if name in label_map:
                raise InvalidAnnotationError(f"Label '{name}' is already defined in the label map")

            if len(label_desc) > 1 and label_desc[1]:
                color = label_desc[1].split(",")
                if len(color) != 3:
                    raise InvalidAnnotationError(
                        f"Label '{name}' has wrong color '{color}'. Expected an 'r,g,b' triplet."
                    )
                color = tuple(int(c) for c in color)
            else:
                color = None

            if len(label_desc) > 2 and label_desc[2]:
                parts = [s.strip() for s in label_desc[2].split(",")]
            else:
                parts = []

            if len(label_desc) > 3 and label_desc[3]:
                actions = [s.strip() for s in label_desc[3].split(",")]
            else:
                actions = []

            label_map[name] = [color, parts, actions]
    return label_map


def parse_meta_file(path: str) -> LabelMapConfig:
    # Uses custom format with extra fields
    meta_file = path
    if osp.isdir(path):
        meta_file = get_meta_file(path)

    dataset_meta = parse_json_file(meta_file)

    label_map = OrderedDict()
    parts = dataset_meta.get("parts", {})
    actions = dataset_meta.get("actions", {})

    for i, label in enumerate(dataset_meta.get("labels", [])):
        label_map[label] = [None, parts.get(str(i), []), actions.get(str(i), [])]

    colors = dataset_meta.get("segmentation_colors", [])

    for i, label in enumerate(dataset_meta.get("label_map", {}).values()):
        if label not in label_map:
            label_map[label] = [None, [], []]

        if any(colors) and colors[i] is not None:
            label_map[label][0] = tuple(colors[i])

    return label_map


def write_label_map(path: str, label_map: LabelMapConfig):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# label:color_rgb:parts:actions\n")
        for label_name, label_desc in label_map.items():
            if label_desc[0]:
                color_rgb = ",".join(str(c) for c in label_desc[0])
            else:
                color_rgb = ""

            parts = ",".join(str(p) for p in label_desc[1])
            actions = ",".join(str(a) for a in label_desc[2])

            f.write("%s\n" % ":".join([label_name, color_rgb, parts, actions]))


def write_meta_file(path: str, label_map: LabelMapConfig):
    # Uses custom format with extra fields
    dataset_meta = {}

    labels = []
    labels_dict = {}
    segmentation_colors = []
    parts = {}
    actions = {}

    for i, (label_name, label_desc) in enumerate(label_map.items()):
        labels.append(label_name)
        if label_desc[0]:
            labels_dict[str(i)] = label_name
            segmentation_colors.append([int(label_desc[0][0]), int(label_desc[0][1]), int(label_desc[0][2])])

        parts[str(i)] = label_desc[1]
        actions[str(i)] = label_desc[2]

    dataset_meta["labels"] = labels

    if any(segmentation_colors):
        dataset_meta["label_map"] = labels_dict
        dataset_meta["segmentation_colors"] = segmentation_colors

        bg_label = find(label_map.items(), lambda x: x[1] == (0, 0, 0))
        if bg_label is not None:
            dataset_meta["background_label"] = str(bg_label[0])

    if any(parts):
        dataset_meta["parts"] = parts

    if any(actions):
        dataset_meta["actions"] = actions

    dump_json_file(get_meta_file(path), dataset_meta)


def make_voc_categories(
    label_map: Optional[LabelMapConfig] = None, task: Optional[VocTask] = VocTask.voc
) -> CategoriesInfo:
    if label_map is None:
        label_map = make_voc_label_map(task)

    categories = {}

    label_categories = LabelCategories()
    label_categories.attributes.update(["difficult", "truncated", "occluded"])

    for label, desc in label_map.items():
        label_categories.add(label, attributes=desc[2])

    if task in [VocTask.voc, VocTask.voc_layout]:
        for part in OrderedDict((k, None) for k in chain(*(desc[1] for desc in label_map.values()))):
            label_categories.add(part)

    categories[AnnotationType.label] = label_categories

    if task not in [VocTask.voc, VocTask.voc_segmentation, VocTask.voc_instance_segmentation]:
        return categories

    has_colors = any(v[0] is not None for v in label_map.values())
    if not has_colors:  # generate new colors
        colormap = generate_colormap(len(label_map))
    else:  # only copy defined colors
        label_id = lambda label: label_categories.find(label)[0]
        colormap = {label_id(name): desc[0] for name, desc in label_map.items() if desc[0] is not None}
    mask_categories = MaskCategories(colormap)
    mask_categories.inverse_colormap  # pylint: disable=pointless-statement
    categories[AnnotationType.mask] = mask_categories

    return categories
