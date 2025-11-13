# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from enum import Enum, auto

from datumaro.components.annotation import AnnotationType, LabelCategories, MaskCategories
from datumaro.util.mask_tools import generate_colormap


class KittiTask(Enum):
    segmentation = auto()
    detection = auto()


KittiLabelMap = OrderedDict(
    [
        ("unlabeled", (0, 0, 0)),
        ("egovehicle", (0, 0, 0)),
        ("rectificationborder", (0, 0, 0)),
        ("outofroi", (0, 0, 0)),
        ("static", (0, 0, 0)),
        ("dynamic", (111, 74, 0)),
        ("ground", (81, 0, 81)),
        ("road", (128, 64, 128)),
        ("sidewalk", (244, 35, 232)),
        ("parking", (250, 170, 160)),
        ("railtrack", (230, 150, 140)),
        ("building", (70, 70, 70)),
        ("wall", (102, 102, 156)),
        ("fence", (190, 153, 153)),
        ("guardrail", (180, 165, 180)),
        ("bridge", (150, 100, 100)),
        ("tunnel", (150, 120, 90)),
        ("pole", (153, 153, 153)),
        ("polegroup", (153, 153, 153)),
        ("trafficlight", (250, 170, 30)),
        ("trafficsign", (220, 220, 0)),
        ("vegetation", (107, 142, 35)),
        ("terrain", (152, 251, 152)),
        ("sky", (70, 130, 180)),
        ("person", (220, 20, 60)),
        ("rider", (255, 0, 0)),
        ("car", (0, 0, 142)),
        ("truck", (0, 0, 70)),
        ("bus", (0, 60, 100)),
        ("caravan", (0, 0, 90)),
        ("trailer", (0, 0, 110)),
        ("train", (0, 80, 100)),
        ("motorcycle", (0, 0, 230)),
        ("bicycle", (119, 11, 32)),
        ("licenseplate", (0, 0, 142)),
    ]
)


class KittiPath:
    IMAGES_DIR = "image_2"
    INSTANCES_DIR = "instance"
    LABELS_DIR = "label_2"
    SEMANTIC_RGB_DIR = "semantic_rgb"
    SEMANTIC_DIR = "semantic"
    IMAGE_EXT = ".png"
    MASK_EXT = ".png"

    LABELMAP_FILE = "label_colors.txt"

    DEFAULT_TRUNCATED = 0.0  # 0% truncated
    DEFAULT_OCCLUDED = 0  # fully visible
    DEFAULT_SCORE = 1.0


def make_kitti_categories(label_map=None):
    if label_map is None:
        label_map = KittiLabelMap

    categories = {}
    label_categories = LabelCategories()
    for label in label_map:
        label_categories.add(label)
    categories[AnnotationType.label] = label_categories

    has_colors = any(v is not None for v in label_map.values())
    if not has_colors:  # generate new colors
        colormap = generate_colormap(len(label_map))
    else:  # only copy defined colors

        def label_id(label):
            return label_categories.find(label)[0]

        colormap = {label_id(name): (desc[0], desc[1], desc[2]) for name, desc in label_map.items()}
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
            if not line or (line and line[0] == "#"):
                continue

            # color, name
            label_desc = line.strip().split()

            if len(label_desc) > 2:
                name = label_desc[3]
                color = tuple([int(c) for c in label_desc[:-1]])
            else:
                name = label_desc[0]
                color = None

            if name in label_map:
                raise ValueError("Label '%s' is already defined" % name)

            label_map[name] = color
    return label_map


def write_label_map(path, label_map):
    with open(path, "w", encoding="utf-8") as f:
        for label_name, label_desc in label_map.items():
            if label_desc:
                color_rgb = " ".join(str(c) for c in label_desc)
            else:
                color_rgb = ""
            f.write("%s %s\n" % (color_rgb, label_name))
