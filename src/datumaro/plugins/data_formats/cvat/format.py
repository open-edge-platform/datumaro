# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT


from datumaro.components.annotation import AnnotationType


class CvatPath:
    IMAGES_DIR = "images"

    IMAGE_EXT = ".jpg"

    BUILTIN_ATTRS = {"occluded", "outside", "keyframe", "track_id"}

    SUPPORTED_IMPORT_SHAPES = {
        "box",
        "polygon",
        "polyline",
        "points",
        "mask",
    }
    SUPPORTED_EXPORT_SHAPES = {
        AnnotationType.bbox,
        AnnotationType.rotated_bbox,
        AnnotationType.polygon,
        AnnotationType.polyline,
        AnnotationType.points,
        AnnotationType.mask,
    }
