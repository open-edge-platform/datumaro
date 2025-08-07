# Copyright (C) 2021-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import typing

import pytest
from attrs import frozen


class DatumaroComponent:
    Datumaro = "datumaro"


class Requirements:
    # Please, maintain the ordering when adding new lines

    # Exact requirements
    DATUM_GENERAL_REQ = "Datumaro general requirement"
    DATUM_TELEMETRY = "Datumaro telemetry requirement"
    DATUM_API = "Datumaro API"
    DATUM_PROGRESS_REPORTING = "Datumaro progress reporting requirement"
    DATUM_ERROR_REPORTING = "Datumaro error reporting requirement"
    DATUM_GENERIC_MEDIA = "Datumaro generic media support"

    # GitHub issues (not bugs)
    # https://github.com/open-edge-platform/datumaro/issues
    DATUM_231 = "Readable formats for CJK"
    DATUM_244 = "Add Snyk integration"
    DATUM_267 = "Add Image zip format"
    DATUM_274 = "Support the Open Images dataset"
    DATUM_280 = "Support KITTI dataset formats"
    DATUM_283 = "Create cli tests for testing convert command for VOC format"
    DATUM_399 = "Implement import for ADE20K dataset"
    DATUM_475 = "Support import for CelebA dataset"
    DATUM_497 = "Support import for SYNTHIA dataset"
    DATUM_542 = "Images missing after merging two datasets"
    DATUM_565 = "Export in YOLO with custom subset name"
    DATUM_580 = "Import for MPII Human Pose Dataset"
    DATUM_609 = "Allow not to prepend data/ prefix in YOLO export"
    DATUM_616 = "Import for BraTS dataset"
    DATUM_673 = "Pickling for Dataset and Annotations"
    DATUM_677 = "Generation of fractal images for model compression"

    # GitHub issues (bugs)
    # https://github.com/open-edge-platform/datumaro/issues
    DATUM_BUG_219 = "Return format is not uniform"
    DATUM_BUG_257 = "Dataset.filter doesn't count removed items"
    DATUM_BUG_259 = "Dataset.filter fails on merged datasets"
    DATUM_BUG_289 = "Cannot convert LabelMe dataset, that has no subsets"
    DATUM_BUG_314 = "Unsuccessful remap_labels"
    DATUM_BUG_402 = "Troubles running 'remap_labels' on ProjectDataset"
    DATUM_BUG_404 = "custom importer/extractor not loading"
    DATUM_BUG_425 = "Bug: concatenation for the different types in COCO format"
    DATUM_BUG_466 = "Can't correct import Open Images dataset without images"
    DATUM_BUG_470 = "Cannot to import Cityscapes dataset without images"
    DATUM_BUG_560 = "Reading MOT dataset with seqinfo produces 0-based indexing in frames"
    DATUM_BUG_583 = "Empty lines in VOC subset lists are not ignored"
    DATUM_BUG_602 = "Patch command example error"
    DATUM_BUG_606 = "transform with resize also changed the image extension from .jpg to .png"
    DATUM_BUG_618 = "ResizeTransform returns broken image pixels"
    DATUM_BUG_873 = "Error using datum stats"
    DATUM_BUG_1204 = (
        "Statistics raise an error when there is a label annotation not in the category"
    )
    DATUM_BUG_1214 = (
        "Dataset.import_from() can import nested datasets that exist in the given path."
    )


class SkipMessages:
    NOT_IMPLEMENTED = "NOT IMPLEMENTED"


@frozen
class _CombinedDecorator:
    decorators: typing.List[typing.Callable]

    def __call__(self, function):
        for d in reversed(self.decorators):
            function = d(function)

        return function


_SHARED_DECORATORS = [
    pytest.mark.components(DatumaroComponent.Datumaro),
    pytest.mark.component,
    pytest.mark.priority_medium,
]


def mark_requirement(requirement):
    return _CombinedDecorator(
        [
            *_SHARED_DECORATORS,
            pytest.mark.reqids(requirement),
        ]
    )


def mark_bug(bugs):
    return _CombinedDecorator(
        [
            *_SHARED_DECORATORS,
            pytest.mark.bugs(bugs),
        ]
    )
