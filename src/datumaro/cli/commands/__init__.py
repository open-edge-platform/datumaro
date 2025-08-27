# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=redefined-builtin

from . import (
    compare,
    convert,
    detect_format,
    download,
    filter,
    info,
    merge,
    patch,
    stats,
    transform,
    validate,
)

__all__ = [
    "get_non_project_commands",
]


def get_non_project_commands():
    return [
        ("convert", convert, "Convert dataset between formats"),
        ("detect", detect_format, "Detect the format of a dataset"),
        ("compare", compare, "Compare datasets"),
        ("dinfo", info, "Print dataset info"),
        ("download", download, "Download a publicly available dataset"),
        ("filter", filter, "Filter dataset items"),
        ("merge", merge, "Merge datasets"),
        ("patch", patch, "Update dataset from another one"),
        ("stats", stats, "Compute dataset statistics"),
        ("transform", transform, "Modify dataset items"),
        ("validate", validate, "Validate dataset"),
    ]
