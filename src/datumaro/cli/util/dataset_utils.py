# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import re
from enum import Enum
from typing import Optional

from datumaro.components.dataset import Dataset
from datumaro.components.environment import DEFAULT_ENVIRONMENT, Environment
from datumaro.util.os_util import generate_next_name


def generate_next_file_name(basename, basedir=".", sep=".", ext=""):
    """
    If basedir does not contain basename, returns basename,
    otherwise generates a name by appending sep to the basename
    and the number, next to the last used number in the basedir for
    files with basename prefix. Optionally, appends ext.
    """
    return generate_next_name(os.listdir(basedir), basename, sep, ext)


def parse_dataset_pathspec(s: str, env: Optional[Environment] = None) -> Dataset:
    """
    Parses Dataset paths. The syntax is:
        - <dataset path>[ :<format> ]

    Returns: a dataset from the parsed path
    """
    match = re.fullmatch(
        r"""
        (?P<dataset_path>(?: [^:] | :[/\\] )+)
        (:(?P<format>.+))?
        """,
        s,
        flags=re.VERBOSE,
    )
    if not match:
        raise ValueError("Failed to recognize dataset pathspec in '%s'" % s)
    match = match.groupdict()

    path = match["dataset_path"]
    format = match["format"]
    return Dataset.import_from(path, format, env=env or DEFAULT_ENVIRONMENT)


class FilterModes(Enum):
    # primary
    items = 1
    annotations = 2
    items_annotations = 3

    # shortcuts
    i = 1
    a = 2
    i_a = 3
    a_i = 3
    annotations_items = 3

    @staticmethod
    def parse(s):
        s = s.lower()
        s = s.replace("+", "_")
        return FilterModes[s]

    @classmethod
    def make_filter_args(cls, mode):
        if mode == cls.items:
            return {}
        elif mode == cls.annotations:
            return {"filter_annotations": True}
        elif mode == cls.items_annotations:
            return {
                "filter_annotations": True,
                "remove_empty": True,
            }
        else:
            raise NotImplementedError()

    @classmethod
    def list_options(cls):
        return [m.name.replace("_", "+") for m in cls]
