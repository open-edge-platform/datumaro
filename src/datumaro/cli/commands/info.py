# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from datumaro.components.annotation import AnnotationType
from datumaro.components.errors import MissingObjectError
from datumaro.util.scope import scoped

from ..util import MultilineFormatter
from ..util.dataset_utils import parse_dataset_pathspec


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Prints dataset overview",
        description="""
        Prints info about the dataset at the specified path.|n
        |n
        <dataset_path> - dataset path with optional format specification:|n
        |s|s- <dataset path>[ :<format> ]|n
        |n
        Examples:|n
        - Print dataset info for a path and a format name:|n
        |s|s%(prog)s path/to/dataset:voc|n
        |n
        - Print dataset info for a COCO dataset:|n
        |s|s%(prog)s path/to/dataset:coco
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "target",
        metavar="dataset_path",
        help="Target dataset path with optional format (path:format)",
    )
    parser.add_argument("--all", action="store_true", help="Print all information")
    parser.set_defaults(command=info_command)

    return parser


@scoped
def info_command(args):
    dataset = None
    dataset_problem = ""
    try:
        dataset = parse_dataset_pathspec(args.target)
    except MissingObjectError as e:
        dataset_problem = str(e)
    except Exception as e:
        dataset_problem = f"Error loading dataset: {e}"

    def print_dataset_info(dataset, indent=""):
        print("%slength:" % indent, len(dataset))

        categories = dataset.categories()
        print("%scategories:" % indent, ", ".join(c.name for c in categories))

        for cat_type, cat in categories.items():
            print("%s  %s:" % (indent, cat_type.name))
            if cat_type == AnnotationType.label:
                print("%s    count:" % indent, len(cat.items))

                count_threshold = 10
                if args.all:
                    count_threshold = len(cat.items)
                labels = ", ".join(c.name for c in cat.items[:count_threshold])
                if count_threshold < len(cat.items):
                    labels += " (and %s more)" % (len(cat.items) - count_threshold)
                print("%s    labels:" % indent, labels)

    if dataset is not None:
        print_dataset_info(dataset)

        subsets = dataset.subsets()
        print("subsets:", ", ".join(subsets))
        for subset_name in subsets:
            subset = dataset.get_subset(subset_name)
            print("  '%s':" % subset_name)
            print_dataset_info(subset, indent="    ")
    else:
        print("Dataset info is not available: ", dataset_problem)

    return 0
