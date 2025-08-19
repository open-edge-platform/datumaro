# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from datumaro.components.operations import compute_ann_statistics, compute_image_statistics
from datumaro.util import dump_json_file, str_to_bool
from datumaro.util.scope import scoped

from ..util import MultilineFormatter
from ..util.dataset_utils import generate_next_file_name, parse_dataset_pathspec

__all__ = [
    "build_parser",
]


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Get dataset statistics",
        description="""
        Outputs various dataset statistics like image mean and std (RGB),
        annotations count etc.|n
        |n
        Target dataset is specified by a dataset path:|n
        - Dataset paths:|n
        |s|s- <dataset path>[ :<format> ]|n
        |n
        Examples:|n
        - Compute dataset statistics:|n
        |s|s%(prog)s /path/to/dataset:coco
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument("target", help="Target dataset path")
    parser.add_argument("-s", "--subset", help="Compute stats only for a specific subset")
    parser.add_argument(
        "--image-stats",
        type=str_to_bool,
        default=True,
        help="Compute image mean and std (RGB) (default: %(default)s)",
    )
    parser.add_argument(
        "--ann-stats",
        type=str_to_bool,
        default=True,
        help="Compute annotation statistics (default: %(default)s)",
    )
    parser.set_defaults(command=stats_command)

    return parser


@scoped
def stats_command(args):
    from datumaro.components.environment import DEFAULT_ENVIRONMENT

    # Parse target dataset
    dataset = parse_dataset_pathspec(args.target, DEFAULT_ENVIRONMENT)

    if args.subset:
        dataset = dataset.get_subset(args.subset)

    stats = {}
    if args.image_stats:
        stats.update(compute_image_statistics(dataset))
    if args.ann_stats:
        stats.update(compute_ann_statistics(dataset))

    dst_file = generate_next_file_name("statistics", ext=".json")
    log.info("Writing dataset statistics to '%s'" % dst_file)
    dump_json_file(dst_file, stats, indent=True)
