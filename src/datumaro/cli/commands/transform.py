# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.util.scope import scoped

from ..util import MultilineFormatter
from ..util.dataset_utils import parse_dataset_pathspec
from ..util.errors import CliException


def build_parser(parser_ctor=argparse.ArgumentParser):
    builtins = sorted(DEFAULT_ENVIRONMENT.transforms)

    parser = parser_ctor(
        help="Transform dataset",
        description="""
        Applies a batch operation to a dataset and produces a new dataset.|n
        |n
        By default, datasets are updated in-place. The '-o/--output-dir'
        option can be used to specify another output directory. When
        updating in-place, use the '--overwrite' parameter (in-place
        updates fail by default to prevent data loss).|n
        |n
        Builtin transforms: {}|n
        |n
        This command has the following invocation syntax:
        - %(prog)s <dataset path> -t <transform>|n
        |n
        <dataset path> - a dataset path with optional format specification:|n
        |s|s- <dataset path>[ :<format> ]|n
        |n
        Examples:|n
        - Convert instance polygons to masks:|n |n
        |s|s%(prog)s dataset_path:coco -t polygons_to_masks|n
        |n
        - Rename dataset items by a regular expression:|n |n
        |s|s- Replace 'pattern' with 'replacement':|n |n
        |s|s|s|s%(prog)s dataset_path:voc -t rename -- -e '|pattern|replacement|'|n
        |n
        - Split a dataset randomly:|n |n
        |s|s%(prog)s --overwrite path/to/dataset:voc -t random_split
        """.format(
            ", ".join(builtins)
        ),
        formatter_class=MultilineFormatter,
    )

    parser.add_argument("target", help="Target dataset path")
    parser.add_argument(
        "-t", "--transform", required=True, help="Transform to apply to the dataset"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        help="Output directory. If not specified, the results will be saved inplace.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for transformation (pass '-- -h' for help). "
        "Must be specified after the main command arguments and after "
        "the '--' separator",
    )
    parser.set_defaults(command=transform_command)

    return parser


@scoped
def transform_command(args):
    env = DEFAULT_ENVIRONMENT

    # Parse target dataset
    dataset = parse_dataset_pathspec(args.target, env)

    # Get the transform
    try:
        transform = env.transforms[args.transform]
    except KeyError:
        raise CliException("Transform '%s' is not found" % args.transform)

    extra_args = transform.parse_cmdline(args.extra_args)

    # Determine output directory
    dst_dir = args.dst_dir or dataset.data_path

    if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
        raise CliException(
            "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
        )
    dst_dir = osp.abspath(dst_dir)

    # Apply the transform
    log.info("Transforming...")
    dataset.transform(args.transform, **extra_args)
    dataset.save(dst_dir, save_media=True)

    log.info("Results have been saved to '%s'" % dst_dir)

    return 0
