# Copyright (C) 2021 Intel Corporation
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
    parser = parser_ctor(
        help="Updates dataset from another one",
        description="""
        Updates items of the first dataset with items from the second one.|n
        |n
        By default, datasets are updated in-place. The '-o/--output-dir'
        option can be used to specify another output directory. When
        updating in-place, use the '--overwrite' parameter along with the
        '--save-media' export option (in-place updates fail by default
        to prevent data loss).|n
        |n
        The datasets are not
        required to have the same labels. The labels from the "patch"
        dataset are projected onto the labels of the patched dataset,
        so only the annotations with the matching labels are used, i.e.
        all the annotations having unknown labels are ignored. Currently,
        this command doesn't allow to update the label information in the
        patched dataset.|n
        |n
        The command supports passing extra exporting options for the output
        dataset. The extra options should be passed after the main arguments
        and after the '--' separator. Particularly, this is useful to include
        images in the output dataset with '--save-media'.|n
        |n
        This command can be applied to arbitrary datasets.|n
        |n
        This command has the following invocation syntax:
        - %(prog)s <target dataset path> <patch dataset path>|n
        |n
        <dataset path> - a dataset path with optional format specification:|n
        |s|s- <dataset path>[ :<format> ]|n
        |n
        Examples:|n
        - Update a VOC-like dataset with COCO-like annotations:|n
        |s|s%(prog)s --overwrite dataset1/:voc dataset2/:coco -- --save-media|n
        |n
        - Generate a patched dataset:|n
        |s|s%(prog)s -o patched_dataset/ dataset1/ dataset2/|n
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument("target", help="Target dataset path")
    parser.add_argument("patch", help="Patch dataset path")
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        default=None,
        help="Output directory (default: save in-place)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the save directory, " "if it is not empty",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for exporting (pass '-- -h' for help). "
        "Must be specified after the main command arguments and after "
        "the '--' separator",
    )
    parser.set_defaults(command=patch_command)

    return parser


@scoped
def patch_command(args):
    env = DEFAULT_ENVIRONMENT

    # Parse target and patch datasets
    target_dataset = parse_dataset_pathspec(args.target, env)
    patch_dataset = parse_dataset_pathspec(args.patch, env)

    # Determine output directory
    dst_dir = args.dst_dir or target_dataset.data_path

    if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
        raise CliException(
            "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
        )
    dst_dir = osp.abspath(dst_dir)

    # Get the exporter for the target format
    try:
        exporter = env.exporters[target_dataset.format]
    except KeyError:
        raise CliException("Exporter for format '%s' is not found" % target_dataset.format)

    extra_args = exporter.parse_cmdline(args.extra_args)

    # Apply the patch
    target_dataset.update(patch_dataset)

    # Save the updated dataset
    target_dataset.save(save_dir=dst_dir, **extra_args)

    log.info("Patched dataset has been saved to '%s'" % dst_dir)

    return 0
