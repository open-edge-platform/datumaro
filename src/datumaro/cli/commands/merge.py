# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.dataset import DEFAULT_FORMAT
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.hl_ops import HLOps
from datumaro.components.merge.intersect_merge import IntersectMerge

from ..util import MultilineFormatter, join_cli_args
from ..util.dataset_utils import generate_next_file_name, parse_dataset_pathspec
from ..util.errors import CliException


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Merge multiple datasets",
        description="""
        Merges multiple datasets into one and produces a new dataset.
        The command can be useful if you want to merge several datasets
        which have homogeneous or heterogeneous label categories.
        We provide three merge policies: 1) "exact" for homogenous datasets;
        2) "union", or 3) "intersect" for heterogenous datasets. The default merge
        policy is "union".
        |n
        |n
        In simple cases, when dataset images do not intersect and new
        labels are not added, we can use the "exact" merge policy for this situation.
        |n
        |n
        On the other hand, when two datasets have different label categories or
        have the same (id, subset) pairs in their dataset items. you have to use
        the "union" or "intersect" merge.
        |n
        The "union" merge is more simple way to merge them.
        It tries to obtain a union set from their label categories and attempts to merge
        their dataset items. However, if there exist the same (id, subset) pairs, it appends
        a suffix to each item's id to prevent the same (id, subset) pair in the merged dataset.
        |n
        The "intersect" policy is a more complicated way to merge heterogeneous datasets. If you want to merge
        the annotations in the same (id, subset) pairs which coming from the different datasets
        each other you wish to merge. You have to taking into consideration potential overlaps
        and conflicts between the annotations. "intersect" merge policy can try to find common
        ground by voting or return a list of conflicts.
        |n
        |n
        Dataset paths can optionally specify format:|n
        |s|s- <dataset path>[ :<format> ]|n
        |n
        The output format can be specified with the '-f/--format' option.
        Each dataset format has its own export
        options, which are passed after the '--' separator (see examples),
        pass '-- -h' for more info. If not stated otherwise, by default
        only annotations are exported; to include images pass
        '--save-media' parameter.|n
        |n
        Examples:|n
        - Merge annotations from 3 (or more) datasets:|n
        |s|s%(prog)s dataset1/ dataset2/ dataset3/|n
        |n
        - Merge datasets with varying merge policies:|n
        |s|s%(prog)s dataset1/ dataset2/ -m <union|intersect|exact>|n
        |n
        - Check groups of the merged dataset for consistency:|n
        |s|s|slook for groups consising of 'person', 'hand' 'head', 'foot'|n
        |s|s%(prog)s dataset1/ dataset2/ -g 'person,hand?,head,foot?'|n
        |n
        - Merge two datasets, specify formats:|n
        |s|s%(prog)s path/to/dataset1:voc path/to/dataset2:coco|n
        |n
        - Merge datasets and save in different format:|n
        |s|s%(prog)s -f voc dataset1/:yolo path2/:coco -- --save-media
        """,
        formatter_class=MultilineFormatter,
    )

    def _group(s):
        return s.split(",")

    parser.add_argument(
        "_positionals", nargs=argparse.REMAINDER, help=argparse.SUPPRESS
    )  # workaround for -- eaten by positionals
    parser.add_argument("targets", nargs="+", help="Target dataset revpaths (repeatable)")
    parser.add_argument(
        "-m",
        "--merge-policy",
        default="union",
        type=str,
        help="Policy for how to merge datasets (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        default=None,
        help="Output directory (default: generate a new one)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.add_argument(
        "-f", "--format", default=DEFAULT_FORMAT, help="Output format (default: %(default)s)"
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for exporter (pass '-- -h' for help). "
        "Must be specified after the main command arguments and after "
        "the '--' separator",
    )

    intersect_args = parser.add_argument_group(
        title="intersect policy arguments",
        description="These parameters are optional for the intersect policy only.",
    )
    intersect_args.add_argument(
        "-iou",
        "--iou-thresh",
        default=0.25,
        type=float,
        help="IoU match threshold for segments (default: %(default)s)",
    )
    intersect_args.add_argument(
        "-oconf",
        "--output-conf-thresh",
        default=0.0,
        type=float,
        help="Confidence threshold for output " "annotations (default: %(default)s)",
    )
    intersect_args.add_argument(
        "--quorum",
        default=0,
        type=int,
        help="Minimum count for a label and attribute voting "
        "results to be counted (default: %(default)s)",
    )
    intersect_args.add_argument(
        "-g",
        "--groups",
        action="append",
        type=_group,
        help="A comma-separated list of labels in "
        "annotation groups to check. '?' postfix can be added to a label to "
        "make it optional in the group (repeatable)",
    )

    parser.set_defaults(command=merge_command)

    return parser


def merge_command(args):
    # Workaround. Required positionals consume positionals from the end
    args._positionals += join_cli_args(args, "targets", "extra_args")

    has_sep = "--" in args._positionals
    if has_sep:
        pos = args._positionals.index("--")
        if pos == 0:
            raise argparse.ArgumentError(None, message="Expected at least 1 target argument")
    else:
        pos = len(args._positionals)
    args.targets = args._positionals[:pos]
    args.extra_args = args._positionals[pos + has_sep :]

    if not args.targets:
        raise CliException("Expected at least 1 target dataset")

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException(
                "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
            )
    else:
        dst_dir = generate_next_file_name("merged")
    dst_dir = osp.abspath(dst_dir)

    env = DEFAULT_ENVIRONMENT

    try:
        exporter = env.exporters[args.format]
    except KeyError:
        raise CliException("Exporter for format '%s' is not found" % args.format)

    export_args = exporter.parse_cmdline(args.extra_args)

    source_datasets = []
    try:
        for target in args.targets:
            try:
                dataset = parse_dataset_pathspec(target, env=env)
            except Exception as e:
                raise CliException(str(e))
            source_datasets.append(dataset)
    except Exception as e:
        raise CliException(str(e))

    report_path = osp.join(dst_dir, "merge_report.json")
    options = (
        {
            "conf": IntersectMerge.Conf(
                pairwise_dist=args.iou_thresh,
                groups=args.groups or [],
                output_conf_thresh=args.output_conf_thresh,
                quorum=args.quorum,
            )
        }
        if args.merge_policy == "intersect"
        else {}
    )
    merged_dataset = HLOps.merge(
        *source_datasets, merge_policy=args.merge_policy, report_path=report_path, **options
    )

    merged_dataset.export(save_dir=dst_dir, format=exporter, **export_args)

    log.info("Merge results have been saved to '%s'" % dst_dir)
    log.info("Report has been saved to '%s'" % report_path)

    return 0
