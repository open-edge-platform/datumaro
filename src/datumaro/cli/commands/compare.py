# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp
import shutil
from enum import Enum, auto

from datumaro.components.comparator import DistanceComparator, EqualityComparator, TableComparator
from datumaro.util.scope import on_error_do, scoped

from ..util import MultilineFormatter
from ..util.compare import DistanceCompareVisualizer
from ..util.dataset_utils import generate_next_file_name, parse_dataset_pathspec
from ..util.errors import CliException


class ComparisonMethod(Enum):
    table = auto()
    equality = auto()
    distance = auto()


eq_default_if = ["id", "group"]  # avoid https://bugs.python.org/issue16399


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Compares two datasets",
        description="""
        Compares two datasets by specifying their paths.|n
        |n
        Usage: %(prog)s <dataset1> <dataset2>|n
        |n
        <dataset> - dataset path with optional format specification:|n
        |s|s- <dataset path>[ :<format> ]|n
        |n
        Annotations can be matched 3 ways:|n
        - by comparison table|n
        - by equality checking|n
        - by distance computation|n
        |n
        Examples:|n
        - Compare two datasets by distance, match boxes if IoU > 0.7,|n
        |s|s|s|ssave results to Tensorboard:|n
        |s|s%(prog)s dataset1/ dataset2/ -o diff/ -f tensorboard --iou-thresh 0.7|n
        |n
        - Compare two datasets for equality, exclude annotation groups |n
        |s|s|s|sand the 'is_crowd' attribute from comparison:|n
        |s|s%(prog)s dataset1/ dataset2/ -if group -ia is_crowd -m equality|n
        |n
        - Compare two datasets, specify formats:|n
        |s|s%(prog)s path/to/dataset1:voc path/to/dataset2:coco|n
        """,
        formatter_class=MultilineFormatter,
    )

    formats = ", ".join(f.name for f in DistanceCompareVisualizer.OutputFormat)
    comp_methods = ", ".join(m.name for m in ComparisonMethod)

    def _parse_output_format(s):
        try:
            return DistanceCompareVisualizer.OutputFormat[s.lower()]
        except KeyError:
            raise argparse.ArgumentError(
                "format",
                message="Unknown output " "format '%s', the only available are: %s" % (s, formats),
            )

    def _parse_comparison_method(s):
        try:
            return ComparisonMethod[s.lower()]
        except KeyError:
            raise argparse.ArgumentError(
                "method",
                message="Unknown comparison "
                "method '%s', the only available are: %s" % (s, comp_methods),
            )

    parser.add_argument("first_target", help="The first dataset path to be compared")
    parser.add_argument("second_target", help="The second dataset path to be compared")
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        default=None,
        help="Directory to save comparison results " "(default: generate automatically)",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=_parse_comparison_method,
        default=ComparisonMethod.table.name,
        help="Comparison method, one of {} (default: %(default)s)".format(comp_methods),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.set_defaults(command=compare_command)

    distance_parser = parser.add_argument_group("Distance comparison options")
    distance_parser.add_argument(
        "--iou-thresh",
        default=0.5,
        type=float,
        help="IoU match threshold for shapes (default: %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=_parse_output_format,
        default=DistanceCompareVisualizer.DEFAULT_FORMAT.name,
        help="Output format, one of {} (default: %(default)s)".format(formats),
    )

    equality_parser = parser.add_argument_group("Equality comparison options")
    equality_parser.add_argument(
        "-iia", "--ignore-item-attr", action="append", help="Ignore item attribute (repeatable)"
    )
    equality_parser.add_argument(
        "-ia", "--ignore-attr", action="append", help="Ignore annotation attribute (repeatable)"
    )
    equality_parser.add_argument(
        "-if",
        "--ignore-field",
        action="append",
        help="Ignore annotation field (repeatable, default: %s)" % eq_default_if,
    )
    equality_parser.add_argument(
        "--match-images",
        action="store_true",
        help="Match dataset items by image pixels instead of ids",
    )
    equality_parser.add_argument("--all", action="store_true", help="Include matches in the output")

    return parser


@scoped
def compare_command(args):
    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException(
                "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
            )
    else:
        dst_dir = generate_next_file_name("compare")
    dst_dir = osp.abspath(dst_dir)

    if not osp.exists(dst_dir):
        on_error_do(shutil.rmtree, dst_dir, ignore_errors=True)
        os.makedirs(dst_dir)

    try:
        first_dataset = parse_dataset_pathspec(args.first_target)
        second_dataset = parse_dataset_pathspec(args.second_target)
    except Exception as e:
        raise CliException(str(e))

    if args.method is ComparisonMethod.table:
        comparator = TableComparator()
        (
            high_level_table,
            mid_level_table,
            comparison_dict,
        ) = comparator.compare_datasets(first_dataset, second_dataset)
        if args.dst_dir:
            comparator.save_compare_report(
                high_level_table, mid_level_table, comparison_dict, args.dst_dir
            )

    elif args.method is ComparisonMethod.equality:
        if not args.ignore_field:
            args.ignore_field = eq_default_if

        comparator = EqualityComparator(
            match_images=args.match_images,
            ignored_fields=args.ignore_field,
            ignored_attrs=args.ignore_attr,
            ignored_item_attrs=args.ignore_item_attr,
            all=args.all,
        )
        output = comparator.compare_datasets(first_dataset, second_dataset)
        if args.dst_dir:
            comparator.save_compare_report(output, args.dst_dir)

    elif args.method is ComparisonMethod.distance:
        comparator = DistanceComparator(iou_threshold=args.iou_thresh)
        with DistanceCompareVisualizer(
            save_dir=dst_dir, comparator=comparator, output_format=args.format
        ) as visualizer:
            log.info("Saving compare to '%s'" % dst_dir)
            visualizer.save(first_dataset, second_dataset)

    return 0
