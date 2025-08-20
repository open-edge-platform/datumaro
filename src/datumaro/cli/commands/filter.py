# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.filter import DatasetItemEncoder
from datumaro.util.scope import scoped

from ..util import MultilineFormatter
from ..util.dataset_utils import FilterModes, parse_dataset_pathspec
from ..util.errors import CliException


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Extract subdataset",
        description="""
        Extracts a subdataset that contains only items matching filter.|n
        |n
        By default, datasets are updated in-place. The '-o/--output-dir'
        option can be used to specify another output directory. When
        updating in-place, use the '--overwrite' parameter.|n
        |n
        A filter is an XPath expression, which is applied to XML
        representation of a dataset item. Check '--dry-run' parameter
        to see XML representations of the dataset items.|n
        |n
        To filter annotations use the mode ('-m') parameter.|n
        Supported modes:|n
        - 'i', 'items'|n
        - 'a', 'annotations'|n
        - 'i+a', 'a+i', 'items+annotations', 'annotations+items'|n
        When filtering annotations, use the 'items+annotations'
        mode to point that annotation-less dataset items should be
        removed. To select an annotation, write an XPath that
        returns 'annotation' elements (see examples).|n
        |n
        Usage: %(prog)s <dataset_path>|n
        |n
        <dataset_path> - dataset path with optional format:|n
        |s|s- <dataset path>[ :<format> ]|n
        |n
        Examples:|n
        - Filter images with width < height:|n
        |s|s%(prog)s -e '/item[image/width < image/height]' dataset/|n
        |n
        - Filter images with large-area bboxes:|n
        |s|s%(prog)s -e '/item[annotation/type="bbox" and
            annotation/area>2000]' dataset/|n
        |n
        - Filter out all irrelevant annotations from items:|n
        |s|s%(prog)s -m a -e '/item/annotation[label = "person"]' dataset/|n
        |n
        - Filter out all irrelevant annotations from items:|n
        |s|s%(prog)s -m a -e '/item/annotation[label="cat" and
        area > 99.5]' dataset/|n
        |n
        - Filter occluded annotations and items, if no annotations left:|n
        |s|s%(prog)s -m i+a -e '/item/annotation[occluded="True"]' dataset/|n
        |n
        - Filter a VOC-like dataset inplace:|n
        |s|s%(prog)s -e '/item/annotation[label = "bus"]' --overwrite dataset/:voc
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "target",
        help="Target dataset path with optional format (e.g., 'dataset/' or 'dataset/:voc')",
    )
    parser.add_argument("-e", "--filter", help="XML XPath filter expression for dataset items")
    parser.add_argument(
        "-m",
        "--mode",
        default=FilterModes.i.name,
        type=FilterModes.parse,
        help="Filter mode (options: %s; default: %s)"
        % (", ".join(FilterModes.list_options()), "%(default)s"),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print XML representations to be filtered and exit"
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
    parser.set_defaults(command=filter_command)

    return parser


@scoped
def filter_command(args):
    filter_args = FilterModes.make_filter_args(args.mode)
    filter_expr = args.filter

    try:
        dataset = parse_dataset_pathspec(args.target)
    except Exception as e:
        raise CliException(str(e))

    if args.dry_run:
        dataset = dataset.filter(filter_expr, **filter_args)

        for item in dataset:
            encoded_item = DatasetItemEncoder.encode(item, dataset.categories())
            xml_item = DatasetItemEncoder.to_string(encoded_item)
            print(xml_item)
        return 0

    if not args.filter:
        raise CliException("Expected a filter expression ('-e' argument)")

    log.info("Filtering...")

    dst_dir = args.dst_dir or dataset.data_path
    if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
        raise CliException(
            "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
        )
    dst_dir = osp.abspath(dst_dir)

    dataset = dataset.filter(filter_expr, **filter_args)
    dataset.save(dst_dir, save_media=True)

    log.info("Results have been saved to '%s'" % dst_dir)
    return 0
