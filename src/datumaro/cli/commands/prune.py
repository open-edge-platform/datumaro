# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os.path as osp

from datumaro.components.algorithms.hash_key_inference.prune import Prune
from datumaro.components.errors import ProjectNotFoundError
from datumaro.util.scope import scope_add, scoped

from ..util import MultilineFormatter
from ..util.project import load_project, parse_full_revpath

deprecated = "[DEPRECATED, will be removed in 1.12]"


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help=f"{deprecated} Prune dataset and make a representative subset",
        description=f"""
        {deprecated} Apply data pruning to a dataset.|n
        The command can be useful if you have to extract representative subset.
        |n
        The current project (-p/--project) is used as a context for plugins
        and models. It is used when there is a dataset path in target.
        When not specified, the current project's working tree is used.|n
        |n
        By default, datasets are updated in-place. The '-o/--output-dir'
        option can be used to specify another output directory. When
        updating in-place, use the '--overwrite' parameter (in-place
        updates fail by default to prevent data loss), unless a project
        target is modified.|n
        |n
        The command can be applied to a dataset or a project build target,
        a stage or the combined 'project' target, in which case all the
        targets will be affected.|n
        |n
        Examples:|n
        - Prune dataset with selecting random and ratio 80%%:|n
        |s|s%(prog)s -m random -r 0.8|n
        - Prune dataset with clustering in image hash and ratio 50%%:|n
        |s|s%(prog)s -m query_clust -h img -r 0.5|n
        - Prune dataset based on entropy with clustering in image hash and ratio 50%%:|n
        |s|s%(prog)s -m entropy -h img -r 0.5
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "target",
        nargs="?",
        default="project",
        metavar="revpath",
        help="Target dataset revpath (default: project)",
    )
    parser.add_argument("-m", "--method", dest="method", help="Method to apply to the dataset")
    parser.add_argument(
        "-r", "--ratio", type=float, dest="ratio", help="How much to remain dataset after pruning"
    )
    parser.add_argument(
        "--hash-type",
        type=str,
        dest="hash_type",
        default="img",
        help="Hashtype to extract feature from data information between image and text(label)",
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        help="""
            Output directory. Can be omitted for main project targets
            (i.e. data sources and the 'project' target, but not
            intermediate stages) and dataset targets.
            If not specified, the results will be saved inplace.
            """,
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.set_defaults(command=prune_command)

    return parser


def get_sensitive_args():
    return {
        prune_command: [
            "target",
            "method",
            "ratio",
            "hash_type",
            "project_dir",
            "dst_dir",
        ]
    }


@scoped
def prune_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if args.project_dir:
            raise

    targets = [args.target] if args.target else list(project.working_tree.sources)

    source_dataset = [parse_full_revpath(target, project)[0] for target in targets][0]

    prune = Prune(source_dataset, cluster_method=args.method, hash_type=args.hash_type)
    result = prune.get_pruned(args.ratio)

    dst_dir = args.dst_dir or source_dataset.data_path
    dst_dir = (
        dst_dir if dst_dir else osp.join(args.project_dir, list(project.working_tree.sources)[0])
    )
    result.save(dst_dir, save_media=True)

    log.info("Results have been saved to '%s'" % dst_dir)
