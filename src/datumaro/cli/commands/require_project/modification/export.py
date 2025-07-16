# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.project import ProjectBuildTargets
from datumaro.util.os_util import make_file_name
from datumaro.util.scope import scope_add, scoped

from ....util import MultilineFormatter
from ....util.errors import CliException
from ....util.project import FilterModes, generate_next_file_name, load_project

__all__ = [
    "build_parser",
    "get_sensitive_args",
]


deprecated = "[DEPRECATED, will be removed in 1.12]"


def build_parser(parser_ctor=argparse.ArgumentParser):
    builtins = sorted(DEFAULT_ENVIRONMENT.exporters)

    parser = parser_ctor(
        help=f"{deprecated} Export project",
        description=f"""
        {deprecated} Exports a project in some format.|n
        |n
        Each dataset format has its own export
        options, which are passed after the '--' separator (see examples),
        pass '-- -h' for more info. If not stated otherwise, by default
        only annotations are exported, to include images pass
        '--save-media' parameter.|n
        |n
        A filter can be passed, check the 'filter' command description for
        more info.|n
        |n
        Formats:|n
        Datasets come in a wide variety of formats. Each dataset
        format defines its own data structure and rules on how to
        interpret the data. Check the user manual for the list of
        supported formats, examples and documentation.
        |n
        The list of supported formats can be extended by plugins.
        Check the "plugins" section of the developer guide for information
        about plugin implementation.|n
        |n
        List of builtin dataset formats: {{}}|n
        |n
        The command can only be applied to a project build target, a stage
        or the combined 'project' target, in which case all the targets will
        be affected.
        |n
        Examples:|n
        - Export project as a VOC-like dataset, include images:|n
        |s|s%(prog)s -f voc -- --save-media|n
        |n
        - Export project as a COCO-like dataset in other directory:|n
        |s|s%(prog)s -f coco -o path/I/like/
        """.format(
            ", ".join(builtins)
        ),
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "_positionals", nargs=argparse.REMAINDER, help=argparse.SUPPRESS
    )  # workaround for -- eaten by positionals
    parser.add_argument(
        "target",
        nargs="?",
        default="project",
        help="A project target to be exported (default: %(default)s)",
    )
    parser.add_argument("-e", "--filter", help="XML XPath filter expression for dataset items")
    parser.add_argument(
        "--filter-mode",
        default=FilterModes.i.name,
        type=FilterModes.parse,
        help="Filter mode (options: %s; default: %s)"
        % (", ".join(FilterModes.list_options()), "%(default)s"),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        help="Directory to save output (default: a subdir in the current one)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.add_argument("-f", "--format", required=True, help="Output format")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for exporter (pass '-- -h' for help). "
        "Must be specified after the main command arguments and after "
        "the '--' separator",
    )
    parser.set_defaults(command=export_command)

    return parser


def get_sensitive_args():
    return {
        export_command: ["dst_dir", "project_dir", "name", "extra_args", "target", "filter"],
    }


@scoped
def export_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    has_sep = "--" in args._positionals
    if has_sep:
        pos = args._positionals.index("--")
        if 1 < pos:
            raise argparse.ArgumentError(None, message="Expected no more than 1 target argument")
    else:
        pos = 1
    args.target = (args._positionals[:pos] or [ProjectBuildTargets.MAIN_TARGET])[0]
    args.extra_args = args._positionals[pos + has_sep :]

    show_plugin_help = "-h" in args.extra_args or "--help" in args.extra_args

    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if not show_plugin_help:
            raise

    if project is not None:
        env = project.env
    else:
        env = DEFAULT_ENVIRONMENT

    try:
        exporter = env.exporters[args.format]
    except KeyError:
        raise CliException("Exporter for format '%s' is not found" % args.format)

    extra_args = exporter.parse_cmdline(args.extra_args)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException(
                "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
            )
    else:
        dst_dir = generate_next_file_name("export-%s" % make_file_name(args.format))
    dst_dir = osp.abspath(dst_dir)

    if args.filter:
        filter_args = FilterModes.make_filter_args(args.filter_mode)
        filter_expr = args.filter

    log.info("Loading the project...")

    dataset = project.working_tree.make_dataset(args.target)
    if args.filter:
        dataset.filter(filter_expr, **filter_args)

    log.info("Exporting...")

    dataset.export(save_dir=dst_dir, format=exporter, **extra_args)

    log.info("Results have been saved to '%s'" % dst_dir)

    return 0
