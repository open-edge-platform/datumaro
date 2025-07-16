# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT


import argparse
import logging as log

from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.errors import ProjectNotFoundError
from datumaro.util.scope import on_error_do, scope_add, scoped

from ....util import MultilineFormatter, join_cli_args, show_video_import_warning
from ....util.errors import CliException
from ....util.project import load_project

__all__ = [
    "build_parser",
    "get_sensitive_args",
]


deprecated = "[DEPRECATED, will be removed in 1.12]"


def build_parser(parser_ctor=argparse.ArgumentParser):
    env = DEFAULT_ENVIRONMENT
    builtins = sorted(set(env.extractors) | set(env.importers))

    parser = parser_ctor(
        help=f"{deprecated} Add data source to project",
        description=f"""
        {deprecated} Adds a data source to a project. A data source is a dataset
        in a supported format (check 'formats' section below).|n
        |n
        The command adds a project-local directory as a data source in
        the project. Unlike the "import" command, it does not copy datasets
        and only works with local directories. The source name is defined by
        the directory name.|n
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
        Each dataset format has its own import options, which are passed
        after the '--' separator (see examples), pass '-- -h' for more info.|n
        |n
        Builtin formats: {{}}|n
        |n
        Examples:|n
        - Add a local directory with a VOC-like dataset:|n
        |s|s%(prog)s -f voc my_dataset/|n
        |n
        - Add a directory with a COCO dataset, use only a specific file:|n
        |s|s%(prog)s -f coco_instances -r anns/train.json my_source/
        """.format(
            ", ".join(builtins)
        ),
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "_positionals", nargs=argparse.REMAINDER, help=argparse.SUPPRESS
    )  # workaround for -- eaten by positionals
    parser.add_argument("path", help="A path to a dataset directory")
    parser.add_argument("-f", "--format", required=True, help="Source dataset format")
    parser.add_argument(
        "-r",
        "--path",
        dest="rpath",
        help="A path relative to source root to the source data. Useful to "
        "specify a path to subset, subtask, or a specific file in dataset.",
    )
    parser.add_argument("--no-check", action="store_true", help="Don't try to read the source")
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        default=".",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for extractor (pass '-- -h' for help). "
        "Must be specified after the main command arguments and after "
        "the '--' separator",
    )
    parser.set_defaults(command=add_command)

    return parser


def get_sensitive_args():
    return {
        add_command: ["url", "project_dir", "rpath", "extra_args"],
    }


@scoped
def add_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")

    # Workaround. Required positionals consume positionals from the end
    args._positionals += join_cli_args(args, "path", "extra_args")

    has_sep = "--" in args._positionals
    if has_sep:
        pos = args._positionals.index("--")
    else:
        pos = 1
    args.path = (args._positionals[:pos] or [""])[0]
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

    fmt = args.format
    if fmt in env.importers:
        arg_parser = env.importers[fmt]
    elif fmt in env.extractors:
        arg_parser = env.extractors[fmt]
    else:
        raise CliException(
            "Unknown format '%s'. A format can be added"
            " by providing an Extractor and Importer plugins" % fmt
        )

    extra_args = arg_parser.parse_cmdline(args.extra_args)

    if fmt == "video_frames":
        show_video_import_warning()

    name, _ = project.add_source(
        args.path, format=args.format, options=extra_args, rpath=args.rpath
    )
    on_error_do(
        project.remove_source, name, ignore_errors=True, kwargs={"force": True, "keep_data": True}
    )

    if not args.no_check:
        log.info("Checking the source...")
        project.working_tree.make_dataset(name)

    project.working_tree.save()

    log.info("Source '%s' with format '%s' has been added to the project", name, args.format)

    return 0
