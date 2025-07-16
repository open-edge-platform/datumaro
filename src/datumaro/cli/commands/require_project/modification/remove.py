# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from datumaro.util.scope import scope_add, scoped

from ....util.errors import CliException
from ....util.project import load_project

__all__ = [
    "build_parser",
    "get_sensitive_args",
]


deprecated = "[DEPRECATED, will be removed in 1.12]"


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help=f"{deprecated} Remove source from project",
        description=f"{deprecated} Remove a source from a project",
    )

    parser.add_argument("names", nargs="+", help="Names of the sources to be removed")
    parser.add_argument(
        "--force", action="store_true", help="Do not fail and stop on errors during removal"
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Do not remove source data from the working directory, remove "
        "only project metainfo.",
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.set_defaults(command=remove_command)

    return parser


def get_sensitive_args():
    return {
        remove_command: [
            "project_dir",
            "names",
        ],
    }


@scoped
def remove_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    project = scope_add(load_project(args.project_dir))

    if not args.names:
        raise CliException("Expected source name")

    for name in args.names:
        project.remove_source(name, force=args.force, keep_data=args.keep_data)
    project.working_tree.save()

    log.info("Sources '%s' have been removed from the project" % ", ".join(args.names))

    return 0
