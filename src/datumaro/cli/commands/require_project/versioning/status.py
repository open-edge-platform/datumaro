# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from datumaro.cli.util import MultilineFormatter
from datumaro.util.scope import scope_add, scoped

from ....util.project import load_project

deprecated = "[DEPRECATED, will be removed in 1.12]"


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help=f"{deprecated} Prints project status.",
        description=f"""
        {deprecated} This command prints the summary of the project changes between
        the working tree of a project and its HEAD revision.
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.set_defaults(command=status_command)

    return parser


def get_sensitive_args():
    return {
        status_command: [
            "project_dir",
        ],
    }


@scoped
def status_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    project = scope_add(load_project(args.project_dir))

    statuses = project.status()

    if project.branch:
        print("On branch '%s', commit %s" % (project.branch, project.head_rev))
    else:
        print("HEAD is detached at commit %s" % project.head_rev)

    if statuses:
        for target, status in statuses.items():
            print("%s\t%s" % (status.name, target))
    else:
        print("Working directory clean")

    return 0
