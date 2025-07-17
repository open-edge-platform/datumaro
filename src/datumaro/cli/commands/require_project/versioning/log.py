# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from datumaro.util.scope import scope_add, scoped

from ....util.project import load_project

deprecated = "[DEPRECATED, will be removed in 1.12]"


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(description=f"{deprecated} Prints project history.")

    parser.add_argument(
        "-n",
        "--max-count",
        default=10,
        type=int,
        help="Count of last commits to print (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.set_defaults(command=log_command)

    return parser


def get_sensitive_args():
    return {
        log_command: [
            "project_dir",
        ],
    }


@scoped
def log_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    project = scope_add(load_project(args.project_dir))

    revisions = project.history(args.max_count)
    if revisions:
        for rev, message in revisions:
            print("%s %s" % (rev, message))
    else:
        print("(Project history is empty)")

    return 0
