# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from datumaro.util.scope import scope_add, scoped

from ....util import MultilineFormatter
from ....util.project import load_project

deprecated = "[DEPRECATED, will be removed in 1.12]"


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help=f"{deprecated} Create a revision",
        description=f"""
        {deprecated} Creates a new revision from the current state of the working directory.|n
        |n
        Examples:|n
        - Create a revision:|n
        |s|s%(prog)s -m "Added data"
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument("-m", "--message", required=True, help="Commit message")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow commits with no changes (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-foreign",
        action="store_true",
        help="Allow commits with non-Datumaro changes (default: %(default)s)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't put committed datasets into cache, "
        "save only metainfo (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.set_defaults(command=commit_command)

    return parser


def get_sensitive_args():
    return {
        commit_command: [
            "message",
            "project_dir",
        ],
    }


@scoped
def commit_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    project = scope_add(load_project(args.project_dir))

    old_tree = project.head

    new_commit = project.commit(
        args.message,
        allow_empty=args.allow_empty,
        allow_foreign=args.allow_foreign,
        no_cache=args.no_cache,
    )

    new_tree = project.working_tree
    diff = project.diff(old_tree, new_tree)

    print("Moved to commit '%s' %s" % (new_commit, args.message))
    print(" %s targets changed" % len(diff))
    for t, s in diff.items():
        print(" %s %s" % (s.name, t))

    return 0
