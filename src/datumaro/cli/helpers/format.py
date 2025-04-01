# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import os

from datumaro.cli.util import MultilineFormatter


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="List supported import/export data formats",
        description="""
        List supported import/export data format names.
        For more detailed guides on each data format,
        please visit our documentation website:
        https://open-edge-platform.github.io/datumaro/stable/docs/data-formats/formats
        |n
        |n
        Examples:|n
        - List supported import/export data format names:|n
        |s|s%(prog)s|n
        |n
        - List only supported import data format names:|n
        |s|s%(prog)s --list-import|n
        |n
        - List only supported export data format names:|n
        |s|s%(prog)s --list-export|n
        |n
        - List with comma delimiter:|n
        |s|s%(prog)s --delimiter ','
        """,
        formatter_class=MultilineFormatter,
    )

    group = parser.add_argument_group()
    exclusive_group = group.add_mutually_exclusive_group(required=False)

    exclusive_group.add_argument(
        "-li",
        "--list-import",
        action="store_true",
        help="List all supported import data format names",
    )
    exclusive_group.add_argument(
        "-le",
        "--list-export",
        action="store_true",
        help="List all supported export data format names",
    )
    parser.add_argument(
        "-d",
        "--delimiter",
        type=str,
        default=os.linesep,
        help="Seperator used to list data format names (default: \\n)",
    )

    parser.set_defaults(command=format_command)
    return parser


def get_sensitive_args():
    return {
        format_command: ["list", "show"],
    }


def format_command(args: argparse.Namespace) -> None:
    from datumaro.components.environment import DEFAULT_ENVIRONMENT

    delimiter = args.delimiter

    if args.list_import:
        builtin_readers = sorted(
            set(DEFAULT_ENVIRONMENT.importers) | set(DEFAULT_ENVIRONMENT.extractors)
        )
        print(delimiter.join(builtin_readers))
        return

    if args.list_export:
        builtin_writers = sorted(DEFAULT_ENVIRONMENT.exporters)
        print(delimiter.join(builtin_writers))
        return

    builtin_readers = sorted(
        set(DEFAULT_ENVIRONMENT.importers) | set(DEFAULT_ENVIRONMENT.extractors)
    )
    builtin_writers = sorted(DEFAULT_ENVIRONMENT.exporters)
    print(f"Supported import formats:\n{delimiter.join(builtin_readers)}")
    print(f"Supported export formats:\n{delimiter.join(builtin_writers)}")
