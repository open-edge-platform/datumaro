# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.validator import TaskType
from datumaro.util import dump_json_file
from datumaro.util.scope import scoped

from ..util import MultilineFormatter
from ..util.dataset_utils import generate_next_file_name, parse_dataset_pathspec
from ..util.errors import CliException


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Validate dataset",
        description="""
        Validates a dataset according to the task type and
        reports summary in a JSON file.|n
        |n
        Target dataset is specified by a dataset path:|n
        - Dataset paths:|n
        |s|s- <dataset path>[ :<format> ]|n
        |n
        Examples:|n
        - Validate a dataset as a classification dataset:|n |n
        |s|s%(prog)s /path/to/dataset:coco -t classification
        """,
        formatter_class=MultilineFormatter,
    )

    task_types = ", ".join(t.name for t in TaskType)

    def _parse_task_type(s):
        try:
            return TaskType[s.lower()].name
        except Exception:
            import sys

            print(
                "Unknown task type '%s'. Available task types: %s" % (s, task_types),
                file=sys.stderr,
            )
            sys.exit(1)

    parser.add_argument("target", help="Target dataset path")
    parser.add_argument(
        "-t",
        "--task",
        type=_parse_task_type,
        required=True,
        help="Task type for validation, one of %s" % task_types,
    )
    parser.add_argument(
        "-s", "--subset", dest="subset_name", help="Subset to validate (default: whole dataset)"
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments for validator (pass '-- -h' for help)",
    )
    parser.set_defaults(command=validate_command)

    return parser


@scoped
def validate_command(args):
    env = DEFAULT_ENVIRONMENT

    try:
        validator_type = env.validators[args.task]
    except KeyError:
        raise CliException("Validator for '%s' task is not found" % args.task)

    extra_args = validator_type.parse_cmdline(args.extra_args)

    # Parse target dataset
    dataset = parse_dataset_pathspec(args.target, env)

    if args.subset_name:
        dataset = dataset.get_subset(args.subset_name)

    # Validate the dataset
    validator = validator_type(**extra_args)
    reports = validator.validate(dataset)

    def _make_serializable(d):
        for key, val in list(d.items()):
            # tuple key to str
            if isinstance(key, tuple) or isinstance(key, int):
                d[str(key)] = val
                d.pop(key)
            if isinstance(val, dict):
                _make_serializable(val)

    _make_serializable(reports)

    # Save the validation report
    dst_file = generate_next_file_name("validation_report", ext=".json")
    log.info("Writing validation report to '%s'" % dst_file)
    dump_json_file(dst_file, reports, indent=True)
