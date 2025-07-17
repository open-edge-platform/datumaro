# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.project import Environment
from datumaro.util.os_util import rmtree
from datumaro.util.scope import on_error_do, scope_add, scoped

from ..util import MultilineFormatter, add_subparser
from ..util.errors import CliException
from ..util.project import (
    generate_next_file_name,
    generate_next_name,
    load_project,
    parse_full_revpath,
)

deprecated = "[DEPRECATED, will be removed in 1.12]"


def build_add_parser(parser_ctor=argparse.ArgumentParser):
    builtins = sorted(Environment().launchers)

    parser = parser_ctor(
        help=f"{deprecated} Add model to project",
        description=f"""
        { deprecated } Adds an executable model into a project. A model requires
        a launcher to be executed. Each launcher has its own options, which
        are passed after the '--' separator, pass '-- -h' for more info.
        |n
        List of builtin launchers: {{}}|n
        |n
        Examples:|n
        - Add an OpenVINO model into a project:|n
        |s|s%(prog)s -l openvino -- -d model.xml -w model.bin -i parse_outs.py
        """.format(
            ", ".join(builtins)
        ),
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "-n",
        "--name",
        default=None,
        help="Name of the model to be added (default: generate automatically)",
    )
    parser.add_argument("-l", "--launcher", required=True, help="Model launcher")
    parser.add_argument(
        "--copy", action="store_true", help="Copy model data into project (default: %(default)s)"
    )
    parser.add_argument(
        "--no-check", action="store_true", help="Don't check model loading (default: %(default)s)"
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Additional arguments for exporter (pass '-- -h' for help)",
    )
    parser.set_defaults(command=add_command)

    return parser


def get_add_sensitive_args():
    return {
        add_command: ["project_dir", "name", "extra_args"],
    }


@scoped
def add_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    show_plugin_help = "-h" in args.extra_args or "--help" in args.extra_args

    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if not show_plugin_help and args.project_dir:
            raise

    if project is not None:
        env = project.env
    else:
        env = Environment()

    name = args.name
    if name:
        if name in project.models:
            raise CliException("Model '%s' already exists" % name)
    else:
        name = generate_next_name(list(project.models), "model", sep="-", default=0)

    try:
        launcher = env.launchers[args.launcher]
    except KeyError:
        raise CliException("Launcher '%s' is not found" % args.launcher)

    cli_plugin = getattr(launcher, "cli_plugin", launcher)
    model_args = cli_plugin.parse_cmdline(args.extra_args)

    if args.copy:
        log.info("Copying model data")

        model_dir = project.model_data_dir(name)
        os.makedirs(model_dir, exist_ok=False)
        on_error_do(rmtree, model_dir, ignore_errors=True)

        try:
            cli_plugin.copy_model(model_dir, model_args)
        except (AttributeError, NotImplementedError):
            raise NotImplementedError(
                "Can't copy: copying is not available for '%s' models. " % args.launcher
            )

    project.add_model(name, launcher=args.launcher, options=model_args)
    on_error_do(project.remove_model, name, ignore_errors=True)

    if not args.no_check:
        log.info("Checking the model...")
        project.make_model(name)

    project.save()

    log.info("Model '%s' with launcher '%s' has been added to project", name, args.launcher)

    return 0


def build_remove_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help=f"{deprecated} Remove model from project",
        description=f"{deprecated } Remove a model from a project",
    )

    parser.add_argument("name", help="Name of the model to be removed")
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.set_defaults(command=remove_command)

    return parser


def get_remove_sensitive_args():
    return {
        remove_command: [
            "project_dir",
            "name",
        ],
    }


@scoped
def remove_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    project = scope_add(load_project(args.project_dir))

    project.remove_model(args.name)
    project.save()

    log.info(f"Model {args.name} has been removed from the project.")

    return 0


def build_run_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help=f"{deprecated} Launches model inference",
        description=f"""
        { deprecated } Launches model inference on a dataset.|n
        |n
        Target dataset is specified by a revpath. The full syntax is:|n
        - Dataset paths:|n
        |s|s- <dataset path>[ :<format> ]|n
        - Revision paths:|n
        |s|s- <project path> [ @<rev> ] [ :<target> ]|n
        |s|s- <rev> [ :<target> ]|n
        |s|s- <target>|n
        |n
        Both forms use the -p/--project as a context for plugins and models.
        When not specified, the current project's working tree is used.|n
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "target", nargs="?", default="project", help="Target dataset revpath (default: %(default)s)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        help="Directory to save output (default: auto-generated)",
    )
    parser.add_argument(
        "-m", "--model", dest="model_name", required=True, help="Model to apply to the project"
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output directory if exists"
    )
    parser.set_defaults(command=run_command)

    return parser


def get_run_sensitive_args():
    return {
        run_command: [
            "target",
            "dst_dir",
            "model_name",
            "project_dir",
        ],
    }


@scoped
def run_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException(
                "Directory '%s' already exists " "(pass --overwrite overwrite)" % dst_dir
            )
    else:
        dst_dir = generate_next_file_name("%s-inference" % args.model_name)
    dst_dir = osp.abspath(dst_dir)

    project = scope_add(load_project(args.project_dir))

    dataset, target_project = parse_full_revpath(args.target, project)
    if target_project:
        scope_add(target_project)

    model = project.make_model(args.model_name)
    inference = dataset.run_model(model)
    inference.save(dst_dir)

    log.info("Inference results have been saved to '%s'" % dst_dir)

    return 0


def build_info_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help=f"{deprecated} Print model information in this project",
        description=f"{deprecated } Print model information in this project",
        formatter_class=MultilineFormatter,
    )

    parser.add_argument("-n", "--name", help="Model name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show details")
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.set_defaults(command=info_command)

    return parser


def get_info_sensitive_args():
    return {
        info_command: [
            "name",
            "project_dir",
        ],
    }


@scoped
def info_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    project = scope_add(load_project(args.project_dir))

    if args.name:
        print(project.models[args.name])
    else:
        for name, conf in project.models.items():
            print(name)
            if args.verbose:
                print(dict(conf))


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    subparsers = parser.add_subparsers()
    add_subparser(subparsers, "add", build_add_parser)
    add_subparser(subparsers, "remove", build_remove_parser)
    add_subparser(subparsers, "run", build_run_parser)
    add_subparser(subparsers, "info", build_info_parser)

    return parser


def get_sensitive_args():
    return {
        **get_add_sensitive_args(),
        **get_remove_sensitive_args(),
        **get_run_sensitive_args(),
        **get_info_sensitive_args(),
    }
