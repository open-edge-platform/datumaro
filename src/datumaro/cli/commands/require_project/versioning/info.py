# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os.path as osp

from datumaro.util.scope import scope_add, scoped

from ....util import MultilineFormatter
from ....util.project import load_project

deprecated = "[DEPRECATED, will be removed in 1.12]"


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help=f"{deprecated} Get project info",
        description=f"""
        {deprecated} Outputs project info - information about plugins,
        sources, build tree, models and revisions.|n
        |n
        Examples:|n
        - Print project info for the current working tree:|n |n
        |s|s%(prog)s|n
        |n
        - Print project info for the previous revision:|n |n
        |s|s%(prog)s HEAD~1
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "revision", default="", nargs="?", help="Target revision (default: current working tree)"
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.set_defaults(command=info_command)

    return parser


def get_sensitive_args():
    return {
        info_command: ["project_dir", "revision"],
    }


@scoped
def info_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    project = scope_add(load_project(args.project_dir))
    rev = project.get_rev(args.revision)
    env = rev.env

    print("Project:")
    print("  location:", project._root_dir)
    print("Plugins:")
    print("  extractors:", ", ".join(sorted(set(env.extractors) | set(env.importers))))
    print("  exporters:", ", ".join(env.exporters))
    print("  launchers:", ", ".join(env.launchers))

    print("Models:")
    for model_name, model in project.models.items():
        print("  model '%s':" % model_name)
        print("    type:", model.launcher)

    print("Sources:")
    for source_name, source in rev.sources.items():
        print("  '%s':" % source_name)
        print("    format:", source.format)
        print("    url:", osp.abspath(source.url) if source.url else "")
        print(
            "    location:",
            osp.abspath(osp.join(project.source_data_dir(source_name), source.path)),
        )
        print("    options:", source.options)

        print("    stages:")
        for stage in rev.build_targets[source_name].stages:
            print("      '%s':" % stage.name)
            print("        type:", stage.type)
            print("        hash:", stage.hash)
            print("        cached:", project.is_obj_cached(stage.hash) if stage.hash else "n/a")
            if stage.kind:
                print("        kind:", stage.kind)
            if stage.params:
                print("        parameters:", stage.params)

    return 0
