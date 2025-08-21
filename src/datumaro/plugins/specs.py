# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from pathlib import Path

from datumaro.components.lazy_plugin import get_extra_deps, get_lazy_plugin
from datumaro.util import parse_json_file

_SOURCE_PATH = Path(__file__).resolve()
_SOURCE_DIR = _SOURCE_PATH.parent
_SPECS_JSON_PATH = _SOURCE_DIR / "specs.json"


def get_lazy_plugins():
    return [
        plugin
        for plugin in [
            get_lazy_plugin(
                spec["import_path"],
                spec["plugin_name"],
                spec["plugin_type"],
                spec.get("extra_deps", []),
                spec.get("metadata", {}),
            )
            for spec in parse_json_file(str(_SPECS_JSON_PATH))
        ]
        if plugin is not None
    ]


if __name__ == "__main__":
    from datumaro.components.environment import Environment
    from datumaro.util import dump_json_file

    env = Environment(use_lazy_import=False)
    plugin_specs = []

    def _enroll_to_plugin_specs(plugins, plugin_type):
        global plugin_specs

        for _, plugin in plugins.items():
            mod = plugin.__module__
            class_name = plugin.__name__
            plugin_name = plugin.NAME
            extra_deps = get_extra_deps(plugin)
            plugin_spec = {
                "import_path": f"{mod}.{class_name}",
                "plugin_name": plugin_name,
                "plugin_type": plugin_type,
            }
            if extra_deps:
                plugin_spec["extra_deps"] = extra_deps
            # Setting metadata for importers
            if hasattr(plugin, "get_file_extensions"):
                plugin_spec["metadata"] = {"file_extensions": sorted(plugin.get_file_extensions())}
            plugin_specs.append(plugin_spec)

    _enroll_to_plugin_specs(env.extractors, "DatasetBase")
    _enroll_to_plugin_specs(env.importers, "Importer")
    _enroll_to_plugin_specs(env.exporters, "Exporter")
    _enroll_to_plugin_specs(env.generators, "DatasetGenerator")
    _enroll_to_plugin_specs(env.transforms, "Transform")
    _enroll_to_plugin_specs(env.validators, "Validator")

    dump_json_file(
        _SPECS_JSON_PATH,
        sorted(plugin_specs, key=lambda spec: spec["import_path"]),
        indent=True,
        append_newline=True,
    )
