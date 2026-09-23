# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

import datumaro.components.environment
import datumaro.components.lazy_plugin
from datumaro.components.environment import DEFAULT_ENVIRONMENT, Environment, PluginRegistry
from datumaro.components.exporter import Exporter
from datumaro.components.lazy_plugin import get_lazy_plugin
from datumaro.plugins import specs as plugin_specs
from datumaro.util import parse_json_file

real_find_spec = datumaro.components.lazy_plugin.find_spec


class EnvironmentTest:
    @pytest.fixture
    def fxt_lazy_import(self):
        Environment.release_builtin_plugins()
        env = Environment(use_lazy_import=True)
        _ = env.importers
        yield env
        Environment.release_builtin_plugins()

    @pytest.fixture
    def fxt_no_lazy_import(self):
        Environment.release_builtin_plugins()
        env = Environment(use_lazy_import=False)
        _ = env.importers
        yield env
        Environment.release_builtin_plugins()

    def _test_equivalance(self, lazy_registry: PluginRegistry, no_lazy_registry: PluginRegistry):
        lazy_plugin_names = set(sorted(lazy_registry))
        no_lazy_plugin_names = set(sorted(no_lazy_registry))

        misregistered_names = lazy_plugin_names.difference(no_lazy_plugin_names)
        unregistered_names = no_lazy_plugin_names.difference(lazy_plugin_names)
        assert lazy_plugin_names == no_lazy_plugin_names, (
            f"misregistered_names={misregistered_names}, unregistered_names={unregistered_names}"
        )

    def test_equivalance(self, fxt_lazy_import: Environment, fxt_no_lazy_import: Environment):
        self._test_equivalance(fxt_lazy_import.extractors, fxt_no_lazy_import.extractors)
        self._test_equivalance(fxt_lazy_import.importers, fxt_no_lazy_import.importers)
        self._test_equivalance(fxt_lazy_import.exporters, fxt_no_lazy_import.exporters)
        self._test_equivalance(fxt_lazy_import.generators, fxt_no_lazy_import.generators)
        self._test_equivalance(fxt_lazy_import.transforms, fxt_no_lazy_import.transforms)
        self._test_equivalance(fxt_lazy_import.validators, fxt_no_lazy_import.validators)

    @pytest.fixture
    def fxt_tf_failure_env(self, monkeypatch):
        def _patch(name, package=None):
            if name == "tensorflow":
                return None
            return real_find_spec(name, package)

        monkeypatch.setattr(datumaro.components.lazy_plugin, "find_spec", _patch)

        Environment.release_builtin_plugins()
        env = Environment(use_lazy_import=True)
        _ = env.importers
        yield env
        Environment.release_builtin_plugins()

    def test_extra_deps_req(self, fxt_tf_failure_env):
        """Plugins affected by the import failure: `ac` and `tf_detection_api`."""

        env = fxt_tf_failure_env

        loaded_plugin_names = set(
            sorted(env.extractors)
            + sorted(env.importers)
            + sorted(env.exporters)
            + sorted(env.generators)
            + sorted(env.transforms)
            + sorted(env.validators)
        )

        assert "tf_detection_api" not in loaded_plugin_names

    def test_merge_default_env(self):
        merged_env = Environment.merge([DEFAULT_ENVIRONMENT, DEFAULT_ENVIRONMENT])
        assert merged_env is DEFAULT_ENVIRONMENT

    def test_merge_custom_env(self):
        class TestPlugin(Exporter):
            pass

        envs = [Environment(), Environment()]
        envs[0].exporters.register("test_plugin", TestPlugin)

        merged = Environment.merge(envs)
        assert "test_plugin" in merged.exporters


class TestLazyPluginImportSafety:
    """Safety checks for lazy plugin import path restrictions."""

    _SPECS_JSON_PATH = Path(plugin_specs.__file__).resolve().with_name("specs.json")

    def test_all_specs_import_paths_are_datumaro_internal(self):
        """Every entry in specs.json must start with 'datumaro.' to stay within the trusted package."""
        specs = parse_json_file(str(self._SPECS_JSON_PATH))
        for spec in specs:
            import_path = spec["import_path"]
            assert import_path.startswith("datumaro."), (
                f"specs.json contains a non-datumaro import_path: {import_path!r}. "
                "All plugin paths must be internal to prevent module injection."
            )

    def test_get_lazy_plugin_resolves_correctly(self):
        """get_lazy_plugin with a valid datumaro import_path resolves to the correct class."""
        from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter

        plugin_cls_factory = get_lazy_plugin(
            "datumaro.plugins.data_formats.datumaro.exporter.DatumaroExporter",
            "datumaro",
            "Exporter",
        )
        assert plugin_cls_factory is not None
        resolved = plugin_cls_factory.get_plugin_cls()
        assert resolved is DatumaroExporter
        assert issubclass(resolved, Exporter)

    def test_get_lazy_plugin_rejects_non_datumaro_import_path(self):
        """get_lazy_plugin must refuse import_paths outside the datumaro package."""
        # Untrusted packages, including ones an attacker could pick for RCE
        assert get_lazy_plugin("subprocess.getoutput", "bad", "Exporter") is None
        assert get_lazy_plugin("os.system", "bad", "Exporter") is None
        assert get_lazy_plugin("builtins.eval", "bad", "Exporter") is None
        assert get_lazy_plugin("importlib.import_module", "bad", "Exporter") is None
        assert get_lazy_plugin("datumaroevil.plugin.Class", "bad", "Exporter") is None


def test_load_plugins_only_imports_names_found_by_find_plugins(tmp_path, monkeypatch):
    """Environment.load_plugins must only ever import module names that were discovered by
    scanning `plugins_dir` on disk (via `_find_plugins`); it must not accept a module name from
    any other source (e.g. dataset content).

    This documents the trust boundary for this API: `plugins_dir` is an operator-supplied local
    path, so a malicious file placed there is treated as intentionally trusted, unlike the
    externally-triggerable arbitrary-import vulnerability fixed in schema.py.
    """
    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text(
        "from datumaro.components.exporter import Exporter\n\n\nclass MyExporter(Exporter):\n    pass\n"
    )

    expected_names = Environment._find_plugins(str(tmp_path))
    assert expected_names == ["my_plugin"]

    seen_names = []
    real_importer = datumaro.components.environment.import_foreign_module

    def _tracking_importer(name, path):
        seen_names.append(name)
        return real_importer(name, path)

    monkeypatch.setattr(datumaro.components.environment, "import_foreign_module", _tracking_importer)

    env = Environment()
    # Skip loading the (unrelated, heavy) builtin plugin set that self.exporters etc. would
    # otherwise trigger lazily on first access from within register_plugins().
    env._builtins_initialized = True
    env.load_plugins(str(tmp_path))

    assert seen_names == expected_names
    assert "my" in env.exporters
