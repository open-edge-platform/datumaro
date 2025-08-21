# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest

import datumaro.components.lazy_plugin
from datumaro.components.environment import DEFAULT_ENVIRONMENT, Environment, PluginRegistry
from datumaro.components.exporter import Exporter

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
        assert (
            lazy_plugin_names == no_lazy_plugin_names
        ), f"misregistered_names={misregistered_names}, unregistered_names={unregistered_names}"

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
