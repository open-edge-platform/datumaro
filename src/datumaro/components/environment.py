# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import importlib
import logging as log
import os.path as osp
from functools import partial
from inspect import getmodule, isclass
from typing import Callable, List, Optional, Sequence, Set

from datumaro.components.cli_plugin import plugin_types
from datumaro.components.format_detection import (
    DetectedFormat,
    FormatDetectionConfidence,
    RejectionReason,
    detect_dataset_format,
)
from datumaro.components.registry import (
    DatasetBaseRegistry,
    ExporterRegistry,
    GeneratorRegistry,
    ImporterRegistry,
    PluginRegistry,
    TransformRegistry,
    ValidatorRegistry,
)
from datumaro.util.os_util import get_all_file_extensions, import_foreign_module, split_path


class Environment:
    _builtin_plugins = None

    def __init__(self, use_lazy_import: bool = True):
        self._extractors = DatasetBaseRegistry()
        self._importers = ImporterRegistry()
        self._exporters = ExporterRegistry()
        self._generators = GeneratorRegistry()
        self._transforms = TransformRegistry()
        self._validators = ValidatorRegistry()
        self._builtins_initialized = False
        self._use_lazy_import = use_lazy_import

    def _get_plugin_registry(self, name):
        if not self._builtins_initialized:
            self._builtins_initialized = True
            self._register_builtin_plugins()
        return getattr(self, name)

    @property
    def extractors(self) -> DatasetBaseRegistry:
        return self._get_plugin_registry("_extractors")

    @property
    def importers(self) -> ImporterRegistry:
        return self._get_plugin_registry("_importers")

    @property
    def exporters(self) -> ExporterRegistry:
        return self._get_plugin_registry("_exporters")

    @property
    def generators(self) -> GeneratorRegistry:
        return self._get_plugin_registry("_generators")

    @property
    def transforms(self) -> TransformRegistry:
        return self._get_plugin_registry("_transforms")

    @property
    def validators(self) -> ValidatorRegistry:
        return self._get_plugin_registry("_validators")

    @staticmethod
    def _find_plugins(plugins_dir):
        plugins = []

        for pattern in ("*.py", "*/*.py", "*/*/*.py"):
            for path in glob.glob(osp.join(glob.escape(plugins_dir), pattern)):
                if not osp.isfile(path):
                    continue

                path_rel = osp.relpath(path, plugins_dir)
                name_parts = split_path(osp.splitext(path_rel)[0])

                # a module with a dot in the name won't load correctly
                if any("." in part for part in name_parts):
                    log.warning(
                        "Python file '%s' in directory '%s' can't be imported "
                        "due to a dot in the name; skipping.",
                        path_rel,
                        plugins_dir,
                    )
                    continue
                plugins.append(".".join(name_parts))

        return plugins

    @classmethod
    def _get_plugin_exports(cls, module, types):
        exports = []
        if hasattr(module, "exports"):
            exports = module.exports
        else:
            for symbol in dir(module):
                if symbol.startswith("_"):
                    continue
                exports.append(getattr(module, symbol))

        exports = [
            s
            for s in exports
            if isclass(s)
            and issubclass(s, types)
            and s not in types
            and (
                getmodule(s)
                is None  # Custom plugin (in the Datumaro project) can be a single file and have no module
                or not getmodule(s).__package__.startswith("datumaro.components")
            )
        ]

        return exports

    @classmethod
    def _load_plugins(cls, module_names, *, importer, types=None):
        types = tuple(types or plugin_types())

        all_exports = []
        for module_name in module_names:
            try:
                module = importer(module_name)
                exports = cls._get_plugin_exports(module, types)
            except Exception as e:
                module_search_error = ModuleNotFoundError

                message = ["Failed to import module '%s': %s", module_name, e]
                if isinstance(e, module_search_error):
                    log.debug(*message)
                else:
                    log.warning(*message)
                continue

            log.debug(
                "Imported the following symbols from %s: %s"
                % (module_name, ", ".join(s.__name__ for s in exports))
            )
            all_exports.extend(exports)

        return all_exports

    @classmethod
    def _load_builtin_plugins(cls):
        """Load builtin plugins which will be imported lazily using plugin spec files"""
        if cls._builtin_plugins is None:
            from datumaro.plugins.specs import get_lazy_plugins

            cls._builtin_plugins = get_lazy_plugins()
        return cls._builtin_plugins

    @classmethod
    def _load_builtin_plugins_from_importlib(cls):
        """Load builtin plugins from importlib, not lazy import from plugin spec files"""
        if cls._builtin_plugins is None:
            import datumaro.plugins

            plugins_dir = osp.dirname(datumaro.plugins.__file__)
            module_names = [
                datumaro.plugins.__name__ + "." + name for name in cls._find_plugins(plugins_dir)
            ]
            cls._builtin_plugins = cls._load_plugins(module_names, importer=importlib.import_module)
        return cls._builtin_plugins

    def load_plugins(self, plugins_dir):
        module_names = self._find_plugins(plugins_dir)
        plugins = self._load_plugins(
            module_names, importer=partial(import_foreign_module, path=plugins_dir)
        )

        self.register_plugins(plugins)

    def _register_builtin_plugins(self):
        self.register_plugins(
            self._load_builtin_plugins()
            if self._use_lazy_import
            else self._load_builtin_plugins_from_importlib()
        )

    def register_plugins(self, plugins):
        self.extractors.batch_register(plugins)
        self.importers.batch_register(plugins)
        self.exporters.batch_register(plugins)
        self.generators.batch_register(plugins)
        self.transforms.batch_register(plugins)
        self.validators.batch_register(plugins)

    def make_extractor(self, name, *args, **kwargs):
        return self.extractors.get(name)(*args, **kwargs)

    def make_importer(self, name, *args, **kwargs):
        return self.importers.get(name)(*args, **kwargs)

    def make_exporter(self, name, *args, **kwargs):
        result = self.exporters.get(name)
        if isclass(result):
            result = result.convert
        return partial(result, *args, **kwargs)

    def make_transform(self, name, *args, **kwargs):
        return partial(self.transforms.get(name), *args, **kwargs)

    def is_format_known(self, name):
        return name in self.importers or name in self.extractors

    def detect_dataset(
        self,
        path: str,
        depth: int = 1,
        rejection_callback: Optional[Callable[[str, RejectionReason, str], None]] = None,
    ) -> List[str]:
        ignore_dirs = {"__MSOSX", "__MACOSX"}
        all_matched_formats: Set[DetectedFormat] = set()

        extensions = get_all_file_extensions(path, ignore_dirs) or [""]

        importers = {
            (name, importer.get_plugin_cls() if self._use_lazy_import else importer)
            for extension in extensions
            for name, importer in self.importers.extension_groups.get(extension, [])
        }
        for _ in range(depth + 1):
            detected_formats = detect_dataset_format(
                ((format_name, importer.detect) for format_name, importer in importers),
                path,
                rejection_callback=rejection_callback,
            )

            if detected_formats:
                all_matched_formats |= set(detected_formats)

            paths = glob.glob(osp.join(path, "*"))
            path = "" if len(paths) != 1 else paths[0]
            if not osp.isdir(path) or osp.basename(path) in ignore_dirs:
                break

        max_conf = (
            max(all_matched_formats).confidence
            if len(all_matched_formats) > 0
            else FormatDetectionConfidence.NONE
        )

        return sorted(
            [str(format) for format in all_matched_formats if format.confidence == max_conf]
        )

    def __reduce__(self):
        return (self.__class__, ())

    @classmethod
    def merge(cls, envs: Sequence["Environment"]) -> "Environment":
        if all([env == DEFAULT_ENVIRONMENT for env in envs]):
            return DEFAULT_ENVIRONMENT

        merged = Environment()

        def _register(registry: PluginRegistry):
            merged.register_plugins(list(registry._items.values()))

        for env in envs:
            _register(env.extractors)
            _register(env.importers)
            _register(env.exporters)
            _register(env.generators)
            _register(env.transforms)
            _register(env.validators)

        return merged

    @classmethod
    def release_builtin_plugins(cls):
        cls._builtin_plugins = None


DEFAULT_ENVIRONMENT = Environment()
