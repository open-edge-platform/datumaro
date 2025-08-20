# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from abc import ABC, abstractclassmethod
from importlib import import_module
from importlib.util import find_spec
from typing import Dict, List, Optional, Sequence, Type, Union

from datumaro.components.dataset_base import DatasetBase
from datumaro.components.errors import DatumaroError
from datumaro.components.exporter import Exporter
from datumaro.components.generator import DatasetGenerator
from datumaro.components.importer import Importer
from datumaro.components.transformer import Transform
from datumaro.components.validator import Validator

# Transform should be in front of DatasetBase,
# since Transform inherits DatasetBase.
_PLUGIN_TYPES = Union[
    Transform,
    Exporter,
    DatasetGenerator,
    Importer,
    Validator,
    DatasetBase,
]
PLUGIN_TYPES = Type[_PLUGIN_TYPES]
STR_TO_PLUGIN_TYPES = {
    t.__name__: t
    for t in [
        Transform,
        Exporter,
        DatasetGenerator,
        Importer,
        Validator,
        DatasetBase,
    ]
}
_EXTRA_DEPS_ATTR_NAME = "__extra_deps__"


class LazyPlugin(ABC):
    NAME: str

    @abstractclassmethod
    def get_plugin_cls(cls) -> PLUGIN_TYPES:
        pass


def get_lazy_plugin(
    import_path: str,
    plugin_name: str,
    plugin_type: str,
    extra_deps: List[str] = [],
    metadata: Dict = {},
) -> Optional[LazyPlugin]:
    for extra_dep in extra_deps:
        spec = find_spec(extra_dep)
        if spec is None:
            log.debug(f"Cannot import extra dep={extra_dep} for plugin_name={plugin_name}.")
            return None

    plugin_type_cls = STR_TO_PLUGIN_TYPES[plugin_type]

    class LazyPluginImpl(LazyPlugin, plugin_type_cls):
        NAME = plugin_name
        METADATA = metadata

        @classmethod
        def get_plugin_cls(cls) -> PLUGIN_TYPES:
            splits = import_path.split(".")
            module_name = ".".join(splits[:-1])
            class_name = splits[-1]
            module = import_module(module_name)
            plugin_cls = getattr(module, class_name)
            return plugin_cls

    return LazyPluginImpl


def extra_deps(*extra_dep_names: Sequence[str]):
    """Decorator to assign extra deps for the plugin class.

    There exist some plugins that cannot be executable with the default installation setup.
    For example, Tensorflow Detection API plugin needs `tensorflow` as an extra dependency.
    In this case, you have to add this decorator to that plugin class definition as follows.

    @extra_deps("tensorflow")
    class TfDetectionApiBase(SubsetBase):
    ...
    """

    def inner(plugin_cls: object):
        if hasattr(plugin_cls, _EXTRA_DEPS_ATTR_NAME):
            raise DatumaroError(f"It already has {_EXTRA_DEPS_ATTR_NAME} attribute!")

        setattr(plugin_cls, _EXTRA_DEPS_ATTR_NAME, list(extra_dep_names))
        return plugin_cls

    return inner


def get_extra_deps(plugin_cls: object) -> List[str]:
    """Get extra deps of the given plugin class. If not exists, return an empty list."""
    return getattr(plugin_cls, _EXTRA_DEPS_ATTR_NAME, [])
