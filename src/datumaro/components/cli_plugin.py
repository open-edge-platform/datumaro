# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
from typing import List, Type

from datumaro.cli.util import MultilineFormatter
from datumaro.util import to_snake_case

_plugin_types = None


def plugin_types() -> List[Type["CliPlugin"]]:
    global _plugin_types
    if _plugin_types is None:
        from datumaro.components.dataset_base import DatasetBase
        from datumaro.components.exporter import Exporter
        from datumaro.components.importer import Importer
        from datumaro.components.transformer import Transform
        from datumaro.components.validator import Validator

        _plugin_types = [DatasetBase, Transform, Importer, Exporter, Validator]

    return _plugin_types


def remove_plugin_type(s):
    for t in {"transform", "base", "exporter", "importer", "validator"}:
        s = s.replace("_" + t, "")
    return s


class _PluginNameDescriptor:
    def __get__(self, obj, objtype=None):
        if not objtype:
            objtype = type(obj)
        return remove_plugin_type(to_snake_case(objtype.__name__))


class CliPlugin:
    NAME = _PluginNameDescriptor()

    @staticmethod
    def _get_doc(cls):
        doc = getattr(cls, "__doc__", "")
        if doc:
            if any(getattr(t, "__doc__", "") == doc for t in plugin_types()):
                doc = ""
        return doc

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        args = {
            "prog": cls.NAME,
            "description": cls._get_doc(cls),
            "formatter_class": MultilineFormatter,
        }
        args.update(kwargs)

        return argparse.ArgumentParser(**args)

    @classmethod
    def parse_cmdline(cls, args=None):
        if args and args[0] == "--":
            args = args[1:]
        parser = cls.build_cmdline_parser()
        args = parser.parse_args(args)
        args = vars(args)

        log.debug(
            "Parsed parameters: \n\t%s", "\n\t".join("%s: %s" % (k, v) for k, v in args.items())
        )

        return args
