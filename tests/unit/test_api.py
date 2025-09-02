# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

# ruff: noqa: F401


class ApiTest:
    def test_can_import_core(self):
        import datumaro as dm

        assert hasattr(dm, "Dataset")

    def test_can_reach_module_alias_symbols_from_base(self):
        import datumaro as dm

        assert hasattr(dm.errors, "DatumaroError")

    def test_can_import_from_module_aliases(self):
        from datumaro.components.dataset import Dataset
        from datumaro.errors import DatumaroError
