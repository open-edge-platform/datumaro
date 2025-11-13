# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser, Namespace

import pytest

from datumaro.cli.helpers.format import build_parser, format_command

# from datumaro.cli.__main__ import make_parser


class FormatTest:
    def test_build_parser(self):
        parser = build_parser(lambda help, **kwargs: ArgumentParser(**kwargs))
        assert isinstance(parser, ArgumentParser)

        args = parser.parse_args(["--list-import"])
        assert args.list_import

        args = parser.parse_args(["--list-export"])
        assert args.list_export

        args = parser.parse_args(["--delimiter", ","])
        assert args.delimiter == ","

    @pytest.mark.parametrize("list_import,list_export", [(True, False), (False, True), (False, False)])
    def test_format_command(self, list_import: bool, list_export: bool, capsys: pytest.CaptureFixture):
        format_command(Namespace(delimiter="\n", list_import=list_import, list_export=list_export))
        out, _ = capsys.readouterr()
        assert "coco" in out
