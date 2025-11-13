# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Any


class IDatasetDownloader:
    @classmethod
    def download(
        cls,
        dataset_id: str,
        dst_dir: str,
        overwrite: bool,
        output_format: str,
        subset: str,
        extra_args: Any,
    ):
        raise NotImplementedError

    @classmethod
    def describe(cls, report_format, report_file=None) -> str:
        raise NotImplementedError

    @classmethod
    def get_command_description(cls, *args, **kwargs) -> str:
        raise NotImplementedError

    @classmethod
    def describe_command_description(cls):
        raise NotImplementedError
