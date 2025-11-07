# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod

from datumaro.components.cli_plugin import CliPlugin


class DatasetGenerator(ABC, CliPlugin):
    """
    ImageGenerator generates synthetic data with provided shape.
    """

    @classmethod
    def generate(cls, output_dir, count, shape, **options):
        generator = cls(output_dir, count, shape, **options)
        return generator.generate_dataset()

    @abstractmethod
    def generate_dataset(self): ...
