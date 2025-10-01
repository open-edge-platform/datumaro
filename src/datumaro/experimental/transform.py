# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
from abc import abstractmethod
from typing import Sequence

import polars as pl

from .schema import Schema


class Transform:
    def __init__(self, schema: Schema):
        self._schema = schema

    @abstractmethod
    def apply(self, fields: Sequence[str]) -> pl.DataFrame:
        raise NotImplementedError

    @property
    def schema(self) -> Schema:
        return self._schema

    def get_batch_attributes(self) -> set[str]:
        attributes = set(self.schema.attributes.keys())
        attributes -= self.get_lazy_attributes()
        return attributes

    @abstractmethod
    def get_lazy_attributes(self) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def slice(self, offset: int, length: int | None = None) -> "Transform":
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self, df: pl.DataFrame, schema: Schema):
        super().__init__(schema)
        self._df = df

    def apply(self, _: Sequence[str]) -> pl.DataFrame:
        return self._df

    def get_lazy_attributes(self) -> set[str]:
        return set()

    def slice(self, offset: int, length: int | None = None) -> "Transform":
        return IdentityTransform(self._df.slice(offset, length), self.schema)

    def __len__(self):
        return len(self._df)
