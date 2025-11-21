# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

import polars as pl
import pytest

from datumaro.experimental.fields.annotations import BBoxField, Semantic


def test_field_dtype_validation():
    # pl.Float32 (type) should be converted to pl.Float32() (instance)
    field_type = BBoxField(semantic=Semantic.Default, dtype=pl.Float32)
    assert isinstance(field_type.dtype, pl.DataType)
    assert str(field_type.dtype) == str(pl.Float32())

    # pl.Float32() (instance) should be accepted
    field_instance = BBoxField(semantic=Semantic.Default, dtype=pl.Float32())
    assert isinstance(field_instance.dtype, pl.DataType)
    assert str(field_instance.dtype) == str(pl.Float32())

    # Invalid dtype should raise TypeError
    with pytest.raises(TypeError):
        BBoxField(semantic=Semantic.Default, dtype=float)
