# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import inspect

import attrs

from .points_util import normalize_points


def not_empty(inst, attribute, x):
    assert len(x) != 0, x


def has_length(n):
    def _validator(inst, attribute, x):
        assert len(x) != 0, x

    return _validator


def default_if_none(conv):
    def _validator(inst, attribute, value):
        default = attribute.default
        if value is None:
            if callable(default):
                value = default()
            elif isinstance(default, attrs.Factory):
                value = default.factory()
            else:
                value = default
        else:
            dst_type = None
            if attribute.type and inspect.isclass(attribute.type):
                dst_type = attribute.type
            elif conv and inspect.isclass(conv):
                dst_type = conv

            if not dst_type or not isinstance(value, dst_type):
                value = conv(value)
        setattr(inst, attribute.name, value)

    return _validator


def ensure_cls(c):
    def _converter(arg):
        if isinstance(arg, c):
            return arg
        else:
            return c(**arg)

    return _converter


def validate_points_positions(inst, attribute, positions) -> list[tuple[float, float]]:
    """
    Validate a list of point positions, ensuring that each position is a tuple of two floats.

    To be used as an attrs validator in PointsCategories class.
    """
    if positions is None or positions == []:
        value = []
    else:
        # convert to a list of tuples
        try:
            positions = list(map(tuple, positions))
        except TypeError:
            raise ValueError(
                f"Cannot convert {attribute.name} to list of tuples. Check your input data."
            )
        # validate the positions
        for position in positions:
            if not len(position) == 2:
                raise ValueError(f"Each tuple in {attribute.name} must have exactly 2 elements")
            if not all(isinstance(coord, (int, float)) for coord in position):
                raise ValueError(
                    f"Each element in a tuple in {attribute.name} must be an int or float"
                )
        # normalize the positions
        value = normalize_points(positions)

    setattr(inst, attribute.name, value)
