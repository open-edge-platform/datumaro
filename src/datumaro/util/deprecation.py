# Copyright (C) 2019-2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

import warnings
from functools import wraps


def deprecated(deprecated_version: str, removed_version: str):
    """Class decorator that marks a class as deprecated."""

    def decorator(cls):
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            warnings.warn(
                f"The {cls.__name__} class will be deprecated in version {deprecated_version} "
                f"and will be removed in version {removed_version}.",
                DeprecationWarning,
                stacklevel=2,
            )
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator
