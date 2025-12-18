# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
Custom exceptions for the experimental datumaro module.
"""


class ArrayStructureError(TypeError):
    """Raised when a numpy array has an improper structure for conversion.

    This error indicates that the array's dtype=object structure is incorrect,
    typically because inner arrays also have object dtype instead of a numeric dtype.
    """

    _GUIDANCE = (
        "This typically means the numpy array structure is incorrect.\n"
        "For fields expecting arrays (e.g., polygon_field), ensure:\n"
        "  - The outer array has dtype=object and contains numpy arrays as elements\n"
        "  - Each inner array has a numeric dtype (e.g., float32), NOT dtype=object\n\n"
        "Example:\n"
        "  # Correct: inner arrays have float32 dtype\n"
        "  inner = np.array([[x1, y1], [x2, y2]], dtype=np.float32)\n"
        "  outer = np.array([inner], dtype=object)\n\n"
        "  # Incorrect: causes deeply nested object arrays\n"
        "  outer = np.array([[[x1, y1], [x2, y2]]], dtype=object)"
    )

    def __init__(self, message: str, include_guidance: bool = True):
        if include_guidance:
            message = f"{message}\n\n{self._GUIDANCE}"
        super().__init__(message)
