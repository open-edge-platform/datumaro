# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np


def normalize_points(positions: list[float]) -> list[float]:
    """
    Normalize a set of keypoints to fit within a unit bounding box [0, 1] x [0, 1],
    maintaining the aspect ratio.

    Parameters:
        keypoints (list[float]): A list of point positions in the format [x1, y1, x2, y2, ..., xn, yn].

    Returns:
        list[float]: Normalized keypoints within the unit bounding box, preserving aspect ratio.
    """
    # Convert keypoints to a NumPy array for easier manipulation
    positions = np.array(positions, dtype=float).reshape(-1, 2)

    # Find the minimum and maximum values for x and y
    min = positions.min(axis=0)
    max = positions.max(axis=0)

    # Compute the width and height of the bounding box
    size = max - min

    # Handle edge case where all keypoints are the same (zero width or height)
    size[np.where(size == 0)] = 1e-6

    # Determine the scaling factor to maintain aspect ratio
    scale = size.max()

    # Normalize the keypoints to fit within [0, 1] x [0, 1], preserving aspect ratio
    normalized_positions = (positions - min) / scale

    return list(map(float, normalized_positions.flatten()))
