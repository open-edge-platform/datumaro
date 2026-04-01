# Copyright (C) 2022-2026 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Constants for YOLO dataset format.
"""

from datumaro.experimental.fields import Subset

# Mapping from subset enum to YOLO Ultralytics directory names
SUBSET_TO_DIR_NAME = {
    Subset.TRAINING: "train",
    Subset.VALIDATION: "val",
    Subset.TESTING: "test",
    Subset.UNASSIGNED: "unassigned",
}

# Mapping from YOLO directory names to subset enum
DIR_NAME_TO_SUBSET = {v: k for k, v in SUBSET_TO_DIR_NAME.items()}

# Mapping from subset enum to traditional YOLO directory names
TRADITIONAL_SUBSET_DIR_NAMES = {
    Subset.TRAINING: "obj_train_data",
    Subset.VALIDATION: "obj_valid_data",
    Subset.TESTING: "obj_test_data",
    Subset.UNASSIGNED: "obj_data",
}

# Mapping from subset enum to traditional YOLO config keys
TRADITIONAL_SUBSET_CONFIG_KEYS = {
    Subset.TRAINING: "train",
    Subset.VALIDATION: "valid",
    Subset.TESTING: "test",
}

# Mapping from traditional YOLO directory names to subset enum
TRADITIONAL_DIR_NAME_TO_SUBSET = {
    "obj_train_data": Subset.TRAINING,
    "train": Subset.TRAINING,
    "obj_valid_data": Subset.VALIDATION,
    "valid": Subset.VALIDATION,
    "val": Subset.VALIDATION,
    "obj_test_data": Subset.TESTING,
    "test": Subset.TESTING,
    "obj_data": Subset.UNASSIGNED,
}
