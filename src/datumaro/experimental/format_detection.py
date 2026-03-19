# Copyright (C) 2022-2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
Dataset format detection and auto-import functionality.

This module detects dataset formats and dispatches to the appropriate loader.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from datumaro.experimental.data_formats.base import DataFormat

# File names for native Datumaro format
METADATA_FILE = "metadata.json"
DATAFRAME_FILE = "data.parquet"


def is_datumaro_format(input_dir: Path) -> bool:
    """
    Detect if a directory contains a Datumaro-exported dataset.

    Datumaro format is identified by:
    - metadata.json file
    - data.parquet file

    Args:
        input_dir: Path to the directory to check

    Returns:
        True if the directory contains a Datumaro-format dataset
    """
    return (input_dir / METADATA_FILE).exists() and (input_dir / DATAFRAME_FILE).exists()


def is_legacy_datumaro_format(input_dir: Path) -> bool:
    """
    Detect if a directory contains a legacy Datumaro-format dataset.

    Legacy Datumaro format is identified by annotations/default.json file with 'dm_format_version' key

    Args:
        input_dir: Path to the directory to check

    Returns:
        True if the directory contains a legacy Datumaro-format dataset
    """
    default_json = input_dir / "annotations" / "default.json"
    return default_json.exists() and _is_legacy_datumaro_json(default_json)


def _is_legacy_datumaro_json(json_path: Path) -> bool:
    """Check if a JSON file has legacy Datumaro structure (dm_format_version key)."""
    try:
        with open(json_path) as f:
            data = json.load(f)
            if isinstance(data, dict):
                return "dm_format_version" in data
    except (json.JSONDecodeError, OSError):
        pass
    return False


def detect_dataset_format(input_dir: Path) -> DataFormat:
    """
    Detect the format of a dataset directory.

    Checks formats in order of specificity:
    1. Datumaro (most specific - requires both metadata.json and data.parquet)
    2. Legacy Datumaro
    3. COCO (JSON files with images/annotations structure)
    4. YOLO (data.yaml, obj.names, or directory structure)
    5. VOC (JPEGImages/, Annotations/, ImageSets/ directories)

    Args:
        input_dir: Path to the directory to check

    Returns:
        DataFormat enum value: DATUMARO, DATUMARO_LEGACY, COCO, YOLO, VOC, or UNKNOWN
    """
    if is_datumaro_format(input_dir):
        return DataFormat.DATUMARO
    if is_legacy_datumaro_format(input_dir):
        return DataFormat.DATUMARO_LEGACY

    # Inline imports to avoid circular dependency:
    from datumaro.experimental.data_formats.coco.io import is_coco_format
    from datumaro.experimental.data_formats.voc.io import is_voc_format
    from datumaro.experimental.data_formats.yolo.io import is_yolo_format

    if is_coco_format(input_dir):
        return DataFormat.COCO
    if is_yolo_format(input_dir):
        return DataFormat.YOLO
    if is_voc_format(input_dir):
        return DataFormat.VOC
    return DataFormat.UNKNOWN


def find_dataset_root(input_dir: Path) -> Path:
    """
    Find the actual dataset root, descending at most one level into a single child directory.

    Many third-party dataset zips extract with a single top-level folder
    (e.g., dataset.zip -> dataset/annotations/...). This function checks if
    the format is detectable at the input level, and if not, descends into
    a single child directory (if one exists) to check there.

    Args:
        input_dir: The directory to start searching from

    Returns:
        The detected dataset root directory (may be the input directory itself
        if format is detectable there, or a single nested child directory)
    """
    if detect_dataset_format(input_dir) != DataFormat.UNKNOWN:
        return input_dir

    children = list(input_dir.iterdir())
    child_dirs = [c for c in children if c.is_dir()]
    child_files = [c for c in children if c.is_file()]

    if len(child_dirs) == 1 and not child_files:
        return child_dirs[0]

    return input_dir
