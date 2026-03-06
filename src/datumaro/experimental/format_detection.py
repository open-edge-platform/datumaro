# Copyright (C) 2022-2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
Dataset format detection and auto-import functionality.

This module provides automatic detection of dataset formats (Datumaro, COCO, YOLO)
and dispatches to the appropriate loader.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from datumaro.experimental.data_formats.base import DataFormat

if TYPE_CHECKING:
    from pathlib import Path

    from datumaro.experimental.dataset import Dataset

# File names for native Datumaro format
METADATA_FILE = "metadata.json"
DATAFRAME_FILE = "data.parquet"


def is_coco_format(input_dir: Path) -> bool:
    """
    Detect if a directory contains a COCO-format dataset.

    COCO format is identified by:
    - An 'annotations' subdirectory containing JSON files with COCO structure
    - OR JSON files in the root containing 'images' and 'annotations' keys

    Args:
        input_dir: Path to the directory to check

    Returns:
        True if the directory contains a COCO-format dataset
    """
    # Check for annotations directory with COCO JSON files
    annotations_dir = input_dir / "annotations"
    if annotations_dir.is_dir():
        for json_file in annotations_dir.glob("*.json"):
            if is_coco_json_file(json_file):
                return True

    # Check for COCO JSON files in the root directory
    for json_file in input_dir.glob("*.json"):
        # Skip metadata.json which is Datumaro's format
        if json_file.name == METADATA_FILE:
            continue
        if is_coco_json_file(json_file):
            return True

    return False


def is_coco_json_file(json_path: Path) -> bool:
    """
    Check if a JSON file has COCO structure.

    Args:
        json_path: Path to the JSON file to check

    Returns:
        True if the file contains COCO-style data (images and annotations/categories keys)
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
            if isinstance(data, dict):
                # COCO format has 'images' array and typically 'annotations' or 'categories'
                return "images" in data and ("annotations" in data or "categories" in data)
    except (json.JSONDecodeError, OSError):
        pass
    return False


def is_yolo_format(input_dir: Path) -> bool:
    """
    Detect if a directory contains a YOLO-format dataset.

    YOLO format is identified by:
    - data.yaml file (Ultralytics format)
    - OR obj.names/obj.data files (traditional format)
    - OR images/ and labels/ directories (Ultralytics structure)

    Args:
        input_dir: Path to the directory to check

    Returns:
        True if the directory contains a YOLO-format dataset
    """
    # Check for Ultralytics format (data.yaml)
    if (input_dir / "data.yaml").exists():
        return True

    # Check for traditional YOLO format (obj.names or obj.data)
    if (input_dir / "obj.names").exists() or (input_dir / "obj.data").exists():
        return True

    # Check for Ultralytics-style directory structure
    return (input_dir / "images").is_dir() and (input_dir / "labels").is_dir()


def is_voc_format(input_dir: Path) -> bool:
    """
    Detect if a directory contains a Pascal VOC-format dataset.

    Pascal VOC format is identified by:
    - JPEGImages/ directory AND (Annotations/ or ImageSets/ directory)
    - OR labelmap.txt file with VOC structure

    Args:
        input_dir: Path to the directory to check

    Returns:
        True if the directory contains a VOC-format dataset
    """
    # Check for standard VOC directory structure
    jpeg_images_dir = input_dir / "JPEGImages"
    annotations_dir = input_dir / "Annotations"
    imagesets_dir = input_dir / "ImageSets"

    if jpeg_images_dir.is_dir() and (annotations_dir.is_dir() or imagesets_dir.is_dir()):
        return True

    # Check for labelmap.txt with VOC structure
    labelmap_file = input_dir / "labelmap.txt"
    return labelmap_file.exists() and _is_voc_labelmap(labelmap_file)


def _is_voc_labelmap(labelmap_path: Path) -> bool:
    """Check if a file has VOC labelmap structure (name:color:parts:actions)."""
    try:
        with open(labelmap_path) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                # VOC labelmap has format: name:color:parts:actions
                parts = line.split(":")
                if len(parts) == 4:
                    return True
    except OSError:
        pass
    return False


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
                # Legacy Datumaro format has 'dm_format_version' key
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
    # If format is detectable at current level, return it
    if detect_dataset_format(input_dir) != DataFormat.UNKNOWN:
        return input_dir

    # Check if there's exactly one child directory (and no files)
    children = list(input_dir.iterdir())
    child_dirs = [c for c in children if c.is_dir()]
    child_files = [c for c in children if c.is_file()]

    # Only descend if there's exactly one directory and no files
    if len(child_dirs) == 1 and not child_files:
        return child_dirs[0]

    # Otherwise return the original directory
    return input_dir


def _find_coco_annotations(input_dir: Path) -> list[str]:
    """Find all COCO annotation files in a directory."""
    annotation_files: list[str] = []

    # Check annotations/ subdirectory first
    annotations_dir = input_dir / "annotations"
    if annotations_dir.is_dir():
        for json_file in annotations_dir.glob("*.json"):
            if is_coco_json_file(json_file):
                annotation_files.append(str(json_file))
    else:
        # Fall back to root directory
        for json_file in input_dir.glob("*.json"):
            if json_file.name != METADATA_FILE and is_coco_json_file(json_file):
                annotation_files.append(str(json_file))

    return annotation_files


def _find_coco_images_dir(input_dir: Path) -> str:
    """Find the images directory for a COCO dataset."""
    # Common COCO image directory patterns
    candidates = ["train2017", "val2017", "images", "train", "val"]
    for candidate in candidates:
        candidate_dir = input_dir / candidate
        if candidate_dir.is_dir():
            return str(candidate_dir)

    # Default to root directory
    return str(input_dir)


def import_coco_dataset(input_dir: Path) -> Dataset:
    """
    Import a COCO-format dataset from a directory.

    Automatically discovers annotation files and image directories.

    Args:
        input_dir: Path to the COCO dataset directory

    Returns:
        Dataset containing CocoSample instances

    Raises:
        FileNotFoundError: If no COCO annotation files are found
    """
    annotation_files = _find_coco_annotations(input_dir)
    if not annotation_files:
        raise FileNotFoundError(f"No COCO annotation files found in {input_dir}")

    images_dir = _find_coco_images_dir(input_dir)

    # Import inline to avoid circular imports
    from datumaro.experimental.data_formats.coco.io import load_coco_dataset

    annotations_path = annotation_files if len(annotation_files) > 1 else annotation_files[0]
    return load_coco_dataset(images_dir_path=images_dir, annotations_path=annotations_path)


def import_yolo_dataset(input_dir: Path) -> Dataset:
    """
    Import a YOLO-format dataset from a directory.

    Supports both Ultralytics and traditional YOLO formats.

    Args:
        input_dir: Path to the YOLO dataset directory

    Returns:
        Dataset containing YoloSample instances
    """
    # Import inline to avoid circular imports
    from datumaro.experimental.data_formats.yolo.io import load_yolo_dataset

    return load_yolo_dataset(root_dir=str(input_dir))


def import_voc_dataset(input_dir: Path) -> Dataset:
    """
    Import a Pascal VOC-format dataset from a directory.

    Supports standard VOC directory structure with:
    - JPEGImages/ for images
    - Annotations/ for XML annotations
    - ImageSets/Main/ for train/val/test splits

    Args:
        input_dir: Path to the VOC dataset directory

    Returns:
        Dataset containing VocSample instances
    """
    # Import inline to avoid circular imports
    from datumaro.experimental.data_formats.voc.io import load_voc_dataset

    return load_voc_dataset(root_dir=str(input_dir))
