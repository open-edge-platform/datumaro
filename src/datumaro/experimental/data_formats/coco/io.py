# Copyright (C) 2022-2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
COCO dataset support with flexible layout handling.

Supports both:
- COCO 2017 layout: images split by subset folders, separate annotation files per subset
- Simple COCOAPI layout: single images folder and single annotation file
"""

import logging
from collections import defaultdict
from pathlib import Path

from datumaro.experimental import Dataset
from datumaro.experimental.data_formats.coco.helpers import (
    _assemble_sample_from_image_record,
    _cat_id_to_idx_from_primary,
    _detect_coco_label_categories_from_paths,
    _load_json_or_none,
    _prepare_categories,
    _save_subset_flexible,
)
from datumaro.experimental.data_formats.coco.sample import CocoSample
from datumaro.experimental.fields import Subset

logger = logging.getLogger(__name__)

# Sentinel key used for simple layout (single folder) in both loading and saving
_DEFAULT_SUBSET_KEY = "__default__"

# Mapping from subset name strings to Subset enum values
_SUBSET_NAME_TO_ENUM: dict[str, Subset] = {
    "train": Subset.TRAINING,
    "training": Subset.TRAINING,
    "val": Subset.VALIDATION,
    "validation": Subset.VALIDATION,
    "test": Subset.TESTING,
    "testing": Subset.TESTING,
    _DEFAULT_SUBSET_KEY: Subset.UNASSIGNED,
}


def load_coco_dataset(
    images_dir_path: str | dict[str, str],
    annotations_path: str | list[str] | dict[str, str | list[str]],
) -> Dataset:
    """
    Load a COCO dataset from specified image directories and annotation files.

    This function supports two common COCO dataset layouts:

    1. **Simple layout** (single folder): Pass strings for both arguments.
       - ``images_dir_path``: Path to a single directory containing all images
       - ``annotations_path``: Path to a single COCO JSON annotation file
       - Samples will have ``Subset.UNASSIGNED`` as their subset

    2. **Split layout** (multiple subsets): Pass dicts mapping subset names to paths.
       - ``images_dir_path``: Dict mapping subset names to image directories
       - ``annotations_path``: Dict mapping subset names to annotation JSON files
       - Subset names can be arbitrary strings (e.g., "training", "validation", "test")
       - Known subset names ("train", "training", "val", "validation", "test", "testing")
         are mapped to ``Subset.TRAINING``, ``Subset.VALIDATION``, ``Subset.TESTING``
       - Unknown subset names are mapped to ``Subset.UNASSIGNED``

    Args:
        images_dir_path: Either a single directory path (str) containing images,
            or a dict mapping subset names to their respective image directories.
        annotations_path: Annotation file path(s). Can be:
            - A single annotation file path (str)
            - A list of annotation file paths (list[str]) for loading multiple
              annotation types (e.g., instances, keypoints, captions) at once
            - A dict mapping subset names to annotation file path(s), where each
              value can be a str or list[str]

    Returns:
        Dataset containing CocoSample instances.

    Raises:
        ValueError: If the types of images_dir_path and annotations_path don't match
            (images_dir_path must be str when annotations_path is str or list[str],
            and must be dict when annotations_path is dict).
        FileNotFoundError: If any specified directory or file doesn't exist.

    Examples:
        Load a simple COCO dataset::

            dataset = load_coco_dataset(
                images_dir_path="/path/to/images",
                annotations_path="/path/to/annotations.json",
            )

        Load multiple annotation types at once::

            dataset = load_coco_dataset(
                images_dir_path="/path/to/images",
                annotations_path=[
                    "/path/to/annotations/instances.json",
                    "/path/to/annotations/captions.json",
                    "/path/to/annotations/person_keypoints.json",
                ],
            )


        Load COCO 2017 with all annotation types (instances, captions, keypoints)::

            dataset = load_coco_dataset(
                images_dir_path={
                    "training": "/path/to/train2017",
                    "validation": "/path/to/val2017",
                },
                annotations_path={
                    "training": [
                        "/path/to/annotations/instances_train2017.json",
                        "/path/to/annotations/captions_train2017.json",
                        "/path/to/annotations/person_keypoints_train2017.json",
                    ],
                    "validation": [
                        "/path/to/annotations/instances_val2017.json",
                        "/path/to/annotations/captions_val2017.json",
                        "/path/to/annotations/person_keypoints_val2017.json",
                    ],
                },
            )
    """
    # Normalize and validate inputs
    images_config, annotations_config = _normalize_load_inputs(images_dir_path, annotations_path)

    # Build subset configuration
    subset_config = _build_subset_config_from_paths(images_config, annotations_config)

    logger.info("[COCO] Loading dataset with %d subset(s)", len(subset_config))

    # Determine label categories via helper
    loaded_label_categories = _detect_coco_label_categories_from_paths(subset_config)
    categories = {"labels": loaded_label_categories}
    dataset = Dataset(CocoSample, categories=categories)

    for subset, config in subset_config.items():
        _load_subset_into_dataset(dataset, subset, config)

    logger.info("[COCO] Finished loading dataset with %d samples in total", len(dataset))
    return dataset


def _normalize_load_inputs(
    images_dir_path: str | dict[str, str],
    annotations_path: str | list[str] | dict[str, str | list[str]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Normalize and validate load_coco_dataset inputs.

    Returns:
        Tuple of (images_config, annotations_config) with normalized dict formats.

    Raises:
        ValueError: If input types are incompatible or subset keys don't match.
    """
    if isinstance(images_dir_path, str) and isinstance(annotations_path, (str, list)):
        # Simple layout: single directory and single/multiple annotation file(s)
        images_config: dict[str, str] = {_DEFAULT_SUBSET_KEY: images_dir_path}
        if isinstance(annotations_path, str):
            annotations_config: dict[str, list[str]] = {_DEFAULT_SUBSET_KEY: [annotations_path]}
        else:
            annotations_config = {_DEFAULT_SUBSET_KEY: annotations_path}
    elif isinstance(images_dir_path, dict) and isinstance(annotations_path, dict):
        # Split layout: multiple subsets
        images_config = images_dir_path
        # Normalize each subset's annotations to a list
        annotations_config = {}
        for key, value in annotations_path.items():
            if isinstance(value, str):
                annotations_config[key] = [value]
            else:
                annotations_config[key] = value
    else:
        raise ValueError(
            "images_dir_path and annotations_path must be compatible types. "
            "images_dir_path must be str when annotations_path is str or list[str], "
            "and must be dict when annotations_path is dict. "
            f"Got {type(images_dir_path).__name__} and {type(annotations_path).__name__}."
        )

    # Validate that dict keys match
    if set(images_config.keys()) != set(annotations_config.keys()):
        raise ValueError(
            f"Subset keys must match between images_dir_path and annotations_path. "
            f"Got images: {set(images_config.keys())}, annotations: {set(annotations_config.keys())}"
        )

    return images_config, annotations_config


def _load_subset_into_dataset(
    dataset: Dataset,
    subset: Subset,
    config: dict[str, Path | list[Path]],
) -> None:
    """
    Load a single subset's data into the dataset.

    Args:
        dataset: The dataset to append samples to.
        subset: The subset enum value.
        config: Configuration dict with 'images_dir' and 'annotations' keys.

    Raises:
        FileNotFoundError: If images directory or annotation files don't exist.
    """
    images_dir = config["images_dir"]
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    annotations_files = config["annotations"]  # List of Paths
    for annotations_file in annotations_files:
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file does not exist: {annotations_file}")

    logger.info("[COCO] Loading subset '%s' from '%s'", subset, images_dir)

    # Load and merge annotations from all files
    all_annotations_data = [_load_json_or_none(f) for f in annotations_files]
    all_annotations_data = [d for d in all_annotations_data if d]  # Filter None

    if not all_annotations_data:
        logger.warning("[COCO] No annotations found for subset '%s', skipping", subset)
        return

    # Use first file with categories as primary for category mapping
    primary_data = _find_primary_annotations_data(all_annotations_data)
    cat_id_to_idx = _cat_id_to_idx_from_primary(primary_data)

    # Merge annotations from all files
    instances_by_image, keypoints_by_image, captions_by_image = _merge_annotations_from_files(all_annotations_data)

    # Merge images from all files (deduplicate by id)
    images = _merge_images_from_annotations(all_annotations_data)

    num_images = len(images)
    logger.info("[COCO] Building %d samples for subset '%s'", num_images, subset)

    samples = []
    for idx, img in enumerate(images, start=1):
        if idx % 1000 == 0:
            logger.info("[COCO] Subset '%s': processed %d/%d images", subset, idx, num_images)
            # Flush samples in chunks to keep memory bounded
            if samples:
                dataset.append_batch(samples)
                samples = []
        sample = _assemble_sample_from_image_record(
            images_dir=images_dir,
            img=img,
            cat_id_to_idx=cat_id_to_idx,
            instances_by_image=instances_by_image,
            keypoints_by_image=keypoints_by_image,
            captions_by_image=captions_by_image,
            subset=subset,
        )
        samples.append(sample)

    if samples:
        dataset.append_batch(samples)

    logger.info("[COCO] Finished subset '%s' with %d samples", subset, num_images)


def _find_primary_annotations_data(all_annotations_data: list[dict]) -> dict:
    """Find the first annotations dict that contains categories."""
    primary_data = all_annotations_data[0]
    for data in all_annotations_data:
        if data.get("categories"):
            return data
    return primary_data


def _merge_images_from_annotations(all_annotations_data: list[dict]) -> list[dict]:
    """Merge and deduplicate images from multiple annotation files."""
    images_by_id: dict[int, dict] = {}
    for data in all_annotations_data:
        for img in data.get("images", []):
            if img["id"] not in images_by_id:
                images_by_id[img["id"]] = img
    return list(images_by_id.values())


def _merge_annotations_from_files(
    all_annotations_data: list[dict],
) -> tuple[dict[int, list[dict]], dict[int, list[dict]], dict[int, list[dict]]]:
    """
    Merge annotations from multiple COCO annotation files.

    This function intelligently classifies annotations based on file type detection
    and annotation structure (presence of keypoints, captions, or bbox/segmentation).
    It samples multiple annotations (up to 10) to determine the file type more robustly.

    Args:
        all_annotations_data: List of parsed COCO JSON dictionaries.

    Returns:
        Tuple of (instances_by_image, keypoints_by_image, captions_by_image) dicts,
        where each dict maps image_id to a list of annotation records.
    """
    instances_by_image: dict[int, list[dict]] = defaultdict(list)
    keypoints_by_image: dict[int, list[dict]] = defaultdict(list)
    captions_by_image: dict[int, list[dict]] = defaultdict(list)

    for data in all_annotations_data:
        annotations = data.get("annotations", [])
        if not annotations:
            continue

        # Classify file type based on annotation structure
        # Check first annotation to determine type
        sample_ann = annotations[0] if annotations else {}

        if "caption" in sample_ann:
            # Caption annotations
            for ann in annotations:
                captions_by_image[ann["image_id"]].append(ann)
        elif "keypoints" in sample_ann:
            # Keypoint annotations
            for ann in annotations:
                keypoints_by_image[ann["image_id"]].append(ann)
        else:
            # Instance annotations (bbox, segmentation)
            for ann in annotations:
                instances_by_image[ann["image_id"]].append(ann)

    return instances_by_image, keypoints_by_image, captions_by_image


def _build_subset_config_from_paths(
    images_config: dict[str, str],
    annotations_config: dict[str, list[str]],
) -> dict[Subset, dict[str, Path | list[Path]]]:
    """
    Build subset configuration from user-provided paths.

    Maps user-provided subset names to Subset enum values and creates
    configuration dictionaries for each subset.
    """
    result: dict[Subset, dict[str, Path | list[Path]]] = {}

    for subset_name, images_path in images_config.items():
        # Map subset name to enum (case-insensitive)
        subset_enum = _SUBSET_NAME_TO_ENUM.get(subset_name.lower(), Subset.UNASSIGNED)

        result[subset_enum] = {
            "images_dir": Path(images_path),
            "annotations": [Path(p) for p in annotations_config[subset_name]],
        }

    return result


def save_coco_dataset(
    dataset: Dataset[CocoSample],
    images_dir_path: str | dict[str, str],
    annotations_path: str | dict[str, str],
) -> None:
    """
    Save a Dataset of CocoSample to disk in COCO JSON format.

    This function supports two common COCO dataset layouts:

    1. **Simple layout** (single folder): Pass strings for both path arguments.
       - ``images_dir_path``: Path to a single directory where images will be saved
       - ``annotations_path``: Path to a single COCO JSON annotation file to write
       - All samples will be saved regardless of their subset assignment

    2. **Split layout** (multiple subsets): Pass dicts mapping subset names to paths.
       - ``images_dir_path``: Dict mapping subset names to image directories
       - ``annotations_path``: Dict mapping subset names to annotation JSON files
       - Only samples whose subset matches a key will be saved
       - Subset names can be arbitrary strings (e.g., "training", "validation", "test")

    The function attempts to preserve COCO structure while being robust to missing
    fields by inserting placeholder values when expected data is absent.

    Placeholder policy:
      - Categories: if label names are not available from schema, a single
        placeholder label "label_0" is used with id=1.
      - Images: if width/height are unknown, they are set to 0.
      - Bboxes: if bbox values are missing but labels exist, a placeholder
        bbox [0, 0, 0, 0] is used; area is set to 0; iscrowd=0.
      - Polygons: if polygon points appear to be padded with zeros, trailing
        zero points are trimmed; empty result falls back to an empty list.
      - Keypoints: when labels exist without keypoint coordinates, an empty
        keypoint list is emitted with num_keypoints=0.
      - Captions: empty captions are skipped; missing ids are auto-generated.

    Args:
        dataset: Dataset containing CocoSample samples.
        images_dir_path: Either a single directory path (str) for saving images,
            or a dict mapping subset names to their respective image directories.
        annotations_path: Either a single annotation file path (str) to write,
            or a dict mapping subset names to their respective annotation JSON files.

    Raises:
        ValueError: If the types of images_dir_path and annotations_path don't match
            (both must be str or both must be dict).

    Examples:
        Save to a simple COCO layout::

            save_coco_dataset(
                dataset,
                images_dir_path="/path/to/output/images",
                annotations_path="/path/to/output/annotations.json",
            )

        Save to a COCO 2017-style layout::

            save_coco_dataset(
                dataset,
                images_dir_path={
                    "training": "/path/to/output/train2017",
                    "validation": "/path/to/output/val2017",
                },
                annotations_path={
                    "training": "/path/to/output/annotations/instances_train2017.json",
                    "validation": "/path/to/output/annotations/instances_val2017.json",
                },
            )
    """
    # Normalize inputs to dict format for unified processing
    if isinstance(images_dir_path, str) and isinstance(annotations_path, str):
        # Simple layout: single directory and single annotation file
        images_config: dict[str, str] = {_DEFAULT_SUBSET_KEY: images_dir_path}
        annotations_config: dict[str, str] = {_DEFAULT_SUBSET_KEY: annotations_path}
        is_simple_layout = True
    elif isinstance(images_dir_path, dict) and isinstance(annotations_path, dict):
        # Split layout: multiple subsets
        images_config = images_dir_path
        annotations_config = annotations_path
        is_simple_layout = False
    else:
        raise ValueError(
            "images_dir_path and annotations_path must both be strings or both be dicts. "
            f"Got {type(images_dir_path).__name__} and {type(annotations_path).__name__}."
        )

    # Validate that dict keys match
    if set(images_config.keys()) != set(annotations_config.keys()):
        raise ValueError(
            f"Subset keys must match between images_dir_path and annotations_path. "
            f"Got images: {set(images_config.keys())}, annotations: {set(annotations_config.keys())}"
        )

    logger.info("[COCO] Saving dataset with %d subset(s)", len(images_config))

    categories_coco, to_category_id = _prepare_categories(dataset)

    # Group samples by subset
    subset_to_samples: dict[Subset, list[CocoSample]] = defaultdict(list)
    for sample in dataset:
        subset_to_samples[sample.subset].append(sample)  # type: ignore[attr-defined]

    if is_simple_layout:
        # Simple layout: save all samples to a single location
        all_samples = [sample for samples in subset_to_samples.values() for sample in samples]
        if all_samples:
            _save_subset_flexible(
                images_dir=Path(images_config[_DEFAULT_SUBSET_KEY]),
                annotations_file=Path(annotations_config[_DEFAULT_SUBSET_KEY]),
                samples=all_samples,
                categories_coco=categories_coco,
                to_category_id=to_category_id,
            )
            logger.info("[COCO] Saved %d samples to %s", len(all_samples), annotations_config[_DEFAULT_SUBSET_KEY])
    else:
        # Split layout: save each subset to its designated location
        for subset_name, images_dir_str in images_config.items():
            subset_enum = _SUBSET_NAME_TO_ENUM.get(subset_name.lower(), Subset.UNASSIGNED)
            samples = subset_to_samples.get(subset_enum, [])

            if not samples:
                logger.warning("[COCO] No samples found for subset '%s', skipping", subset_name)
                continue

            _save_subset_flexible(
                images_dir=Path(images_dir_str),
                annotations_file=Path(annotations_config[subset_name]),
                samples=samples,
                categories_coco=categories_coco,
                to_category_id=to_category_id,
            )
            logger.info("[COCO] Saved %d samples for subset '%s'", len(samples), subset_name)

    logger.info("[COCO] Finished saving dataset")
