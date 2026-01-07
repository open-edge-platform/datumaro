# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE
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
    _index_annotations_by_image,
    _load_json_or_none,
    _prepare_categories,
    _save_subset_flexible,
)
from datumaro.experimental.data_formats.coco.sample import CocoSample
from datumaro.experimental.fields import Subset

logger = logging.getLogger(__name__)


def load_coco_dataset(
    images_dir_path: str | dict[str, str],
    annotations_path: str | dict[str, str],
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
        annotations_path: Either a single annotation file path (str),
            or a dict mapping subset names to their respective annotation JSON files.

    Returns:
        Dataset containing CocoSample instances.

    Raises:
        ValueError: If the types of images_dir_path and annotations_path don't match
            (both must be str or both must be dict).
        FileNotFoundError: If any specified directory or file doesn't exist.

    Examples:
        Load a simple COCO dataset::

            dataset = load_coco_dataset(
                images_dir_path="/path/to/images",
                annotations_path="/path/to/annotations.json",
            )

        Load a COCO 2017-style dataset::

            dataset = load_coco_dataset(
                images_dir_path={
                    "training": "/path/to/train2017",
                    "validation": "/path/to/val2017",
                },
                annotations_path={
                    "training": "/path/to/annotations/instances_train2017.json",
                    "validation": "/path/to/annotations/instances_val2017.json",
                },
            )
    """
    # Normalize inputs to dict format for unified processing
    if isinstance(images_dir_path, str) and isinstance(annotations_path, str):
        # Simple layout: single directory and single annotation file
        images_config: dict[str, str] = {"__unassigned__": images_dir_path}
        annotations_config: dict[str, str] = {"__unassigned__": annotations_path}
    elif isinstance(images_dir_path, dict) and isinstance(annotations_path, dict):
        # Split layout: multiple subsets
        images_config = images_dir_path
        annotations_config = annotations_path
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

    # Build subset configuration
    subset_config = _build_subset_config_from_paths(images_config, annotations_config)

    logger.info("[COCO] Loading dataset with %d subset(s)", len(subset_config))

    # Determine label categories via helper
    loaded_label_categories = _detect_coco_label_categories_from_paths(subset_config)
    categories = {"labels": loaded_label_categories}
    dataset = Dataset(CocoSample, categories=categories)

    for subset, config in subset_config.items():
        images_dir = config["images_dir"]
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

        annotations_file = config["annotations"]
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file does not exist: {annotations_file}")

        logger.info("[COCO] Loading subset '%s' from '%s'", subset, images_dir)

        annotations_data = _load_json_or_none(annotations_file)

        if not annotations_data:
            logger.warning("[COCO] No annotations found for subset '%s', skipping", subset)
            continue

        cat_id_to_idx = _cat_id_to_idx_from_primary(annotations_data)
        instances_by_image, keypoints_by_image, captions_by_image = _index_annotations_by_image(
            annotations_data, None, None
        )

        images = annotations_data.get("images", [])
        num_images = len(images)
        logger.info("[COCO] Building %d samples for subset '%s'", num_images, subset)

        for idx, img in enumerate(images, start=1):
            if idx % 1000 == 0:
                logger.info("[COCO] Subset '%s': processed %d/%d images", subset, idx, num_images)
            sample = _assemble_sample_from_image_record(
                images_dir=images_dir,
                img=img,
                cat_id_to_idx=cat_id_to_idx,
                instances_by_image=instances_by_image,
                keypoints_by_image=keypoints_by_image,
                captions_by_image=captions_by_image,
                subset=subset,
            )
            dataset.append(sample)

        logger.info("[COCO] Finished subset '%s' with %d samples", subset, num_images)

    logger.info("[COCO] Finished loading dataset with %d samples in total", len(dataset))
    return dataset


def _build_subset_config_from_paths(
    images_config: dict[str, str],
    annotations_config: dict[str, str],
) -> dict[Subset, dict[str, Path]]:
    """
    Build subset configuration from user-provided paths.

    Maps user-provided subset names to Subset enum values and creates
    configuration dictionaries for each subset.
    """
    subset_name_to_enum = {
        "train": Subset.TRAINING,
        "training": Subset.TRAINING,
        "val": Subset.VALIDATION,
        "validation": Subset.VALIDATION,
        "test": Subset.TESTING,
        "testing": Subset.TESTING,
        "__unassigned__": Subset.UNASSIGNED,
    }

    result: dict[Subset, dict[str, Path]] = {}

    for subset_name, images_path in images_config.items():
        # Map subset name to enum (case-insensitive)
        subset_enum = subset_name_to_enum.get(subset_name.lower(), Subset.UNASSIGNED)

        result[subset_enum] = {
            "images_dir": Path(images_path),
            "annotations": Path(annotations_config[subset_name]),
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
        images_config: dict[str, str] = {"__all__": images_dir_path}
        annotations_config: dict[str, str] = {"__all__": annotations_path}
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

    # Build mapping from subset names to Subset enum
    subset_name_to_enum = {
        "train": Subset.TRAINING,
        "training": Subset.TRAINING,
        "val": Subset.VALIDATION,
        "validation": Subset.VALIDATION,
        "test": Subset.TESTING,
        "testing": Subset.TESTING,
    }

    if is_simple_layout:
        # Simple layout: save all samples to a single location
        all_samples = [sample for samples in subset_to_samples.values() for sample in samples]
        if all_samples:
            _save_subset_flexible(
                images_dir=Path(images_config["__all__"]),
                annotations_file=Path(annotations_config["__all__"]),
                samples=all_samples,
                categories_coco=categories_coco,
                to_category_id=to_category_id,
            )
            logger.info("[COCO] Saved %d samples to %s", len(all_samples), annotations_config["__all__"])
    else:
        # Split layout: save each subset to its designated location
        for subset_name, images_dir_str in images_config.items():
            subset_enum = subset_name_to_enum.get(subset_name.lower(), Subset.UNASSIGNED)
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
