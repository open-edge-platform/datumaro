# Copyright (C) 2022-2026 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Pascal VOC dataset I/O support.
Supports loading and saving datasets in Pascal VOC format with the following structure::
    dataset_root/
        JPEGImages/
            image1.jpg
            image2.jpg
        Annotations/
            image1.xml
            image2.xml
        ImageSets/
            Main/
                train.txt
                val.txt
                test.txt
        labelmap.txt (optional)
"""

import logging
from pathlib import Path

from datumaro.experimental import Dataset
from datumaro.experimental.categories import MaskCategories
from datumaro.experimental.data_formats.voc.helpers import (
    _load_voc_categories,
    _load_voc_from_imagesets,
    _load_voc_simple,
    _save_voc_dataset,
)
from datumaro.experimental.data_formats.voc.sample import VocSample

logger = logging.getLogger(__name__)


def load_voc_dataset(
    root_dir: str | None = None,
    images_dir_path: str | None = None,
    annotations_dir_path: str | None = None,
) -> Dataset:
    """
    Load a Pascal VOC dataset.
    This function supports two ways of loading:
    1. **Standard VOC layout**: Pass only ``root_dir`` pointing to the dataset root.
       The function will auto-detect the structure and load from ImageSets if available::
            dataset_root/
                JPEGImages/
                Annotations/
                ImageSets/
                    Main/
                        train.txt
                        val.txt
                labelmap.txt (optional)
    2. **Simple layout**: Pass ``images_dir_path`` and ``annotations_dir_path`` separately.
       All images will be assigned ``Subset.UNASSIGNED``.
    Args:
        root_dir: Path to the VOC dataset root directory (standard layout).
        images_dir_path: Path to images directory (simple layout).
        annotations_dir_path: Path to annotations directory (simple layout).
    Returns:
        Dataset containing VocSample instances.
    Raises:
        ValueError: If arguments are invalid or conflicting.
        FileNotFoundError: If directories don't exist.
    Examples:
        Load standard VOC dataset::
            dataset = load_voc_dataset(root_dir="/path/to/VOCdevkit/VOC2012")
        Load from separate directories::
            dataset = load_voc_dataset(
                images_dir_path="/path/to/images",
                annotations_dir_path="/path/to/annotations",
            )
    """
    # Validate arguments
    if root_dir is not None and (images_dir_path is not None or annotations_dir_path is not None):
        raise ValueError(
            "Cannot specify both 'root_dir' and 'images_dir_path'/'annotations_dir_path'. "
            "Use either root_dir for standard VOC layout, or both images_dir_path and "
            "annotations_dir_path for simple layout."
        )
    if root_dir is None and images_dir_path is None:
        raise ValueError(
            "Must specify either 'root_dir' or 'images_dir_path'. "
            "Use root_dir for standard VOC layout, or both images_dir_path and "
            "annotations_dir_path for simple layout."
        )
    # Simple layout
    if images_dir_path is not None:
        if annotations_dir_path is None:
            raise ValueError("'annotations_dir_path' is required when using 'images_dir_path'")
        images_path = Path(images_dir_path)
        if not images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir_path}")
        logger.info("[VOC] Loading simple layout from '%s'", images_dir_path)
        # Load categories from parent or current directory
        parent_dir = images_path.parent
        categories = _load_voc_categories(parent_dir)
        samples = _load_voc_simple(images_dir_path, annotations_dir_path, categories)
        # Build categories dict with MaskCategories for mask fields
        mask_categories = MaskCategories.generate(size=len(categories.labels), include_background=True)
        mask_categories = MaskCategories(labels=list(categories.labels), colormap=mask_categories.colormap)
        # Note: instance_mask shares categories with class_mask via categories_from field attribute
        dataset_categories: dict[str, object] = {
            "labels": categories,
            "class_mask": mask_categories,
        }
        dataset = Dataset(VocSample, categories=dataset_categories)
        for sample in samples:
            dataset.append(sample)
        logger.info("[VOC] Loaded %d samples", len(dataset))
        return dataset
    # Standard VOC layout
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"VOC root directory not found: {root_dir}")
    logger.info("[VOC] Loading standard layout from '%s'", root_dir)
    categories = _load_voc_categories(root_path)
    samples = _load_voc_from_imagesets(root_path, categories)
    # Build categories dict - start with labels
    dataset_categories: dict[str, object] = {"labels": categories}
    # Note: instance_mask shares categories with class_mask via categories_from field attribute
    mask_categories = MaskCategories.generate(
        size=len(categories.labels), include_background=True, labels=list(categories.labels)
    )
    dataset_categories["class_mask"] = mask_categories
    dataset = Dataset(VocSample, categories=dataset_categories)
    for sample in samples:
        dataset.append(sample)
    logger.info("[VOC] Loaded %d samples", len(dataset))
    return dataset


def save_voc_dataset(
    dataset: Dataset[VocSample],
    root_dir: str,
    save_images: bool = True,
) -> dict[str, Path]:
    """
    Save a Dataset of VocSample to disk in Pascal VOC format.
    Creates the following structure::
        root_dir/
            JPEGImages/
                image1.jpg
                ...
            Annotations/
                image1.xml
                ...
            ImageSets/
                Main/
                    train.txt
                    val.txt
                    ...
            labelmap.txt
    Args:
        dataset: Dataset containing VocSample samples.
        root_dir: Path to the output directory.
        save_images: Whether to copy images to the output directory.
    Returns:
        Dictionary mapping logical names to written paths.
    """
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    logger.info("[VOC] Saving dataset to '%s'", root_dir)
    written_paths = _save_voc_dataset(dataset, root_path, save_images)
    logger.info("[VOC] Saved %d samples", len(dataset))
    return written_paths
