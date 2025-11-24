# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE
"""
Coco2017 dataset support
"""

import logging
from collections import defaultdict
from pathlib import Path

from datumaro.experimental import Dataset
from datumaro.experimental.data_formats.coco.helpers import (
    _assemble_sample_from_image_record,
    _build_subset_config,
    _cat_id_to_idx_from_primary,
    _index_annotations_by_image,
    _load_json_or_none,
    _prepare_categories,
    _save_subset,
)
from datumaro.experimental.data_formats.coco.sample import CocoCategories, CocoSample
from datumaro.experimental.fields import Subset

logger = logging.getLogger(__name__)


def load_coco_dataset(root_dir: str, version: str = "2017") -> Dataset:
    """
    Load a COCO dataset (2017 layout) from its top-level directory.

    The expected directory structure under ``root_dir`` is::

        annotations/
            instances_train{version}.json
            instances_val{version}.json
            person_keypoints_train{version}.json
            person_keypoints_val{version}.json
            (optional) instances_test{version}.json
            (optional) person_keypoints_test{version}.json
        train{version}/
        val{version}/
        (optional) test{version}/

    Args:
        root_dir: Path to the top-level COCO dataset directory.
        version: Dataset year version string (default: "2017").

    Returns:
        Dataset containing CocoSample instances.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"COCO root directory does not exist: {root_dir}")

    annotations_path = root_path / "annotations"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing 'annotations' directory under COCO root: {annotations_path}")

    subset_config = _build_subset_config(root_path, version)

    logger.info("[COCO] Loading %s dataset from '%s' with %d subsets", version, root_dir, len(subset_config))

    categories = {"labels": CocoCategories()}
    dataset = Dataset(CocoSample, categories=categories)

    for subset, config in subset_config.items():
        images_dir = config["images_dir"]
        if not images_dir.exists():
            logger.warning("[COCO] Skipping subset '%s': images directory '%s' does not exist", subset, images_dir)
            continue

        logger.info("[COCO] Loading subset '%s' from '%s'", subset, images_dir)

        instances_data = _load_json_or_none(config["instances"])  # type: ignore[arg-type]
        keypoints_data = _load_json_or_none(config["keypoints"])  # type: ignore[arg-type]
        captions_data = _load_json_or_none(config["captions"])  # type: ignore[arg-type]

        if not instances_data and not keypoints_data:
            logger.warning("[COCO] No instances/keypoints annotations found for subset '%s', skipping", subset)
            continue

        primary_data = instances_data or keypoints_data or {"categories": [], "images": []}

        cat_id_to_idx = _cat_id_to_idx_from_primary(primary_data)
        instances_by_image, keypoints_by_image, captions_by_image = _index_annotations_by_image(
            instances_data, keypoints_data, captions_data
        )

        images = primary_data.get("images", [])
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

    logger.info("[COCO] Finished loading dataset from '%s' with %d samples in total", root_dir, len(dataset))
    return dataset


def save_coco_dataset(dataset: Dataset[CocoSample], root_dir: str, version: str = "2017") -> dict[str, Path]:
    """
    Save a Dataset of CocoSample back to disk in COCO (2017) JSON format.

    This function writes COCO annotation JSON files under ``{root_dir}/annotations``
    for each subset present in the dataset. It attempts to preserve COCO structure
    while being robust to missing fields by inserting placeholder values when
    expected data is absent.

    Files written per subset (if there is relevant content):
      - instances_{subset}{version}.json          (bboxes & polygons)
      - person_keypoints_{subset}{version}.json   (keypoints)
      - captions_{subset}{version}.json           (captions)

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
        root_dir: Path to the top-level COCO dataset directory where
            annotation files will be written.
        version: Dataset year version string (default: "2017").

    Returns:
        Mapping from a logical name (e.g., "instances_train") to the written Path.
    """

    root_path = Path(root_dir)
    annotations_path = root_path / "annotations"
    annotations_path.mkdir(parents=True, exist_ok=True)

    logger.info("[COCO] Saving %s dataset to '%s' (annotations dir: '%s')", version, root_dir, annotations_path)

    categories_coco, to_category_id = _prepare_categories(dataset)

    subset_to_samples: dict[Subset, list[CocoSample]] = defaultdict(list)
    for sample in dataset:
        subset_to_samples[sample.subset].append(sample)  # type: ignore[attr-defined]

    written: dict[str, Path] = {}
    for subset, samples in subset_to_samples.items():
        if not samples:
            continue

        logger.info("[COCO] Saving subset '%s' with %d samples", subset, len(samples))

        result_paths = _save_subset(
            root_path=root_path,
            annotations_path=annotations_path,
            version=version,
            subset=subset,
            samples=samples,
            categories_coco=categories_coco,
            to_category_id=to_category_id,
        )
        for logical_name, path in result_paths.items():
            logger.info("[COCO] Written %s -> %s", logical_name, path)
        written.update(result_paths)

    logger.info("[COCO] Finished saving dataset to '%s'", root_dir)
    return written
