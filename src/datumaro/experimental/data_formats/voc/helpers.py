# Copyright (C) 2022-2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
Helper functions for Pascal VOC dataset I/O.
"""

import logging
import shutil
from collections import defaultdict
from pathlib import Path
from xml.etree.ElementTree import Element  # nosec B405 - only used for type hints and writing, not parsing

import numpy as np
from defusedxml import ElementTree

from datumaro.experimental import Dataset
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.data_formats.voc.constants import (
    SUBSET_TO_VOC_NAME,
    VOC_ANNOTATIONS_DIR,
    VOC_IMAGES_DIR,
    VOC_IMAGESETS_DIR,
    VOC_LABELMAP_FILE,
    VOC_LABELS,
    VOC_MAIN_DIR,
    VOC_SUBSET_NAME_TO_ENUM,
)
from datumaro.experimental.data_formats.voc.sample import VocSample
from datumaro.experimental.fields import ImageInfo, Subset
from datumaro.util.image import IMAGE_EXTENSIONS, find_images

logger = logging.getLogger(__name__)


def _get_image_size(image_path: Path) -> tuple[int, int]:
    """Get image dimensions (height, width) using PIL."""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size[1], img.size[0]  # height, width
    except Exception as e:
        logger.warning("Failed to read image size from %s: %s", image_path, e)
        return 0, 0


def _find_image_file(images_dir: Path, stem: str) -> Path | None:
    """Find an image file with any supported extension."""
    for candidate in images_dir.glob(f"{stem}.*"):
        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    return None


def _parse_voc_labelmap(labelmap_path: Path) -> list[str]:
    """
    Parse a VOC labelmap file.

    Format: 'name : color (r, g, b) : parts : actions'
    Only the name is extracted.

    Returns:
        List of label names in order.
    """
    labels = []
    with open(labelmap_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # name : color : parts : actions
            parts = line.split(":")
            if parts:
                name = parts[0].strip()
                if name:
                    labels.append(name)
    return labels


def _load_voc_categories(root_path: Path) -> LabelCategories:
    """Load categories from VOC labelmap file or use defaults."""
    labelmap_path = root_path / VOC_LABELMAP_FILE
    if labelmap_path.exists():
        labels = _parse_voc_labelmap(labelmap_path)
        if labels:
            return LabelCategories(labels=tuple(labels))

    # Check for meta file
    meta_file = root_path / "dataset_meta.json"
    if meta_file.exists():
        import json

        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
            labels = meta.get("labels", [])
            if labels:
                return LabelCategories(labels=tuple(labels))

    # Use default VOC labels
    return LabelCategories(labels=VOC_LABELS)


def _detect_voc_subsets(root_path: Path) -> dict[str, Path]:
    """
    Detect available subsets in a VOC dataset.

    Returns:
        Dictionary mapping subset names to their txt file paths.
    """
    subsets = {}
    imagesets_main = root_path / VOC_IMAGESETS_DIR / VOC_MAIN_DIR

    if imagesets_main.exists():
        for txt_file in imagesets_main.glob("*.txt"):
            subset_name = txt_file.stem
            # Skip label-specific files like 'aeroplane_train.txt'
            if "_" not in subset_name or subset_name in VOC_SUBSET_NAME_TO_ENUM:
                subsets[subset_name] = txt_file

    return subsets


def _parse_subset_list(subset_file: Path) -> list[str]:
    """
    Parse a VOC ImageSets subset file.

    Returns:
        List of image IDs in the subset.
    """
    image_ids = []
    with open(subset_file, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # VOC format can have 'image_id label' for classification
            # We only want the image_id
            parts = line.split()
            if parts:
                image_ids.append(parts[0])
    return image_ids


def _parse_voc_annotation(anno_path: Path, categories: LabelCategories) -> dict:
    """
    Parse a VOC XML annotation file.

    Returns:
        Dictionary with keys: width, height, bboxes, labels, difficult, truncated, occluded, pose
    """
    result = {
        "width": 0,
        "height": 0,
        "bboxes": [],
        "labels": [],
        "difficult": [],
        "truncated": [],
        "occluded": [],
        "pose": [],
    }

    if not anno_path.exists():
        return result

    try:
        tree = ElementTree.parse(anno_path)
        root = tree.getroot()

        # Parse size
        size_elem = root.find("size")
        if size_elem is not None:
            width_elem = size_elem.find("width")
            height_elem = size_elem.find("height")
            if width_elem is not None and width_elem.text:
                result["width"] = int(width_elem.text)
            if height_elem is not None and height_elem.text:
                result["height"] = int(height_elem.text)

        # Parse objects
        for obj_elem in root.findall("object"):
            _parse_object_element(obj_elem, result, categories)

    except ElementTree.ParseError as e:
        logger.warning("Failed to parse VOC annotation %s: %s", anno_path, e)

    return result


def _parse_object_element(obj_elem: Element, result: dict, categories: LabelCategories) -> None:
    """Parse a single object element from VOC XML."""
    # Get label name
    name_elem = obj_elem.find("name")
    if name_elem is None or not name_elem.text:
        return

    label_name = name_elem.text.strip()

    # Get label index from categories
    try:
        label_idx = categories.labels.index(label_name)
    except ValueError:
        logger.warning("Unknown label '%s' in VOC annotation, skipping", label_name)
        return

    # Parse bounding box
    bndbox = obj_elem.find("bndbox")
    if bndbox is None:
        return

    try:
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
    except (AttributeError, ValueError, TypeError) as e:
        logger.warning("Invalid bounding box in VOC annotation: %s", e)
        return

    result["bboxes"].append([xmin, ymin, xmax, ymax])
    result["labels"].append(label_idx)

    # Parse attributes
    difficult_elem = obj_elem.find("difficult")
    result["difficult"].append(difficult_elem is not None and difficult_elem.text == "1")

    truncated_elem = obj_elem.find("truncated")
    result["truncated"].append(truncated_elem is not None and truncated_elem.text == "1")

    occluded_elem = obj_elem.find("occluded")
    result["occluded"].append(occluded_elem is not None and occluded_elem.text == "1")

    pose_elem = obj_elem.find("pose")
    result["pose"].append(pose_elem.text if pose_elem is not None else "Unspecified")


def _create_sample_from_annotation(
    image_id: str,
    images_dir: Path,
    annotations_dir: Path,
    categories: LabelCategories,
    subset_enum: Subset,
) -> VocSample | None:
    """Create a VocSample from image ID and annotation."""
    # Find image file
    image_path = _find_image_file(images_dir, image_id)
    if image_path is None:
        logger.warning("Image not found for ID '%s', skipping", image_id)
        return None

    # Parse annotation
    anno_path = annotations_dir / f"{image_id}.xml"
    annotation = _parse_voc_annotation(anno_path, categories)

    # Get image size from annotation or read from image
    height = annotation["height"]
    width = annotation["width"]
    if height == 0 or width == 0:
        height, width = _get_image_size(image_path)

    # Create arrays
    bboxes = np.array(annotation["bboxes"], dtype=np.float32) if annotation["bboxes"] else None
    labels = np.array(annotation["labels"], dtype=np.uint32) if annotation["labels"] else None
    difficult = np.array(annotation["difficult"], dtype=bool) if annotation["difficult"] else None
    truncated = np.array(annotation["truncated"], dtype=bool) if annotation["truncated"] else None
    occluded = np.array(annotation["occluded"], dtype=bool) if annotation["occluded"] else None
    pose = np.array(annotation["pose"], dtype=object) if annotation["pose"] else None

    return VocSample(
        image=str(image_path),
        image_info=ImageInfo(height=height, width=width),
        bboxes=bboxes,
        labels=labels,
        difficult=difficult,
        truncated=truncated,
        occluded=occluded,
        pose=pose,
        subset=subset_enum,
    )


def _load_voc_from_imagesets(root_path: Path, categories: LabelCategories) -> list[VocSample]:
    """Load VOC dataset using ImageSets structure."""
    samples = []

    images_dir = root_path / VOC_IMAGES_DIR
    annotations_dir = root_path / VOC_ANNOTATIONS_DIR

    if not images_dir.exists():
        logger.warning("Images directory not found: %s", images_dir)
        return samples

    subsets = _detect_voc_subsets(root_path)

    if not subsets:
        # No ImageSets found, try to load all images
        logger.info("No ImageSets found, loading all images from %s", images_dir)
        image_files = list(find_images(str(images_dir)))
        for image_path in image_files:
            image_id = Path(image_path).stem
            sample = _create_sample_from_annotation(
                image_id, images_dir, annotations_dir, categories, Subset.UNASSIGNED
            )
            if sample:
                samples.append(sample)
    else:
        # Load from ImageSets
        for subset_name, subset_file in subsets.items():
            subset_enum = VOC_SUBSET_NAME_TO_ENUM.get(subset_name, Subset.UNASSIGNED)
            image_ids = _parse_subset_list(subset_file)

            logger.info(
                "[VOC] Loading subset '%s' with %d images",
                subset_name,
                len(image_ids),
            )

            for image_id in image_ids:
                sample = _create_sample_from_annotation(image_id, images_dir, annotations_dir, categories, subset_enum)
                if sample:
                    samples.append(sample)

    return samples


def _load_voc_simple(
    images_dir_path: str,
    annotations_dir_path: str,
    categories: LabelCategories,
) -> list[VocSample]:
    """Load VOC dataset from simple image and annotations directories."""
    samples = []

    images_dir = Path(images_dir_path)
    annotations_dir = Path(annotations_dir_path)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_files = list(find_images(str(images_dir)))

    for image_path in image_files:
        image_id = Path(image_path).stem
        sample = _create_sample_from_annotation(image_id, images_dir, annotations_dir, categories, Subset.UNASSIGNED)
        if sample:
            samples.append(sample)

    return samples


# ============= Export helpers =============


def _create_voc_xml_annotation(
    sample: VocSample,
    image_filename: str,
    categories: LabelCategories,
) -> Element:
    """Create a VOC XML annotation element from a sample."""
    root = Element("annotation")

    # Folder (optional)
    folder_elem = Element("folder")
    folder_elem.text = VOC_IMAGES_DIR
    root.append(folder_elem)

    # Filename
    filename_elem = Element("filename")
    filename_elem.text = image_filename
    root.append(filename_elem)

    # Size
    size_elem = Element("size")
    width_elem = Element("width")
    width_elem.text = str(sample.image_info.width)
    size_elem.append(width_elem)
    height_elem = Element("height")
    height_elem.text = str(sample.image_info.height)
    size_elem.append(height_elem)
    depth_elem = Element("depth")
    depth_elem.text = "3"  # Assume RGB
    size_elem.append(depth_elem)
    root.append(size_elem)

    # Objects
    if sample.bboxes is not None and len(sample.bboxes) > 0:
        for i, bbox in enumerate(sample.bboxes):
            obj_elem = _create_object_element(sample, i, bbox, categories)
            root.append(obj_elem)

    return root


def _create_object_element(
    sample: VocSample,
    idx: int,
    bbox: np.ndarray,
    categories: LabelCategories,
) -> Element:
    """Create a single object element for VOC XML."""
    obj_elem = Element("object")

    # Name (label)
    name_elem = Element("name")
    if sample.labels is not None and idx < len(sample.labels):
        label_idx = int(sample.labels[idx])
        if label_idx < len(categories.labels):
            name_elem.text = categories.labels[label_idx]
        else:
            name_elem.text = f"unknown_{label_idx}"
    else:
        name_elem.text = "unknown"
    obj_elem.append(name_elem)

    # Pose
    pose_elem = Element("pose")
    if sample.pose is not None and idx < len(sample.pose):
        pose_elem.text = str(sample.pose[idx])
    else:
        pose_elem.text = "Unspecified"
    obj_elem.append(pose_elem)

    # Truncated
    truncated_elem = Element("truncated")
    if sample.truncated is not None and idx < len(sample.truncated):
        truncated_elem.text = "1" if sample.truncated[idx] else "0"
    else:
        truncated_elem.text = "0"
    obj_elem.append(truncated_elem)

    # Difficult
    difficult_elem = Element("difficult")
    if sample.difficult is not None and idx < len(sample.difficult):
        difficult_elem.text = "1" if sample.difficult[idx] else "0"
    else:
        difficult_elem.text = "0"
    obj_elem.append(difficult_elem)

    # Occluded
    occluded_elem = Element("occluded")
    if sample.occluded is not None and idx < len(sample.occluded):
        occluded_elem.text = "1" if sample.occluded[idx] else "0"
    else:
        occluded_elem.text = "0"
    obj_elem.append(occluded_elem)

    # Bounding box
    bndbox_elem = Element("bndbox")
    xmin_elem = Element("xmin")
    xmin_elem.text = str(int(bbox[0]))
    bndbox_elem.append(xmin_elem)
    ymin_elem = Element("ymin")
    ymin_elem.text = str(int(bbox[1]))
    bndbox_elem.append(ymin_elem)
    xmax_elem = Element("xmax")
    xmax_elem.text = str(int(bbox[2]))
    bndbox_elem.append(xmax_elem)
    ymax_elem = Element("ymax")
    ymax_elem.text = str(int(bbox[3]))
    bndbox_elem.append(ymax_elem)
    obj_elem.append(bndbox_elem)

    return obj_elem


def _write_voc_xml(root: Element, output_path: Path) -> None:
    """Write VOC XML annotation to file with pretty formatting."""
    import xml.etree.ElementTree as ET  # nosec B405 - safe for writing XML, not parsing untrusted data

    rough_string = ET.tostring(root, encoding="unicode")

    # Simple indentation without minidom for security
    lines = []
    indent = 0
    for raw_token in rough_string.replace("><", ">\n<").split("\n"):
        token = raw_token.strip()
        if not token:
            continue
        if token.startswith("</"):
            indent -= 1
        lines.append("  " * indent + token)
        if not token.startswith("</") and not token.endswith("/>") and "</" not in token:
            indent += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_labelmap(categories: LabelCategories, output_path: Path) -> None:
    """Write VOC labelmap file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# label:color_rgb:parts:actions\n")
        for label in categories.labels:
            f.write(f"{label}:::\n")


def _get_image_path_from_sample(sample: VocSample) -> Path | None:
    """Extract image path from a VocSample."""
    if hasattr(sample.image, "path"):
        return Path(sample.image.path)
    if isinstance(sample.image, str):
        return Path(sample.image)
    return None


def _get_image_id_from_sample(sample: VocSample, idx: int) -> str:
    """Generate a unique image ID from a sample."""
    image_path = _get_image_path_from_sample(sample)
    if image_path:
        return f"{image_path.stem}_{idx:06d}"
    return f"image_{idx:06d}"


def _process_sample_image(
    sample: VocSample,
    image_id: str,
    images_dir: Path,
    save_images: bool,
) -> str:
    """Process and optionally copy sample image, returning the filename."""
    image_path = _get_image_path_from_sample(sample)

    if save_images and image_path is not None and image_path.exists():
        dst_path = images_dir / f"{image_id}{image_path.suffix}"
        if image_path != dst_path:
            shutil.copy2(image_path, dst_path)
        return dst_path.name
    if image_path is not None:
        return f"{image_id}{image_path.suffix}"
    return f"{image_id}.jpg"


def _process_subset(
    sample_list: list[tuple[str, VocSample]],
    images_dir: Path,
    annotations_dir: Path,
    categories: LabelCategories,
    save_images: bool,
) -> list[str]:
    """Process all samples in a subset, saving images and annotations."""
    image_ids = []
    for image_id, sample in sample_list:
        image_ids.append(image_id)
        image_filename = _process_sample_image(sample, image_id, images_dir, save_images)
        xml_root = _create_voc_xml_annotation(sample, image_filename, categories)
        xml_path = annotations_dir / f"{image_id}.xml"
        _write_voc_xml(xml_root, xml_path)
    return image_ids


def _save_voc_dataset(
    dataset: Dataset[VocSample],
    root_dir: Path,
    save_images: bool = True,
) -> dict[str, Path]:
    """
    Save a VOC dataset to disk.

    Returns:
        Dictionary mapping logical names to written paths.
    """
    written_paths: dict[str, Path] = {}

    # Create directory structure
    images_dir = root_dir / VOC_IMAGES_DIR
    annotations_dir = root_dir / VOC_ANNOTATIONS_DIR
    imagesets_dir = root_dir / VOC_IMAGESETS_DIR / VOC_MAIN_DIR

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    imagesets_dir.mkdir(parents=True, exist_ok=True)

    # Get categories from schema
    label_attr_info = dataset.schema.attributes.get("labels")
    categories = (
        label_attr_info.categories
        if label_attr_info and label_attr_info.categories
        else LabelCategories(labels=VOC_LABELS)
    )

    # Group samples by subset
    samples_by_subset: dict[Subset, list[tuple[str, VocSample]]] = defaultdict(list)
    for idx, sample in enumerate(dataset):
        image_id = _get_image_id_from_sample(sample, idx)
        samples_by_subset[sample.subset].append((image_id, sample))

    # Group by VOC subset name to handle multiple Subset enums mapping to the same name
    samples_by_voc_name: dict[str, list[tuple[str, VocSample]]] = defaultdict(list)
    for subset, sample_list in samples_by_subset.items():
        voc_name = SUBSET_TO_VOC_NAME.get(subset, "train")
        samples_by_voc_name[voc_name].extend(sample_list)

    # Process each VOC subset
    for subset_name, sample_list in samples_by_voc_name.items():
        subset_file = imagesets_dir / f"{subset_name}.txt"

        image_ids = _process_subset(sample_list, images_dir, annotations_dir, categories, save_images)

        # Write ImageSets file
        with open(subset_file, "w", encoding="utf-8") as f:
            for image_id in image_ids:
                f.write(f"{image_id}\n")

        written_paths[f"imageset_{subset_name}"] = subset_file
        logger.info("[VOC] Wrote subset '%s' with %d images", subset_name, len(image_ids))

    # Write labelmap
    labelmap_path = root_dir / VOC_LABELMAP_FILE
    _write_labelmap(categories, labelmap_path)
    written_paths["labelmap"] = labelmap_path

    written_paths["images_dir"] = images_dir
    written_paths["annotations_dir"] = annotations_dir

    return written_paths
