# Copyright (C) 2021-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import contextlib
import csv
import errno
import fnmatch
import functools
import glob
import itertools
import json
import logging as log
import os
import os.path as osp
import re
import types
from typing import List, Optional, Union

import cv2
import numpy as np
from attr import attrs

from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories, Mask
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.dataset_item_storage import ItemStatus
from datumaro.components.errors import DatasetError, MediaTypeError, RepeatedItemError, UndefinedLabel
from datumaro.components.exporter import Exporter
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.components.validator import Severity
from datumaro.util import parse_json_file
from datumaro.util.annotation_util import find_instances
from datumaro.util.image import (
    DEFAULT_IMAGE_META_FILE_NAME,
    find_images,
    lazy_image,
    load_image,
    load_image_meta_file,
    save_image,
    save_image_meta_file,
)
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file
from datumaro.util.os_util import make_file_name, split_path

# A regex to check whether a string can be used as a "normal" path
# component.
# Some strings that are loaded from annotation files are used in paths,
# and we need to validate them with this, so that a specially crafted
# annotation file can't access files outside of the expected directory.
_RE_INVALID_PATH_COMPONENT = re.compile(
    r"""
    # empty
    | \.\.? # special path component
    | .*[/\\\0].* # contains special characters
""",
    re.VERBOSE,
)


@attrs(auto_attribs=True)
class UnsupportedSubsetNameError(DatasetError):
    item_id: str
    subset: str

    def __str__(self):
        return "Item %s has an unsupported subset name %r." % (self.item_id, self.subset)


@attrs(auto_attribs=True)
class UnsupportedBoxIdError(DatasetError):
    item_id: str
    box_id: str

    def __str__(self):
        return "Item %s has a mask with an unsupported box ID %r." % (self.item_id, self.box_id)


@attrs(auto_attribs=True)
class UnsupportedMaskPathError(DatasetError):
    item_id: str
    mask_path: str

    def __str__(self):
        return "Item %s has a mask with an unsupported path %r." % (self.item_id, self.mask_path)


class OpenImagesPath:
    ANNOTATIONS_DIR = "annotations"
    IMAGES_DIR = "images"
    MASKS_DIR = "masks"

    FULL_IMAGE_DESCRIPTION_FILE_NAME = "image_ids_and_rotation.csv"
    SUBSET_IMAGE_DESCRIPTION_FILE_PATTERNS = (
        "*-images-with-rotation.csv",
        "*-images-with-labels-with-rotation.csv",
    )
    V5_CLASS_DESCRIPTION_FILE_NAME = "class-descriptions.csv"
    V5_CLASS_DESCRIPTION_BBOX_FILE_NAME = "class-descriptions-boxable.csv"
    HIERARCHY_FILE_NAME = "bbox_labels_600_hierarchy.json"

    LABEL_DESCRIPTION_FILE_SUFFIX = "-annotations-human-imagelabels.csv"
    BBOX_DESCRIPTION_FILE_SUFFIX = "-annotations-bbox.csv"
    MASK_DESCRIPTION_FILE_SUFFIX = "-annotations-object-segmentation.csv"

    IMAGE_DESCRIPTION_FIELDS = (
        "ImageID",
        "Subset",
        "OriginalURL",
        "OriginalLandingURL",
        "License",
        "AuthorProfileURL",
        "Author",
        "Title",
        "OriginalSize",
        "OriginalMD5",
        "Thumbnail300KURL",
        "Rotation",
    )

    LABEL_DESCRIPTION_FIELDS = (
        "ImageID",
        "Source",
        "LabelName",
        "Confidence",
    )

    BBOX_DESCRIPTION_FIELDS = (
        "ImageID",
        "Source",
        "LabelName",
        "Confidence",
        "XMin",
        "XMax",
        "YMin",
        "YMax",
        "IsOccluded",
        "IsTruncated",
        "IsGroupOf",
        "IsDepiction",
        "IsInside",
    )

    BBOX_BOOLEAN_ATTRIBUTES = (
        types.SimpleNamespace(datumaro_name="occluded", oid_name="IsOccluded"),
        types.SimpleNamespace(datumaro_name="truncated", oid_name="IsTruncated"),
        types.SimpleNamespace(datumaro_name="is_group_of", oid_name="IsGroupOf"),
        types.SimpleNamespace(datumaro_name="is_depiction", oid_name="IsDepiction"),
        types.SimpleNamespace(datumaro_name="is_inside", oid_name="IsInside"),
    )

    MASK_DESCRIPTION_FIELDS = (
        "MaskPath",
        "ImageID",
        "LabelName",
        "BoxID",
        "BoxXMin",
        "BoxXMax",
        "BoxYMin",
        "BoxYMax",
        "PredictedIoU",
        "Clicks",
    )


class OpenImagesBase(DatasetBase):
    def __init__(
        self,
        path: str,
        *,
        image_meta: Optional[Union[dict, str]] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isdir(path):
            raise NotADirectoryError(errno.ENOTDIR, "Can't find dataset directory", path)

        super().__init__(ctx=ctx)

        self._dataset_dir = path

        self._annotation_dir = osp.join(path, OpenImagesPath.ANNOTATIONS_DIR)
        if not osp.exists(self._annotation_dir):
            self._annotation_dir = path
        self._annotation_files = os.listdir(self._annotation_dir)

        self._categories = {}
        self._items = []

        assert image_meta is None or isinstance(image_meta, (dict, str))
        if isinstance(image_meta, dict):
            self._image_meta = dict(image_meta)
        elif isinstance(image_meta, str):
            self._image_meta = load_image_meta_file(osp.join(path, image_meta))
        elif image_meta is None:
            try:
                self._image_meta = load_image_meta_file(osp.join(self._annotation_dir, DEFAULT_IMAGE_META_FILE_NAME))
            except FileNotFoundError:
                self._image_meta = {}

        if has_meta_file(path):
            self._categories = {AnnotationType.label: LabelCategories.from_iterable(parse_meta_file(path).keys())}
        else:
            self._load_categories()
        self._load_items()

    def __iter__(self):
        yield from self._items

    def categories(self):
        return self._categories

    @contextlib.contextmanager
    def _open_csv_annotation(self, file_name):
        absolute_path = osp.join(self._annotation_dir, file_name)

        with open(absolute_path, "r", encoding="utf-8", newline="") as f:
            yield csv.DictReader(f)

    def _glob_annotations(self, pattern):
        for annotation_file in self._annotation_files:
            if fnmatch.fnmatch(annotation_file, pattern):
                yield annotation_file

    def _load_categories(self):
        label_categories = LabelCategories()

        class_desc_patterns = [
            "oidv*-class-descriptions.csv",
            OpenImagesPath.V5_CLASS_DESCRIPTION_BBOX_FILE_NAME,
            OpenImagesPath.V5_CLASS_DESCRIPTION_FILE_NAME,
        ]

        class_desc_files = [file for pattern in class_desc_patterns for file in self._glob_annotations(pattern)]

        if not class_desc_files:
            raise FileNotFoundError(
                errno.ENOENT,
                "Can't find class description file, the annotations directory does't contain any of these files",
                ", ".join(class_desc_patterns),
            )

        # In OID v6, the class description file is prefixed with `oidv6-`, whereas
        # in the previous versions, it isn't. We try to find it regardless.
        # We use a wildcard so that if, say, OID v7 is released in the future with
        # a similar layout as v6, it's automatically supported.
        # If the file doesn't exist with either name, we'll fail trying to open
        # `class-descriptions.csv`.

        annotation_name = class_desc_files[0]
        with self._open_csv_annotation(annotation_name) as class_description_reader:
            # Prior to OID v6, this file didn't contain a header row.
            if annotation_name in {
                OpenImagesPath.V5_CLASS_DESCRIPTION_BBOX_FILE_NAME,
                OpenImagesPath.V5_CLASS_DESCRIPTION_FILE_NAME,
            }:
                class_description_reader.fieldnames = ("LabelName", "DisplayName")

            for class_description in class_description_reader:
                label_name = class_description["LabelName"]
                label_categories.add(label_name)

        self._categories[AnnotationType.label] = label_categories

        self._load_label_category_parents()

    def _load_label_category_parents(self):
        label_categories = self._categories[AnnotationType.label]

        hierarchy_path = osp.join(self._annotation_dir, OpenImagesPath.HIERARCHY_FILE_NAME)

        try:
            root_node = parse_json_file(hierarchy_path)
        except FileNotFoundError:
            return

        def set_parents_from_node(node, category):
            for child_node in node.get("Subcategory", []):
                _, child_category = label_categories.find(child_node["LabelName"])

                if category is not None and child_category is not None:
                    child_category.parent = category.name

                set_parents_from_node(child_node, child_category)

        _, root_category = label_categories.find(root_node["LabelName"])
        set_parents_from_node(root_node, root_category)

    def _load_items(self):
        images_dir = osp.join(self._dataset_dir, OpenImagesPath.IMAGES_DIR)

        self._image_paths_by_id = {
            # the first component of `path_parts` is the subset name
            "/".join(path_parts[1:]): path
            for path in find_images(images_dir, recursive=True)
            for path_parts in [
                split_path(
                    osp.splitext(osp.relpath(path, images_dir))[0],
                )
            ]
            if len(path_parts) > 1
        }

        items_by_id = {}

        def load_from(annotation_name):
            with self._open_csv_annotation(annotation_name) as image_reader:
                for image_description in image_reader:
                    image_id = image_description["ImageID"]
                    if image_id in items_by_id:
                        raise RepeatedItemError(item_id=image_id)

                    subset = image_description["Subset"]

                    if _RE_INVALID_PATH_COMPONENT.fullmatch(subset):
                        raise UnsupportedSubsetNameError(item_id=image_id, subset=subset)

                    if image_id in items_by_id:
                        log.warning("Item %s is repeated" % image_id)
                        continue

                    items_by_id[image_id] = self._add_item(image_id, subset)

        # It's preferable to load the combined image description file,
        # because it contains descriptions for training images without human-annotated labels
        # (the file specific to the training set doesn't).
        # However, if it's missing, we'll try loading subset-specific files instead, so that
        # this extractor can be used on individual subsets of the dataset.
        try:
            load_from(OpenImagesPath.FULL_IMAGE_DESCRIPTION_FILE_NAME)
        except FileNotFoundError:
            for pattern in OpenImagesPath.SUBSET_IMAGE_DESCRIPTION_FILE_PATTERNS:
                for path in self._glob_annotations(pattern):
                    load_from(path)

        self._load_labels(items_by_id)
        normalized_coords = self._load_bboxes(items_by_id)
        self._load_masks(items_by_id, normalized_coords)

    def _add_item(self, item_id, subset):
        image_path = self._image_paths_by_id.get(item_id)
        image = None
        if image_path is None:
            log.warning(
                "Can't find image for item: %s. "
                "It should be in the '%s' directory" % (item_id, OpenImagesPath.IMAGES_DIR)
            )
        else:
            image = Image.from_file(path=image_path, size=self._image_meta.get(item_id))

        item = DatasetItem(id=item_id, media=image, subset=subset)
        self._items.append(item)
        return item

    def _get_subset_name(self, filename):
        parts = filename.split("-")
        return parts[1] if parts[0] == "oidv6" else parts[0]

    def _load_labels(self, items_by_id):
        label_categories = self._categories[AnnotationType.label]

        # TODO: implement reading of machine-annotated labels

        for label_path in self._glob_annotations("*" + OpenImagesPath.LABEL_DESCRIPTION_FILE_SUFFIX):
            with self._open_csv_annotation(label_path) as label_reader:
                for label_description in label_reader:
                    image_id = label_description["ImageID"]
                    item = items_by_id.get(image_id)
                    if item is None:
                        item = items_by_id.setdefault(
                            image_id, self._add_item(image_id, self._get_subset_name(label_path))
                        )

                    confidence = float(label_description["Confidence"])

                    label_name = label_description["LabelName"]
                    label_index, _ = label_categories.find(label_name)
                    if label_index is None:
                        raise UndefinedLabel(
                            item_id=item.id,
                            subset=item.subset,
                            label_name=label_name,
                            severity=Severity.error,
                        )
                    item.annotations.append(Label(label=label_index, attributes={"score": confidence}))
                    self._ann_types.add(AnnotationType.label)

    def _load_bboxes(self, items_by_id):
        label_categories = self._categories[AnnotationType.label]

        # OID specifies box coordinates in the normalized form, which we have to
        # convert to the unnormalized form to fit the Datumaro data model.
        # However, we need to temporarily preserve the normalized form as well,
        # because we will need it later to match up box and mask annotations.
        # So we store each box's normalized coordinates in this dictionary.
        normalized_coords = {}

        for bbox_path in self._glob_annotations("*" + OpenImagesPath.BBOX_DESCRIPTION_FILE_SUFFIX):
            with self._open_csv_annotation(bbox_path) as bbox_reader:
                for bbox_description in bbox_reader:
                    image_id = bbox_description["ImageID"]
                    item = items_by_id.get(image_id)
                    if item is None:
                        item = items_by_id.setdefault(
                            image_id, self._add_item(image_id, self._get_subset_name(bbox_path))
                        )

                    label_name = bbox_description["LabelName"]
                    label_index, _ = label_categories.find(label_name)
                    if label_index is None:
                        raise UndefinedLabel(
                            item_id=item.id,
                            subset=item.subset,
                            label_name=label_name,
                            severity=Severity.error,
                        )

                    if item.media and item.media.size is not None:
                        height, width = item.media.size
                    elif self._image_meta.get(item.id):
                        height, width = self._image_meta[item.id]
                    else:
                        log.warning("Can't decode box for item '%s' due to missing image file", item.id)
                        continue

                    x_min_norm, x_max_norm, y_min_norm, y_max_norm = [
                        float(bbox_description[field]) for field in ["XMin", "XMax", "YMin", "YMax"]
                    ]

                    x_min = x_min_norm * width
                    x_max = x_max_norm * width
                    y_min = y_min_norm * height
                    y_max = y_max_norm * height

                    attributes = {
                        "score": float(bbox_description["Confidence"]),
                    }

                    for bool_attr in OpenImagesPath.BBOX_BOOLEAN_ATTRIBUTES:
                        int_value = int(bbox_description[bool_attr.oid_name])
                        if int_value >= 0:
                            attributes[bool_attr.datumaro_name] = bool(int_value)

                    # Give each box within an item a distinct group ID,
                    # so that we can later group them together with the corresponding masks.
                    if item.annotations and item.annotations[-1].type is AnnotationType.bbox:
                        group = item.annotations[-1].group + 1
                    else:
                        group = 1

                    item.annotations.append(
                        Bbox(
                            label=label_index,
                            x=x_min,
                            y=y_min,
                            w=x_max - x_min,
                            h=y_max - y_min,
                            attributes=attributes,
                            group=group,
                        )
                    )
                    self._ann_types.add(AnnotationType.bbox)

                    normalized_coords[id(item.annotations[-1])] = np.array(
                        [x_min_norm, x_max_norm, y_min_norm, y_max_norm]
                    )

        return normalized_coords

    def _load_masks(self, items_by_id, normalized_coords):
        label_categories = self._categories[AnnotationType.label]

        for mask_path in self._glob_annotations("*" + OpenImagesPath.MASK_DESCRIPTION_FILE_SUFFIX):
            with self._open_csv_annotation(mask_path) as mask_reader:
                for mask_description in mask_reader:
                    mask_path = mask_description["MaskPath"]
                    image_id = mask_description["ImageID"]

                    if _RE_INVALID_PATH_COMPONENT.fullmatch(mask_path):
                        raise UnsupportedMaskPathError(item_id=image_id, mask_path=mask_path)

                    item = items_by_id.get(image_id)
                    if item is None:
                        item = items_by_id.setdefault(
                            image_id, self._add_item(image_id, self._get_subset_name(mask_path))
                        )

                    label_name = mask_description["LabelName"]
                    label_index, _ = label_categories.find(label_name)
                    if label_index is None:
                        raise UndefinedLabel(
                            item_id=item.id,
                            subset=item.subset,
                            label_name=label_name,
                            severity=Severity.error,
                        )

                    if item.media and item.media.has_size:
                        image_size = item.media.size
                    elif self._image_meta.get(item.id):
                        image_size = self._image_meta.get(item.id)
                    else:
                        log.warning("Can't decode mask for item '%s' due to missing image file", item.id)
                        continue

                    attributes = {}

                    # The box IDs are rather useless, because the _box_ annotations
                    # don't include them, so they cannot be used to match masks to boxes.
                    # However, it is still desirable to record them, because they are
                    # included in the mask file names, so in order to save each mask to the
                    # file it was loaded from when saving in-places, we need to know
                    # the original box ID.
                    box_id = mask_description["BoxID"]
                    if _RE_INVALID_PATH_COMPONENT.fullmatch(box_id):
                        raise UnsupportedBoxIdError(item_id=item.id, box_id=box_id)
                    attributes["box_id"] = box_id

                    group = 0

                    box_coord_fields = ("BoxXMin", "BoxXMax", "BoxYMin", "BoxYMax")

                    # The original OID has box coordinates for all masks, but
                    # a dataset converted from another dataset might not.
                    if all(mask_description[f] for f in box_coord_fields):
                        # Try to find the box annotation corresponding to the
                        # current mask.
                        mask_box_coords = np.array([float(mask_description[field]) for field in box_coord_fields])

                        for annotation in item.annotations:
                            if (
                                annotation.type is AnnotationType.bbox
                                and annotation.label == label_index
                                and np.allclose(
                                    mask_box_coords,
                                    normalized_coords[id(annotation)],
                                    rtol=0,
                                    atol=1e-6,
                                )
                            ):
                                # In the original OID, mask box coordinates are stored
                                # with 6 digit precision, hence the tolerance.
                                group = annotation.group

                    if mask_description["PredictedIoU"]:
                        attributes["predicted_iou"] = float(mask_description["PredictedIoU"])

                    item.annotations.append(
                        Mask(
                            image=lazy_image(
                                osp.join(
                                    self._dataset_dir,
                                    OpenImagesPath.MASKS_DIR,
                                    item.subset,
                                    mask_path,
                                ),
                                loader=functools.partial(self._load_and_resize_mask, size=image_size),
                            ),
                            label=label_index,
                            attributes=attributes,
                            group=group,
                        )
                    )
                    self._ann_types.add(AnnotationType.mask)

    @staticmethod
    def _load_and_resize_mask(path, size):
        raw = load_image(path, dtype=np.uint8)
        resized = cv2.resize(raw, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        return resized.astype(bool)


class OpenImagesImporter(Importer):
    POSSIBLE_ANNOTATION_PATTERNS = (
        OpenImagesPath.FULL_IMAGE_DESCRIPTION_FILE_NAME,
        *OpenImagesPath.SUBSET_IMAGE_DESCRIPTION_FILE_PATTERNS,
        "*" + OpenImagesPath.LABEL_DESCRIPTION_FILE_SUFFIX,
        "*" + OpenImagesPath.BBOX_DESCRIPTION_FILE_SUFFIX,
        "*" + OpenImagesPath.MASK_DESCRIPTION_FILE_SUFFIX,
    )

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        ann_dirs = [f"{OpenImagesPath.ANNOTATIONS_DIR}/", ""]
        ann_patterns = itertools.product(ann_dirs, cls.POSSIBLE_ANNOTATION_PATTERNS)
        with context.require_any():
            for ann_dir, ann_file in ann_patterns:
                with context.alternative():
                    context.require_file(ann_dir + ann_file)

    @classmethod
    def find_sources(cls, path):
        for pattern in cls.POSSIBLE_ANNOTATION_PATTERNS:
            if glob.glob(osp.join(glob.escape(path), OpenImagesPath.ANNOTATIONS_DIR, pattern)) or glob.glob(
                osp.join(glob.escape(path), pattern)
            ):
                return [{"url": path, "format": OpenImagesBase.NAME}]
        return []

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return list({osp.splitext(p)[1] for p in cls.POSSIBLE_ANNOTATION_PATTERNS})


class _LazyCsvDictWriter:
    """
    For annotation files, we only learn that the file is required after
    we find at least one occurrence of the corresponding annotation type.
    However, it's convenient to create the writer ahead of time, so that
    it can be used in a `with` statement.

    This class behaves like a csv.DictWriter, but it only creates the file
    once the first row is written.
    """

    def __init__(self, writer_manager_factory):
        self._writer_manager_factory = writer_manager_factory
        self._writer_manager = None
        self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._writer_manager:
            self._writer_manager.__exit__(exc_type, exc_value, traceback)

    def writerow(self, rowdict):
        if not self._writer_manager:
            self._writer_manager = self._writer_manager_factory()
            self._writer = self._writer_manager.__enter__()

        self._writer.writerow(rowdict)


class _AnnotationWriter:
    _POSSIBLE_ANNOTATION_FILE_PATTERNS = (
        # class description files don't need to be listed,
        # because they are always written
        OpenImagesPath.FULL_IMAGE_DESCRIPTION_FILE_NAME,
        *OpenImagesPath.SUBSET_IMAGE_DESCRIPTION_FILE_PATTERNS,
        "*" + OpenImagesPath.LABEL_DESCRIPTION_FILE_SUFFIX,
        "*" + OpenImagesPath.BBOX_DESCRIPTION_FILE_SUFFIX,
        "*" + OpenImagesPath.MASK_DESCRIPTION_FILE_SUFFIX,
        DEFAULT_IMAGE_META_FILE_NAME,
    )

    def __init__(self, root_dir):
        self._annotations_dir = osp.join(root_dir, OpenImagesPath.ANNOTATIONS_DIR)
        self._written_annotations = set()

        os.makedirs(self._annotations_dir, exist_ok=True)

    @contextlib.contextmanager
    def open(self, file_name, newline=None):
        self._written_annotations.add(file_name)

        file_path = osp.join(self._annotations_dir, file_name)

        # Write to a temporary file first, to avoid data loss if we're patching
        # an existing dataset and the process is interrupted.
        temp_file_path = file_path + ".tmp"

        with open(
            temp_file_path,
            "w",
            encoding="utf-8",
            newline=newline,
        ) as f:
            yield f

        os.replace(temp_file_path, file_path)

    @contextlib.contextmanager
    def open_csv(self, file_name, field_names, *, write_header=True):
        with self.open(file_name, newline="") as f:
            writer = csv.DictWriter(f, field_names)
            if write_header:
                writer.writeheader()
            yield writer

    def open_csv_lazy(self, file_name, field_names):
        return _LazyCsvDictWriter(lambda: self.open_csv(file_name, field_names))

    def remove_unwritten(self):
        for file_name in os.listdir(self._annotations_dir):
            if file_name not in self._written_annotations and any(
                fnmatch.fnmatch(file_name, pattern) for pattern in self._POSSIBLE_ANNOTATION_FILE_PATTERNS
            ):
                os.unlink(osp.join(self._annotations_dir, file_name))


class OpenImagesExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".jpg"

    def _apply_impl(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        self._save(_AnnotationWriter(self._save_dir))

    @classmethod
    def patch(cls, dataset, patch, save_dir, **options):
        converter = cls(dataset, save_dir, **options)
        annotation_writer = _AnnotationWriter(save_dir)
        converter._save(annotation_writer)
        annotation_writer.remove_unwritten()

        images_dir = osp.join(save_dir, OpenImagesPath.IMAGES_DIR)
        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                continue

            item = DatasetItem(item_id, subset=subset)

            image_path = osp.join(images_dir, converter._make_image_filename(item, subdir=subset))

            if osp.isfile(image_path):
                os.unlink(image_path)

    def _save(self, annotation_writer):
        self._save_categories(annotation_writer)
        self._save_label_category_parents(annotation_writer)
        self._save_subsets(annotation_writer)

    def _save_categories(self, annotation_writer):
        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)

        with annotation_writer.open_csv(
            OpenImagesPath.V5_CLASS_DESCRIPTION_FILE_NAME,
            ["LabelName", "DisplayName"],
            # no header, since we're saving it in the V5 format
            write_header=False,
        ) as class_description_writer:
            for category in self._extractor.categories()[AnnotationType.label]:
                class_description_writer.writerow(
                    {
                        "LabelName": category.name,
                        "DisplayName": category.name,
                    }
                )

    def _save_label_category_parents(self, annotation_writer):
        all_label_names = set()
        hierarchy_nodes = {}
        orphan_nodes = []

        def get_node(name):
            return hierarchy_nodes.setdefault(name, {"LabelName": name})

        for category in self._extractor.categories()[AnnotationType.label]:
            all_label_names.add(category.name)

            child_node = get_node(category.name)

            if category.parent:
                parent_node = get_node(category.parent)
                parent_node.setdefault("Subcategory", []).append(child_node)
            else:
                orphan_nodes.append(child_node)

        # The hierarchy has to be rooted in a single node. However, there's
        # no guarantee that there exists only one orphan (label without a parent).
        # Therefore, we create a fake root node and make it the parent of every
        # orphan label.
        # This is not a violation of the format, because the original OID does
        # the same thing.
        root_node = {
            # Create an OID-like label name that isn't already used by a real label
            "LabelName": next(
                root_name for i in itertools.count() for root_name in [f"/m/{i}"] if root_name not in all_label_names
            ),
            # If an orphan has no children, then it makes no semantic difference
            # whether it's listed in the hierarchy file or not. So strip such nodes
            # to avoid recording meaningless data.
            "Subcategory": [node for node in orphan_nodes if "Subcategory" in node],
        }

        hierarchy_path = osp.join(self._save_dir, OpenImagesPath.ANNOTATIONS_DIR, OpenImagesPath.HIERARCHY_FILE_NAME)

        with annotation_writer.open(hierarchy_path) as hierarchy_file:
            json.dump(root_node, hierarchy_file, indent=4, ensure_ascii=False)
            hierarchy_file.write("\n")

    def _save_subsets(self, annotation_writer):
        label_categories = self._extractor.categories().get(AnnotationType.label, LabelCategories())

        image_meta = {}

        for subset_name, subset in self._extractor.subsets().items():
            if _RE_INVALID_PATH_COMPONENT.fullmatch(subset_name):
                raise UnsupportedSubsetNameError(item_id=next(iter(subset)).id, subset=subset)

            image_description_name = f"{subset_name}-images-with-rotation.csv"

            with (
                annotation_writer.open_csv(
                    image_description_name,
                    OpenImagesPath.IMAGE_DESCRIPTION_FIELDS,
                ) as image_description_writer,
                annotation_writer.open_csv_lazy(
                    subset_name + OpenImagesPath.LABEL_DESCRIPTION_FILE_SUFFIX,
                    OpenImagesPath.LABEL_DESCRIPTION_FIELDS,
                ) as label_description_writer,
                annotation_writer.open_csv_lazy(
                    subset_name + OpenImagesPath.BBOX_DESCRIPTION_FILE_SUFFIX,
                    OpenImagesPath.BBOX_DESCRIPTION_FIELDS,
                ) as bbox_description_writer,
                annotation_writer.open_csv_lazy(
                    subset_name + OpenImagesPath.MASK_DESCRIPTION_FILE_SUFFIX,
                    OpenImagesPath.MASK_DESCRIPTION_FIELDS,
                ) as mask_description_writer,
            ):
                for item in subset:
                    image_description_writer.writerow(
                        {
                            "ImageID": item.id,
                            "Subset": subset_name,
                        }
                    )

                    if self._save_media:
                        if item.media:
                            self._save_image(item, subdir=osp.join(OpenImagesPath.IMAGES_DIR, subset_name))
                        else:
                            log.debug("Item '%s' has no image", item.id)

                    self._save_item_annotations(
                        item,
                        label_description_writer,
                        bbox_description_writer,
                        mask_description_writer,
                        label_categories,
                        image_meta,
                    )

        if image_meta:
            image_meta_file_path = osp.join(
                self._save_dir, OpenImagesPath.ANNOTATIONS_DIR, DEFAULT_IMAGE_META_FILE_NAME
            )

            save_image_meta_file(image_meta, image_meta_file_path)

    def _save_item_annotations(
        self,
        item,
        label_description_writer,
        bbox_description_writer,
        mask_description_writer,
        label_categories,
        image_meta,
    ):
        next_box_id = 0

        existing_box_ids = {
            annotation.attributes["box_id"]
            for annotation in item.annotations
            if annotation.type is AnnotationType.mask
            if "box_id" in annotation.attributes
        }

        for instance in find_instances(item.annotations):
            instance_box = next((a for a in instance if a.type is AnnotationType.bbox), None)

            for annotation in instance:
                if annotation.type is AnnotationType.label:
                    label_description_writer.writerow(
                        {
                            "ImageID": item.id,
                            "LabelName": label_categories[annotation.label].name,
                            "Confidence": str(annotation.attributes.get("score", 1)),
                        }
                    )
                elif annotation.type is AnnotationType.bbox:
                    if item.media and item.media.size is not None:
                        image_meta[item.id] = item.media.size
                        height, width = item.media.size
                    else:
                        log.warning("Can't encode box for item '%s' due to missing image file", item.id)
                        continue

                    bbox_description_writer.writerow(
                        {
                            "ImageID": item.id,
                            "LabelName": label_categories[annotation.label].name,
                            "Confidence": str(annotation.attributes.get("score", 1)),
                            "XMin": annotation.x / width,
                            "YMin": annotation.y / height,
                            "XMax": (annotation.x + annotation.w) / width,
                            "YMax": (annotation.y + annotation.h) / height,
                            **{
                                bool_attr.oid_name: int(annotation.attributes.get(bool_attr.datumaro_name, -1))
                                for bool_attr in OpenImagesPath.BBOX_BOOLEAN_ATTRIBUTES
                            },
                        }
                    )
                elif annotation.type is AnnotationType.mask:
                    mask_dir = osp.join(self._save_dir, OpenImagesPath.MASKS_DIR, item.subset)

                    box_id_str = annotation.attributes.get("box_id")

                    if box_id_str:
                        if _RE_INVALID_PATH_COMPONENT.fullmatch(box_id_str):
                            raise UnsupportedBoxIdError(item_id=item.id, box_id=box_id_str)
                    else:
                        # find a box ID that isn't used in any other annotations
                        while True:
                            box_id_str = format(next_box_id, "08x")
                            next_box_id += 1
                            if box_id_str not in existing_box_ids:
                                break

                    label_name = label_categories[annotation.label].name
                    mask_file_name = "%s_%s_%s.png" % (
                        make_file_name(item.id),
                        make_file_name(label_name),
                        box_id_str,
                    )

                    box_coords = {}

                    if instance_box is not None:
                        if item.media and item.media.size is not None:
                            image_meta[item.id] = item.media.size
                            height, width = item.media.size

                            box_coords = {
                                "BoxXMin": instance_box.x / width,
                                "BoxXMax": (instance_box.x + instance_box.w) / width,
                                "BoxYMin": instance_box.y / height,
                                "BoxYMax": (instance_box.y + instance_box.h) / height,
                            }
                        else:
                            log.warning(
                                "Can't encode box coordinates for a mask for item '%s' due to missing image file",
                                item.id,
                            )

                    mask_description_writer.writerow(
                        {
                            "MaskPath": mask_file_name,
                            "ImageID": item.id,
                            "LabelName": label_name,
                            "BoxID": box_id_str,
                            **box_coords,
                            "PredictedIoU": annotation.attributes.get("predicted_iou", ""),
                        }
                    )

                    save_image(osp.join(mask_dir, mask_file_name), annotation.image, create_dir=True)
