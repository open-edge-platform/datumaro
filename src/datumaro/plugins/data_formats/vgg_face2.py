# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import csv
import errno
import os
import os.path as osp
from typing import List, Optional

from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories, Points
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.errors import AnnotationExportError, DatasetImportError, InvalidAnnotationError, MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util.image import find_images
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class VggFace2Path:
    ANNOTATION_DIR = "bb_landmark"
    IMAGE_EXT = ".jpg"
    BBOXES_FILE = "loose_bb_"
    LANDMARKS_FILE = "loose_landmark_"
    LABELS_FILE = "labels.txt"
    IMAGES_DIR_NO_LABEL = "no_label"


class VggFace2Base(DatasetBase):
    def __init__(self, path: str, *, ctx: Optional[ImportContext] = None):
        subset = None
        if osp.isdir(path):
            self._path = path
        elif osp.isfile(path):
            subset = osp.splitext(osp.basename(path).split("_")[2])[0]
            self._path = osp.dirname(path)
        else:
            raise DatasetImportError("Can't read annotations from '%s'" % path)

        annotation_files = [
            p
            for p in os.listdir(self._path)
            if (
                osp.basename(p).startswith(VggFace2Path.BBOXES_FILE)
                or osp.basename(p).startswith(VggFace2Path.LANDMARKS_FILE)
            )
            and p.endswith(".csv")
        ]

        if len(annotation_files) < 1:
            raise FileNotFoundError(errno.ENOENT, "Can't find annotations in the directory", path)

        super().__init__(ctx=ctx)

        self._dataset_dir = osp.dirname(self._path)
        self._subsets = {subset} if subset else set(osp.splitext(f.split("_")[2])[0] for f in annotation_files)

        self._categories = {}
        self._items = []

        self._load_categories()
        for subset in self._subsets:
            self._items.extend(list(self._load_items(subset).values()))

    def __iter__(self):
        return iter(self._items)

    def categories(self):
        return self._categories

    def _load_categories(self):
        label_cat = LabelCategories()
        path = osp.join(self._dataset_dir, VggFace2Path.LABELS_FILE)
        if has_meta_file(self._dataset_dir):
            labels = parse_meta_file(self._dataset_dir).keys()
            for label in labels:
                label_cat.add(label)
        elif osp.isfile(path):
            with open(path, encoding="utf-8") as labels_file:
                lines = [s.strip() for s in labels_file]
            for line in lines:
                objects = line.split()
                label = objects[0]
                class_name = None
                if len(objects) > 1:
                    class_name = objects[1]
                label_cat.add(label, parent=class_name)
        else:
            for subset in self._subsets:
                subset_path = osp.join(self._dataset_dir, subset)
                if osp.isdir(subset_path):
                    for images_dir in sorted(os.listdir(subset_path)):
                        if (
                            osp.isdir(osp.join(subset_path, images_dir))
                            and images_dir != VggFace2Path.IMAGES_DIR_NO_LABEL
                        ):
                            label_cat.add(images_dir)
        self._categories[AnnotationType.label] = label_cat

    def _load_items(self, subset):
        def _get_label(path):
            label_name = path.split("/")[0]
            label = None
            if label_name != VggFace2Path.IMAGES_DIR_NO_LABEL:
                label = self._categories[AnnotationType.label].find(label_name)[0]
            return label

        items = {}

        image_dir = osp.join(self._dataset_dir, subset)
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace("\\", "/"): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        landmarks_path = osp.join(
            self._dataset_dir,
            VggFace2Path.ANNOTATION_DIR,
            VggFace2Path.LANDMARKS_FILE + subset + ".csv",
        )
        if osp.isfile(landmarks_path):
            with open(landmarks_path, encoding="utf-8") as content:
                landmarks_table = list(csv.DictReader(content))
            for row in landmarks_table:
                item_id = row["NAME_ID"]
                label = None
                if "/" in item_id:
                    label = _get_label(item_id)

                if item_id not in items:
                    image = images.get(row["NAME_ID"])
                    if image:
                        image = Image.from_file(path=image)
                    items[item_id] = DatasetItem(id=item_id, subset=subset, media=image)

                annotations = items[item_id].annotations
                if [a for a in annotations if a.type == AnnotationType.points]:
                    raise InvalidAnnotationError("Item %s: an image can have only one set of landmarks" % item_id)

                if len([p for p in row if row[p] == ""]) == 0 and len(row) == 11:
                    annotations.append(Points([float(row[p]) for p in row if p != "NAME_ID"], label=label))
                    self._ann_types.add(AnnotationType.points)
                elif label is not None:
                    annotations.append(Label(label=label))
                    self._ann_types.add(AnnotationType.label)

        bboxes_path = osp.join(
            self._dataset_dir,
            VggFace2Path.ANNOTATION_DIR,
            VggFace2Path.BBOXES_FILE + subset + ".csv",
        )
        if osp.isfile(bboxes_path):
            with open(bboxes_path, encoding="utf-8") as content:
                bboxes_table = list(csv.DictReader(content))
            for row in bboxes_table:
                item_id = row["NAME_ID"]
                label = None
                if "/" in item_id:
                    label = _get_label(item_id)

                if item_id not in items:
                    image = images.get(row["NAME_ID"])
                    if image:
                        image = Image.from_file(path=image)
                    items[item_id] = DatasetItem(id=item_id, subset=subset, media=image)

                annotations = items[item_id].annotations
                if [a for a in annotations if a.type == AnnotationType.bbox]:
                    raise InvalidAnnotationError("Item %s: an image can have only one bbox" % item_id)

                if len([p for p in row if row[p] == ""]) == 0 and len(row) == 5:
                    annotations.append(
                        Bbox(
                            float(row["X"]),
                            float(row["Y"]),
                            float(row["W"]),
                            float(row["H"]),
                            label=label,
                        )
                    )
                    self._ann_types.add(AnnotationType.bbox)
        return items


class VggFace2Importer(Importer):
    _ANNO_EXT = ".csv"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        with context.require_any():
            for prefix in (VggFace2Path.BBOXES_FILE, VggFace2Path.LANDMARKS_FILE):
                with context.alternative():
                    context.require_file(f"{VggFace2Path.ANNOTATION_DIR}/{prefix}*{cls._ANNO_EXT}")

    @classmethod
    def find_sources(cls, path):
        if osp.isdir(path):
            annotation_dir = osp.join(path, VggFace2Path.ANNOTATION_DIR)
            if osp.isdir(annotation_dir):
                return [
                    {
                        "url": annotation_dir,
                        "format": VggFace2Base.NAME,
                    }
                ]
        elif osp.isfile(path):
            if (
                osp.basename(path).startswith(VggFace2Path.LANDMARKS_FILE)
                or osp.basename(path).startswith(VggFace2Path.BBOXES_FILE)
            ) and path.endswith(".csv"):
                return [
                    {
                        "url": path,
                        "format": VggFace2Base.NAME,
                    }
                ]
        return []

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]


class VggFace2Exporter(Exporter):
    DEFAULT_IMAGE_EXT = VggFace2Path.IMAGE_EXT

    def _apply_impl(self):
        def _get_name_id(item_parts, label_name):
            if len(item_parts) > 1 and item_parts[0] == label_name:
                return "/".join([label_name, *item_parts[1:]])
            return "/".join([label_name, *item_parts])

        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        save_dir = self._save_dir
        os.makedirs(save_dir, exist_ok=True)

        if self._save_dataset_meta:
            self._save_meta_file(save_dir)
        else:
            labels_path = osp.join(save_dir, VggFace2Path.LABELS_FILE)
            labels_file = ""
            for label in self._extractor.categories()[AnnotationType.label]:
                labels_file += "%s" % label.name
                if label.parent:
                    labels_file += " %s" % label.parent
                labels_file += "\n"
            with open(labels_path, "w", encoding="utf-8") as f:
                f.write(labels_file)

        label_categories = self._extractor.categories()[AnnotationType.label]

        for subset_name, subset in self._extractor.subsets().items():
            bboxes_table = []
            landmarks_table = []
            for item in subset:
                item_parts = item.id.split("/")
                if item.media and self._save_media:
                    labels = set(p.label for p in item.annotations if (getattr(p, "label", None) is not None))
                    if labels:
                        for label in labels:
                            image_dir = label_categories[label].name
                            if len(item_parts) > 1 and image_dir == item_parts[0]:
                                image_dir = ""
                            self._save_image(item, subdir=osp.join(subset_name, image_dir))
                    else:
                        image_dir = VggFace2Path.IMAGES_DIR_NO_LABEL
                        if len(item_parts) > 1 and image_dir == item_parts[0]:
                            image_dir = ""
                        self._save_image(item, subdir=osp.join(subset_name, image_dir))

                landmarks = [a for a in item.annotations if a.type == AnnotationType.points]
                if len(landmarks) > 1:
                    raise AnnotationExportError(
                        "Item (%s, %s): an image can have only one set of landmarks" % (item.id, item.subset)
                    )
                if landmarks:
                    if landmarks[0].label is not None and label_categories[landmarks[0].label].name:
                        name_id = _get_name_id(item_parts, label_categories[landmarks[0].label].name)
                    else:
                        name_id = _get_name_id(item_parts, VggFace2Path.IMAGES_DIR_NO_LABEL)
                    points = landmarks[0].points
                    if len(points) != 10:
                        landmarks_table.append({"NAME_ID": name_id})
                    else:
                        landmarks_table.append(
                            {
                                "NAME_ID": name_id,
                                "P1X": points[0],
                                "P1Y": points[1],
                                "P2X": points[2],
                                "P2Y": points[3],
                                "P3X": points[4],
                                "P3Y": points[5],
                                "P4X": points[6],
                                "P4Y": points[7],
                                "P5X": points[8],
                                "P5Y": points[9],
                            }
                        )

                bboxes = [a for a in item.annotations if a.type == AnnotationType.bbox]
                if len(bboxes) > 1:
                    raise AnnotationExportError(
                        "Item (%s, %s): an image can have only one bbox" % (item.id, item.subset)
                    )
                if bboxes:
                    if bboxes[0].label is not None and label_categories[bboxes[0].label].name:
                        name_id = _get_name_id(item_parts, label_categories[bboxes[0].label].name)
                    else:
                        name_id = _get_name_id(item_parts, VggFace2Path.IMAGES_DIR_NO_LABEL)
                    bboxes_table.append(
                        {
                            "NAME_ID": name_id,
                            "X": bboxes[0].x,
                            "Y": bboxes[0].y,
                            "W": bboxes[0].w,
                            "H": bboxes[0].h,
                        }
                    )

                labels = [a for a in item.annotations if a.type == AnnotationType.label]
                for label in labels:
                    if label.label is not None and label_categories[label.label].name:
                        name_id = _get_name_id(item_parts, label_categories[labels[0].label].name)
                    else:
                        name_id = _get_name_id(item_parts, VggFace2Path.IMAGES_DIR_NO_LABEL)
                    landmarks_table.append({"NAME_ID": name_id})

                if not landmarks and not bboxes and not labels:
                    landmarks_table.append({"NAME_ID": _get_name_id(item_parts, VggFace2Path.IMAGES_DIR_NO_LABEL)})

            landmarks_path = osp.join(
                save_dir,
                VggFace2Path.ANNOTATION_DIR,
                VggFace2Path.LANDMARKS_FILE + subset_name + ".csv",
            )
            os.makedirs(osp.dirname(landmarks_path), exist_ok=True)
            with open(landmarks_path, "w", encoding="utf-8", newline="") as file:
                columns = [
                    "NAME_ID",
                    "P1X",
                    "P1Y",
                    "P2X",
                    "P2Y",
                    "P3X",
                    "P3Y",
                    "P4X",
                    "P4Y",
                    "P5X",
                    "P5Y",
                ]
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(landmarks_table)

            if bboxes_table:
                bboxes_path = osp.join(
                    save_dir,
                    VggFace2Path.ANNOTATION_DIR,
                    VggFace2Path.BBOXES_FILE + subset_name + ".csv",
                )
                os.makedirs(osp.dirname(bboxes_path), exist_ok=True)
                with open(bboxes_path, "w", encoding="utf-8", newline="") as file:
                    columns = ["NAME_ID", "X", "Y", "W", "H"]
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(bboxes_table)
