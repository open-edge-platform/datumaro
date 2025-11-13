# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import os
import os.path as osp
import re
from typing import List, Optional

from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util import str_to_bool
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class WiderFacePath:
    IMAGE_EXT = ".jpg"
    ANNOTATIONS_DIR = "wider_face_split"
    IMAGES_DIR = "images"
    SUBSET_DIR = "WIDER_"
    LABELS_FILE = "labels.txt"
    IMAGES_DIR_NO_LABEL = "no_label"
    BBOX_ATTRIBUTES = ["blur", "expression", "illumination", "occluded", "pose", "invalid"]
    DEFAULT_LABEL = "face"


class WiderFaceBase(SubsetBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isfile(path):
            raise FileNotFoundError(errno.ENOENT, "Can't find annotations file", path)
        self._path = path
        self._dataset_dir = osp.dirname(osp.dirname(path))

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
            if re.fullmatch(r"wider_face_\S+((_bbx_gt)|(_filelist))", subset):
                subset = subset.split("_")[2]

        super().__init__(subset=subset, ctx=ctx)

        self._categories = self._load_categories()
        self._items = list(self._load_items(path).values())

    def _load_categories(self):
        label_cat = LabelCategories()
        if has_meta_file(self._dataset_dir):
            labels = parse_meta_file(self._dataset_dir).keys()
            for label in labels:
                label_cat.add(label)
        elif osp.isfile(osp.join(self._dataset_dir, WiderFacePath.LABELS_FILE)):
            path = osp.join(self._dataset_dir, WiderFacePath.LABELS_FILE)
            with open(path, encoding="utf-8") as labels_file:
                for line in labels_file:
                    label_cat.add(line.strip())
        else:
            label_cat.add(WiderFacePath.DEFAULT_LABEL)
            subset_path = osp.join(self._dataset_dir, WiderFacePath.SUBSET_DIR + self._subset, WiderFacePath.IMAGES_DIR)
            if osp.isdir(subset_path):
                for images_dir in sorted(os.listdir(subset_path)):
                    if osp.isdir(osp.join(subset_path, images_dir)) and images_dir != WiderFacePath.IMAGES_DIR_NO_LABEL:
                        if "--" in images_dir:
                            images_dir = images_dir.split("--")[1]
                        if images_dir != WiderFacePath.DEFAULT_LABEL:
                            label_cat.add(images_dir)
            if len(label_cat) == 1:
                label_cat = LabelCategories()
        return {AnnotationType.label: label_cat}

    def _load_items(self, path):
        items = {}
        label_categories = self._categories[AnnotationType.label]

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        line_ids = [
            line_idx for line_idx, line in enumerate(lines) if ("/" in line or "\\" in line) and "." in line
        ]  # a heuristic for paths

        for line_idx in line_ids:
            image_path = lines[line_idx].strip()
            item_id = osp.splitext(image_path)[0]
            item_id = item_id.replace("\\", "/")

            image_path = osp.join(
                self._dataset_dir,
                WiderFacePath.SUBSET_DIR + self._subset,
                WiderFacePath.IMAGES_DIR,
                image_path,
            )

            annotations = []
            if "/" in item_id:
                label_name = item_id.split("/")[0]
                if "--" in label_name:
                    label_name = label_name.split("--")[1]
                if label_name != WiderFacePath.IMAGES_DIR_NO_LABEL:
                    label = label_categories.find(label_name)[0]
                    if label is not None:
                        annotations.append(Label(label=label))
                        self._ann_types.add(AnnotationType.label)
                item_id = item_id[len(item_id.split("/")[0]) + 1 :]

            items[item_id] = DatasetItem(
                id=item_id,
                subset=self._subset,
                media=Image.from_file(path=image_path),
                annotations=annotations,
            )

            try:
                bbox_count = int(lines[line_idx + 1])
            except ValueError:  # can be the next image
                continue
            except IndexError:  # the file can only contain names of images
                continue

            bbox_lines = lines[line_idx + 2 : line_idx + bbox_count + 2]
            for bbox in bbox_lines:
                bbox_list = bbox.split()
                if len(bbox_list) >= 4:
                    label = label_categories.find(WiderFacePath.DEFAULT_LABEL)[0]
                    if len(bbox_list) == 5 or len(bbox_list) == 11:
                        label_name = bbox_list[-1]
                        label = label_categories.find(label_name)[0]
                    if label is None and len(label_categories) == 0:
                        label_categories.add(WiderFacePath.DEFAULT_LABEL)
                        label = label_categories.find(WiderFacePath.DEFAULT_LABEL)[0]

                    attributes = {}
                    if len(bbox_list) >= 10:
                        i = 4
                        for attr in WiderFacePath.BBOX_ATTRIBUTES:
                            if bbox_list[i] != "-":
                                if bbox_list[i] in ["True", "False"]:
                                    attributes[attr] = str_to_bool(bbox_list[i])
                                else:
                                    attributes[attr] = bbox_list[i]
                            i += 1

                    items[item_id].annotations.append(
                        Bbox(
                            float(bbox_list[0]),
                            float(bbox_list[1]),
                            float(bbox_list[2]),
                            float(bbox_list[3]),
                            attributes=attributes,
                            label=label,
                        )
                    )
                    self._ann_types.add(AnnotationType.bbox)

        return items


class WiderFaceImporter(Importer):
    _ANNO_EXT = ".txt"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(f"{WiderFacePath.ANNOTATIONS_DIR}/*{cls._ANNO_EXT}")

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, cls._ANNO_EXT, "wider_face", dirname=WiderFacePath.ANNOTATIONS_DIR)

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]


class WiderFaceExporter(Exporter):
    DEFAULT_IMAGE_EXT = WiderFacePath.IMAGE_EXT

    def _apply_impl(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        save_dir = self._save_dir
        os.makedirs(save_dir, exist_ok=True)

        label_categories = self._extractor.categories()[AnnotationType.label]

        if self._save_dataset_meta:
            self._save_meta_file(save_dir)
        else:
            labels_path = osp.join(save_dir, WiderFacePath.LABELS_FILE)
            with open(labels_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label.name for label in label_categories))

        for subset_name, subset in self._extractor.subsets().items():
            subset_dir = osp.join(save_dir, WiderFacePath.SUBSET_DIR + subset_name)

            wider_annotation = ""
            for item in subset:
                labels = [a.label for a in item.annotations if a.type == AnnotationType.label]
                if labels:
                    image_path = self._make_image_filename(
                        item, subdir="%s--%s" % (labels[0], label_categories[labels[0]].name)
                    )
                else:
                    image_path = self._make_image_filename(item, subdir=WiderFacePath.IMAGES_DIR_NO_LABEL)
                wider_annotation += image_path + "\n"
                if item.media and self._save_media:
                    self._save_image(item, osp.join(save_dir, subset_dir, WiderFacePath.IMAGES_DIR, image_path))

                bboxes = [a for a in item.annotations if a.type == AnnotationType.bbox]
                if len(bboxes) > 0:
                    wider_annotation += "%s\n" % len(bboxes)
                for bbox in bboxes:
                    wider_bb = " ".join("%s" % p for p in bbox.get_bbox())
                    wider_annotation += "%s " % wider_bb
                    if bbox.attributes:
                        wider_attr = ""
                        attr_counter = 0
                        for attr in WiderFacePath.BBOX_ATTRIBUTES:
                            if attr in bbox.attributes:
                                wider_attr += "%s " % bbox.attributes[attr]
                                attr_counter += 1
                            else:
                                wider_attr += "- "
                        if attr_counter > 0:
                            wider_annotation += wider_attr
                    if label_categories[bbox.label].name != WiderFacePath.DEFAULT_LABEL and bbox.label is not None:
                        wider_annotation += "%s" % label_categories[bbox.label].name
                    wider_annotation += "\n"

            annotation_path = osp.join(
                save_dir, WiderFacePath.ANNOTATIONS_DIR, "wider_face_" + subset_name + "_bbx_gt.txt"
            )
            os.makedirs(osp.dirname(annotation_path), exist_ok=True)
            with open(annotation_path, "w", encoding="utf-8") as f:
                f.write(wider_annotation)
