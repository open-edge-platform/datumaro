# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os.path as osp
from typing import Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
from defusedxml import ElementTree

from datumaro.components.annotation import Annotation, AnnotationType, Bbox, CompiledMask, ExtractedMask, Label
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import (
    DatasetImportError,
    InvalidAnnotationError,
    InvalidFieldError,
    MissingFieldError,
    UndeclaredLabelError,
)
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image
from datumaro.util.image import find_images
from datumaro.util.mask_tools import invert_colormap, lazy_mask
from datumaro.util.meta_file_util import has_meta_file

from .format import (
    VocImporterType,
    VocInstColormap,
    VocPath,
    VocTask,
    make_voc_categories,
    parse_label_map,
    parse_meta_file,
)

_inverse_inst_colormap = invert_colormap(VocInstColormap)

T = TypeVar("T")


class VocBase(SubsetBase):
    def __init__(
        self,
        path: str,
        task: Optional[VocTask] = VocTask.voc,
        *,
        subset: Optional[str] = None,
        voc_importer_type: VocImporterType = VocImporterType.default,
        ctx: Optional[ImportContext] = None,
        **kwargs,
    ):
        if not subset:
            subset = osp.splitext(osp.basename(path))[0]

        super().__init__(subset=subset, ctx=ctx)

        if voc_importer_type == VocImporterType.default:
            dataset_dir = osp.dirname(osp.dirname(osp.dirname(path)))
            self._image_dir = osp.join(dataset_dir, VocPath.IMAGES_DIR)
            self._anno_dir = osp.join(dataset_dir, VocPath.ANNOTATIONS_DIR)
            self._mask_dir = osp.join(dataset_dir, VocPath.SEGMENTATION_DIR)
            self._inst_dir = osp.join(dataset_dir, VocPath.INSTANCES_DIR)
        elif voc_importer_type == VocImporterType.roboflow:
            dataset_dir = path
            self._image_dir = dataset_dir
            self._anno_dir = dataset_dir
        else:
            raise DatasetImportError(f"Not supported type: {voc_importer_type}")

        self._path = path
        self._task = task

        self._categories = self._load_categories(dataset_dir)

        if self._task in [VocTask.voc, VocTask.voc_segmentation, VocTask.voc_instance_segmentation]:

            def label_color(label_idx):
                return self._categories[AnnotationType.mask].colormap.get(label_idx, None)

            log.debug(
                "Loaded labels: %s",
                ", ".join(
                    "'%s' %s" % (l.name, ("(%s, %s, %s)" % c) if c else "")
                    for i, l, c in (
                        (i, l, label_color(i)) for i, l in enumerate(self._categories[AnnotationType.label].items)
                    )
                ),
            )
        self._items = {item: None for item in self._load_subset_list(path)}

    def _get_label_id(self, label: str) -> int:
        label_id, _ = self._categories[AnnotationType.label].find(label)
        if label_id is None:
            raise UndeclaredLabelError(label)
        return label_id

    def _load_categories(self, dataset_path):
        label_map = None
        if has_meta_file(dataset_path):
            label_map = parse_meta_file(dataset_path)
        else:
            label_map_path = osp.join(dataset_path, VocPath.LABELMAP_FILE)
            if osp.isfile(label_map_path):
                label_map = parse_label_map(label_map_path)

        return make_voc_categories(label_map, self._task)

    def _load_subset_list(self, subset_path):
        if not osp.isfile(subset_path):
            raise DatasetImportError(f"Can't find txt subset list file at '{subset_path}'")

        subset_list = []
        with open(subset_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line[0] == "#":
                    continue

                if self._task == VocTask.voc_layout:
                    objects = line.split('"')
                    if len(objects) > 1:
                        if len(objects) == 3:
                            line = objects[1]
                        else:
                            raise InvalidAnnotationError(
                                f"{osp.basename(subset_path)}:{i + 1}: "
                                "unexpected number of quotes in filename, expected 0 or 2"
                            )
                    else:
                        line = line.split()[0]
                else:
                    line = line.strip()
                subset_list.append(line)
            return subset_list

    def __iter__(self):
        if osp.isdir(self._image_dir):
            images = {
                osp.splitext(osp.relpath(p, self._image_dir))[0].replace("\\", "/"): p
                for p in find_images(self._image_dir, recursive=True)
            }
        else:
            images = {}

        annotations = self._parse_labels() if self._task in [VocTask.voc, VocTask.voc_classification] else {}

        for item_id in self._ctx.progress_reporter.iter(self._items, desc=f"Importing '{self._subset}'"):
            log.debug("Reading item '%s'" % item_id)
            size = None

            try:
                anns = annotations.get(item_id, [])
                image = None

                ann_file = osp.join(self._anno_dir, item_id + ".xml")
                if osp.isfile(ann_file) and self._task not in [
                    VocTask.voc_classification,
                    VocTask.voc_segmentation,
                ]:
                    root_elem = ElementTree.parse(ann_file).getroot()
                    if root_elem.tag != "annotation":
                        raise MissingFieldError("annotation")

                    height = self._parse_field(root_elem, "size/height", int, required=False)
                    width = self._parse_field(root_elem, "size/width", int, required=False)
                    if height and width:
                        size = (height, width)

                    filename_elem = root_elem.find("filename")
                    if filename_elem is not None:
                        image = osp.join(self._image_dir, filename_elem.text)

                    anns += self._parse_annotations(root_elem, item_id=(item_id, self._subset))

                if self._task in [
                    VocTask.voc,
                    VocTask.voc_segmentation,
                    VocTask.voc_instance_segmentation,
                ]:
                    anns += self._parse_masks(item_id)

                if image is None:
                    image = images.pop(item_id, None)

                if image or size:
                    image = Image.from_file(path=image, size=size)

                yield DatasetItem(id=item_id, subset=self._subset, media=image, annotations=anns)

                for ann in anns:
                    self._ann_types.add(ann.type)

            except ElementTree.ParseError as e:
                readable_wrapper = InvalidAnnotationError("Failed to parse XML file")
                readable_wrapper.__cause__ = e
                self._ctx.error_policy.report_item_error(readable_wrapper, item_id=(item_id, self._subset))
            except Exception as e:
                self._ctx.error_policy.report_item_error(e, item_id=(item_id, self._subset))

    @staticmethod
    def _parse_field(root, xpath: str, cls: Type[T] = str, required: bool = True) -> Optional[T]:
        elem = root.find(xpath)
        if elem is None:
            if required:
                raise MissingFieldError(xpath)
            return None

        if cls is str:
            return elem.text

        try:
            return cls(elem.text)
        except Exception as e:
            raise InvalidFieldError(xpath) from e

    @staticmethod
    def _parse_bool_field(root, xpath: str, default: bool = False) -> Optional[bool]:
        elem = root.find(xpath)
        if elem is None:
            return default

        if elem.text not in ["0", "1"]:
            raise InvalidFieldError(xpath)
        return elem.text == "1"

    def _parse_attribute(self, object_elem):
        attributes = {}

        for key in ["difficult", "truncated", "occluded"]:
            attributes[key] = self._parse_bool_field(object_elem, key, default=False)

        pose_elem = object_elem.find("pose")
        if pose_elem is not None:
            attributes["pose"] = pose_elem.text

        point_elem = object_elem.find("point")
        if point_elem is not None:
            point_x = self._parse_field(point_elem, "x", float)
            point_y = self._parse_field(point_elem, "y", float)
            attributes["point"] = (point_x, point_y)

        attributes_elem = object_elem.find("attributes")
        if attributes_elem is not None:
            for attr_elem in attributes_elem.iter("attribute"):
                attributes[self._parse_field(attr_elem, "name")] = self._parse_field(attr_elem, "value")

        return attributes

    @classmethod
    def _parse_bbox(cls, object_elem):
        bbox_elem = object_elem.find("bndbox")
        if not bbox_elem:
            raise MissingFieldError("bndbox")

        xmin = cls._parse_field(bbox_elem, "xmin", float)
        xmax = cls._parse_field(bbox_elem, "xmax", float)
        ymin = cls._parse_field(bbox_elem, "ymin", float)
        ymax = cls._parse_field(bbox_elem, "ymax", float)
        return [xmin, ymin, xmax - xmin, ymax - ymin]

    def _parse_annotations(self, root_elem, *, item_id: Tuple[str, str]) -> List[Annotation]:
        item_annotations = []

        obj_id = 0
        for object_elem in root_elem.iterfind("object"):
            try:
                label_name = self._parse_field(object_elem, "name")

                # person_layout and action_classification are only available for background and person
                if self._task in [VocTask.voc_layout, VocTask.voc_action] and (
                    label_name not in ["person", "background"]
                ):
                    continue

                obj_label_id = self._get_label_id(label_name)
                obj_bbox = self._parse_bbox(object_elem)
                attributes = self._parse_attribute(object_elem)

                group = obj_id

                if self._task in [VocTask.voc, VocTask.voc_layout]:
                    for part_elem in object_elem.findall("part"):
                        part_label_id = self._get_label_id(self._parse_field(part_elem, "name"))
                        part_bbox = self._parse_bbox(part_elem)

                        item_annotations.append(Bbox(*part_bbox, label=part_label_id, group=group))

                if self._task in [VocTask.voc, VocTask.voc_action]:
                    actions_elem = object_elem.find("actions")
                    actions = {a: False for a in self._categories[AnnotationType.label].items[obj_label_id].attributes}
                    if actions_elem is not None:
                        for action_elem in actions_elem:
                            actions[action_elem.tag] = self._parse_bool_field(actions_elem, action_elem.tag)
                    for action, present in actions.items():
                        attributes[action] = present

                item_annotations.append(
                    Bbox(*obj_bbox, label=obj_label_id, attributes=attributes, id=obj_id, group=group)
                )
                obj_id += 1
            except Exception as e:
                self._ctx.error_policy.report_annotation_error(e, item_id=item_id)

        return item_annotations

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c

    def _parse_masks(self, item_id):
        item_annotations = []

        class_mask = None
        segm_path = osp.join(self._mask_dir, item_id + VocPath.SEGM_EXT)
        if osp.isfile(segm_path):
            inverse_cls_colormap = self._categories[AnnotationType.mask].inverse_colormap
            class_mask = lazy_mask(segm_path, inverse_cls_colormap)

        instances_mask = None
        inst_path = osp.join(self._inst_dir, item_id + VocPath.SEGM_EXT)
        if osp.isfile(inst_path):
            instances_mask = lazy_mask(inst_path, _inverse_inst_colormap)

        label_cat = self._categories[AnnotationType.label]

        if instances_mask is not None:
            compiled_mask = CompiledMask(class_mask, instances_mask)

            if class_mask is not None:
                instance_labels = compiled_mask.get_instance_labels()
            else:
                instance_labels = {i: None for i in range(compiled_mask.instance_count)}

            for instance_id, label_id in instance_labels.items():
                if len(label_cat) <= label_id:
                    self._ctx.error_policy.report_annotation_error(
                        UndeclaredLabelError(str(label_id)), item_id=(item_id, self._subset)
                    )

                item_annotations.append(
                    ExtractedMask(
                        index_mask=instances_mask,
                        index=instance_id,
                        label=label_id,
                        group=instance_id,
                    )
                )
        elif class_mask is not None:
            log.warning("Item %s: only class segmentations available", item_id)

            np_class_mask = class_mask()
            classes = np.unique(np_class_mask)
            for label_id in classes:
                if len(label_cat) <= label_id:
                    self._ctx.error_policy.report_annotation_error(
                        UndeclaredLabelError(str(label_id)), item_id=(item_id, self._subset)
                    )

                item_annotations.append(ExtractedMask(index_mask=class_mask, index=label_id, label=label_id))

        return item_annotations

    def _parse_labels(self) -> Dict[str, List[Label]]:
        annotations = {}
        task_dir = osp.dirname(self._path)
        for label_id, label in enumerate(self._categories[AnnotationType.label]):
            ann_file = osp.join(task_dir, f"{label.name}_{self._subset}.txt")
            if not osp.isfile(ann_file):
                continue

            with open(ann_file, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line[0] == "#":
                        continue

                    parts = line.rsplit(maxsplit=1)
                    if len(parts) != 2:
                        raise InvalidAnnotationError(
                            f"{osp.basename(ann_file)}:{i + 1}: invalid number of fields in line, expected 2"
                        )

                    item, present = parts
                    if present not in ["-1", "0", "1"]:
                        # Both -1 and 0 are used in the original VOC, they mean the same
                        raise InvalidAnnotationError(
                            f"{osp.basename(ann_file)}:{i + 1}: "
                            f"unexpected class existence value '{present}', expected -1, 0 or 1"
                        )

                    if present == "1":
                        annotations.setdefault(item, []).append(Label(label_id))

        return annotations

    @property
    def is_stream(self) -> bool:
        return True


class VocClassificationBase(VocBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, task=VocTask.voc_classification, **kwargs)


class VocDetectionBase(VocBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, task=VocTask.voc_detection, **kwargs)


class VocSegmentationBase(VocBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, task=VocTask.voc_segmentation, **kwargs)


class VocInstanceSegmentationBase(VocBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, task=VocTask.voc_instance_segmentation, **kwargs)


class VocLayoutBase(VocBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, task=VocTask.voc_layout, **kwargs)


class VocActionBase(VocBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, task=VocTask.voc_action, **kwargs)
