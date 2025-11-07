# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import logging as log
import os
import os.path as osp
from collections import defaultdict
from glob import glob, iglob
from typing import List, Optional

import numpy as np
from defusedxml import ElementTree

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories, Mask, Polygon
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util import cast, escape, unescape
from datumaro.util.image import save_image
from datumaro.util.mask_tools import find_mask_bbox, load_mask
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class LabelMePath:
    IMAGES_DIR = "Images"
    ANNOTATIONS_DIR = "Annotations"
    MASKS_DIR = "Masks"
    IMAGE_EXT = ".jpg"

    ATTR_IMPORT_ESCAPES = [
        ("\\=", r"%%{eq}%%"),
        ('\\"', r"%%{doublequote}%%"),
        ("\\,", r"%%{comma}%%"),
        ("\\\\", r"%%{backslash}%%"),  # keep last
    ]
    ATTR_EXPORT_ESCAPES = [
        ("\\", "\\\\"),  # keep first
        ("=", "\\="),
        ('"', '\\"'),
        (",", "\\,"),
    ]


class LabelMeBase(DatasetBase):
    def __init__(self, path: str, *, ctx: Optional[ImportContext] = None):
        assert osp.isdir(path), path
        super().__init__(ctx=ctx)

        self._items, self._categories, self._subsets = self._parse(path)
        self._length = len(self._items)

    def _parse(self, path):
        items = []
        subsets = set()

        if has_meta_file(path):
            categories = {
                AnnotationType.label: LabelCategories(attributes={"occluded", "username"}).from_iterable(
                    parse_meta_file(path).keys()
                )
            }
        else:
            categories = {AnnotationType.label: LabelCategories(attributes={"occluded", "username"})}

        for xml_path in sorted(glob(osp.join(path, "**", "*.xml"), recursive=True)):
            root = ElementTree.parse(xml_path)

            subset = root.find("folder").text or ""
            item_id = osp.splitext(root.find("filename").text)[0]
            image_path = osp.join(path, "Images", subset, root.find("filename").text)

            image_size = None
            imagesize_elem = root.find("imagesize")
            if imagesize_elem is not None:
                width_elem = imagesize_elem.find("ncols").text
                height_elem = imagesize_elem.find("nrows").text
                image_size = (int(height_elem), int(width_elem)) if height_elem and width_elem else None

            image = Image.from_file(path=image_path, size=image_size)

            annotations = self._parse_annotations(root, path, subset, categories)

            for ann in annotations:
                self._ann_types.add(ann.type)

            items.append(DatasetItem(id=item_id, subset=subset, media=image, annotations=annotations))
            subsets.add(items[-1].subset)
        return items, categories, subsets

    @staticmethod
    def _escape(s):
        return escape(s, LabelMePath.ATTR_IMPORT_ESCAPES)

    @staticmethod
    def _unescape(s):
        s = unescape(s, LabelMePath.ATTR_IMPORT_ESCAPES)
        s = unescape(s, LabelMePath.ATTR_EXPORT_ESCAPES)
        return s

    @classmethod
    def _parse_annotations(cls, xml_root, path, subset, categories):
        def _parse_attributes(attr_str):
            parsed = []
            if not attr_str:
                return parsed

            for attr in [a.strip() for a in cls._escape(attr_str).split(",")]:
                if not attr:
                    continue

                if "=" in attr:
                    name, value = attr.split("=", maxsplit=1)
                    if value.lower() in {"true", "false"}:
                        value = value.lower() == "true"
                    elif len(value) > 1 and value[0] == '"' and value[-1] == '"':
                        value = value[1:-1]
                    else:
                        for t in [int, float]:
                            casted = cast(value, t)
                            if casted is not None and str(casted) == value:
                                value = casted
                                break
                    if isinstance(value, str):
                        value = cls._unescape(value)
                    parsed.append((cls._unescape(name), value))
                else:
                    parsed.append((cls._unescape(attr), True))

            return parsed

        label_cat = categories[AnnotationType.label]

        def _get_label_id(label):
            if not label:
                return None
            idx, _ = label_cat.find(label)
            if idx is None:
                idx = label_cat.add(label)
            return idx

        image_annotations = []

        parsed_annotations = dict()
        group_assignments = dict()
        root_annotations = set()
        for obj_elem in xml_root.iter("object"):
            obj_id = int(obj_elem.find("id").text)

            ann_items = []

            label = _get_label_id(obj_elem.find("name").text)

            attributes = []
            attributes_elem = obj_elem.find("attributes")
            if attributes_elem is not None and attributes_elem.text:
                attributes = _parse_attributes(attributes_elem.text)

            occluded = False
            occluded_elem = obj_elem.find("occluded")
            if occluded_elem is not None and occluded_elem.text:
                occluded = occluded_elem.text == "yes"
            attributes.append(("occluded", occluded))

            deleted = False
            deleted_elem = obj_elem.find("deleted")
            if deleted_elem is not None and deleted_elem.text:
                deleted = bool(int(deleted_elem.text))

            user = ""

            poly_elem = obj_elem.find("polygon")
            segm_elem = obj_elem.find("segm")
            type_elem = obj_elem.find("type")  # the only value is 'bounding_box'
            if poly_elem is not None:
                user_elem = poly_elem.find("username")
                if user_elem is not None and user_elem.text:
                    user = user_elem.text
                attributes.append(("username", user))

                points = []
                for point_elem in poly_elem.iter("pt"):
                    x = float(point_elem.find("x").text)
                    y = float(point_elem.find("y").text)
                    points.append(x)
                    points.append(y)

                if type_elem is not None and type_elem.text == "bounding_box":
                    xmin = min(points[::2])
                    xmax = max(points[::2])
                    ymin = min(points[1::2])
                    ymax = max(points[1::2])
                    ann_items.append(
                        Bbox(
                            xmin,
                            ymin,
                            xmax - xmin,
                            ymax - ymin,
                            label=label,
                            attributes=attributes,
                            id=obj_id,
                        )
                    )
                else:
                    ann_items.append(
                        Polygon(
                            points,
                            label=label,
                            attributes=attributes,
                            id=obj_id,
                        )
                    )
            elif segm_elem is not None:
                user_elem = segm_elem.find("username")
                if user_elem is not None and user_elem.text:
                    user = user_elem.text
                attributes.append(("username", user))

                mask_path = osp.join(path, LabelMePath.MASKS_DIR, subset, segm_elem.find("mask").text)
                if not osp.isfile(mask_path):
                    raise FileNotFoundError(errno.ENOENT, "Can't find mask", mask_path)
                mask = load_mask(mask_path)
                mask = np.any(mask, axis=2)
                ann_items.append(Mask(image=mask, label=label, id=obj_id, attributes=attributes))

            if not deleted:
                parsed_annotations[obj_id] = ann_items

            # Find parents and children
            parts_elem = obj_elem.find("parts")
            if parts_elem is not None:
                children_ids = []
                hasparts_elem = parts_elem.find("hasparts")
                if hasparts_elem is not None and hasparts_elem.text:
                    children_ids = [int(c) for c in hasparts_elem.text.split(",")]

                parent_ids = []
                ispartof_elem = parts_elem.find("ispartof")
                if ispartof_elem is not None and ispartof_elem.text:
                    parent_ids = [int(c) for c in ispartof_elem.text.split(",")]

                if children_ids and not parent_ids and hasparts_elem.text:
                    root_annotations.add(obj_id)
                group_assignments[obj_id] = [None, children_ids]

        # assign single group to all grouped annotations
        current_group_id = 0
        annotations_to_visit = list(root_annotations)
        while annotations_to_visit:
            ann_id = annotations_to_visit.pop()
            ann_assignment = group_assignments[ann_id]
            group_id, children_ids = ann_assignment
            if group_id:
                continue

            if ann_id in root_annotations:
                current_group_id += 1  # start a new group

            group_id = current_group_id
            ann_assignment[0] = group_id

            # continue with children
            annotations_to_visit.extend(children_ids)

        assert current_group_id == len(root_annotations)

        for ann_id, ann_items in parsed_annotations.items():
            group_id = 0
            if ann_id in group_assignments:
                ann_assignment = group_assignments[ann_id]
                group_id = ann_assignment[0]

            for ann_item in ann_items:
                if group_id:
                    ann_item.group = group_id

                image_annotations.append(ann_item)

        return image_annotations

    def categories(self):
        return self._categories

    def __iter__(self):
        yield from self._items


class LabelMeImporter(Importer):
    _ANNO_EXT = ".xml"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        annot_paths = context.require_files(f"**/*{cls._ANNO_EXT}")

        for annot_path in annot_paths:
            with context.probe_text_file(
                annot_path,
                "must be a LabelMe annotation file",
            ) as f:
                elem_parents = []

                for event, elem in ElementTree.iterparse(f, events=("start", "end")):
                    if event == "start":
                        if elem_parents == [] and elem.tag != "annotation":
                            raise Exception

                        if elem_parents == ["annotation", "object"] and elem.tag in {
                            "polygon",
                            "segm",
                        }:
                            return

                        elem_parents.append(elem.tag)
                    elif event == "end":
                        elem_parents.pop()

                        if elem_parents == ["annotation"] and elem.tag == "object":
                            # If we got here, then we found an object with no
                            # polygon and no mask, so it's probably the wrong
                            # format.
                            raise Exception

            # If we got here, then the current file has no objects and is thus
            # ambiguous - it could be ours or it could be from the VOC format.
            # We'll proceed to test the next one.

        # If we got here, then every file was ambiguous. We'll have to
        # (implicitly) return a match.

    @classmethod
    def find_sources(cls, path):
        subsets = []
        if not osp.isdir(path):
            return []

        try:
            next(iglob(osp.join(path, "**", "*.xml"), recursive=True))
            subsets.append(
                {
                    "url": osp.normpath(path),
                    "format": LabelMeBase.NAME,
                }
            )
        except StopIteration:
            pass
        return subsets

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]


class LabelMeExporter(Exporter):
    DEFAULT_IMAGE_EXT = LabelMePath.IMAGE_EXT

    def _apply_impl(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(self._save_dir, exist_ok=True)

        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)

        for subset_name, subset in self._extractor.subsets().items():
            for item in subset:
                self._save_item(item, subset_name)

    def _get_label(self, label_id):
        if label_id is None:
            return ""
        return self._extractor.categories()[AnnotationType.label][label_id].name

    @staticmethod
    def _escape(s: str):
        return escape(s, escapes=LabelMePath.ATTR_EXPORT_ESCAPES)

    def _save_item(self, item, subset_dir):
        # Disable B410: import_lxml - the library is used for writing here
        from lxml import etree as ET  # nosec

        log.debug("Converting item '%s'", item.id)

        image_filename = self._make_image_filename(item)
        if self._save_media:
            if item.media and item.media.has_data:
                image_dir = osp.join(self._save_dir, LabelMePath.IMAGES_DIR, subset_dir)
                os.makedirs(image_dir, exist_ok=True)
                self._save_image(item, osp.join(image_dir, image_filename))
            else:
                log.debug("Item '%s' has no image", item.id)

        root_elem = ET.Element("annotation")
        ET.SubElement(root_elem, "filename").text = image_filename
        ET.SubElement(root_elem, "folder").text = subset_dir

        source_elem = ET.SubElement(root_elem, "source")
        ET.SubElement(source_elem, "sourceImage").text = ""
        ET.SubElement(source_elem, "sourceAnnotation").text = "Datumaro"

        if item.media:
            image_elem = ET.SubElement(root_elem, "imagesize")
            image_size = item.media.size
            ET.SubElement(image_elem, "nrows").text = str(image_size[0])
            ET.SubElement(image_elem, "ncols").text = str(image_size[1])

        groups = defaultdict(list)

        obj_id = 0
        for ann in item.annotations:
            if ann.type not in {AnnotationType.polygon, AnnotationType.bbox, AnnotationType.mask}:
                continue

            obj_elem = ET.SubElement(root_elem, "object")
            ET.SubElement(obj_elem, "name").text = self._get_label(ann.label)
            ET.SubElement(obj_elem, "deleted").text = "0"
            ET.SubElement(obj_elem, "verified").text = "0"
            ET.SubElement(obj_elem, "occluded").text = "yes" if ann.attributes.get("occluded") is True else "no"
            ET.SubElement(obj_elem, "date").text = ""
            ET.SubElement(obj_elem, "id").text = str(obj_id)

            parts_elem = ET.SubElement(obj_elem, "parts")
            if ann.group:
                groups[ann.group].append((obj_id, parts_elem))
            else:
                ET.SubElement(parts_elem, "hasparts").text = ""
                ET.SubElement(parts_elem, "ispartof").text = ""

            if ann.type == AnnotationType.bbox:
                ET.SubElement(obj_elem, "type").text = "bounding_box"

                poly_elem = ET.SubElement(obj_elem, "polygon")
                x0, y0, x1, y1 = ann.points
                points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                for x, y in points:
                    point_elem = ET.SubElement(poly_elem, "pt")
                    ET.SubElement(point_elem, "x").text = "%.2f" % x
                    ET.SubElement(point_elem, "y").text = "%.2f" % y

                ET.SubElement(poly_elem, "username").text = str(ann.attributes.get("username", ""))
            elif ann.type == AnnotationType.polygon:
                poly_elem = ET.SubElement(obj_elem, "polygon")
                for x, y in zip(ann.points[::2], ann.points[1::2]):
                    point_elem = ET.SubElement(poly_elem, "pt")
                    ET.SubElement(point_elem, "x").text = "%.2f" % x
                    ET.SubElement(point_elem, "y").text = "%.2f" % y

                ET.SubElement(poly_elem, "username").text = str(ann.attributes.get("username", ""))
            elif ann.type == AnnotationType.mask:
                mask_filename = "%s_mask_%s.png" % (item.id, obj_id)
                save_image(
                    osp.join(self._save_dir, LabelMePath.MASKS_DIR, subset_dir, mask_filename),
                    self._paint_mask(ann.image),
                    create_dir=True,
                )

                segm_elem = ET.SubElement(obj_elem, "segm")
                ET.SubElement(segm_elem, "mask").text = mask_filename

                bbox = find_mask_bbox(ann.image)
                box_elem = ET.SubElement(segm_elem, "box")
                ET.SubElement(box_elem, "xmin").text = "%.2f" % bbox[0]
                ET.SubElement(box_elem, "ymin").text = "%.2f" % bbox[1]
                ET.SubElement(box_elem, "xmax").text = "%.2f" % (bbox[0] + bbox[2])
                ET.SubElement(box_elem, "ymax").text = "%.2f" % (bbox[1] + bbox[3])

                ET.SubElement(segm_elem, "username").text = str(ann.attributes.get("username", ""))
            else:
                raise NotImplementedError("Unknown shape type '%s'" % ann.type)

            attrs = []
            for k, v in ann.attributes.items():
                if k in {"username", "occluded"}:
                    continue
                if isinstance(v, str):
                    if (cast(v, float) is not None and str(float(v)) == v) or (
                        cast(v, int) is not None and str(int(v)) == v
                    ):
                        v = f'"{v}"'  # add escaping for string values
                    else:
                        v = self._escape(v)
                attrs.append("%s=%s" % (self._escape(k), v))
            ET.SubElement(obj_elem, "attributes").text = ", ".join(attrs)

            obj_id += 1

        for _, group in groups.items():
            leader_id, leader_parts_elem = group[0]
            leader_parts = [str(o_id) for o_id, _ in group[1:]]
            ET.SubElement(leader_parts_elem, "hasparts").text = ",".join(leader_parts)
            ET.SubElement(leader_parts_elem, "ispartof").text = ""

            for obj_id, parts_elem in group[1:]:
                ET.SubElement(parts_elem, "hasparts").text = ""
                ET.SubElement(parts_elem, "ispartof").text = str(leader_id)

        ann_path = osp.join(self._save_dir, LabelMePath.ANNOTATIONS_DIR, subset_dir)
        os.makedirs(osp.join(ann_path, osp.dirname(image_filename)), exist_ok=True)
        xml_path = osp.join(ann_path, osp.splitext(image_filename)[0] + ".xml")
        if osp.exists(xml_path):
            xml_path = osp.join(ann_path, image_filename + ".xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            xml_data = ET.tostring(root_elem, encoding="unicode", pretty_print=True)
            f.write(xml_data)

    @staticmethod
    def _paint_mask(mask):
        # TODO: check if mask colors are random
        return np.array([[0, 0, 0, 0], [255, 203, 0, 153]], dtype=np.uint8)[mask.astype(np.uint8)]
