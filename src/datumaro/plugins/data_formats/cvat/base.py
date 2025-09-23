# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional

import numpy as np
from defusedxml import ElementTree

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Label,
    LabelCategories,
    Mask,
    Points,
    Polygon,
    PolyLine,
    RotatedBbox,
)
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import DatasetImportError
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util import mask_tools

from .format import CvatPath


def _find_meta_root(path: str):
    context = ElementTree.iterparse(path, events=("start", "end"))
    context = iter(context)

    meta_root = None

    for event, elem in context:
        if elem.tag == "meta" and event == "start":
            meta_root = elem
        elif elem.tag == "meta" and event == "end":
            break

    if meta_root is None:
        raise DatasetImportError("CVAT XML file should have <meta> tag.")

    return meta_root, context


class CvatBase(SubsetBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        assert osp.isfile(path), path
        rootpath = osp.dirname(path)
        images_dir = ""
        if osp.isdir(osp.join(rootpath, CvatPath.IMAGES_DIR)):
            images_dir = osp.join(rootpath, CvatPath.IMAGES_DIR)
        self._images_dir = images_dir
        self._path = path

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
        super().__init__(subset=subset, ctx=ctx)

        items, categories = self._parse(path)
        self._categories = categories
        self._items = list(self._load_items(items).values())

    def _parse(self, path):
        meta_root, context = _find_meta_root(path)

        categories, frame_size, attribute_types = self._parse_meta(meta_root)

        items = OrderedDict()

        track = None
        shape = None
        tag = None
        attributes = None
        image = None
        subset = None
        for ev, el in context:
            if ev == "start":
                if el.tag == "track":
                    track = {
                        "id": el.attrib["id"],
                        "label": el.attrib.get("label"),
                        "group": int(el.attrib.get("group_id", 0)),
                        "height": frame_size[0],
                        "width": frame_size[1],
                    }
                    subset = el.attrib.get("subset")
                elif el.tag == "image":
                    image = {
                        "name": el.attrib.get("name"),
                        "frame": el.attrib["id"],
                        "width": el.attrib.get("width"),
                        "height": el.attrib.get("height"),
                    }
                    subset = el.attrib.get("subset")
                elif el.tag in CvatPath.SUPPORTED_IMPORT_SHAPES and (track or image):
                    attributes = {}
                    shape = {
                        "type": None,
                        "attributes": attributes,
                    }
                    if track:
                        shape.update(track)
                        shape["track_id"] = int(track["id"])
                    if image:
                        shape.update(image)
                elif el.tag == "tag" and image:
                    attributes = {}
                    tag = {
                        "frame": image["frame"],
                        "attributes": attributes,
                        "group": int(el.attrib.get("group_id", 0)),
                        "label": el.attrib["label"],
                    }
            elif ev == "end":
                if el.tag == "attribute" and attributes is not None:
                    attr_value = el.text or ""
                    attr_type = attribute_types.get(el.attrib["name"])
                    if el.text in ["true", "false"]:
                        attr_value = attr_value == "true"
                    elif attr_type is not None and attr_type != "text":
                        try:
                            attr_value = float(attr_value)
                        except ValueError:
                            pass
                    attributes[el.attrib["name"]] = attr_value
                elif el.tag in CvatPath.SUPPORTED_IMPORT_SHAPES:
                    if track is not None:
                        shape["frame"] = el.attrib["frame"]
                        shape["outside"] = el.attrib.get("outside") == "1"
                        shape["keyframe"] = el.attrib.get("keyframe") == "1"
                    if image is not None:
                        shape["label"] = el.attrib.get("label")
                        shape["group"] = int(el.attrib.get("group_id", 0))

                    shape["type"] = el.tag
                    shape["occluded"] = el.attrib.get("occluded") == "1"
                    shape["z_order"] = int(el.attrib.get("z_order", 0))

                    if el.tag == "box":
                        # Get box coordinates
                        shape["points"] = list(
                            map(
                                float,
                                [
                                    el.attrib["xtl"],
                                    el.attrib["ytl"],
                                    el.attrib["xbr"],
                                    el.attrib["ybr"],
                                ],
                            )
                        )

                        # Check if rotation is present
                        if "rotation" in el.attrib:
                            shape["rotation"] = float(el.attrib.get("rotation", "0"))
                    elif el.tag == "mask":
                        shape["rle"] = el.attrib["rle"]
                        shape["left"] = el.attrib["left"]
                        shape["top"] = el.attrib["top"]
                        shape["width"] = el.attrib["width"]
                        shape["height"] = el.attrib["height"]
                    else:
                        shape["points"] = []
                        for pair in el.attrib["points"].split(";"):
                            shape["points"].extend(map(float, pair.split(",")))

                    if subset is None or subset == self._subset:
                        frame_desc = items.get(shape["frame"], {"annotations": []})
                        frame_desc["annotations"].append(
                            self._parse_shape_ann(shape, categories, image)
                        )
                        items[shape["frame"]] = frame_desc

                    shape = None

                elif el.tag == "tag":
                    if subset is None or subset == self._subset:
                        frame_desc = items.get(tag["frame"], {"annotations": []})
                        frame_desc["annotations"].append(self._parse_tag_ann(tag, categories))
                        items[tag["frame"]] = frame_desc
                    tag = None
                elif el.tag == "track":
                    track = None
                elif el.tag == "image":
                    if subset is None or subset == self._subset:
                        frame_desc = items.get(image["frame"], {"annotations": []})
                        frame_desc.update(
                            {
                                "name": image.get("name"),
                                "height": image.get("height"),
                                "width": image.get("width"),
                            }
                        )
                        items[image["frame"]] = frame_desc
                    image = None
                el.clear()

        return items, categories

    @staticmethod
    def _parse_meta(meta_root):
        categories = {}

        frame_size = None
        original_size = [item for item in meta_root.iter("original_size")]

        if len(original_size) > 1:
            raise DatasetImportError("CVAT XML file should have only one <original_size> tag.")
        elif len(original_size) == 1:
            frame_size = (
                int(original_size[0].find("height").text),
                int(original_size[0].find("width").text),
            )

        mode = None
        labels = OrderedDict()

        for label in meta_root.iter("label"):
            name = label.find("name").text
            labels[name] = [
                {
                    "name": attr.find("name").text,
                    "input_type": attr.find("input_type").text,
                }
                for attr in label.iter("attribute")
            ]

        common_attrs = ["occluded"]
        if mode == "interpolation":
            common_attrs.append("keyframe")
            common_attrs.append("outside")
            common_attrs.append("track_id")

        label_cat = LabelCategories(attributes=common_attrs)
        attribute_types = {}
        for label, attrs in labels.items():
            attr_names = {v["name"] for v in attrs}
            label_cat.add(label, attributes=attr_names)
            for attr in attrs:
                attribute_types[attr["name"]] = attr["input_type"]

        categories[AnnotationType.label] = label_cat
        return categories, frame_size, attribute_types

    @classmethod
    def _parse_shape_ann(cls, ann, categories, image):
        ann_id = ann.get("id", 0)
        ann_type = ann["type"]

        attributes = ann.get("attributes") or {}
        if "occluded" in categories[AnnotationType.label].attributes:
            attributes["occluded"] = ann.get("occluded", False)
        if "outside" in ann:
            attributes["outside"] = ann["outside"]
        if "keyframe" in ann:
            attributes["keyframe"] = ann["keyframe"]
        if "track_id" in ann:
            attributes["track_id"] = ann["track_id"]

        group = ann.get("group")

        label = ann.get("label")
        label_id = categories[AnnotationType.label].find(label)[0]

        z_order = ann.get("z_order", 0)
        points = ann.get("points", [])

        if ann_type == "polyline":
            return PolyLine(
                points,
                label=label_id,
                z_order=z_order,
                id=ann_id,
                attributes=attributes,
                group=group,
            )

        elif ann_type == "polygon":
            return Polygon(
                points,
                label=label_id,
                z_order=z_order,
                id=ann_id,
                attributes=attributes,
                group=group,
            )

        elif ann_type == "points":
            return Points(
                points,
                label=label_id,
                z_order=z_order,
                id=ann_id,
                attributes=attributes,
                group=group,
            )

        elif ann_type == "box":
            x1, y1 = points[0], points[1]  # xtl, ytl
            x2, y2 = points[2], points[3]  # xbr, ybr
            w = x2 - x1
            h = y2 - y1

            # If rotation is present, convert to RotatedBbox
            if "rotation" in ann:
                # Convert from CVAT format (xtl,ytl,xbr,ybr + rotation)
                # to Datumaro format (cx,cy,w,h,r)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                return RotatedBbox(
                    cx,
                    cy,
                    w,
                    h,
                    float(ann["rotation"]),
                    label=label_id,
                    z_order=z_order,
                    id=ann_id,
                    attributes=attributes,
                    group=group,
                )

            # Otherwise use regular Bbox
            return Bbox(
                x1,
                y1,
                w,
                h,
                label=label_id,
                z_order=z_order,
                id=ann_id,
                attributes=attributes,
                group=group,
            )

        elif ann_type == "mask":
            rle = ann.get("rle")
            mask_w, mask_h = int(ann.get("width")), int(ann.get("height"))
            mask_l, mask_t = int(ann.get("left")), int(ann.get("top"))
            img_w, img_h = int(image.get("width")), int(image.get("height"))

            rle_uncompressed = {
                "counts": np.array([int(str_num) for str_num in rle.split(",")], dtype=np.uint32),
                "size": np.array([mask_w, mask_h]),
            }

            def _gen_mask():
                # From the manual test for the dataset exported from the CVAT 2.5,
                # the RLE encoding in the dataset has (W, H) binary 2D np.ndarray, not (H, W)
                # Therefore, we need to tranpose it to make its shape as (H, W).
                mask = mask_tools.rle_to_mask(rle_uncompressed).transpose()
                canvas = np.zeros(shape=[img_h, img_w], dtype=np.uint8)
                canvas[mask_t : mask_t + mask_h, mask_l : mask_l + mask_w] = mask
                return canvas

            return Mask(
                image=_gen_mask,
                label=label_id,
                z_order=z_order,
                id=ann_id,
                attributes=attributes,
                group=group,
            )

        else:
            raise NotImplementedError("Unknown annotation type '%s'" % ann_type)

    @classmethod
    def _parse_tag_ann(cls, ann, categories):
        label = ann.get("label")
        label_id = categories[AnnotationType.label].find(label)[0]
        group = ann.get("group")
        attributes = ann.get("attributes")
        return Label(label_id, attributes=attributes, group=group)

    def _load_items(self, parsed):
        for frame_id, item_desc in parsed.items():
            name = item_desc.get("name", "frame_%06d.png" % int(frame_id))

            image_path_opt_1 = osp.join(self._images_dir, name)
            image_path_opt_2 = (
                osp.join(self._images_dir, self._subset, name) if self._subset is not None else None
            )
            if osp.exists(image_path_opt_1):
                image = image_path_opt_1
            elif image_path_opt_2 and osp.exists(image_path_opt_2):
                image = image_path_opt_2
            elif "name" not in item_desc:
                # If --use-track flag is on
                # TODO: Revisit all the CVAT import/export parts.
                image = image_path_opt_1
            else:
                raise DatasetImportError(f"Cannot find an image which has name={name}.")

            image_size = (item_desc.get("height"), item_desc.get("width"))
            if all(image_size):
                image = Image.from_file(path=image, size=tuple(map(int, image_size)))
            else:
                image = Image.from_file(path=image)

            parsed[frame_id] = DatasetItem(
                id=osp.splitext(name)[0],
                subset=self._subset,
                media=image,
                annotations=item_desc.get("annotations"),
                attributes={"frame": int(frame_id)},
            )
            for ann in item_desc.get("annotations"):
                self._ann_types.add(ann.type)

        return parsed


class CvatImporter(Importer):
    _ANNO_EXT = ".xml"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        annot_file = context.require_file(f"*{cls._ANNO_EXT}")

        with context.probe_text_file(
            annot_file,
            'must be an XML file with an "annotations" root element',
        ) as f:
            _, root_elem = next(ElementTree.iterparse(f, events=("start",)))
            if root_elem.tag != "annotations":
                raise Exception

    @staticmethod
    def find_subsets(meta_root):
        subsets = [item.text for item in meta_root.iter("subset")]
        if len(subsets) == 0:
            raise DatasetImportError("CVAT XML should include <subset> tags.")
        return subsets

    @classmethod
    def find_sources(cls, path):
        source_files = cls._find_sources_recursive(path, ".xml", "cvat")
        sources = []

        for source in source_files:
            path = source["url"]
            meta_root, _ = _find_meta_root(path)

            if meta_root.find("project") is not None:
                for subset in cls.find_subsets(meta_root):
                    source_clone = deepcopy(source)
                    source_clone["options"] = {"subset": subset}
                    sources += [source_clone]
            elif meta_root.find("task") is not None:
                sources += [source]
            else:
                raise DatasetImportError(
                    "CVAT XML file should have a <meta> -> <task> or <meta> -> <project> subtree."
                )

        return sources

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]
