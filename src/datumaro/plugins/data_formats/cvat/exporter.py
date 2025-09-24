# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from collections import OrderedDict, defaultdict
from itertools import chain

# Disable B406: import_xml_sax - the library is used for writing
from xml.sax.saxutils import XMLGenerator  # nosec

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.dataset_item_storage import ItemStatus
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.media import Image
from datumaro.util import cast, mask_tools, pairs

from .format import CvatPath


class XmlAnnotationWriter:
    VERSION = "1.1"

    def __init__(self, f):
        self.xmlgen = XMLGenerator(f, "utf-8")
        self._level = 0

    def _indent(self, newline=True):
        if newline:
            self.xmlgen.ignorableWhitespace("\n")
        self.xmlgen.ignorableWhitespace("  " * self._level)

    def _add_version(self):
        self._indent()
        self.xmlgen.startElement("version", {})
        self.xmlgen.characters(self.VERSION)
        self.xmlgen.endElement("version")

    def open_root(self):
        self.xmlgen.startDocument()
        self.xmlgen.startElement("annotations", {})
        self._level += 1
        self._add_version()

    def _add_meta(self, meta):
        self._level += 1
        for k, v in meta.items():
            if isinstance(v, OrderedDict):
                self._indent()
                self.xmlgen.startElement(k, {})
                self._add_meta(v)
                self._indent()
                self.xmlgen.endElement(k)
            elif isinstance(v, list):
                self._indent()
                self.xmlgen.startElement(k, {})
                for tup in v:
                    self._add_meta(OrderedDict([tup]))
                self._indent()
                self.xmlgen.endElement(k)
            else:
                self._indent()
                self.xmlgen.startElement(k, {})
                self.xmlgen.characters(v)
                self.xmlgen.endElement(k)
        self._level -= 1

    def write_meta(self, meta):
        self._indent()
        self.xmlgen.startElement("meta", {})
        self._add_meta(meta)
        self._indent()
        self.xmlgen.endElement("meta")

    def open_track(self, track):
        self._indent()
        self.xmlgen.startElement("track", track)
        self._level += 1

    def open_image(self, image):
        self._indent()
        self.xmlgen.startElement("image", image)
        self._level += 1

    def open_box(self, box):
        self._indent()
        self.xmlgen.startElement("box", box)
        self._level += 1

    def open_polygon(self, polygon):
        self._indent()
        self.xmlgen.startElement("polygon", polygon)
        self._level += 1

    def open_polyline(self, polyline):
        self._indent()
        self.xmlgen.startElement("polyline", polyline)
        self._level += 1

    def open_points(self, points):
        self._indent()
        self.xmlgen.startElement("points", points)
        self._level += 1

    def open_mask(self, mask):
        self._indent()
        self.xmlgen.startElement("mask", mask)
        self._level += 1

    def open_tag(self, tag):
        self._indent()
        self.xmlgen.startElement("tag", tag)
        self._level += 1

    def add_attribute(self, attribute):
        self._indent()
        self.xmlgen.startElement("attribute", {"name": attribute["name"]})
        self.xmlgen.characters(attribute["value"])
        self.xmlgen.endElement("attribute")

    def _close_element(self, element):
        self._level -= 1
        self._indent()
        self.xmlgen.endElement(element)

    def close_box(self):
        self._close_element("box")

    def close_polygon(self):
        self._close_element("polygon")

    def close_polyline(self):
        self._close_element("polyline")

    def close_points(self):
        self._close_element("points")

    def close_mask(self):
        self._close_element("mask")

    def close_tag(self):
        self._close_element("tag")

    def close_image(self):
        self._close_element("image")

    def close_track(self):
        self._close_element("track")

    def close_root(self):
        self._close_element("annotations")
        self.xmlgen.endDocument()


class _SubsetWriter:
    def __init__(self, file, name, extractor, context):
        self._writer = XmlAnnotationWriter(file)
        self._name = name
        self._extractor = extractor
        self._context = context
        self._item_count = 0

    def is_empty(self):
        return self._item_count == 0

    def write(self):
        self._writer.open_root()
        self._write_meta()

        if self._context._use_track:
            tracks = self._get_tracks()
            for track in tracks:
                self._write_track(track)
        else:
            for index, item in enumerate(self._extractor):
                self._write_item(item, index)

        self._writer.close_root()

    def _write_track(self, track):
        track_id = track["track_id"]
        label_name = self._get_label(track["label"]).name
        annotations = track["annotations"]

        track_info = {"id": str(track_id), "label": label_name}

        self._writer.open_track(track_info)
        for ann in annotations:
            if ann.type in CvatPath.SUPPORTED_EXPORT_SHAPES:
                self._write_shape(ann, write_label_info=False, write_frame=True)
        self._writer.close_track()

    def _get_tracks(self):
        track_infos = defaultdict(lambda: defaultdict(list))
        for item in self._extractor:
            if "frame" not in item.attributes:
                continue
            frame = item.attributes["frame"]
            for annotation in item.annotations:
                if "track_id" not in annotation.attributes:
                    continue
                track_id = int(annotation.attributes["track_id"])
                track_infos[track_id][frame].append(annotation)

        tracks = []

        for tid in sorted(list(track_infos.keys())):
            annotations = []
            for frame in sorted(list(track_infos[tid].keys())):
                for annotation in track_infos[tid][frame]:
                    annotation.attributes["frame"] = frame
                    del annotation.attributes["track_id"]
                    annotations.append(annotation)

            if len(annotations) == 0:
                continue

            valid_label = True
            for annotation in annotations[1:]:
                if annotation.label != annotations[0].label:
                    valid_label = False

            if not valid_label:
                continue

            tracks.append(
                {"track_id": tid, "label": annotations[0].label, "annotations": annotations}
            )

        return tracks

    def _write_item(self, item, index):
        if not self._context._reindex:
            index = cast(item.attributes.get("frame"), int, index)
        image_info = OrderedDict(
            [
                ("id", str(index)),
            ]
        )
        filename = self._context._make_image_filename(item)
        image_info["name"] = filename
        if item.media:
            size = item.media.size
            if size:
                h, w = size
                image_info["width"] = str(w)
                image_info["height"] = str(h)

            if self._context._save_media:
                self._context._save_image(item, osp.join(self._context._images_dir, filename))
        else:
            log.debug("Item '%s' has no image info", item.id)
        self._writer.open_image(image_info)

        for ann in item.annotations:
            if ann.type in CvatPath.SUPPORTED_EXPORT_SHAPES:
                self._write_shape(ann, item)
            elif ann.type == AnnotationType.label:
                self._write_tag(ann, item)
            else:
                continue

        self._writer.close_image()

        self._item_count += 1

    def _write_meta(self):
        label_cat = self._extractor.categories().get(AnnotationType.label, LabelCategories())

        task_items = [
            ("id", ""),
            ("name", self._name),
            ("size", str(len(self._extractor))),
            ("mode", "annotation"),
            ("overlap", ""),
            ("start_frame", "0"),
            ("stop_frame", str(len(self._extractor))),
            ("frame_filter", ""),
            ("z_order", "True"),
            (
                "labels",
                [
                    (
                        "label",
                        OrderedDict(
                            [
                                ("name", label.name),
                                (
                                    "attributes",
                                    [
                                        (
                                            "attribute",
                                            OrderedDict(
                                                [
                                                    ("name", attr),
                                                    ("mutable", "True"),
                                                    ("input_type", "text"),
                                                    ("default_value", ""),
                                                    ("values", ""),
                                                ]
                                            ),
                                        )
                                        for attr in self._get_label_attrs(label)
                                    ],
                                ),
                            ]
                        ),
                    )
                    for label in label_cat.items
                ],
            ),
        ]

        if self._context._original_size is not None:
            task_items.append(
                (
                    "original_size",
                    OrderedDict(
                        [
                            ("width", str(self._context._original_size[0])),
                            ("height", str(self._context._original_size[1])),
                        ]
                    ),
                )
            )

        meta = OrderedDict(
            [
                ("task", OrderedDict(task_items)),
            ]
        )
        self._writer.write_meta(meta)

    def _get_label(self, label_id):
        if label_id is None:
            return ""
        label_cat = self._extractor.categories().get(AnnotationType.label, LabelCategories())
        return label_cat.items[label_id]

    def _get_label_attrs(self, label):
        label_cat = self._extractor.categories().get(AnnotationType.label, LabelCategories())
        if isinstance(label, int):
            label = label_cat[label]
        return set(chain(label.attributes, label_cat.attributes)) - self._context._builtin_attrs

    def _write_shape(self, shape, item=None, write_label_info=True, write_frame=False):
        item_id = "None" if item is None else item.id

        if write_label_info and shape.label is None:
            log.warning("Item %s: skipping a %s with no label", item_id, shape.type.name)
            return

        if write_frame and "frame" not in shape.attributes:
            log.warning("Skipping a %s with no frame", shape.type.name)
            return

        if write_label_info:
            label_name = self._get_label(shape.label).name
            shape_data = OrderedDict([("label", label_name)])
        else:
            shape_data = OrderedDict()

        if write_frame:
            shape_data.update(OrderedDict([("frame", str(int(shape.attributes.get("frame", 0))))]))
        shape_data.update(
            OrderedDict([("occluded", str(int(shape.attributes.get("occluded", False))))])
        )
        shape_data.update(
            OrderedDict([("outside", str(int(shape.attributes.get("outside", False))))])
        )
        shape_data.update(
            OrderedDict([("keyframe", str(int(shape.attributes.get("keyframe", False))))])
        )

        if shape.type == AnnotationType.bbox:
            shape_data.update(
                OrderedDict(
                    [
                        ("xtl", "{:.2f}".format(shape.points[0])),
                        ("ytl", "{:.2f}".format(shape.points[1])),
                        ("xbr", "{:.2f}".format(shape.points[2])),
                        ("ybr", "{:.2f}".format(shape.points[3])),
                    ]
                )
            )
        elif shape.type == AnnotationType.rotated_bbox:
            # Convert from Datumaro format (cx,cy,w,h,r)
            # to CVAT format (xtl,ytl,xbr,ybr + rotation)
            cx, cy = shape.points[0], shape.points[1]
            w, h = shape.points[2], shape.points[3]
            rotation = shape.points[4]

            # Calculate corners from center coordinates
            xtl = cx - w / 2
            ytl = cy - h / 2
            xbr = cx + w / 2
            ybr = cy + h / 2

            shape_data.update(
                OrderedDict(
                    [
                        ("xtl", "{:.2f}".format(xtl)),
                        ("ytl", "{:.2f}".format(ytl)),
                        ("xbr", "{:.2f}".format(xbr)),
                        ("ybr", "{:.2f}".format(ybr)),
                        ("rotation", "{:.2f}".format(rotation)),
                    ]
                )
            )
        elif shape.type == AnnotationType.mask:
            # From the manual test for the dataset exported from the CVAT 2.5,
            # the RLE encoding in the dataset has (W, H) binary 2D np.ndarray, not (H, W)
            # Therefore, we need to tranpose it to make its shape as (H, W).
            mask = shape.image.transpose()
            rle_uncompressed = mask_tools.mask_to_rle(mask)
            width, height = mask.shape
            shape_data.update(
                OrderedDict(
                    rle=", ".join([str(c) for c in rle_uncompressed["counts"]]),
                    left=str(0),
                    top=str(0),
                    width=str(width),
                    height=str(height),
                )
            )
        else:
            shape_data.update(
                OrderedDict(
                    [
                        (
                            "points",
                            ";".join(
                                (
                                    ",".join(("{:.2f}".format(x), "{:.2f}".format(y)))
                                    for x, y in pairs(shape.points)
                                )
                            ),
                        ),
                    ]
                )
            )

        shape_data["z_order"] = str(int(shape.z_order))
        if shape.group:
            shape_data["group_id"] = str(shape.group)

        if shape.type in [AnnotationType.bbox, AnnotationType.rotated_bbox]:
            self._writer.open_box(shape_data)
        elif shape.type == AnnotationType.polygon:
            self._writer.open_polygon(shape_data)
        elif shape.type == AnnotationType.polyline:
            self._writer.open_polyline(shape_data)
        elif shape.type == AnnotationType.points:
            self._writer.open_points(shape_data)
        elif shape.type == AnnotationType.mask:
            self._writer.open_mask(shape_data)
        else:
            raise NotImplementedError("unknown shape type")

        if write_label_info:
            for attr_name, attr_value in shape.attributes.items():
                if attr_name in self._context._builtin_attrs:
                    continue
                if isinstance(attr_value, bool):
                    attr_value = "true" if attr_value else "false"
                if self._context._allow_undeclared_attrs or attr_name in self._get_label_attrs(
                    shape.label
                ):
                    self._writer.add_attribute(
                        OrderedDict(
                            [
                                ("name", str(attr_name)),
                                ("value", str(attr_value)),
                            ]
                        )
                    )
                else:
                    log.warning(
                        "Item %s: skipping undeclared "
                        "attribute '%s' for label '%s' "
                        "(allow with --allow-undeclared-attrs option)",
                        item_id,
                        attr_name,
                        label_name,
                    )

        if shape.type in [AnnotationType.bbox, AnnotationType.rotated_bbox]:
            self._writer.close_box()
        elif shape.type == AnnotationType.polygon:
            self._writer.close_polygon()
        elif shape.type == AnnotationType.polyline:
            self._writer.close_polyline()
        elif shape.type == AnnotationType.points:
            self._writer.close_points()
        elif shape.type == AnnotationType.mask:
            self._writer.close_mask()
        else:
            raise NotImplementedError("unknown shape type")

    def _write_tag(self, label, item):
        if label.label is None:
            log.warning("Item %s: skipping a %s with no label", item.id, label.type.name)
            return

        label_name = self._get_label(label.label).name
        tag_data = OrderedDict(
            [
                ("label", label_name),
            ]
        )
        if label.group:
            tag_data["group_id"] = str(label.group)
        self._writer.open_tag(tag_data)

        for attr_name, attr_value in label.attributes.items():
            if attr_name in self._context._builtin_attrs:
                continue
            if isinstance(attr_value, bool):
                attr_value = "true" if attr_value else "false"
            if self._context._allow_undeclared_attrs or attr_name in self._get_label_attrs(
                label.label
            ):
                self._writer.add_attribute(
                    OrderedDict(
                        [
                            ("name", str(attr_name)),
                            ("value", str(attr_value)),
                        ]
                    )
                )
            else:
                log.warning(
                    "Item %s: skipping undeclared "
                    "attribute '%s' for label '%s' "
                    "(allow with --allow-undeclared-attrs option)",
                    item.id,
                    attr_name,
                    label_name,
                )

        self._writer.close_tag()


class CvatExporter(Exporter):
    DEFAULT_IMAGE_EXT = CvatPath.IMAGE_EXT

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--reindex",
            action="store_true",
            help="Assign new indices to frames (default: %(default)s)",
        )
        parser.add_argument(
            "--allow-undeclared-attrs",
            action="store_true",
            help="Write annotation attributes even if they are not present in "
            "the input dataset metainfo (default: %(default)s)",
        )
        return parser

    def __init__(
        self,
        extractor,
        save_dir,
        reindex=False,
        allow_undeclared_attrs=False,
        use_track=False,
        **kwargs,
    ):
        super().__init__(extractor, save_dir, **kwargs)

        self._reindex = reindex
        self._builtin_attrs = CvatPath.BUILTIN_ATTRS
        self._allow_undeclared_attrs = allow_undeclared_attrs
        self._use_track = use_track
        if use_track:
            self._original_size = self._get_original_size(extractor)
        else:
            self._original_size = None

    def _get_original_size(self, extractor):
        item_hw = None
        for item in extractor:
            if item_hw is None:
                item_hw = item.media.size
            elif item_hw != item.media.size:
                return None
        return (item_hw[1], item_hw[0])

    def _apply_impl(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        self._images_dir = osp.join(self._save_dir, CvatPath.IMAGES_DIR)
        os.makedirs(self._images_dir, exist_ok=True)

        for subset_name, subset in self._extractor.subsets().items():
            ann_path = osp.join(self._save_dir, "%s.xml" % subset_name)
            with open(ann_path, "w", encoding="utf-8") as f:
                writer = _SubsetWriter(f, subset_name, subset, self)
                writer.write()

            if self._patch and subset_name in self._patch.updated_subsets and writer.is_empty():
                os.remove(ann_path)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        for subset in patch.updated_subsets:
            conv = cls(dataset.get_subset(subset), save_dir=save_dir, **kwargs)
            conv._patch = patch
            conv.apply()

        conv = cls(dataset, save_dir=save_dir, **kwargs)
        # Find images that needs to be removed
        # images from different subsets are stored in the common directory
        # Avoid situations like:
        # (a, test): added
        # (a, train): removed
        # where the second line removes images from the first.
        ids_to_remove = {}
        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.media):
                ids_to_remove[item_id] = (item, False)
            else:
                ids_to_remove.setdefault(item_id, (item, True))

        for item, to_remove in ids_to_remove.values():
            if not to_remove:
                continue

            image_path = osp.join(save_dir, CvatPath.IMAGES_DIR, conv._make_image_filename(item))
            if osp.isfile(image_path):
                os.unlink(image_path)
