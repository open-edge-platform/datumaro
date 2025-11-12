# Copyright (C) 2021-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from copy import deepcopy

# Disable B406: import_xml_sax - the library is used for writing
from xml.sax.saxutils import XMLGenerator  # nosec

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.dataset_item_storage import ItemStatus
from datumaro.components.errors import DatasetExportError, MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.media import PointCloud
from datumaro.util import cast
from datumaro.util.image import find_images

from .format import KittiRawPath, OcclusionStates, PoseStates, TruncationStates


class _XmlAnnotationWriter:
    # Format constants
    _tracking_level = 0

    _tracklets_class_id = 0
    _tracklets_version = 0

    _tracklet_class_id = 1
    _tracklet_version = 1

    _poses_class_id = 2
    _poses_version = 0

    _pose_class_id = 3
    _pose_version = 1

    # XML headers
    _header = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"""
    _doctype = "<!DOCTYPE boost_serialization>"

    def __init__(self, file, tracklets):
        self._file = file
        self._tracklets = tracklets

        self._xmlgen = XMLGenerator(self._file, encoding="utf-8")
        self._level = 0

        # See reference for section headers here:
        # https://www.boost.org/doc/libs/1_40_0/libs/serialization/doc/traits.html
        # XML archives have regular structure, so we only include headers once
        self._add_tracklet_header = True
        self._add_poses_header = True
        self._add_pose_header = True

    def _indent(self, newline=True):
        if newline:
            self._xmlgen.ignorableWhitespace("\n")
        self._xmlgen.ignorableWhitespace("  " * self._level)

    def _add_headers(self):
        self._file.write(self._header)

        self._indent(newline=True)
        self._file.write(self._doctype)

    def _open_serialization(self):
        self._indent(newline=True)
        self._xmlgen.startElement("boost_serialization", {"version": "9", "signature": "serialization::archive"})

    def _close_serialization(self):
        self._indent(newline=True)
        self._xmlgen.endElement("boost_serialization")

    def _add_count(self, count):
        self._indent(newline=True)
        self._xmlgen.startElement("count", {})
        self._xmlgen.characters(str(count))
        self._xmlgen.endElement("count")

    def _add_item_version(self, version):
        self._indent(newline=True)
        self._xmlgen.startElement("item_version", {})
        self._xmlgen.characters(str(version))
        self._xmlgen.endElement("item_version")

    def _open_tracklets(self, tracklets):
        self._indent(newline=True)
        self._xmlgen.startElement(
            "tracklets",
            {
                "version": str(self._tracklets_version),
                "tracking_level": str(self._tracking_level),
                "class_id": str(self._tracklets_class_id),
            },
        )
        self._level += 1
        self._add_count(len(tracklets))
        self._add_item_version(self._tracklet_version)

    def _close_tracklets(self):
        self._level -= 1
        self._indent(newline=True)
        self._xmlgen.endElement("tracklets")

    def _open_tracklet(self):
        self._indent(newline=True)
        if self._add_tracklet_header:
            self._xmlgen.startElement(
                "item",
                {
                    "version": str(self._tracklet_class_id),
                    "tracking_level": str(self._tracking_level),
                    "class_id": str(self._tracklet_class_id),
                },
            )
            self._add_tracklet_header = False
        else:
            self._xmlgen.startElement("item", {})
        self._level += 1

    def _close_tracklet(self):
        self._level -= 1
        self._indent(newline=True)
        self._xmlgen.endElement("item")

    def _add_tracklet(self, tracklet):
        self._open_tracklet()

        for key, value in tracklet.items():
            if key == "poses":
                self._add_poses(value)
            elif key == "attributes":
                self._add_attributes(value)
            else:
                self._indent(newline=True)
                self._xmlgen.startElement(key, {})
                self._xmlgen.characters(str(value))
                self._xmlgen.endElement(key)

        self._close_tracklet()

    def _open_poses(self, poses):
        self._indent(newline=True)
        if self._add_poses_header:
            self._xmlgen.startElement(
                "poses",
                {
                    "version": str(self._poses_version),
                    "tracking_level": str(self._tracking_level),
                    "class_id": str(self._poses_class_id),
                },
            )
            self._add_poses_header = False
        else:
            self._xmlgen.startElement("poses", {})
        self._level += 1

        self._add_count(len(poses))
        self._add_item_version(self._poses_version)

    def _close_poses(self):
        self._level -= 1
        self._indent(newline=True)
        self._xmlgen.endElement("poses")

    def _add_poses(self, poses):
        self._open_poses(poses)

        for pose in poses:
            self._add_pose(pose)

        self._close_poses()

    def _open_pose(self):
        self._indent(newline=True)
        if self._add_pose_header:
            self._xmlgen.startElement(
                "item",
                {
                    "version": str(self._pose_version),
                    "tracking_level": str(self._tracking_level),
                    "class_id": str(self._pose_class_id),
                },
            )
            self._add_pose_header = False
        else:
            self._xmlgen.startElement("item", {})
        self._level += 1

    def _close_pose(self):
        self._level -= 1
        self._indent(newline=True)
        self._xmlgen.endElement("item")

    def _add_pose(self, pose):
        self._open_pose()

        for key, value in pose.items():
            if key == "attributes":
                self._add_attributes(value)
            elif key != "frame_id":
                self._indent(newline=True)
                self._xmlgen.startElement(key, {})
                self._xmlgen.characters(str(value))
                self._xmlgen.endElement(key)

        self._close_pose()

    def _open_attributes(self):
        self._indent(newline=True)
        self._xmlgen.startElement("attributes", {})
        self._level += 1

    def _close_attributes(self):
        self._level -= 1
        self._indent(newline=True)
        self._xmlgen.endElement("attributes")

    def _add_attributes(self, attributes):
        self._open_attributes()

        for name, value in attributes.items():
            self._add_attribute(name, value)

        self._close_attributes()

    def _open_attribute(self):
        self._indent(newline=True)
        self._xmlgen.startElement("attribute", {})
        self._level += 1

    def _close_attribute(self):
        self._level -= 1
        self._indent(newline=True)
        self._xmlgen.endElement("attribute")

    def _add_attribute(self, name, value):
        self._open_attribute()

        self._indent(newline=True)
        self._xmlgen.startElement("name", {})
        self._xmlgen.characters(name)
        self._xmlgen.endElement("name")

        self._xmlgen.startElement("value", {})
        self._xmlgen.characters(str(value))
        self._xmlgen.endElement("value")

        self._close_attribute()

    def write(self):
        self._add_headers()
        self._open_serialization()

        self._open_tracklets(self._tracklets)

        for tracklet in self._tracklets:
            self._add_tracklet(tracklet)

        self._close_tracklets()

        self._close_serialization()


class KittiRawExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".jpg"

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--reindex",
            action="store_true",
            help="Assign new indices to frames and tracks. "
            "Allows annotations without 'track_id' (default: %(default)s)",
        )
        parser.add_argument(
            "--allow-attrs",
            action="store_true",
            help="Allow writing annotation attributes (default: %(default)s)",
        )
        return parser

    def __init__(self, extractor, save_dir, reindex=False, allow_attrs=False, **kwargs):
        super().__init__(extractor, save_dir, **kwargs)

        self._reindex = reindex
        self._builtin_attrs = KittiRawPath.BUILTIN_ATTRS | KittiRawPath.SPECIAL_ATTRS
        self._allow_attrs = allow_attrs

    def _create_tracklets(self, subset):
        tracks = {}  # track_id -> track
        name_mapping = {}  # frame_id -> name

        for frame_id, item in enumerate(subset):
            frame_id = self._write_item(item, frame_id)

            if frame_id in name_mapping:
                raise DatasetExportError("Item %s: frame id %s is repeated in the dataset" % (item.id, frame_id))
            name_mapping[frame_id] = item.id

            for ann in item.annotations:
                if ann.type != AnnotationType.cuboid_3d:
                    continue

                if ann.label is None:
                    log.warning(
                        "Item %s: skipping a %s%s with no label",
                        item.id,
                        ann.type.name,
                        "(#%s) " % ann.id if ann.id is not None else "",
                    )
                    continue

                label = self._get_label(ann.label).name

                track_id = cast(ann.attributes.get("track_id"), int, None)
                if self._reindex and track_id is None:
                    # In this format, track id is not used for anything except
                    # annotation grouping. So we only need to pick a definitely
                    # unused id. A negative one, for example.
                    track_id = -(len(tracks) + 1)
                if track_id is None:
                    raise DatasetExportError(
                        "Item %s: expected track annotations "
                        "having 'track_id' (integer) attribute. "
                        "Use --reindex to export single shapes." % item.id
                    )

                track = tracks.get(track_id)
                if not track:
                    track = {
                        "objectType": label,
                        "h": ann.scale[0],
                        "w": ann.scale[1],
                        "l": ann.scale[2],
                        "first_frame": frame_id,
                        "poses": [],
                        "finished": 1,  # keep last
                    }
                    tracks[track_id] = track
                else:
                    if [track["h"], track["w"], track["l"]] != ann.scale:
                        # Tracks have fixed scale in the format
                        raise DatasetExportError("Item %s: mismatching track shapes, track id %s" % (item.id, track_id))

                    if track["objectType"] != label:
                        raise DatasetExportError(
                            "Item %s: mismatching track labels, "
                            "track id %s: %s vs. %s" % (item.id, track_id, track["objectType"], label)
                        )

                    # If there is a skip in track frames, add missing as outside
                    if frame_id != track["poses"][-1]["frame_id"] + 1:
                        last_key_pose = track["poses"][-1]
                        last_keyframe_id = last_key_pose["frame_id"]
                        last_key_pose["occlusion_kf"] = 1
                        for i in range(last_keyframe_id + 1, frame_id):
                            pose = deepcopy(last_key_pose)
                            pose["occlusion"] = OcclusionStates.OCCLUSION_UNSET
                            pose["truncation"] = TruncationStates.OUT_IMAGE
                            pose["frame_id"] = i
                            track["poses"].append(pose)

                occlusion = OcclusionStates.VISIBLE
                if "occlusion" in ann.attributes:
                    occlusion = OcclusionStates(ann.attributes["occlusion"].upper())
                elif ann.attributes.get("occluded"):
                    occlusion = OcclusionStates.PARTLY

                truncation = TruncationStates.IN_IMAGE
                if "truncation" in ann.attributes:
                    truncation = TruncationStates(ann.attributes["truncation"].upper())

                pose = {
                    "tx": ann.position[0],
                    "ty": ann.position[1],
                    "tz": ann.position[2],
                    "rx": ann.rotation[0],
                    "ry": ann.rotation[1],
                    "rz": ann.rotation[2],
                    "state": PoseStates.LABELED.value,
                    "occlusion": occlusion.value,
                    "occlusion_kf": int(ann.attributes.get("keyframe", False) is True),
                    "truncation": truncation.value,
                    "amt_occlusion": -1,
                    "amt_border_l": -1,
                    "amt_border_r": -1,
                    "amt_occlusion_kf": -1,
                    "amt_border_kf": -1,
                    "frame_id": frame_id,
                }

                if self._allow_attrs:
                    attributes = {}
                    for name, value in ann.attributes.items():
                        if name in self._builtin_attrs:
                            continue

                        if isinstance(value, bool):
                            value = "true" if value else "false"
                        attributes[name] = value

                    pose["attributes"] = attributes

                track["poses"].append(pose)

        self._write_name_mapping(name_mapping)

        return [e[1] for e in sorted(tracks.items(), key=lambda e: e[0])]

    def _write_name_mapping(self, name_mapping):
        with open(osp.join(self._save_dir, KittiRawPath.NAME_MAPPING_FILE), "w", encoding="utf-8") as f:
            f.writelines("%s %s\n" % (frame_id, name) for frame_id, name in name_mapping.items())

    def _get_label(self, label_id):
        if label_id is None:
            return ""
        label_cat = self._extractor.categories().get(AnnotationType.label, LabelCategories())
        return label_cat.items[label_id]

    def _write_item(self, item, index):
        if not self._reindex:
            index = cast(item.attributes.get("frame"), int, index)

        if self._save_media and item.media:
            self._save_point_cloud(item, subdir=KittiRawPath.PCD_DIR)

            images = sorted(item.media.extra_images, key=lambda img: img.path if hasattr(img, "path") else "")
            for i, image in enumerate(images):
                if image.has_data:
                    image.save(
                        osp.join(
                            self._save_dir,
                            KittiRawPath.IMG_DIR_PREFIX + ("%02d" % i),
                            "data",
                            item.id + self._find_image_ext(image),
                        )
                    )

        elif self._save_media and not item.media:
            log.debug("Item '%s' has no image info", item.id)

        return index

    def _apply_impl(self):
        if self._extractor.media_type() and self._extractor.media_type() is not PointCloud:
            raise MediaTypeError("Media type is not a point cloud")

        os.makedirs(self._save_dir, exist_ok=True)

        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)

        if len(self._extractor.subsets()) > 1:
            log.warning("Kitti RAW format supports only a single subset. Subset information will be ignored on export.")

        tracklets = self._create_tracklets(self._extractor)
        with open(osp.join(self._save_dir, KittiRawPath.ANNO_FILE), "w", encoding="utf-8") as f:
            writer = _XmlAnnotationWriter(f, tracklets)
            writer.write()

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        conv = cls(patch.as_dataset(dataset), save_dir=save_dir, **kwargs)
        conv.apply()

        pcd_dir = osp.abspath(osp.join(save_dir, KittiRawPath.PCD_DIR))
        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.media):
                continue

            pcd_path = osp.join(pcd_dir, conv._make_pcd_filename(item))
            if osp.isfile(pcd_path):
                os.unlink(pcd_path)

            for d in os.listdir(save_dir):
                image_dir = osp.join(save_dir, d, "data", osp.dirname(item.id))
                if d.startswith(KittiRawPath.IMG_DIR_PREFIX) and osp.isdir(image_dir):
                    for p in find_images(image_dir):
                        if osp.splitext(osp.basename(p))[0] == osp.basename(item.id):
                            os.unlink(p)
