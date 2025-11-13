# Copyright (C) 2021-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

# The format is described here:
# https://docs.supervise.ly/data-organization/00_ann_format_navi

from __future__ import annotations

import logging as log
import os
import os.path as osp
import shutil
import uuid
from datetime import datetime

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset_base import DatasetItem, IDataset
from datumaro.components.dataset_item_storage import ItemStatus
from datumaro.components.errors import DatasetExportError, MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.media import PointCloud
from datumaro.util import cast, dump_json_file

from .format import PointCloudPath


class _SuperviselyPointCloudDumper:
    def __init__(self, extractor: IDataset, context: SuperviselyPointCloudExporter):
        self._extractor = extractor
        self._context = context

        timestamp = str(datetime.now())
        self._default_user_info = {
            "labelerLogin": "",
            "createdAt": timestamp,
            "updatedAt": timestamp,
        }

        self._key_id_data = {"tags": {}, "objects": {}, "figures": {}, "videos": {}}

        self._meta_data = {"classes": [], "tags": [], "projectType": "point_clouds"}

        # Meta info contents
        self._tag_meta = {}  # name -> descriptor

        # Registries of item annotations
        self._objects = {}  # id -> key

        self._label_cat = extractor.categories().get(AnnotationType.label, LabelCategories())

    def _write_related_images(self, item):
        img_dir = self._related_images_dir

        for idx, img in enumerate(item.media.extra_images):
            name = osp.splitext(osp.basename(img.path))[0] if hasattr(img, "path") else f"extra_image_{idx}"
            img_path = osp.join(img_dir, item.id + "_pcd", name + self._find_image_ext(img))
            if img.has_data:
                img.save(img_path)

            img_data = {
                "name": osp.basename(img_path),
                "meta": {
                    "sensorsData": {
                        "extrinsicMatrix": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "intrinsicMatrix": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    }
                },
            }

            dump_json_file(osp.join(img_dir, img_path + ".json"), img_data, indent=True)

    def _write_pcd(self, item):
        self._context._save_point_cloud(item, basedir=self._point_cloud_dir)

    def _write_meta(self):
        for tag in self._tag_meta.values():
            if tag["value_type"] is None:
                tag["value_type"] = "any_string"
            tag["classes"] = list(tag["classes"])
        self._meta_data["tags"] = list(self._tag_meta.values())

        dump_json_file(osp.join(self._save_dir, PointCloudPath.META_FILE), self._meta_data, indent=True)

    def _write_key_id(self):
        objects = self._objects
        key_id_data = self._key_id_data

        key_id_data["objects"] = {v: k for k, v in objects.items()}

        dump_json_file(osp.join(self._save_dir, PointCloudPath.KEY_ID_FILE), key_id_data, indent=True)

    def _write_item_annotations(self, item):
        key_id_data = self._key_id_data

        item_id = cast(item.attributes.get("frame"), int)
        if item_id is None or self._context._reindex:
            item_id = len(key_id_data["videos"]) + 1

        item_key = str(uuid.uuid4())
        key_id_data["videos"][item_key] = item_id

        item_user_info = {k: item.attributes.get(k, default_v) for k, default_v in self._default_user_info.items()}

        item_ann_data = {
            "description": item.attributes.get("description", ""),
            "key": item_key,
            "tags": [],
            "objects": [],
            "figures": [],
        }
        self._export_item_attributes(item, item_ann_data, item_user_info)
        self._export_item_annotations(item, item_ann_data, item_user_info)

        ann_path = osp.join(self._ann_dir, item.id + ".pcd.json")
        os.makedirs(osp.dirname(ann_path), exist_ok=True)
        dump_json_file(ann_path, item_ann_data, indent=True)

    def _export_item_attributes(self, item, item_ann_data, item_user_info):
        for attr_name, attr_value in item.attributes.items():
            if attr_name in PointCloudPath.SPECIAL_ATTRS:
                continue

            attr_value = self._encode_attr_value(attr_value)

            tag = self._register_tag(attr_name, value=attr_value, applicable_type="imagesOnly")

            if tag["applicable_type"] != "imagesOnly":
                tag["applicable_type"] = "all"

            value_type = self._define_attr_type(attr_value)
            if tag["value_type"] is None:
                tag["value_type"] = value_type
            elif tag["value_type"] != value_type:
                raise DatasetExportError(
                    "Item %s: mismatching "
                    "value types for tag %s: %s vs %s" % (item.id, attr_name, tag["value_type"], value_type)
                )

            tag_key = str(uuid.uuid4())
            item_ann_data["tags"].append(
                {
                    "key": tag_key,
                    "name": attr_name,
                    "value": attr_value,
                    **item_user_info,
                }
            )

            # only item attributes are listed in the key_id file
            # meta tag ids have no relation to key_id tag ids!
            tag_id = len(self._key_id_data["tags"]) + 1
            self._key_id_data["tags"][tag_key] = tag_id

    def _export_item_annotations(self, item, item_ann_data, item_user_info):
        objects = self._objects
        tags = self._tag_meta
        label_cat = self._label_cat
        key_id_data = self._key_id_data

        image_objects = set()
        for ann in item.annotations:
            if ann.type != AnnotationType.cuboid_3d:
                continue

            obj_id = cast(ann.attributes.get("track_id", ann.id), int)
            if obj_id is None:
                # should not be affected by reindex
                # because it is used to match figures,
                # including different frames
                obj_id = len(self._objects) + 1

            object_key = objects.setdefault(obj_id, str(uuid.uuid4()))
            object_label = label_cat[ann.label].name
            if obj_id not in image_objects:
                ann_user_info = {k: ann.attributes.get(k, default_v) for k, default_v in item_user_info.items()}

                obj_ann_data = {
                    "key": object_key,
                    "classTitle": object_label,
                    "tags": [],
                    "objects": [],
                    "figures": [],
                    **ann_user_info,
                }

                for attr_name, attr_value in ann.attributes.items():
                    if attr_name in PointCloudPath.SPECIAL_ATTRS:
                        continue

                    attr_value = self._encode_attr_value(attr_value)

                    tag = tags.get(attr_name)
                    if tag is None:
                        if self._context._allow_undeclared_attrs:
                            tag = self._register_tag(attr_name, applicable_type="objectsOnly")
                            tags[attr_name] = tag
                        else:
                            log.warning(
                                "Item %s: skipping undeclared "
                                "attribute '%s' for label '%s' "
                                "(allow with --allow-undeclared-attrs option)",
                                item.id,
                                attr_name,
                                object_label,
                            )
                            continue

                    if tag["applicable_type"] == "imagesOnly":
                        tag["applicable_type"] = "all"
                    elif tag["applicable_type"] == "objectsOnly" and tag["classes"]:
                        tag["classes"].add(object_label)

                    value_type = self._define_attr_type(attr_value)
                    if tag["value_type"] is None:
                        tag["value_type"] = value_type
                    elif tag["value_type"] != value_type:
                        raise DatasetExportError(
                            "Item %s: mismatching "
                            "value types for tag %s: %s vs %s" % (item.id, attr_name, tag["value_type"], value_type)
                        )

                    tag_key = str(uuid.uuid4())
                    obj_ann_data["tags"].append(
                        {
                            "key": tag_key,
                            "name": attr_name,
                            "value": attr_value,
                            **ann_user_info,
                        }
                    )

                item_ann_data["objects"].append(obj_ann_data)

                image_objects.add(obj_id)

            figure_key = str(uuid.uuid4())
            item_ann_data["figures"].append(
                {
                    "key": figure_key,
                    "objectKey": object_key,
                    "geometryType": "cuboid_3d",
                    "geometry": {
                        "position": {
                            "x": float(ann.position[0]),
                            "y": float(ann.position[1]),
                            "z": float(ann.position[2]),
                        },
                        "rotation": {
                            "x": float(ann.rotation[0]),
                            "y": float(ann.rotation[1]),
                            "z": float(ann.rotation[2]),
                        },
                        "dimensions": {
                            "x": float(ann.scale[0]),
                            "y": float(ann.scale[1]),
                            "z": float(ann.scale[2]),
                        },
                    },
                    **ann_user_info,
                }
            )
            figure_id = ann.id
            if self._context._reindex or figure_id is None:
                figure_id = len(key_id_data["figures"]) + 1
            key_id_data["figures"][figure_key] = figure_id

    @staticmethod
    def _encode_attr_value(v):
        if v is True or v is False:  # use is to check the type too
            v = str(v).lower()
        return v

    @staticmethod
    def _define_attr_type(v):
        if isinstance(v, (int, float)):
            t = "any_number"
        else:
            t = "any_string"
        return t

    def _register_tag(self, name, **kwargs):
        tag = {
            "name": name,
            "value_type": None,
            "color": "",
            "id": len(self._tag_meta) + 1,
            "hotkey": "",
            "applicable_type": "all",
            "classes": set(),
        }
        tag.update(kwargs)
        return self._tag_meta.setdefault(name, tag)

    def _make_dirs(self):
        save_dir = self._context._save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._save_dir = save_dir

        base_dir = osp.join(self._save_dir, PointCloudPath.BASE_DIR)
        os.makedirs(base_dir, exist_ok=True)

        ann_dir = osp.join(base_dir, PointCloudPath.ANNNOTATION_DIR)
        os.makedirs(ann_dir, exist_ok=True)
        self._ann_dir = ann_dir

        point_cloud_dir = osp.join(base_dir, PointCloudPath.POINT_CLOUD_DIR)
        os.makedirs(point_cloud_dir, exist_ok=True)
        self._point_cloud_dir = point_cloud_dir

        related_images_dir = osp.join(base_dir, PointCloudPath.RELATED_IMAGES_DIR)
        os.makedirs(related_images_dir, exist_ok=True)
        self._related_images_dir = related_images_dir

    def _init_meta(self):
        for attr in self._label_cat.attributes:
            self._register_tag(attr, applicable_type="objectsOnly")

        for idx, label in enumerate(self._label_cat):
            self._meta_data["classes"].append(
                {
                    "id": idx,
                    "title": label.name,
                    "color": "",
                    "shape": "cuboid_3d",
                    "geometry_config": {},
                }
            )

            for attr in label.attributes:
                tag = self._register_tag(attr, applicable_type="objectsOnly")
                tag["classes"].add(label.name)

    def _find_image_ext(self, image):
        src_ext = image.ext
        return self._context._image_ext or src_ext or self._context._default_image_ext

    def dump(self):
        self._make_dirs()

        self._init_meta()

        for item in self._context._extractor:
            if self._context._save_media:
                if item.media:
                    self._write_pcd(item)
                else:
                    log.debug("Item '%s' has no point cloud info", item.id)

                if item.media and item.media.extra_images:
                    self._write_related_images(item)
                else:
                    log.debug("Item '%s' has no related images info", item.id)

            self._write_item_annotations(item)

        self._write_meta()
        self._write_key_id()


class SuperviselyPointCloudExporter(Exporter):
    NAME = "sly_pointcloud"
    DEFAULT_IMAGE_EXT = PointCloudPath.DEFAULT_IMAGE_EXT

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

    def __init__(self, extractor, save_dir, reindex=False, allow_undeclared_attrs=False, **kwargs):
        super().__init__(extractor, save_dir, **kwargs)

        self._reindex = reindex
        self._allow_undeclared_attrs = allow_undeclared_attrs

    def _apply_impl(self):
        if self._extractor.media_type() and self._extractor.media_type() is not PointCloud:
            raise MediaTypeError("Media type is not an image")

        if len(self._extractor.subsets()) > 1:
            log.warning(
                "Supervisely pointcloud format supports only a single "
                "subset. Subset information will be ignored on export."
            )

        _SuperviselyPointCloudDumper(self._extractor, self).dump()

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        conv = cls(patch.as_dataset(dataset), save_dir=save_dir, **kwargs)
        conv.apply()

        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.media):
                continue

            pcd_name = conv._make_pcd_filename(item)

            ann_path = osp.join(
                save_dir,
                PointCloudPath.BASE_DIR,
                PointCloudPath.ANNNOTATION_DIR,
                pcd_name + ".json",
            )
            if osp.isfile(ann_path):
                os.remove(ann_path)

            pcd_path = osp.join(save_dir, PointCloudPath.BASE_DIR, PointCloudPath.POINT_CLOUD_DIR, pcd_name)
            if osp.isfile(pcd_path):
                os.remove(pcd_path)

            images_dir = osp.join(
                save_dir,
                PointCloudPath.BASE_DIR,
                PointCloudPath.RELATED_IMAGES_DIR,
                osp.splitext(pcd_name)[0] + "_pcd",
            )
            if osp.isdir(images_dir):
                shutil.rmtree(images_dir)
