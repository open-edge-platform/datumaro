# Copyright (C) 2022-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import os.path as osp
from glob import iglob
from typing import List, Optional

from datumaro.components.annotation import AnnotationType, Cuboid3d, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import InvalidFieldError, UndeclaredLabelError
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image, PointCloud
from datumaro.util import parse_json_file
from datumaro.util.image import find_images

from .format import PointCloudPath


class SuperviselyPointCloudBase(SubsetBase):
    NAME = "sly_pointcloud"
    _SUPPORTED_SHAPES = "cuboid"

    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isfile(path):
            raise FileNotFoundError(errno.ENOENT, "Can't find annotations file", path)

        rootdir = osp.abspath(osp.dirname(path))
        self._rootdir = rootdir

        super().__init__(subset=subset, media_type=PointCloud, ctx=ctx)

        items, categories = self._parse(rootdir)
        self._categories = categories
        self._items = list(self._load_items(items).values())

    @classmethod
    def _parse(cls, rootpath):
        mapping = parse_json_file(osp.join(rootpath, PointCloudPath.KEY_ID_FILE))
        meta = parse_json_file(osp.join(rootpath, PointCloudPath.META_FILE))

        label_cat = LabelCategories()
        for label in meta.get("classes", []):
            label_cat.add(label["title"])

        tags = {}
        for tag in meta.get("tags", []):
            # See reference at:
            # https://github.com/supervisely/supervisely/blob/047e52ebe407cfee61464c1bd0beb9c906892253/supervisely_lib/annotation/tag_meta.py#L139
            tags[tag["name"]] = tag

            applicable_to = tag.get("applicable_type", "all")
            if applicable_to == "imagesOnly":
                continue  # an image attribute
            if applicable_to not in {"all", "objectsOnly"}:
                raise InvalidFieldError(applicable_to)

            applicable_classes = tag.get("classes", [])
            if not applicable_classes:
                label_cat.attributes.add(tag["name"])
            else:
                for label_name in applicable_classes:
                    _, label = label_cat.find(label_name)
                    if label is None:
                        raise UndeclaredLabelError(label_name)

                    label.attributes.add(tag["name"])

        categories = {AnnotationType.label: label_cat}

        def _get_label_attrs(label_id):
            attrs = set(label_cat.attributes)
            attrs.update(label_cat[label_id].attributes)
            return attrs

        def _parse_tag(tag):
            if tag["value"] == "true":
                value = True
            elif tag["value"] == "false":
                value = False
            else:
                value = tag["value"]
            return value

        ann_dir = osp.join(rootpath, PointCloudPath.BASE_DIR, PointCloudPath.ANNNOTATION_DIR)
        items = {}
        for ann_file in iglob(osp.join(ann_dir, "**", "*.json"), recursive=True):
            ann_data = parse_json_file(ann_file)

            objects = {}
            for obj in ann_data["objects"]:
                obj["id"] = mapping["objects"][obj["key"]]
                objects[obj["key"]] = obj

            frame_attributes = {"description": ann_data.get("description", "")}
            for tag in ann_data["tags"]:
                frame_attributes[tag["name"]] = _parse_tag(tag)

            frame = mapping["videos"][ann_data["key"]]
            frame_desc = items.setdefault(
                frame,
                {
                    "name": osp.splitext(osp.relpath(ann_file, ann_dir))[0],
                    "annotations": [],
                    "attributes": frame_attributes,
                },
            )

            for figure in ann_data["figures"]:
                geometry = {
                    dst_field: [float(figure["geometry"][src_field][axis]) for axis in ["x", "y", "z"]]
                    for src_field, dst_field in {
                        "position": "position",
                        "rotation": "rotation",
                        "dimensions": "scale",
                    }.items()
                }

                ann_id = mapping["figures"][figure["key"]]

                obj = objects[figure["objectKey"]]
                label = categories[AnnotationType.label].find(obj["classTitle"])[0]

                attributes = {}
                attributes["track_id"] = obj["id"]
                for tag in obj.get("tags", []):
                    attributes[tag["name"]] = _parse_tag(tag)
                for attr in _get_label_attrs(label):
                    if attr in attributes:
                        continue
                    if tags[attr]["value_type"] == "any_string":
                        value = ""
                    elif tags[attr]["value_type"] == "oneof_string":
                        value = (tags[attr]["values"] or [""])[0]
                    elif tags[attr]["value_type"] == "any_number":
                        value = 0
                    else:
                        value = None
                    attributes[attr] = value

                shape = Cuboid3d(**geometry, label=label, id=ann_id, attributes=attributes)

                frame_desc["annotations"].append(shape)

        return items, categories

    def _load_items(self, parsed):
        for frame_id, frame_desc in parsed.items():
            pcd_name = frame_desc["name"]
            name = osp.splitext(pcd_name)[0]
            pcd_path = osp.join(self._rootdir, PointCloudPath.BASE_DIR, PointCloudPath.POINT_CLOUD_DIR, pcd_name)
            assert pcd_path.endswith(".pcd"), pcd_path

            related_images_dir = osp.join(
                self._rootdir,
                PointCloudPath.BASE_DIR,
                PointCloudPath.RELATED_IMAGES_DIR,
                name + "_pcd",
            )
            related_images = None
            if osp.isdir(related_images_dir):
                related_images = [Image.from_file(path=image) for image in find_images(related_images_dir)]

            parsed[frame_id] = DatasetItem(
                id=name,
                subset=self._subset,
                media=PointCloud.from_file(path=pcd_path, extra_images=related_images),
                annotations=frame_desc.get("annotations"),
                attributes={"frame": int(frame_id), **frame_desc["attributes"]},
            )

            for ann in frame_desc.get("annotations"):
                self._ann_types.add(ann.type)

        return parsed


class SuperviselyPointCloudImporter(Importer):
    NAME = "sly_pointcloud"
    _ANNO_EXT = ".json"

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, cls._ANNO_EXT, "sly_pointcloud", filename="meta")

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]
