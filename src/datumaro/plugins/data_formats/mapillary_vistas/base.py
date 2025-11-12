# Copyright (C) 2022-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import logging as log
import os
import os.path as osp
from typing import Optional

import numpy as np

from datumaro.components.annotation import AnnotationType, ExtractedMask, LabelCategories, MaskCategories, Polygon
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import DatasetImportError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image
from datumaro.util import parse_json_file
from datumaro.util.image import find_images, lazy_image, load_image
from datumaro.util.mask_tools import bgr2index
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file

from .format import (
    MapillaryVistasLabelMaps,
    MapillaryVistasPath,
    MapillaryVistasTask,
    make_mapillary_instance_categories,
    parse_config_file,
)


class _MapillaryVistasBase(SubsetBase):
    def __init__(
        self,
        path: str,
        task: MapillaryVistasTask,
        *,
        use_original_config: bool = False,
        keep_original_category_ids: bool = False,
        format_version: str = "v2.0",
        parse_polygon: bool = False,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if format_version == "v1.2" and parse_polygon is True:
            raise DatasetImportError(
                f"Format version {format_version} is not available for polygons. "
                "Please try with v2.0 for parsing polygons."
            )

        assert osp.isdir(path), path
        self._path = path
        if subset is None:
            subset = osp.basename(self._path)
        super().__init__(subset=subset, ctx=ctx)

        annotations_dirs = [d for d in os.listdir(path) if d in MapillaryVistasPath.ANNOTATION_DIRS]

        if len(annotations_dirs) == 0:
            expected_dirs = ",".join(MapillaryVistasPath.ANNOTATION_DIRS[format_version])
            raise NotADirectoryError(
                errno.ENOTDIR,
                f"Can't find annotation directory at {path}. Expected one of these directories: {expected_dirs}.",
            )
        if len(annotations_dirs) > 1:
            skipped_dirs = ",".join(annotations_dirs[1:])
            log.warning(
                f"Directory(-es): {skipped_dirs} will be skipped, dataset should contain only one annotation directory"
            )

        self._use_original_config = use_original_config
        self._format_version = format_version
        self._parse_polygon = parse_polygon
        self._annotations_dir = osp.join(path, format_version)
        self._images_dir = osp.join(path, MapillaryVistasPath.IMAGES_DIR)

        if task == MapillaryVistasTask.instances:
            if has_meta_file(path):
                self._categories = make_mapillary_instance_categories(parse_meta_file(path))
            else:
                self._categories = self._load_instances_categories()
            self._items = self._load_instances_items()
        else:
            panoptic_config = self._load_panoptic_config(self._annotations_dir)
            self._categories = self._load_panoptic_categories(panoptic_config["categories"], keep_original_category_ids)
            self._items = self._load_panoptic_items(panoptic_config)

    def _load_panoptic_config(self, path):
        panoptic_config_path = osp.join(
            path,
            MapillaryVistasPath.PANOPTIC_DIR,
            MapillaryVistasPath.PANOPTIC_CONFIG[self._format_version],
        )

        if not osp.isfile(panoptic_config_path):
            raise FileNotFoundError(errno.ENOENT, "Can't find panoptic config file", panoptic_config_path)

        return parse_json_file(panoptic_config_path)

    def _load_panoptic_categories(self, categories_info, keep_original_ids):
        label_cat = LabelCategories()
        label_map, color_map = {}, {}

        if keep_original_ids:
            for cat in sorted(categories_info, key=lambda cat: cat["id"]):
                label_map[cat["id"]] = cat["id"]
                color_map[cat["id"]] = tuple(map(int, cat["color"]))

                while len(label_cat) < cat["id"]:
                    label_cat.add(name=f"class-{len(label_cat)}")

                label_cat.add(name=cat["name"], parent=cat.get("supercategory"))
        else:
            for idx, cat in enumerate(categories_info):
                label_map[cat["id"]] = idx
                color_map[idx] = tuple(map(int, cat["color"]))

                label_cat.add(name=cat["name"], parent=cat.get("supercategory"))

        self._label_map = label_map
        mask_cat = MaskCategories(color_map)
        mask_cat.inverse_colormap  # pylint: disable=pointless-statement

        return {AnnotationType.label: label_cat, AnnotationType.mask: mask_cat}

    def _load_panoptic_items(self, config):
        items = {}

        images_info = {
            img["id"]: {
                "path": osp.join(self._images_dir, img["file_name"]),
                "height": img.get("height"),
                "width": img.get("width"),
            }
            for img in config["images"]
        }

        polygon_dir = osp.join(self._annotations_dir, MapillaryVistasPath.POLYGON_DIR)
        for item_ann in config["annotations"]:
            item_id = item_ann["image_id"]
            image = None
            if images_info.get(item_id):
                image = Image.from_file(
                    path=images_info[item_id]["path"],
                    size=self._get_image_size(images_info[item_id]),
                )

            mask_path = osp.join(self._annotations_dir, MapillaryVistasPath.PANOPTIC_DIR, item_ann["file_name"])
            mask = lazy_image(mask_path, loader=self._load_pan_mask)

            annotations = []
            for segment_info in item_ann["segments_info"]:
                cat_id = self._get_label_id(segment_info)
                segment_id = segment_info["id"]
                attributes = {"is_crowd": bool(segment_info["iscrowd"])}
                annotations.append(
                    ExtractedMask(
                        index_mask=mask,
                        index=segment_id,
                        label=cat_id,
                        id=segment_id,
                        group=segment_id,
                        attributes=attributes,
                    )
                )

            if self._parse_polygon:
                polygon_path = osp.join(polygon_dir, item_id + ".json")
                item_info = parse_json_file(polygon_path)

                polygons = item_info["objects"]
                for polygon in polygons:
                    label = polygon["label"]
                    label_id = self._categories[AnnotationType.label].find(label)[0]
                    if label_id is None:
                        label_id = self._categories[AnnotationType.label].add(label)

                    points = [int(coord) for point in polygon["polygon"] for coord in point]
                    annotations.append(Polygon(label=label_id, points=points))

            items[item_id] = DatasetItem(id=item_id, subset=self._subset, annotations=annotations, media=image)
            for ann in annotations:
                self._ann_types.add(ann.type)

        return items.values()

    def _load_instances_categories(self):
        config_file = MapillaryVistasPath.CONFIG_FILES[self._format_version]
        label_map = None

        if self._use_original_config:
            label_map = MapillaryVistasLabelMaps[self._format_version]
        else:
            try:
                label_map = parse_config_file(osp.join(self._path, config_file))
            except FileNotFoundError:
                label_map = parse_config_file(osp.join(osp.dirname(self._path), config_file))
        return make_mapillary_instance_categories(label_map)

    def _load_instances_items(self):
        items = {}

        instance_dir = osp.join(self._annotations_dir, MapillaryVistasPath.INSTANCES_DIR)
        polygon_dir = osp.join(self._annotations_dir, MapillaryVistasPath.POLYGON_DIR)
        for image_path in find_images(self._images_dir, recursive=True):
            item_id = osp.splitext(osp.relpath(image_path, self._images_dir))[0]
            image = Image.from_file(path=image_path)

            instance_path = osp.join(instance_dir, item_id + MapillaryVistasPath.MASK_EXT)
            index_mask = lazy_image(instance_path, dtype=np.uint32)
            np_index_mask = index_mask()

            annotations = []
            for uval in np.unique(np_index_mask):
                label_id, instance_id = uval >> 8, uval & 255
                annotations.append(ExtractedMask(index_mask=index_mask, index=uval, label=label_id, id=instance_id))

            if self._parse_polygon:
                polygon_path = osp.join(polygon_dir, item_id + ".json")
                item_info = parse_json_file(polygon_path)

                polygons = item_info["objects"]
                for polygon in polygons:
                    label = polygon["label"]
                    label_id = self._categories[AnnotationType.label].find(label)[0]
                    if label_id is None:
                        label_id = self._categories[AnnotationType.label].add(label)

                    points = [int(coord) for point in polygon["polygon"] for coord in point]
                    annotations.append(Polygon(label=label_id, points=points))

            for ann in annotations:
                self._ann_types.add(ann.type)

            items[item_id] = DatasetItem(id=item_id, subset=self._subset, media=image, annotations=annotations)

        return items.values()

    @staticmethod
    def _get_image_size(image_info):
        image_size = image_info.get("height"), image_info.get("width")
        if all(image_size):
            return int(image_size[0]), int(image_size[1])
        return None

    @staticmethod
    def _load_pan_mask(path):
        mask = load_image(path)
        return bgr2index(mask)

    def _get_label_id(self, ann):
        cat_id = ann.get("category_id")
        if cat_id in [0, None]:
            return None
        return self._label_map[cat_id]


class MapillaryVistasInstancesBase(_MapillaryVistasBase):
    def __init__(self, path, **kwargs):
        kwargs["task"] = MapillaryVistasTask.instances
        super().__init__(path, **kwargs)


class MapillaryVistasPanopticBase(_MapillaryVistasBase):
    def __init__(self, path, **kwargs):
        kwargs["task"] = MapillaryVistasTask.panoptic
        super().__init__(path, **kwargs)
