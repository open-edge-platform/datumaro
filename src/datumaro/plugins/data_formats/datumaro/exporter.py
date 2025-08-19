# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=no-self-use

import json
import os
import os.path as osp
import shutil
from contextlib import contextmanager
from multiprocessing.pool import Pool
from typing import Dict, Optional

import numpy as np
import pycocotools.mask as mask_utils
from json_stream.writer import streamable_dict, streamable_list

from datumaro.components.annotation import (
    Annotation,
    Bbox,
    Caption,
    Cuboid2D,
    Cuboid3d,
    Ellipse,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    RleMask,
    Shape,
)
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.dataset_item_storage import ItemStatus
from datumaro.components.errors import PathSeparatorInSubsetNameError
from datumaro.components.exporter import ExportContextComponent, Exporter
from datumaro.components.media import Image, MediaElement, PointCloud, Video, VideoFrame
from datumaro.util import cast, dump_json_file

from .format import DATUMARO_FORMAT_VERSION, DatumaroPath


class JsonWriter:
    @classmethod
    def _convert_attribute_categories(cls, attributes):
        return sorted(attributes)

    @classmethod
    def _convert_labels_label_groups(cls, labels):
        return sorted(labels)

    @classmethod
    def _convert_label_categories(cls, obj):
        converted = {
            "labels": [],
            "label_groups": [],
            "attributes": cls._convert_attribute_categories(obj.attributes),
        }
        for label in obj.items:
            converted["labels"].append(
                {
                    "name": cast(label.name, str),
                    "parent": cast(label.parent, str),
                    "attributes": cls._convert_attribute_categories(label.attributes),
                }
            )
        for label_group in obj.label_groups:
            converted["label_groups"].append(
                {
                    "name": cast(label_group.name, str),
                    "group_type": label_group.group_type.to_str(),
                    "labels": cls._convert_labels_label_groups(label_group.labels),
                }
            )
        return converted

    @classmethod
    def _convert_mask_categories(cls, obj):
        converted = {
            "colormap": [],
        }
        for label_id, color in obj.colormap.items():
            converted["colormap"].append(
                {
                    "label_id": int(label_id),
                    "r": int(color[0]),
                    "g": int(color[1]),
                    "b": int(color[2]),
                }
            )
        return converted

    @classmethod
    def _convert_points_categories(cls, obj):
        converted = {
            "items": [],
        }
        for label_id, item in obj.items.items():
            converted["items"].append(
                {
                    "label_id": int(label_id),
                    "labels": [cast(label, str) for label in item.labels],
                    "joints": [list(map(int, j)) for j in item.joints],
                    "positions": list(map(float, item.positions)),
                }
            )
        return converted

    @classmethod
    def write_categories(cls, categories) -> Dict[str, Dict]:
        dict_cat = {}
        for ann_type, desc in categories.items():
            if isinstance(desc, LabelCategories):
                converted_desc = cls._convert_label_categories(desc)
            elif isinstance(desc, MaskCategories):
                converted_desc = cls._convert_mask_categories(desc)
            elif isinstance(desc, PointsCategories):
                converted_desc = cls._convert_points_categories(desc)
            else:
                raise NotImplementedError()
            dict_cat[ann_type.name] = converted_desc

        return dict_cat


class _SubsetWriter:
    def __init__(
        self,
        context: Exporter,
        subset: str,
        ann_file: str,
        export_context: ExportContextComponent,
    ):
        self._context = context
        self._subset = subset

        self._data = {
            "dm_format_version": DATUMARO_FORMAT_VERSION,
            "media_type": context._extractor.media_type()._type,
            "infos": {},
            "categories": {},
            "items": [],
        }

        self.ann_file = ann_file
        self.export_context = export_context

    @property
    def infos(self):
        return self._data["infos"]

    @property
    def categories(self):
        return self._data["categories"]

    @property
    def items(self):
        return self._data["items"]

    def is_empty(self):
        return not self.items

    @staticmethod
    @contextmanager
    def context_save_media(
        item: DatasetItem, context: ExportContextComponent, encryption: bool = False
    ) -> None:
        """Implicitly change the media path and save it if save_media=True.
        When done, revert it's path as before.

        Parameters
        ----------
            item: Dataset item to save its media
            context: Context instance to help the media export
            encryption: If false, prevent the media from being encrypted
        """
        if item.media is None:
            yield
        elif isinstance(item.media, Video):
            video = item.media_as(Video)

            if context.save_media:
                fname = context.make_video_filename(item)
                subdir = item.subset.replace(os.sep, "_") if item.subset else None
                context.save_video(item, fname=fname, subdir=subdir)
                item.media = Video(
                    path=fname,
                    step=video._step,
                    start_frame=video._start_frame,
                    end_frame=video._end_frame,
                )

            yield
            item.media = video
        elif isinstance(item.media, VideoFrame):
            video_frame = item.media_as(VideoFrame)

            if context.save_media:
                fname = context.make_video_filename(item)
                subdir = item.subset.replace(os.sep, "_") if item.subset else None
                context.save_video(item, fname=fname, subdir=subdir)
                item.media = VideoFrame(Video(fname), video_frame.index)

            yield
            item.media = video_frame
        elif isinstance(item.media, Image):
            image = item.media_as(Image)

            if context.save_media:
                # Temporarily update image path and save it.
                fname = context.make_image_filename(item, name=str(item.id).replace(os.sep, "_"))
                subdir = item.subset.replace(os.sep, "_") if item.subset else None
                context.save_image(item, encryption=encryption, fname=fname, subdir=subdir)
                item.media = Image.from_file(path=fname, size=image._size)

            yield
            item.media = image
        elif isinstance(item.media, PointCloud):
            pcd = item.media_as(PointCloud)

            if context.save_media:
                pcd_name = str(item.id).replace(os.sep, "_")
                pcd_fname = context.make_pcd_filename(item, name=pcd_name)
                subdir = item.subset.replace(os.sep, "_") if item.subset else None
                context.save_point_cloud(item, fname=pcd_fname, subdir=subdir)

                extra_images = []
                for i, extra_image in enumerate(pcd.extra_images):
                    extra_images.append(
                        Image.from_file(
                            path=context.make_pcd_extra_image_filename(
                                item, i, extra_image, name=f"{pcd_name}/extra_image_{i}"
                            )
                        )
                    )

                # Temporarily update media with a new pcd saved into disk.
                item.media = PointCloud.from_file(path=pcd_fname, extra_images=extra_images)

            yield
            item.media = pcd
        else:
            raise NotImplementedError

    def add_item(self, item: DatasetItem, *args, **kwargs) -> None:
        self.items.append(self._gen_item_desc(item))

    def _gen_item_desc(self, item: DatasetItem, *args, **kwargs) -> Dict:
        annotations = []
        item_desc = {
            "id": item.id,
            "annotations": annotations,
        }

        if item.attributes:
            item_desc["attr"] = item.attributes

        with self.context_save_media(item, self.export_context):
            # Since VideoFrame is a descendant of Image, this condition should be ahead of Image
            if isinstance(item.media, VideoFrame):
                video_frame = item.media_as(VideoFrame)
                item_desc["video_frame"] = {
                    "video_path": getattr(video_frame.video, "path", None),
                    "frame_index": getattr(video_frame, "index", -1),
                }
            elif isinstance(item.media, Video):
                video = item.media_as(Video)
                item_desc["video"] = {
                    "path": getattr(video, "path", None),
                    "step": video._step,
                    "start_frame": video._start_frame,
                }
                if video._end_frame is not None:
                    item_desc["video"]["end_frame"] = video._end_frame
            elif isinstance(item.media, Image):
                image = item.media_as(Image)
                item_desc["image"] = {"path": getattr(image, "path", None)}
                if item.media.has_size:  # avoid occasional loading
                    item_desc["image"]["size"] = image.size
            elif isinstance(item.media, PointCloud):
                pcd = item.media_as(PointCloud)

                item_desc["point_cloud"] = {"path": getattr(pcd, "path", None)}

                related_images = [
                    {"path": getattr(img, "path", None), "size": img.size}
                    if img.has_size
                    else {"path": getattr(img, "path", None)}
                    for img in pcd.extra_images
                ]

                if related_images:
                    item_desc["related_images"] = related_images
            elif isinstance(item.media, MediaElement):
                item_desc["media"] = {"path": getattr(item.media, "path", None)}

        for ann in item.annotations:
            if isinstance(ann, Label):
                converted_ann = self._convert_label_object(ann)
            elif isinstance(ann, Mask):
                converted_ann = self._convert_mask_object(ann)
            elif isinstance(ann, Points):
                converted_ann = self._convert_points_object(ann)
            elif isinstance(ann, PolyLine):
                converted_ann = self._convert_polyline_object(ann)
            elif isinstance(ann, Polygon):
                converted_ann = self._convert_polygon_object(ann)
            elif isinstance(ann, Bbox):
                converted_ann = self._convert_bbox_object(ann)
            elif isinstance(ann, Caption):
                converted_ann = self._convert_caption_object(ann)
            elif isinstance(ann, Cuboid3d):
                converted_ann = self._convert_cuboid_3d_object(ann)
            elif isinstance(ann, Ellipse):
                converted_ann = self._convert_ellipse_object(ann)
            elif isinstance(ann, Cuboid2D):
                converted_ann = self._convert_cuboid_2d_object(ann)
            else:
                raise NotImplementedError()
            annotations.append(converted_ann)

        return item_desc

    def add_infos(self, infos):
        self._data["infos"].update(infos)

    def add_categories(self, categories):
        self._data["categories"] = JsonWriter.write_categories(categories)

    def write(self, *args, **kwargs):
        dump_json_file(self.ann_file, self._data)

    def _convert_annotation(self, obj):
        assert isinstance(obj, Annotation)

        ann_json = {
            "id": cast(obj.id, int),
            "type": cast(obj.type.name, str),
            "attributes": obj.attributes,
            "group": cast(obj.group, int, 0),
        }
        if obj.object_id >= 0:
            ann_json["object_id"] = cast(obj.object_id, int)
        return ann_json

    def _convert_label_object(self, obj):
        converted = self._convert_annotation(obj)

        converted.update(
            {
                "label_id": cast(obj.label, int),
            }
        )
        return converted

    def _convert_mask_object(self, obj):
        converted = self._convert_annotation(obj)

        if isinstance(obj, RleMask):
            rle = obj.rle
        else:
            rle = mask_utils.encode(np.require(obj.image, dtype=np.uint8, requirements="F"))

        if isinstance(rle["counts"], str):
            counts = rle["counts"]
        else:
            counts = rle["counts"].decode("ascii")

        converted.update(
            {
                "label_id": cast(obj.label, int),
                "rle": {
                    # serialize as compressed COCO mask
                    "counts": counts,
                    "size": list(int(c) for c in rle["size"]),
                },
                "z_order": obj.z_order,
            }
        )
        return converted

    def _convert_shape_object(self, obj):
        assert isinstance(obj, Shape)
        converted = self._convert_annotation(obj)

        converted.update(
            {
                "label_id": cast(obj.label, int),
                "points": [float(p) for p in obj.points],
                "z_order": obj.z_order,
            }
        )
        return converted

    def _convert_polyline_object(self, obj):
        return self._convert_shape_object(obj)

    def _convert_polygon_object(self, obj):
        return self._convert_shape_object(obj)

    def _convert_bbox_object(self, obj):
        converted = self._convert_shape_object(obj)
        converted.pop("points", None)
        converted["bbox"] = [float(p) for p in obj.get_bbox()]
        return converted

    def _convert_points_object(self, obj):
        converted = self._convert_shape_object(obj)

        converted.update(
            {
                "visibility": [int(v.value) for v in obj.visibility],
            }
        )
        return converted

    def _convert_caption_object(self, obj):
        converted = self._convert_annotation(obj)

        converted.update(
            {
                "caption": cast(obj.caption, str),
            }
        )
        return converted

    def _convert_cuboid_3d_object(self, obj):
        converted = self._convert_annotation(obj)
        converted.update(
            {
                "label_id": cast(obj.label, int),
                "position": [float(p) for p in obj.position],
                "rotation": [float(p) for p in obj.rotation],
                "scale": [float(p) for p in obj.scale],
            }
        )
        return converted

    def _convert_ellipse_object(self, obj: Ellipse):
        return self._convert_shape_object(obj)

    def _convert_cuboid_2d_object(self, obj: Cuboid2D):
        converted = self._convert_annotation(obj)

        converted.update(
            {
                "label_id": cast(obj.label, int),
                "points": obj.points,
                "z_order": obj.z_order,
            }
        )
        return converted


class _StreamSubsetWriter(_SubsetWriter):
    def __init__(
        self,
        context: Exporter,
        subset: str,
        ann_file: str,
        export_context: ExportContextComponent,
    ):
        super().__init__(context, subset, ann_file, export_context)

    def write(self, *args, **kwargs):
        @streamable_list
        def _item_list():
            subset = self._context._extractor.get_subset(self._subset)
            pbar = self._context._ctx.progress_reporter
            for item in pbar.iter(subset, desc=f"Exporting '{self._subset}'"):
                yield self._gen_item_desc(item)

        @streamable_dict
        def _data():
            yield "dm_format_version", self._data["dm_format_version"]
            yield "media_type", self._data["media_type"]
            yield "infos", self._data["infos"]
            yield "categories", self._data["categories"]
            yield "items", _item_list()

        with open(self.ann_file, "w", encoding="utf-8") as fp:
            json.dump(_data(), fp)

    def is_empty(self):
        # TODO: Force empty to be False, but it should be fixed with refactoring `_SubsetWriter`.`
        return False


class DatumaroExporter(Exporter):
    DEFAULT_IMAGE_EXT = DatumaroPath.IMAGE_EXT
    PATH_CLS = DatumaroPath

    def create_writer(
        self,
        subset: str,
        images_dir: str,
        pcd_dir: str,
        video_dir: str,
    ) -> _SubsetWriter:
        export_context = ExportContextComponent(
            save_dir=self._save_dir,
            save_media=self._save_media,
            images_dir=images_dir,
            pcd_dir=pcd_dir,
            video_dir=video_dir,
            image_ext=self._image_ext,
            default_image_ext=self._default_image_ext,
        )

        if os.path.sep in subset:
            raise PathSeparatorInSubsetNameError(subset)

        return (
            _SubsetWriter(
                context=self,
                subset=subset,
                ann_file=osp.join(
                    self._annotations_dir,
                    subset + self.PATH_CLS.ANNOTATION_EXT,
                ),
                export_context=export_context,
            )
            if not self._stream
            else _StreamSubsetWriter(
                context=self,
                subset=subset,
                ann_file=osp.join(
                    self._annotations_dir,
                    subset + self.PATH_CLS.ANNOTATION_EXT,
                ),
                export_context=export_context,
            )
        )

    def _apply_impl(self, pool: Optional[Pool] = None, *args, **kwargs):
        os.makedirs(self._save_dir, exist_ok=True)

        images_dir = osp.join(self._save_dir, self.PATH_CLS.IMAGES_DIR)
        os.makedirs(images_dir, exist_ok=True)
        self._images_dir = images_dir

        annotations_dir = osp.join(self._save_dir, self.PATH_CLS.ANNOTATIONS_DIR)
        os.makedirs(annotations_dir, exist_ok=True)
        self._annotations_dir = annotations_dir

        self._pcd_dir = osp.join(self._save_dir, self.PATH_CLS.PCD_DIR)
        self._video_dir = osp.join(self._save_dir, self.PATH_CLS.VIDEO_DIR)

        writers = {
            subset: self.create_writer(
                subset,
                self._images_dir,
                self._pcd_dir,
                self._video_dir,
            )
            for subset in self._extractor.subsets()
        }

        for writer in writers.values():
            writer.add_infos(self._extractor.infos())
            writer.add_categories(self._extractor.categories())

        pbar = self._ctx.progress_reporter
        for subset_name, subset in self._extractor.subsets().items():
            if not self._stream:
                for item in pbar.iter(subset, desc=f"Exporting '{subset_name}'"):
                    writers[subset_name].add_item(item, pool)

        for subset, writer in writers.items():
            if self._patch and subset in self._patch.updated_subsets and writer.is_empty():
                if osp.isfile(writer.ann_file):
                    # Remove subsets that became empty
                    os.remove(writer.ann_file)
                continue

            writer.write(pool)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        for subset in patch.updated_subsets:
            conv = cls(dataset.get_subset(subset), save_dir=save_dir, **kwargs)
            conv._patch = patch
            conv.apply()

        conv = cls(dataset, save_dir=save_dir, **kwargs)
        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.media):
                continue

            image_path = osp.join(
                save_dir, cls.PATH_CLS.IMAGES_DIR, item.subset, conv._make_image_filename(item)
            )
            if osp.isfile(image_path):
                os.unlink(image_path)

            pcd_path = osp.join(
                save_dir, cls.PATH_CLS.PCD_DIR, item.subset, conv._make_pcd_filename(item)
            )
            if osp.isfile(pcd_path):
                os.unlink(pcd_path)

            related_images_path = osp.join(save_dir, cls.PATH_CLS.IMAGES_DIR, item.subset, item.id)
            if osp.isdir(related_images_path):
                shutil.rmtree(related_images_path)

    @property
    def can_stream(self) -> bool:
        return True
