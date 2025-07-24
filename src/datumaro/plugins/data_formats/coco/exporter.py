# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import logging as log
import os
import os.path as osp
from dataclasses import dataclass
from enum import Enum, auto
from io import BufferedWriter
from itertools import chain, groupby
from typing import Dict, List, Optional, Type, Union

import numpy as np
import pycocotools.mask as mask_utils
from json_stream.writer import streamable_dict, streamable_list

import datumaro.util.annotation_util as anno_tools
import datumaro.util.mask_tools as mask_tools
from datumaro.components.annotation import (
    COORDINATE_ROUNDING_DIGITS,
    AnnotationType,
    Ellipse,
    Points,
    Polygon,
)
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.dataset_item_storage import ItemStatus
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.media import Image
from datumaro.util import cast, dump_json, dump_json_file, find, parse_json, str_to_bool
from datumaro.util.image import save_image

from .format import CocoPath, CocoTask


class SegmentationMode(Enum):
    guess = auto()
    polygons = auto()
    mask = auto()


@dataclass
class _Writer:
    fp: BufferedWriter
    is_empty: bool


class TemporaryWriters:
    def __init__(self, subset: str, task: CocoTask, ann_dir: str):
        self._subset = subset
        self._task = task
        self._ann_dir = ann_dir
        self._writers = tuple()
        self.reset()

    def close(self):
        for writer in self._writers:
            if not writer.fp.closed:
                writer.fp.close()

    def remove(self):
        self.close()

        for writer in self._writers:
            fpath = writer.fp.name
            if osp.exists(fpath):
                os.remove(fpath)

    def reset(self):
        self.remove()

        self._writers = tuple(
            _Writer(
                fp=open(
                    osp.join(self._ann_dir, f"__{self._task.name}_{self._subset}_{key}.tmp"), "wb"
                ),
                is_empty=True,
            )
            for key in ["imgs", "anns"]
        )

    def __del__(self):
        self.remove()

    @property
    def imgs(self) -> _Writer:
        return self._writers[0]

    @property
    def anns(self) -> _Writer:
        return self._writers[1]

    def add_item(self, data: Dict) -> None:
        self.imgs.is_empty = False
        writer = self.imgs.fp
        writer.write(dump_json(data, append_newline=True))

    def add_anns(self, data: Dict) -> None:
        self.anns.is_empty = False
        writer = self.anns.fp
        writer.write(dump_json(data, append_newline=True))

    def merge(self, path: str, header: Dict, min_ann_id: Optional[int]) -> None:
        self.close()

        @streamable_list
        def _gen_images():
            with open(self.imgs.fp.name, "rb") as fp:
                for line in fp:
                    yield parse_json(line)

        @streamable_list
        def _gen_anns():
            with open(self.anns.fp.name, "rb") as fp:
                next_id = min_ann_id
                for line in fp:
                    ann = parse_json(line)
                    if min_ann_id is not None and not ann["id"]:
                        ann["id"] = next_id
                        next_id += 1
                    yield ann

        @streamable_dict
        def _gen():
            yield "licenses", header["licenses"]
            yield "info", header["info"]
            yield "categories", header["categories"]

            if not self.imgs.is_empty:
                yield "images", _gen_images()
            else:
                yield "images", []

            if not self.anns.is_empty:
                yield "annotations", _gen_anns()
            else:
                yield "annotations", []

        with open(path, "w", encoding="utf-8") as fp:
            json.dump(_gen(), fp)

        self.remove()


class _TaskExporter:
    def __init__(
        self, context: "CocoExporter", subset: str, task: CocoTask, ann_dir: str, stream: bool
    ):
        self._min_ann_id = 1
        self._context = context
        self._subset = subset
        self._task = task
        self._ann_dir = ann_dir
        self._stream = stream

        data = {"licenses": [], "info": {}, "categories": [], "images": [], "annotations": []}

        data["licenses"].append({"name": "", "id": 0, "url": ""})

        data["info"] = {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": "",
        }
        self._data = data
        self._temporary_writers = TemporaryWriters(
            subset=subset,
            task=task,
            ann_dir=ann_dir,
        )

    def is_empty(self):
        return (
            len(self._data["annotations"]) == 0
            if not self._stream
            else self._temporary_writers.anns.is_empty
        )

    def _get_image_id(self, item):
        return self._context._get_image_id(item)

    def save_image_info(self, item, filename):
        h = 0
        w = 0
        if item.media and item.media.size:
            h, w = item.media.size

        item_desc = {
            "id": self._get_image_id(item),
            "width": int(w),
            "height": int(h),
            "file_name": cast(filename, str, ""),
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0,
        }

        if not self._stream:
            self._data["images"].append(item_desc)
        else:
            self._temporary_writers.add_item(item_desc)

    def save_categories(self, dataset):
        raise NotImplementedError()

    def save_annotations(self, item):
        raise NotImplementedError()

    def write(self, path):
        next_id = self._min_ann_id
        for ann in self.annotations:
            if not ann["id"]:
                ann["id"] = next_id
                next_id += 1

        dump_json_file(path, self._data)

        if self._stream:
            self._temporary_writers.merge(path, self._data, self._min_ann_id)

    @property
    def annotations(self):
        return self._data["annotations"]

    @property
    def infos(self):
        return self._data["info"]

    @property
    def categories(self):
        return self._data["categories"]

    def _get_ann_id(self, annotation):
        ann_id = 0 if self._context._reindex else annotation.id
        if ann_id:
            self._min_ann_id = max(ann_id, self._min_ann_id)
        return ann_id

    @staticmethod
    def _convert_attributes(ann):
        return {k: v for k, v in ann.attributes.items() if k not in {"is_crowd", "score"}}


class _ImageInfoExporter(_TaskExporter):
    def is_empty(self):
        return (
            len(self._data["images"]) == 0
            if not self._stream
            else self._temporary_writers.imgs.is_empty
        )

    def save_categories(self, dataset):
        pass

    def save_annotations(self, item):
        pass


class _CaptionsExporter(_TaskExporter):
    def save_categories(self, dataset):
        pass

    def save_annotations(self, item):
        for ann_idx, ann in enumerate(item.annotations):
            if ann.type != AnnotationType.caption:
                continue

            elem = {
                "id": self._get_ann_id(ann),
                "image_id": self._get_image_id(item),
                "category_id": 0,  # NOTE: workaround for a bug in cocoapi
                "caption": ann.caption,
            }
            if "score" in ann.attributes:
                try:
                    elem["score"] = float(ann.attributes["score"])
                except Exception as e:
                    log.warning(
                        "Item '%s', ann #%s: failed to convert "
                        "attribute 'score': %e" % (item.id, ann_idx, e)
                    )
            if self._context._allow_attributes:
                attrs = self._convert_attributes(ann)
                if attrs:
                    elem["attributes"] = attrs

            if not self._stream:
                self.annotations.append(elem)
            else:
                self._temporary_writers.add_anns(elem)


class _InstancesExporter(_TaskExporter):
    _polygon_types = {AnnotationType.polygon, AnnotationType.ellipse}
    _allowed_types = {
        AnnotationType.bbox,
        AnnotationType.polygon,
        AnnotationType.mask,
        AnnotationType.ellipse,
    }

    def save_categories(self, dataset):
        label_categories = dataset.categories().get(AnnotationType.label)
        if label_categories is None:
            return

        for idx, cat in enumerate(label_categories.items):
            self.categories.append(
                {
                    "id": 1 + idx,
                    "name": cast(cat.name, str, ""),
                    "supercategory": cast(cat.parent, str, ""),
                }
            )

    @classmethod
    def crop_segments(cls, instances, img_width, img_height):
        instances = sorted(instances, key=lambda x: x[0].z_order)

        segment_map = []
        segments = []
        for inst_idx, (_, polygons, mask, _) in enumerate(instances):
            if polygons:
                segment_map.extend(inst_idx for p in polygons)
                segments.extend(polygons)
            elif mask is not None:
                segment_map.append(inst_idx)
                segments.append(mask)

        segments = mask_tools.crop_covered_segments(segments, img_width, img_height)

        for inst_idx, inst in enumerate(instances):
            new_segments = [s for si_id, s in zip(segment_map, segments) if si_id == inst_idx]

            if not new_segments:
                inst[1] = []
                inst[2] = None
                continue

            if inst[1]:
                inst[1] = sum(new_segments, [])
            else:
                mask = mask_tools.merge_masks(new_segments)
                inst[2] = mask_tools.mask_to_rle(mask)

        return instances

    def find_instance_parts(self, group, img_width, img_height):
        boxes = [a for a in group if a.type == AnnotationType.bbox]
        polygons: List[Union[Polygon, Ellipse]] = [
            a for a in group if a.type in self._polygon_types
        ]
        masks = [a for a in group if a.type == AnnotationType.mask]

        anns = boxes + polygons + masks
        leader = anno_tools.find_group_leader(anns)
        if len(boxes) > 0:
            bbox = anno_tools.max_bbox(boxes)
        else:
            bbox = anno_tools.max_bbox(anns)
        mask = None
        polygons = [p.as_polygon() for p in polygons]

        if self._context._segmentation_mode == SegmentationMode.guess:
            use_masks = True is leader.attributes.get(
                "is_crowd", find(masks, lambda x: x.label == leader.label) is not None
            )
        elif self._context._segmentation_mode == SegmentationMode.polygons:
            use_masks = False
        elif self._context._segmentation_mode == SegmentationMode.mask:
            use_masks = True
        else:
            raise NotImplementedError(
                "Unexpected segmentation mode '%s'" % self._context._segmentation_mode
            )

        if use_masks:
            if polygons and img_width > 0 and img_height > 0:
                mask = mask_tools.rles_to_mask(polygons, img_width, img_height)

            if masks:
                masks = (m.image for m in masks)
                if mask is not None:
                    masks = chain(masks, [mask])
                mask = mask_tools.merge_masks(masks)

            if mask is not None:
                mask = mask_tools.mask_to_rle(mask)
            polygons = []
        else:
            if masks:
                mask = mask_tools.merge_masks(m.image for m in masks)
                polygons += mask_tools.mask_to_polygons(mask)
            mask = None

        return [leader, polygons, mask, bbox]

    @staticmethod
    def find_instance_anns(annotations):
        return [a for a in annotations if a.type in _InstancesExporter._allowed_types]

    @classmethod
    def find_instances(cls, annotations):
        return anno_tools.find_instances(cls.find_instance_anns(annotations))

    def save_annotations(self, item):
        instances = self.find_instances(item.annotations)
        if not instances:
            return

        if not item.media or not item.media.size:
            h, w = 0, 0
            log.warning(
                "Item '%s': Mask annotations can be skipped since no image info available" % item.id
            )
        else:
            h, w = item.media.size

        instances = [self.find_instance_parts(i, w, h) for i in instances]

        if self._context._crop_covered and w > 0 and h > 0:
            instances = self.crop_segments(instances, w, h)

        for instance in instances:
            elem = self.convert_instance(instance, item)

            if elem is None:
                continue

            if not self._stream:
                self.annotations.append(elem)
            else:
                self._temporary_writers.add_anns(elem)

    def convert_instance(self, instance, item) -> Optional[Dict]:
        ann, polygons, mask, bbox = instance

        is_crowd = mask is not None
        if is_crowd:
            segmentation = {
                "counts": list(int(c) for c in mask["counts"]),
                "size": list(int(c) for c in mask["size"]),
            }
        else:
            segmentation = [list(map(float, p)) for p in polygons]

        area = 0
        if segmentation:
            if item.media and item.media.size:
                h, w = item.media.size
            else:
                # Here we can guess the image size as
                # it is only needed for the area computation
                w = bbox[0] + bbox[2]
                h = bbox[1] + bbox[3]

            rles = mask_utils.frPyObjects(segmentation, h, w)
            if is_crowd:
                rles = [rles]
            else:
                rles = mask_utils.merge(rles)
            area = mask_utils.area(rles)
            if isinstance(area, np.ndarray):
                area = area[0]
        else:
            _, _, w, h = bbox
            segmentation = []
            area = w * h

        elem = {
            "id": self._get_ann_id(ann),
            "image_id": self._get_image_id(item),
            "category_id": cast(ann.label, int, -1) + 1,
            "segmentation": segmentation,
            "area": float(area),
            "bbox": [round(float(n), COORDINATE_ROUNDING_DIGITS) for n in bbox],
            "iscrowd": int(is_crowd),
        }
        if "score" in ann.attributes:
            try:
                elem["score"] = float(ann.attributes["score"])
            except Exception as e:
                log.warning("Item '%s': failed to convert attribute " "'score': %e" % (item.id, e))
        if self._context._allow_attributes:
            attrs = self._convert_attributes(ann)
            if attrs:
                elem["attributes"] = attrs

        return elem


class _KeypointsExporter(_InstancesExporter):
    def save_categories(self, dataset):
        label_categories = dataset.categories().get(AnnotationType.label)
        if label_categories is None:
            return
        point_categories = dataset.categories().get(AnnotationType.points)

        for idx, label_cat in enumerate(label_categories.items):
            cat = {
                "id": 1 + idx,
                "name": cast(label_cat.name, str, ""),
                "supercategory": cast(label_cat.parent, str, ""),
                "keypoints": [],
                "skeleton": [],
            }

            if point_categories is not None:
                kp_cat = point_categories.items.get(idx)
                if kp_cat is not None:
                    cat.update(
                        {
                            "keypoints": [str(l) for l in kp_cat.labels],
                            "skeleton": [list(map(int, j)) for j in kp_cat.joints],
                        }
                    )
            self.categories.append(cat)

    def save_annotations(self, item):
        point_annotations = [a for a in item.annotations if a.type == AnnotationType.points]
        if not point_annotations:
            return

        # Create annotations for solitary keypoints annotations
        for points in self.find_solitary_points(item.annotations):
            instance = [points, [], None, points.get_bbox()]
            elem = super().convert_instance(instance, item)
            elem.update(self.convert_points_object(points))
            if not self._stream:
                self.annotations.append(elem)
            else:
                self._temporary_writers.add_anns(elem)

        # Create annotations for complete instance + keypoints annotations
        super().save_annotations(item)

    @classmethod
    def find_solitary_points(cls, annotations):
        annotations = sorted(annotations, key=lambda a: a.group)
        solitary_points = []

        for g_id, group in groupby(annotations, lambda a: a.group):
            if not g_id or g_id and not cls.find_instance_anns(group):
                group = [a for a in group if a.type == AnnotationType.points]
                solitary_points.extend(group)

        return solitary_points

    @staticmethod
    def convert_points_object(ann):
        keypoints = []
        points = ann.points
        visibility = ann.visibility
        for index in range(0, len(points), 2):
            kp = points[index : index + 2]
            state = visibility[index // 2].value
            keypoints.extend([*kp, state])

        num_annotated = len([v for v in visibility if v != Points.Visibility.absent])

        return {
            "keypoints": keypoints,
            "num_keypoints": num_annotated,
        }

    def convert_instance(self, instance, item) -> Optional[Dict]:
        points_ann = find(
            item.annotations,
            lambda x: x.type == AnnotationType.points
            and instance[0].group
            and x.group == instance[0].group,
        )
        if not points_ann:
            return None

        elem = super().convert_instance(instance, item)
        elem.update(self.convert_points_object(points_ann))

        return elem


class _LabelsExporter(_TaskExporter):
    def save_categories(self, dataset):
        label_categories = dataset.categories().get(AnnotationType.label)
        if label_categories is None:
            return

        for idx, cat in enumerate(label_categories.items):
            self.categories.append(
                {
                    "id": 1 + idx,
                    "name": cast(cat.name, str, ""),
                    "supercategory": cast(cat.parent, str, ""),
                }
            )

    def save_annotations(self, item):
        for ann in item.annotations:
            if ann.type != AnnotationType.label:
                continue

            elem = {
                "id": self._get_ann_id(ann),
                "image_id": self._get_image_id(item),
                "category_id": int(ann.label) + 1,
            }
            if "score" in ann.attributes:
                try:
                    elem["score"] = float(ann.attributes["score"])
                except Exception as e:
                    log.warning(
                        "Item '%s': failed to convert attribute " "'score': %e" % (item.id, e)
                    )
            if self._context._allow_attributes:
                attrs = self._convert_attributes(ann)
                if attrs:
                    elem["attributes"] = attrs

            if not self._stream:
                self.annotations.append(elem)
            else:
                self._temporary_writers.add_anns(elem)


class _StuffExporter(_InstancesExporter):
    pass


class _PanopticExporter(_TaskExporter):
    def write(self, path):
        dump_json_file(path, self._data)

        if self._stream:
            self._temporary_writers.merge(path, self._data, None)

    def save_categories(self, dataset):
        label_categories = dataset.categories().get(AnnotationType.label)
        if label_categories is None:
            return

        for idx, cat in enumerate(label_categories.items):
            self.categories.append(
                {
                    "id": 1 + idx,
                    "name": cast(cat.name, str, ""),
                    "supercategory": cast(cat.parent, str, ""),
                    "isthing": 0,  # TODO: can't represent this information yet
                }
            )

    def save_annotations(self, item):
        if not item.media:
            return

        ann_filename = item.id + CocoPath.PANOPTIC_EXT

        segments_info = list()
        masks = []
        next_id = self._min_ann_id
        for ann in item.annotations:
            if ann.type != AnnotationType.mask:
                continue

            if not ann.id:
                ann.id = next_id
                next_id += 1

            segment_info = {}
            segment_info["id"] = ann.id
            segment_info["category_id"] = cast(ann.label, int, -1) + 1
            segment_info["area"] = float(ann.get_area())
            segment_info["bbox"] = [float(p) for p in ann.get_bbox()]
            segment_info["iscrowd"] = cast(ann.attributes.get("is_crowd"), int, 0)
            segments_info.append(segment_info)
            masks.append(ann)

        if not masks:
            return

        pan_format = mask_tools.merge_masks((m.image, m.id) for m in masks)
        save_image(
            osp.join(self._context._segmentation_dir, ann_filename),
            mask_tools.index2bgr(pan_format),
            create_dir=True,
        )

        elem = {
            "image_id": self._get_image_id(item),
            "file_name": ann_filename,
            "segments_info": segments_info,
        }

        if not self._stream:
            self.annotations.append(elem)
        else:
            self._temporary_writers.add_anns(elem)


class CocoExporter(Exporter):
    @staticmethod
    def _split_tasks_string(s):
        return [CocoTask[i.strip()] for i in s.split(",")]

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        kwargs[
            "description"
        ] = """
            Segmentation mask modes ('--segmentation-mode'):|n
            - '{sm.guess.name}': guess the mode for each instance,|n
            |s|suse 'is_crowd' attribute as hint|n
            - '{sm.polygons.name}': save polygons,|n
            |s|smerge and convert masks, prefer polygons|n
            - '{sm.mask.name}': save masks,|n
            |s|smerge and convert polygons, prefer masks|n
            |n
            The '--reindex' option allows to control if the images and
            annotations must be given new indices. It can be useful, when
            you want to preserve the original indices in the produced dataset.
            Consider having this option enabled when converting from other
            formats.|n
            |n
            The '--allow-attributes' parameter enables or disables writing
            the custom annotation attributes to the "attributes" annotation
            field. This field is an extension to the original COCO format.|n
            |n
            The '--merge-images' parameter controls the output directory for
            images. When enabled, the dataset images are saved into a single
            directory, otherwise they are saved in separate directories
            by subsets.
            """.format(
            sm=SegmentationMode
        )
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--segmentation-mode",
            choices=[m.name for m in SegmentationMode],
            default=SegmentationMode.guess.name,
            help="Save mode for instance segmentation (default: %(default)s)",
        )
        parser.add_argument(
            "--crop-covered",
            action="store_true",
            help="Crop covered segments so that background objects' "
            "segmentation was more accurate (default: %(default)s)",
        )
        parser.add_argument(
            "--allow-attributes",
            type=str_to_bool,
            default=True,
            help="Allow export of attributes (default: %(default)s)",
        )
        parser.add_argument(
            "--reindex",
            type=str_to_bool,
            default=True,
            help="Assign new indices to images and annotations, "
            "useful to avoid merge conflicts (default: %(default)s)",
        )
        parser.add_argument(
            "--merge-images",
            type=str_to_bool,
            default=False,
            help="Save all images into a single " "directory (default: %(default)s)",
        )
        parser.add_argument(
            "--tasks",
            type=cls._split_tasks_string,
            help="COCO task filter, comma-separated list of {%s} "
            "(default: all)" % ", ".join(t.name for t in CocoTask),
        )
        return parser

    DEFAULT_IMAGE_EXT = CocoPath.IMAGE_EXT

    _TASK_CONVERTER: Dict[CocoTask, Type[_TaskExporter]] = {
        CocoTask.image_info: _ImageInfoExporter,
        CocoTask.instances: _InstancesExporter,
        CocoTask.person_keypoints: _KeypointsExporter,
        CocoTask.captions: _CaptionsExporter,
        CocoTask.labels: _LabelsExporter,
        CocoTask.panoptic: _PanopticExporter,
        CocoTask.stuff: _StuffExporter,
    }

    def __init__(
        self,
        extractor,
        save_dir,
        tasks=None,
        segmentation_mode=None,
        crop_covered=False,
        allow_attributes=True,
        reindex=False,
        merge_images=False,
        stream: bool = False,
        **kwargs,
    ):
        super().__init__(extractor, save_dir, stream=stream, **kwargs)

        assert tasks is None or isinstance(tasks, (CocoTask, list, str))
        if isinstance(tasks, CocoTask):
            tasks = [tasks]
        elif isinstance(tasks, str):
            tasks = [CocoTask[tasks]]
        elif tasks:
            for i, t in enumerate(tasks):
                if isinstance(t, str):
                    tasks[i] = CocoTask[t]
                else:
                    assert t in CocoTask, t
        else:
            tasks = set()
        self._tasks = tasks

        assert (
            segmentation_mode is None
            or isinstance(segmentation_mode, str)
            or segmentation_mode in SegmentationMode
        )
        if segmentation_mode is None:
            segmentation_mode = SegmentationMode.guess
        if isinstance(segmentation_mode, str):
            segmentation_mode = SegmentationMode[segmentation_mode]
        self._segmentation_mode = segmentation_mode

        self._crop_covered = crop_covered
        self._allow_attributes = allow_attributes
        self._reindex = reindex
        self._merge_images = merge_images

        self._image_ids = {}

        self._patch = None

    def _make_dirs(self):
        self._images_dir = osp.join(self._save_dir, CocoPath.IMAGES_DIR)
        os.makedirs(self._images_dir, exist_ok=True)

        self._ann_dir = osp.join(self._save_dir, CocoPath.ANNOTATIONS_DIR)
        os.makedirs(self._ann_dir, exist_ok=True)

    def _make_segmentation_dir(self, subset_name):
        self._segmentation_dir = osp.join(
            self._save_dir, CocoPath.ANNOTATIONS_DIR, "panoptic_" + subset_name
        )
        os.makedirs(self._segmentation_dir, exist_ok=True)

    def _make_task_converter(self, task: CocoTask, subset: str) -> _TaskExporter:
        if task not in self._TASK_CONVERTER:
            raise NotImplementedError()
        return self._TASK_CONVERTER[task](
            context=self,
            subset=subset,
            task=task,
            ann_dir=self._ann_dir,
            stream=self._stream,
        )

    def _make_task_converters(self, subset: str):
        return {
            task: self._make_task_converter(task, subset)
            for task in (self._tasks or self._TASK_CONVERTER)
        }

    def _get_image_id(self, item):
        image_id = self._image_ids.get(item.id)
        if image_id is None:
            if not self._reindex:
                image_id = cast(item.attributes.get("id"), int, len(self._image_ids) + 1)
            else:
                image_id = len(self._image_ids) + 1
            self._image_ids[item.id] = image_id
        return image_id

    def _apply_impl(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        self._make_dirs()

        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)

        subsets = self._extractor.subsets()
        pbars = self._ctx.progress_reporter.split(len(subsets))
        for pbar, (subset_name, subset) in zip(pbars, subsets.items()):
            task_converters = self._make_task_converters(subset_name)
            for task_conv in task_converters.values():
                task_conv.save_categories(subset)
            if CocoTask.panoptic in task_converters:
                self._make_segmentation_dir(subset_name)

            for item in pbar.iter(subset, desc=f"Exporting '{subset_name}'"):
                try:
                    if self._save_media:
                        if item.media:
                            self._save_image(
                                item,
                                subdir=osp.join(
                                    self._images_dir, "" if self._merge_images else subset_name
                                ),
                            )
                        else:
                            log.debug("Item '%s' has no image info", item.id)

                    for task_conv in task_converters.values():
                        try:
                            task_conv.save_image_info(item, self._make_image_filename(item))
                            task_conv.save_annotations(item)
                        except Exception as e:
                            self._report_annotation_error(e, item_id=(item.id, item.subset))
                except Exception as e:
                    self._ctx.error_policy.report_item_error(e, item_id=(item.id, item.subset))

            for task, task_conv in task_converters.items():
                ann_file = osp.join(self._ann_dir, "%s_%s.json" % (task.name, subset_name))

                if task_conv.is_empty() and (not self._tasks or self._patch):
                    if task == CocoTask.panoptic:
                        os.rmdir(self._segmentation_dir)
                    if self._patch:
                        if osp.isfile(ann_file):
                            # Remove subsets that became empty
                            os.remove(ann_file)
                    continue

                task_conv.write(ann_file)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        for subset in patch.updated_subsets:
            conv = cls(dataset.get_subset(subset), save_dir=save_dir, **kwargs)
            conv._patch = patch
            conv.apply()

        conv = cls(dataset, save_dir=save_dir, **kwargs)
        images_dir = osp.join(save_dir, CocoPath.IMAGES_DIR)
        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.media):
                continue

            # Exporter supports saving in separate dirs and common image dir

            image_path = osp.join(images_dir, conv._make_image_filename(item))
            if osp.isfile(image_path):
                os.unlink(image_path)

            image_path = osp.join(images_dir, subset, conv._make_image_filename(item))
            if osp.isfile(image_path):
                os.unlink(image_path)

    @property
    def can_stream(self) -> bool:
        return True


class CocoInstancesExporter(CocoExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = CocoTask.instances
        super().__init__(*args, **kwargs)


class CocoImageInfoExporter(CocoExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = CocoTask.image_info
        super().__init__(*args, **kwargs)


class CocoPersonKeypointsExporter(CocoExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = CocoTask.person_keypoints
        super().__init__(*args, **kwargs)


class CocoCaptionsExporter(CocoExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = CocoTask.captions
        super().__init__(*args, **kwargs)


class CocoLabelsExporter(CocoExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = CocoTask.labels
        super().__init__(*args, **kwargs)


class CocoPanopticExporter(CocoExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = CocoTask.panoptic
        super().__init__(*args, **kwargs)


class CocoStuffExporter(CocoExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = CocoTask.stuff
        kwargs["segmentation_mode"] = SegmentationMode.mask
        super().__init__(*args, **kwargs)
