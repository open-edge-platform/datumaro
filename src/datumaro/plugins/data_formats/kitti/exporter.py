# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from collections import OrderedDict
from enum import Enum, auto

import numpy as np

from datumaro.components.annotation import AnnotationType, CompiledMask, LabelCategories
from datumaro.components.errors import InvalidAnnotationError, MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.media import Image
from datumaro.util import cast, parse_str_enum_value, str_to_bool
from datumaro.util.annotation_util import make_label_id_mapping
from datumaro.util.image import save_image
from datumaro.util.mask_tools import paint_mask
from datumaro.util.meta_file_util import is_meta_file, parse_meta_file

from .format import KittiLabelMap, KittiPath, KittiTask, make_kitti_categories, parse_label_map, write_label_map


class LabelmapType(Enum):
    kitti = auto()
    source = auto()


class KittiExporter(Exporter):
    DEFAULT_IMAGE_EXT = KittiPath.IMAGE_EXT

    @staticmethod
    def _split_tasks_string(s):
        return [KittiTask[i.strip().lower()] for i in s.split(",")]

    @staticmethod
    def _get_labelmap(s):
        if osp.isfile(s):
            return s
        try:
            return LabelmapType[s.lower()].name
        except KeyError:
            import argparse

            raise argparse.ArgumentTypeError()

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument(
            "--apply-colormap",
            type=str_to_bool,
            default=True,
            help="Use colormap for class masks (default: %(default)s)",
        )
        parser.add_argument(
            "--label-map",
            type=cls._get_labelmap,
            default=None,
            help="Labelmap file path or one of %s" % ", ".join(t.name for t in LabelmapType),
        )
        parser.add_argument(
            "--tasks",
            type=cls._split_tasks_string,
            help="KITTI task filter, comma-separated list of {%s} "
            "(default: all)" % ", ".join(t.name for t in KittiTask),
        )
        return parser

    def __init__(
        self,
        extractor,
        save_dir,
        tasks=None,
        apply_colormap=True,
        allow_attributes=True,
        label_map=None,
        **kwargs,
    ):
        super().__init__(extractor, save_dir, **kwargs)

        assert tasks is None or isinstance(tasks, (KittiTask, list, set))
        if tasks is None:
            tasks = set(KittiTask)
        elif isinstance(tasks, KittiTask):
            tasks = {tasks}
        else:
            tasks = set(parse_str_enum_value(t, KittiTask) for t in tasks)
        self._tasks = tasks

        self._apply_colormap = apply_colormap

        if label_map is None:
            label_map = LabelmapType.source.name
        if KittiTask.segmentation in self._tasks:
            self._load_categories(label_map)
        elif KittiTask.detection in self._tasks:
            self._categories = {
                AnnotationType.label: self._extractor.categories().get(AnnotationType.label, LabelCategories())
            }

    def _apply_impl(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(self._save_dir, exist_ok=True)

        for subset_name, subset in self._extractor.subsets().items():
            if KittiTask.segmentation in self._tasks:
                os.makedirs(osp.join(self._save_dir, subset_name, KittiPath.INSTANCES_DIR), exist_ok=True)

            for item in subset:
                if self._save_media:
                    self._save_image(item, subdir=osp.join(subset_name, KittiPath.IMAGES_DIR))

                masks = [a for a in item.annotations if a.type == AnnotationType.mask]
                if masks and KittiTask.segmentation in self._tasks:
                    compiled_class_mask = CompiledMask.from_instance_masks(
                        masks, instance_labels=[self._label_id_mapping(m.label) for m in masks]
                    )
                    color_mask_path = osp.join(subset_name, KittiPath.SEMANTIC_RGB_DIR, item.id + KittiPath.MASK_EXT)
                    self.save_mask(osp.join(self._save_dir, color_mask_path), compiled_class_mask.class_mask)

                    labelids_mask_path = osp.join(subset_name, KittiPath.SEMANTIC_DIR, item.id + KittiPath.MASK_EXT)
                    self.save_mask(
                        osp.join(self._save_dir, labelids_mask_path),
                        compiled_class_mask.class_mask,
                        apply_colormap=False,
                        dtype=np.int32,
                    )

                    # TODO: optimize second merging
                    compiled_instance_mask = CompiledMask.from_instance_masks(
                        masks,
                        instance_labels=[(self._label_id_mapping(m.label) << 8) + m.id for m in masks],
                    )
                    inst_path = osp.join(subset_name, KittiPath.INSTANCES_DIR, item.id + KittiPath.MASK_EXT)
                    self.save_mask(
                        osp.join(self._save_dir, inst_path),
                        compiled_instance_mask.class_mask,
                        apply_colormap=False,
                        dtype=np.int32,
                    )

                bboxes = [a for a in item.annotations if a.type == AnnotationType.bbox]
                if bboxes and KittiTask.detection in self._tasks:
                    labels_file = osp.join(self._save_dir, subset_name, KittiPath.LABELS_DIR, "%s.txt" % item.id)
                    os.makedirs(osp.dirname(labels_file), exist_ok=True)
                    with open(labels_file, "w", encoding="utf-8") as f:
                        for bbox in bboxes:
                            label_line = [-1] * 16
                            label_line[0] = self.get_label(bbox.label)
                            label_line[1] = cast(bbox.attributes.get("truncated"), float, KittiPath.DEFAULT_TRUNCATED)
                            label_line[2] = cast(bbox.attributes.get("occluded"), int, KittiPath.DEFAULT_OCCLUDED)
                            x, y, h, w = bbox.get_bbox()
                            label_line[4:8] = x, y, x + h, y + w

                            label_line[15] = cast(bbox.attributes.get("score"), float, KittiPath.DEFAULT_SCORE)

                            label_line = " ".join(str(v) for v in label_line)
                            f.write("%s\n" % label_line)

        if KittiTask.segmentation in self._tasks:
            self.save_label_map()

    def get_label(self, label_id):
        return self._extractor.categories()[AnnotationType.label].items[label_id].name

    def save_label_map(self):
        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)
        else:
            path = osp.join(self._save_dir, KittiPath.LABELMAP_FILE)
            write_label_map(path, self._label_map)

    def _load_categories(self, label_map_source):
        if label_map_source == LabelmapType.kitti.name:
            # use the default KITTI colormap
            label_map = KittiLabelMap

        elif label_map_source == LabelmapType.source.name and AnnotationType.mask not in self._extractor.categories():
            # generate colormap for input labels
            labels = self._extractor.categories().get(AnnotationType.label, LabelCategories())
            label_map = OrderedDict((item.name, None) for item in labels.items)

        elif label_map_source == LabelmapType.source.name and AnnotationType.mask in self._extractor.categories():
            # use source colormap
            labels = self._extractor.categories()[AnnotationType.label]
            colors = self._extractor.categories()[AnnotationType.mask]
            label_map = OrderedDict()
            for idx, item in enumerate(labels.items):
                color = colors.colormap.get(idx)
                if color is not None:
                    label_map[item.name] = color

        elif isinstance(label_map_source, dict):
            label_map = OrderedDict(sorted(label_map_source.items(), key=lambda e: e[0]))

        elif isinstance(label_map_source, str) and osp.isfile(label_map_source):
            if is_meta_file(label_map_source):
                label_map = parse_meta_file(label_map_source)
            else:
                label_map = parse_label_map(label_map_source)

        else:
            raise InvalidAnnotationError(
                "Wrong labelmap specified, expected one of %s or a file path" % ", ".join(t.name for t in LabelmapType)
            )

        self._categories = make_kitti_categories(label_map)
        self._label_map = label_map
        self._label_id_mapping = self._make_label_id_map()

    def _make_label_id_map(self):
        map_id, id_mapping, src_labels, dst_labels = make_label_id_mapping(
            self._extractor.categories().get(AnnotationType.label),
            self._categories[AnnotationType.label],
        )

        void_labels = [src_label for src_id, src_label in src_labels.items() if src_label not in dst_labels]
        if void_labels:
            log.warning("The following labels are remapped to background: %s" % ", ".join(void_labels))
        log.debug(
            "Saving segmentations with the following label mapping: \n%s"
            % "\n".join(
                [
                    "#%s '%s' -> #%s '%s'"
                    % (
                        src_id,
                        src_label,
                        id_mapping[src_id],
                        self._categories[AnnotationType.label].items[id_mapping[src_id]].name,
                    )
                    for src_id, src_label in src_labels.items()
                ]
            )
        )

        return map_id

    def save_mask(self, path, mask, colormap=None, apply_colormap=True, dtype=np.uint8):
        if self._apply_colormap and apply_colormap:
            if colormap is None:
                colormap = self._categories[AnnotationType.mask].colormap
            mask = paint_mask(mask, colormap)
        save_image(path, mask, create_dir=True, dtype=dtype)


class KittiSegmentationExporter(KittiExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = KittiTask.segmentation
        super().__init__(*args, **kwargs)


class KittiDetectionExporter(KittiExporter):
    def __init__(self, *args, **kwargs):
        kwargs["tasks"] = KittiTask.detection
        super().__init__(*args, **kwargs)
