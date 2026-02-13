# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from collections import Counter
from enum import Enum, auto
from importlib.util import find_spec
from typing import TYPE_CHECKING, Union

import cv2
import numpy as np

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset import IDataset
from datumaro.components.media import Image
from datumaro.util import parse_str_enum_value
from datumaro.util.image import save_image
from datumaro.util.import_util import lazy_import

TENSORBOARDX_AVAILABLE = find_spec("tensorboardX") is not None

if TENSORBOARDX_AVAILABLE:
    if TYPE_CHECKING:
        import warnings
    
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import tensorboardX as tb
    else:
        tb = lazy_import("tensorboardX")


class DistanceCompareVisualizer:
    class OutputFormat(Enum):
        simple = auto()
        tensorboard = auto()

    DEFAULT_FORMAT = OutputFormat.simple
    _UNMATCHED_LABEL = -1

    def __init__(self, comparator, save_dir: str, output_format: Union[None, str, OutputFormat] = None):
        self._cmp = comparator

        self._output_format = parse_str_enum_value(output_format, self.OutputFormat, default=self.DEFAULT_FORMAT)

        self._save_dir = save_dir

    def __enter__(self):
        os.makedirs(self._save_dir, exist_ok=True)

        if self._output_format is self.OutputFormat.tensorboard:
            logdir = osp.join(self._save_dir, "logs", "compare")
            self._file_writer = tb.SummaryWriter(logdir)
        elif self._output_format is self.OutputFormat.simple:
            self._label_diff_writer = None

        self._a_classes = {}
        self._b_classes = {}

        self.label_confusion_matrix = Counter()
        self.bbox_confusion_matrix = Counter()
        self.polygon_confusion_matrix = Counter()
        self.mask_confusion_matrix = Counter()

        return self

    def __exit__(self, *args, **kwargs):
        if self._output_format is self.OutputFormat.tensorboard:
            self._file_writer.flush()
            self._file_writer.close()
        elif self._output_format is self.OutputFormat.simple:
            if self._label_diff_writer:
                self._label_diff_writer.flush()
                self._label_diff_writer.close()

    def save(self, a: IDataset, b: IDataset):
        if len(a) != len(b):
            print("Datasets have different lengths: %s vs %s" % (len(a), len(b)))

        a_classes = set(a.get_label_cat_names())
        b_classes = set(b.get_label_cat_names())

        if a_classes ^ b_classes:
            print("Datasets have mismatching labels:")

        for idx, diff in enumerate(a_classes - b_classes):
            print(" #%s: > %s" % (idx, diff))

        for idx, diff in enumerate(b_classes - a_classes, start=len((a_classes - b_classes))):
            print(" #%s: < %s" % (idx, diff))

        self._a_classes = a.categories().get(AnnotationType.label)
        self._b_classes = b.categories().get(AnnotationType.label)

        ids_a = set((item.id, item.subset) for item in a)
        ids_b = set((item.id, item.subset) for item in b)
        ids = ids_a & ids_b

        if len(ids) != len(ids_a):
            print("Unmatched items in the first dataset: ")
            print(ids_a - ids)
        if len(ids) != len(ids_b):
            print("Unmatched items in the second dataset: ")
            print(ids_b - ids)

        for item_id, item_subset in ids:
            item_a = a.get(item_id, item_subset)
            item_b = b.get(item_id, item_subset)

            label_diff = self._cmp.match_labels(item_a, item_b)
            self.update_label_confusion(label_diff)

            bbox_diff = self._cmp.match_boxes(item_a, item_b)
            self.update_bbox_confusion(bbox_diff)

            polygon_diff = self._cmp.match_polygons(item_a, item_b)
            self.update_polygon_confusion(polygon_diff)

            mask_diff = self._cmp.match_masks(item_a, item_b)
            self.update_mask_confusion(mask_diff)

            self.save_item_label_diff(item_a, item_b, label_diff)

            if (
                a.media_type()
                and issubclass(a.media_type(), Image)
                and b.media_type()
                and issubclass(b.media_type(), Image)
            ):
                self.save_item_bbox_diff(item_a, item_b, bbox_diff)

        if len(self.label_confusion_matrix) != 0:
            self.save_conf_matrix(self.label_confusion_matrix, "label_confusion.png")
        if len(self.bbox_confusion_matrix) != 0:
            self.save_conf_matrix(self.bbox_confusion_matrix, "bbox_confusion.png")
        if len(self.polygon_confusion_matrix) != 0:
            self.save_conf_matrix(self.polygon_confusion_matrix, "polygon_confusion.png")
        if len(self.mask_confusion_matrix) != 0:
            self.save_conf_matrix(self.mask_confusion_matrix, "mask_confusion.png")

    def update_label_confusion(self, label_diff):
        matches, a_unmatched, b_unmatched = label_diff
        for label in matches:
            self.label_confusion_matrix[(label, label)] += 1
        for a_label in a_unmatched:
            self.label_confusion_matrix[(a_label, self._UNMATCHED_LABEL)] += 1
        for b_label in b_unmatched:
            self.label_confusion_matrix[(self._UNMATCHED_LABEL, b_label)] += 1

    @classmethod
    def _update_segment_confusion(cls, matrix, diff):
        matches, mispred, a_unmatched, b_unmatched = diff
        for a_segm, b_segm in matches:
            matrix[(a_segm.label, b_segm.label)] += 1
        for a_segm, b_segm in mispred:
            matrix[(a_segm.label, b_segm.label)] += 1
        for a_segm in a_unmatched:
            matrix[(a_segm.label, cls._UNMATCHED_LABEL)] += 1
        for b_segm in b_unmatched:
            matrix[(cls._UNMATCHED_LABEL, b_segm.label)] += 1

    def update_bbox_confusion(self, diff):
        self._update_segment_confusion(self.bbox_confusion_matrix, diff)

    def update_polygon_confusion(self, diff):
        self._update_segment_confusion(self.polygon_confusion_matrix, diff)

    def update_mask_confusion(self, diff):
        self._update_segment_confusion(self.mask_confusion_matrix, diff)

    @classmethod
    def draw_text_with_background(
        cls,
        frame,
        text,
        origin,
        font=None,
        scale=1.0,
        color=(0, 0, 0),
        thickness=1,
        bgcolor=(1, 1, 1),
    ):
        if not font:
            font = cv2.FONT_HERSHEY_SIMPLEX

        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(
            frame,
            tuple((origin + (0, baseline)).astype(int)),  # noqa: RUF005
            tuple((origin + (text_size[0], -text_size[1])).astype(int)),  # noqa: RUF005
            bgcolor,
            cv2.FILLED,
        )
        cv2.putText(frame, text, tuple(origin.astype(int)), font, scale, color, thickness)
        return text_size, baseline

    def draw_detection_roi(self, frame, x, y, w, h, label, conf, color):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        text = "%s %.2f%%" % (label, 100.0 * conf)
        text_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, text_scale, 1)
        line_height = np.array([0, text_size[0][1]])
        self.draw_text_with_background(
            frame,
            text,
            np.array([x, y]) - line_height * 0.5,
            font,
            scale=text_scale,
            color=[255 - c for c in color],
        )

    def get_a_label(self, label_id):
        return self._get_label(self._a_classes, label_id)

    def get_b_label(self, label_id):
        return self._get_label(self._b_classes, label_id)

    @staticmethod
    def _get_label(cat: LabelCategories, label_id):
        if cat is None:
            return str(label_id)
        return cat[label_id].name

    def draw_bbox(self, img, shape, label, color):
        x, y, w, h = shape.get_bbox()
        self.draw_detection_roi(img, int(x), int(y), int(w), int(h), label, shape.attributes.get("score", 1), color)

    def get_label_diff_file(self):
        if self._label_diff_writer is None:
            self._label_diff_writer = open(osp.join(self._save_dir, "label_diff.txt"), "w", encoding="utf-8")  # noqa: SIM115
        return self._label_diff_writer

    def save_item_label_diff(self, item_a, item_b, diff):
        _, a_unmatched, b_unmatched = diff

        if len(a_unmatched) + len(b_unmatched) > 0:
            if self._output_format is self.OutputFormat.simple:
                f = self.get_label_diff_file()
                f.write(item_a.id + "\n")
                for a_label in a_unmatched:
                    f.write("  >%s\n" % self.get_a_label(a_label))
                for b_label in b_unmatched:
                    f.write("  <%s\n" % self.get_b_label(b_label))
            elif self._output_format is self.OutputFormat.tensorboard:
                tag = item_a.id
                for a_label in a_unmatched:
                    self._file_writer.add_text(tag, ">%s\n" % self.get_a_label(a_label))
                for b_label in b_unmatched:
                    self._file_writer.add_text(tag, "<%s\n" % self.get_b_label(b_label))

    def save_item_bbox_diff(self, item_a, item_b, diff):
        _, mispred, a_unmatched, b_unmatched = diff

        if len(a_unmatched) + len(b_unmatched) + len(mispred) > 0:
            if not isinstance(item_a.media, Image) or not item_a.media.has_data:
                log.warning("Item %s: item has no image data, it will be skipped" % (item_a.id))
                return
            img_a = item_a.media.data.copy()
            img_b = img_a.copy()
            for a_bbox, b_bbox in mispred:
                self.draw_bbox(img_a, a_bbox, self.get_a_label(a_bbox.label), (0, 255, 0))
                self.draw_bbox(img_b, b_bbox, self.get_b_label(b_bbox.label), (0, 0, 255))
            for a_bbox in a_unmatched:
                self.draw_bbox(img_a, a_bbox, self.get_a_label(a_bbox.label), (255, 255, 0))
            for b_bbox in b_unmatched:
                self.draw_bbox(img_b, b_bbox, self.get_b_label(b_bbox.label), (255, 255, 0))

            img = np.hstack([img_a, img_b])

            path = osp.join(self._save_dir, item_a.subset, item_a.id)

            if self._output_format is self.OutputFormat.simple:
                save_image(path + ".png", img, create_dir=True)
            elif self._output_format is self.OutputFormat.tensorboard:
                self.save_as_tensorboard(img, path)

    def save_as_tensorboard(self, img, name):
        img = img[:, :, ::-1]  # to RGB
        img = np.transpose(img, (2, 0, 1))  # to (C, H, W)
        img = img.astype(dtype=np.uint8)
        self._file_writer.add_image(name, img)

    def save_conf_matrix(self, conf_matrix, filename):
        import matplotlib.pyplot as plt

        def _get_class_map(label_categories):
            classes = None
            if label_categories is not None:
                classes = {id: c.name for id, c in enumerate(label_categories.items)}
            if classes is None:
                classes = {c: "label_%s" % c for c, _ in conf_matrix}
            classes[self._UNMATCHED_LABEL] = "unmatched"
            classes[None] = "no_class"
            return classes

        a_classes = _get_class_map(self._a_classes)
        b_classes = _get_class_map(self._b_classes)

        a_class_idx = {id: i for i, id in enumerate(a_classes)}
        b_class_idx = {id: i for i, id in enumerate(b_classes)}
        matrix = np.zeros((len(a_classes), len(b_classes)), dtype=int)
        for idx_pair in conf_matrix:
            index = (a_class_idx[idx_pair[0]], b_class_idx[idx_pair[1]])
            matrix[index] = conf_matrix[idx_pair]

        a_labels = [label for id, label in a_classes.items()]
        b_labels = [label for id, label in b_classes.items()]

        fig = plt.figure()
        fig.add_subplot(111)
        table = plt.table(cellText=matrix, rowLabels=a_labels, colLabels=b_labels, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(3, 3)
        # Removing ticks and spines enables you to get the figure only with table
        plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis="y", which="both", right=False, left=False, labelleft=False)
        for pos in ["right", "top", "bottom", "left"]:
            plt.gca().spines[pos].set_visible(False)

        for idx_pair in conf_matrix:
            i = a_class_idx[idx_pair[0]]
            j = b_class_idx[idx_pair[1]]
            if conf_matrix[idx_pair] != 0:
                if a_classes[idx_pair[0]] == b_classes[idx_pair[1]]:
                    table._cells[(i + 1, j)].set_facecolor("#00FF00")
                else:
                    table._cells[(i + 1, j)].set_facecolor("#FF0000")

        plt.savefig(osp.join(self._save_dir, filename), bbox_inches="tight", pad_inches=0.05)
