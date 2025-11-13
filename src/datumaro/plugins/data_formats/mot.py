# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

# The Multiple Object Tracking Benchmark challenge format support
# Format description: https://arxiv.org/pdf/1906.04567.pdf
# Another description: https://motchallenge.net/instructions

import csv
import logging as log
import os
import os.path as osp
from collections import OrderedDict
from enum import Enum
from typing import List, Optional, Union

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util import cast
from datumaro.util.image import find_images
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file

MotLabel = Enum(
    "MotLabel",
    [
        ("pedestrian", 1),
        ("person on vehicle", 2),
        ("car", 3),
        ("bicycle", 4),
        ("motorbike", 5),
        ("non motorized vehicle", 6),
        ("static person", 7),
        ("distractor", 8),
        ("occluder", 9),
        ("occluder on the ground", 10),
        ("occluder full", 11),
        ("reflection", 12),
    ],
)


class MotPath:
    IMAGE_DIR = "img1"
    SEQINFO_FILE = "seqinfo.ini"
    LABELS_FILE = "labels.txt"
    GT_FILENAME = "gt.txt"
    DET_FILENAME = "det.txt"

    IMAGE_EXT = ".jpg"

    FIELDS = [
        "frame_id",
        "track_id",
        "x",
        "y",
        "w",
        "h",
        "confidence",  # or 'not ignored' flag for GT anns
        "class_id",
        "visibility",
    ]


class MotSeqBase(SubsetBase):
    def __init__(
        self,
        path: str,
        *,
        labels: Optional[Union[str, List[str]]] = None,
        occlusion_threshold: float = 0.0,
        is_gt: Optional[bool] = None,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(subset=subset, ctx=ctx)

        assert osp.isfile(path)
        seq_root = osp.dirname(osp.dirname(path))
        self._image_dir = ""
        if osp.isdir(osp.join(seq_root, MotPath.IMAGE_DIR)):
            self._image_dir = osp.join(seq_root, MotPath.IMAGE_DIR)

        seq_info = osp.join(seq_root, MotPath.SEQINFO_FILE)
        if osp.isfile(seq_info):
            seq_info = self._parse_seq_info(seq_info)
            self._image_dir = osp.join(seq_root, seq_info["imdir"])
        else:
            seq_info = None
        self._seq_info = seq_info

        self._occlusion_threshold = float(occlusion_threshold)

        assert is_gt in {None, True, False}
        if is_gt is None:
            if osp.basename(path) == MotPath.DET_FILENAME:
                is_gt = False
            else:
                is_gt = True
        self._is_gt = is_gt

        if has_meta_file(seq_root):
            labels = list(parse_meta_file(seq_root).keys())
        if labels is None:
            labels = osp.join(osp.dirname(path), MotPath.LABELS_FILE)
            if not osp.isfile(labels):
                labels = [lbl.name for lbl in MotLabel]
        if isinstance(labels, str):
            labels = self._parse_labels(labels)
        elif isinstance(labels, list):
            assert all(isinstance(lbl, str) for lbl in labels), labels
        else:
            raise TypeError("Unexpected type of 'labels' argument: %s" % labels)
        self._categories = self._load_categories(labels)
        self._items = list(self._load_items(path).values())

    @staticmethod
    def _parse_labels(path):
        with open(path, encoding="utf-8") as labels_file:
            return [s.strip() for s in labels_file]

    def _load_categories(self, labels):
        attributes = ["track_id"]
        if self._is_gt:
            attributes += ["occluded", "visibility", "ignored"]
        else:
            attributes += ["score"]
        label_cat = LabelCategories(attributes=attributes)
        for label in labels:
            label_cat.add(label)

        return {AnnotationType.label: label_cat}

    def _load_items(self, path):
        labels_count = len(self._categories[AnnotationType.label].items)
        items = OrderedDict()

        if self._seq_info:
            for frame_id in range(1, self._seq_info["seqlength"] + 1):  # base-1 frame ids
                items[frame_id] = DatasetItem(
                    id=frame_id,
                    subset=self._subset,
                    media=Image.from_file(
                        path=osp.join(self._image_dir, "%06d%s" % (frame_id, self._seq_info["imext"])),
                        size=(self._seq_info["imheight"], self._seq_info["imwidth"]),
                    ),
                )
        elif osp.isdir(self._image_dir):
            for p in find_images(self._image_dir):
                frame_id = int(osp.splitext(osp.relpath(p, self._image_dir))[0])
                items[frame_id] = DatasetItem(id=frame_id, subset=self._subset, media=Image.from_file(path=p))

        with open(path, newline="", encoding="utf-8") as csv_file:
            # NOTE: Different MOT files have different count of fields
            # (7, 9 or 10). This is handled by reader:
            # - all extra fields go to a separate field
            # - all unmet fields have None values
            for row in csv.DictReader(csv_file, fieldnames=MotPath.FIELDS):
                frame_id = int(row["frame_id"])
                item = items.get(frame_id)
                if item is None:
                    item = DatasetItem(id=frame_id, subset=self._subset)
                annotations = item.annotations

                x, y = float(row["x"]), float(row["y"])
                w, h = float(row["w"]), float(row["h"])
                label_id = row.get("class_id")
                if label_id and label_id != "-1":
                    label_id = int(label_id) - 1
                    assert label_id < labels_count, label_id
                else:
                    label_id = None

                attributes = {}

                # Annotations for detection task are not related to any track
                track_id = int(row["track_id"])
                if track_id > 0:
                    attributes["track_id"] = track_id

                confidence = cast(row.get("confidence"), float, 1)
                visibility = cast(row.get("visibility"), float, 1)
                if self._is_gt:
                    attributes["visibility"] = visibility
                    attributes["occluded"] = visibility <= self._occlusion_threshold
                    attributes["ignored"] = confidence == 0
                else:
                    attributes["score"] = float(confidence)

                annotations.append(Bbox(x, y, w, h, label=label_id, attributes=attributes))
                self._ann_types.add(AnnotationType.bbox)

                items[frame_id] = item
        return items

    @classmethod
    def _parse_seq_info(cls, path):
        fields = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                entry = line.lower().strip().split("=", maxsplit=1)
                if len(entry) == 2:
                    fields[entry[0]] = entry[1]
        cls._check_seq_info(fields)
        for k in ("framerate", "seqlength", "imwidth", "imheight"):
            fields[k] = int(fields[k])
        return fields

    @staticmethod
    def _check_seq_info(seq_info):
        assert set(seq_info) == {
            "name",
            "imdir",
            "framerate",
            "seqlength",
            "imwidth",
            "imheight",
            "imext",
        }, seq_info


class MotSeqImporter(Importer):
    _ANNO_EXT = ".txt"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(f"gt/gt{cls._ANNO_EXT}")

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(
            path,
            cls._ANNO_EXT,
            "mot_seq",
            dirname="gt",
            filename=osp.splitext(MotPath.GT_FILENAME)[0],
        )

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]


class MotSeqGtExporter(Exporter):
    DEFAULT_IMAGE_EXT = MotPath.IMAGE_EXT

    def _apply_impl(self):
        extractor = self._extractor

        if extractor.media_type() and not issubclass(extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        image_dir = osp.join(self._save_dir, MotPath.IMAGE_DIR)
        os.makedirs(image_dir, exist_ok=True)

        anno_dir = osp.join(self._save_dir, "gt")
        os.makedirs(anno_dir, exist_ok=True)
        anno_file = osp.join(anno_dir, MotPath.GT_FILENAME)
        with open(anno_file, "w", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=MotPath.FIELDS)

            track_id_mapping = {-1: -1}
            for idx, item in enumerate(extractor):
                log.debug("Converting item '%s'", item.id)

                frame_id = cast(item.id, int, 1 + idx)

                for anno in item.annotations:
                    if anno.type != AnnotationType.bbox:
                        continue

                    track_id = int(anno.attributes.get("track_id", -1))
                    if track_id not in track_id_mapping:
                        track_id_mapping[track_id] = len(track_id_mapping)
                    track_id = track_id_mapping[track_id]

                    writer.writerow(
                        {
                            "frame_id": frame_id,
                            "track_id": track_id,
                            "x": anno.x,
                            "y": anno.y,
                            "w": anno.w,
                            "h": anno.h,
                            "confidence": int(anno.attributes.get("ignored") is not True),
                            "class_id": 1 + cast(anno.label, int, -2),
                            "visibility": float(
                                anno.attributes.get("visibility", 1 - float(anno.attributes.get("occluded", False)))
                            ),
                        }
                    )

                if self._save_media:
                    if item.media and item.media.has_data:
                        self._save_image(item, subdir=image_dir, name="%06d" % frame_id)
                    else:
                        log.debug("Item '%s' has no image", item.id)

        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)
        else:
            labels_file = osp.join(anno_dir, MotPath.LABELS_FILE)
            with open(labels_file, "w", encoding="utf-8") as f:
                f.write("\n".join(l.name for l in extractor.categories()[AnnotationType.label]))
