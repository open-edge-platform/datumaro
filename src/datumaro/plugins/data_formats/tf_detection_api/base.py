# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import re
from collections import OrderedDict
from typing import List, Optional

import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories, Mask
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.lazy_plugin import extra_deps
from datumaro.components.media import Image
from datumaro.util.image import decode_image, lazy_image
from datumaro.util.tf_util import has_feature
from datumaro.util.tf_util import import_tf as _import_tf

from .format import DetectionApiPath, TfrecordImporterType

tf = _import_tf()


def clamp(value, _min, _max):
    return max(min(_max, value), _min)


@extra_deps("tensorflow")
class TfDetectionApiBase(SubsetBase):
    def __init__(
        self,
        path: str,
        *,
        tfrecord_importer_type: TfrecordImporterType = TfrecordImporterType.default,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        assert osp.isfile(path), path
        images_dir = ""
        root_dir = osp.dirname(osp.abspath(path))
        if osp.basename(root_dir) == DetectionApiPath.ANNOTATIONS_DIR:
            root_dir = osp.dirname(root_dir)
            images_dir = osp.join(root_dir, DetectionApiPath.IMAGES_DIR)
            if not osp.isdir(images_dir):
                images_dir = ""

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
        super().__init__(subset=subset, ctx=ctx)

        self._features = {
            "image/filename": tf.io.FixedLenFeature([], tf.string),
            "image/source_id": tf.io.FixedLenFeature([], tf.string),
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/format": tf.io.FixedLenFeature([], tf.string),
            # use varlen to avoid errors when this field is missing
            "image/key/sha256": tf.io.VarLenFeature(tf.string),
            # Object boxes and classes.
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
            "image/object/class/label": tf.io.VarLenFeature(tf.int64),
            "image/object/class/text": tf.io.VarLenFeature(tf.string),
            "image/object/mask": tf.io.VarLenFeature(tf.string),
        }
        if tfrecord_importer_type == TfrecordImporterType.roboflow:
            del self._features["image/source_id"]

        items, labels = self._parse_tfrecord_file(path, self._subset, images_dir)
        self._categories = self._load_categories(labels)
        self._items = items

    @staticmethod
    def _load_categories(labels):
        label_categories = LabelCategories().from_iterable(
            e[0] for e in sorted(labels.items(), key=lambda item: item[1])
        )
        return {AnnotationType.label: label_categories}

    @staticmethod
    def _parse_labelmap(text):
        id_pattern = r"(?:id\s*:\s*(?P<id>\d+))"
        name_pattern = r"(?:name\s*:\s*[\'\"](?P<name>.*?)[\'\"])"
        entry_pattern = r"(\{(?:[\s\n]*(?:%(id)s|%(name)s)[\s\n]*){2}\})+" % {
            "id": id_pattern,
            "name": name_pattern,
        }
        matches = re.finditer(entry_pattern, text)

        labelmap = {}
        for match in matches:
            label_id = match.group("id")
            label_name = match.group("name")
            if label_id is not None and label_name is not None:
                labelmap[label_name] = int(label_id)

        return labelmap

    def _parse_tfrecord_file(self, filepath, subset, images_dir):
        dataset = tf.data.TFRecordDataset(filepath)

        files = os.listdir(osp.dirname(filepath))
        for filename in files:
            if DetectionApiPath.LABELMAP_FILE in filename:
                labelmap_path = osp.join(osp.dirname(filepath), filename)
                break

        dataset_labels = OrderedDict()
        if osp.exists(labelmap_path):
            with open(labelmap_path, "r", encoding="utf-8") as f:
                labelmap_text = f.read()
            dataset_labels.update({label: id - 1 for label, id in self._parse_labelmap(labelmap_text).items()})

        dataset_items = []

        for record in dataset:
            parsed_record = tf.io.parse_single_example(record, self._features)
            frame_id = parsed_record.get("image/source_id", None)
            frame_id = frame_id.numpy().decode("utf-8") if frame_id else frame_id
            frame_filename = parsed_record.get("image/filename", None).numpy().decode("utf-8")
            frame_height = tf.cast(parsed_record.get("image/height", 0), tf.int64).numpy().item()
            frame_width = tf.cast(parsed_record.get("image/width", 0), tf.int64).numpy().item()
            frame_image = parsed_record["image/encoded"].numpy()
            xmins = tf.sparse.to_dense(parsed_record["image/object/bbox/xmin"]).numpy()
            ymins = tf.sparse.to_dense(parsed_record["image/object/bbox/ymin"]).numpy()
            xmaxs = tf.sparse.to_dense(parsed_record["image/object/bbox/xmax"]).numpy()
            ymaxs = tf.sparse.to_dense(parsed_record["image/object/bbox/ymax"]).numpy()
            label_ids = tf.sparse.to_dense(parsed_record["image/object/class/label"]).numpy()
            labels = tf.sparse.to_dense(parsed_record["image/object/class/text"], default_value=b"").numpy()
            masks = tf.sparse.to_dense(parsed_record["image/object/mask"], default_value=b"").numpy()

            for label, label_id in zip(labels, label_ids):
                label = label.decode("utf-8")
                if not label:
                    continue
                if label_id <= 0:
                    continue
                if label in dataset_labels:
                    continue
                dataset_labels[label] = label_id - 1

            item_id = osp.splitext(frame_filename)[0]

            annotations = []
            for shape_id, shape in enumerate(np.dstack((labels, xmins, ymins, xmaxs, ymaxs))[0]):
                label = shape[0].decode("utf-8")

                mask = None
                if len(masks) != 0:
                    mask = masks[shape_id]

                if mask is not None:
                    if isinstance(mask, bytes):
                        mask = lazy_image(mask, decode_image)
                    annotations.append(Mask(image=mask, label=dataset_labels.get(label)))
                else:
                    x = clamp(shape[1] * frame_width, 0, frame_width)
                    y = clamp(shape[2] * frame_height, 0, frame_height)
                    w = clamp(shape[3] * frame_width, 0, frame_width) - x
                    h = clamp(shape[4] * frame_height, 0, frame_height) - y
                    annotations.append(Bbox(x, y, w, h, label=dataset_labels.get(label)))

            image_size = None
            if frame_height and frame_width:
                image_size = (frame_height, frame_width)

            image = None
            if frame_image:
                if isinstance(frame_image, np.ndarray):
                    image = Image.from_numpy(data=frame_image, size=image_size)
                else:
                    image = Image.from_bytes(data=frame_image, size=image_size)
            elif frame_filename:
                image = Image.from_file(path=osp.join(images_dir, frame_filename), size=image_size)

            for ann in annotations:
                self._ann_types.add(ann.type)

            dataset_items.append(
                DatasetItem(
                    id=item_id,
                    subset=subset,
                    media=image,
                    annotations=annotations,
                    attributes={"source_id": frame_id},
                )
            )

        return dataset_items, dataset_labels


@extra_deps("tensorflow")
class TfDetectionApiImporter(Importer):
    _FORMAT_EXT = ".tfrecord"

    @classmethod
    def find_sources(cls, path):
        sources = cls._find_sources_recursive(
            path=path,
            ext=cls._FORMAT_EXT,
            extractor_name="tf_detection_api",
        )
        if len(sources) == 0:
            return []

        desired_feature = {
            "image/source_id": tf.io.FixedLenFeature([], tf.string),
        }

        subsets = {}
        for source in sources:
            if has_feature(path=source["url"], feature=desired_feature):
                subset_name = osp.basename(source["url"]).split(".")[-2]
                subsets[subset_name] = source["url"]

        return [
            {
                "url": url,
                "format": "tf_detection_api",
            }
            for _, url in subsets.items()
        ]

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._FORMAT_EXT]
