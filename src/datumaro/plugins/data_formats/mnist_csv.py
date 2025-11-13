# Copyright (C) 2021-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import os
import os.path as osp
from typing import List, Optional

import numpy as np

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.format_detection import FormatDetectionConfidence
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class MnistCsvPath:
    IMAGE_SIZE = 28
    NONE_LABEL = -1


class MnistCsvBase(SubsetBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isfile(path):
            raise FileNotFoundError(errno.ENOENT, "Can't find annotations file", path)

        if not subset:
            file_name = osp.splitext(osp.basename(path))[0]
            subset = file_name.rsplit("_", maxsplit=1)[-1]

        super().__init__(subset=subset, ctx=ctx)
        self._dataset_dir = osp.dirname(path)

        self._categories = self._load_categories()

        self._items = list(self._load_items(path).values())

    def _load_categories(self):
        if has_meta_file(self._dataset_dir):
            return {AnnotationType.label: LabelCategories.from_iterable(parse_meta_file(self._dataset_dir).keys())}

        label_cat = LabelCategories()

        labels_file = osp.join(self._dataset_dir, "labels.txt")
        if osp.isfile(labels_file):
            with open(labels_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    label_cat.add(line)
        else:
            for i in range(10):
                label_cat.add(str(i))

        return {AnnotationType.label: label_cat}

    def _load_items(self, path):
        items = {}
        with open(path, "r", encoding="utf-8") as f:
            annotation_table = f.readlines()

        metafile = osp.join(self._dataset_dir, "meta_%s.csv" % self._subset)
        meta = []
        if osp.isfile(metafile):
            with open(metafile, "r", encoding="utf-8") as f:
                meta = f.readlines()

        for i, data in enumerate(annotation_table):
            data = data.split(",")
            item_anno = []
            try:
                label = int(data[0])
            except ValueError:
                continue
            if label != MnistCsvPath.NONE_LABEL:
                item_anno.append(Label(label))
                self._ann_types.add(AnnotationType.label)

            if len(meta) > 0:
                meta[i] = meta[i].strip().split(",")

            # support for single-channel image only
            image = None
            if len(data) > 1:
                if len(meta) > 0 and len(meta[i]) > 1:
                    image = np.array([int(pix) for pix in data[1:]], dtype="uint8").reshape(
                        int(meta[i][-2]), int(meta[i][-1])
                    )
                else:
                    image = np.array([int(pix) for pix in data[1:]], dtype="uint8").reshape(28, 28)

            if image is not None:
                image = Image.from_numpy(data=image)

            if len(meta) > 0 and len(meta[i]) in [1, 3]:
                i = meta[i][0]

            items[i] = DatasetItem(id=i, subset=self._subset, media=image, annotations=item_anno)

        return items


class MnistCsvImporter(Importer):
    DETECT_CONFIDENCE = FormatDetectionConfidence.MEDIUM
    _ANNO_EXT = ".csv"

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(
            path,
            cls._ANNO_EXT,
            "mnist_csv",
            file_filter=lambda p: osp.basename(p).find("mnist_") != -1,
        )

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]


class MnistCsvExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".png"

    def _apply_impl(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(self._save_dir, exist_ok=True)
        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)

        for subset_name, subset in self._extractor.subsets().items():
            data = []
            item_ids = {}
            image_sizes = {}
            for item in subset:
                anns = [a.label for a in item.annotations if a.type == AnnotationType.label]
                label = MnistCsvPath.NONE_LABEL
                if anns:
                    label = anns[0]

                if item.media and self._save_media:
                    image = item.media
                    if not image.has_data:
                        data.append([label, None])
                    else:
                        if (
                            image.data.shape[0] != MnistCsvPath.IMAGE_SIZE
                            or image.data.shape[1] != MnistCsvPath.IMAGE_SIZE
                        ):
                            image_sizes[len(data)] = [image.data.shape[0], image.data.shape[1]]
                        image = image.data.reshape(-1).astype(np.uint8).tolist()
                        image.insert(0, label)
                        data.append(image)
                else:
                    data.append([label])

                if item.id != str(len(data) - 1):
                    item_ids[len(data) - 1] = item.id

            anno_file = osp.join(self._save_dir, "mnist_%s.csv" % subset_name)
            self.save_in_csv(anno_file, data)

            # it is't in the original format,
            # this is for storng other names and sizes of images
            if item_ids or image_sizes:
                meta = []
                if item_ids and image_sizes:
                    # other names and sizes of images
                    size = [MnistCsvPath.IMAGE_SIZE, MnistCsvPath.IMAGE_SIZE]
                    for i in range(len(data)):
                        w, h = image_sizes.get(i, size)
                        meta.append([item_ids.get(i, i), w, h])

                elif item_ids:
                    # other names of images
                    for i in range(len(data)):
                        meta.append([item_ids.get(i, i)])

                elif image_sizes:
                    # other sizes of images
                    size = [MnistCsvPath.IMAGE_SIZE, MnistCsvPath.IMAGE_SIZE]
                    for i in range(len(data)):
                        meta.append(image_sizes.get(i, size))

                metafile = osp.join(self._save_dir, "meta_%s.csv" % subset_name)
                self.save_in_csv(metafile, meta)

        self.save_labels()

    def save_in_csv(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(",".join([str(p) for p in row]) + "\n")

    def save_labels(self):
        labels_file = osp.join(self._save_dir, "labels.txt")
        with open(labels_file, "w", encoding="utf-8") as f:
            f.writelines(
                l.name + "\n" for l in self._extractor.categories().get(AnnotationType.label, LabelCategories())
            )
