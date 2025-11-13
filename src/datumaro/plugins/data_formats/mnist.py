# Copyright (C) 2021-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import gzip
import os
import os.path as osp
from typing import List, Optional

import numpy as np

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class MnistPath:
    TEST_LABELS_FILE = "t10k-labels-idx1-ubyte.gz"
    TEST_IMAGES_FILE = "t10k-images-idx3-ubyte.gz"
    LABELS_FILE = "-labels-idx1-ubyte.gz"
    IMAGES_FILE = "-images-idx3-ubyte.gz"
    IMAGE_SIZE = 28
    NONE_LABEL = 255


class MnistBase(SubsetBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isfile(path):
            raise FileNotFoundError(errno.ENOENT, "Can't find annotations file", path)
        self._dataset_dir = osp.dirname(path)

        if not subset:
            file_name = osp.splitext(osp.basename(path))[0]
            if file_name.startswith("t10k"):
                subset = "test"
            else:
                subset = file_name.split("-", maxsplit=1)[0]

        super().__init__(subset=subset, ctx=ctx)

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
        with gzip.open(path, "rb") as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        meta = []
        metafile = osp.join(self._dataset_dir, self._subset + "-meta.gz")
        if osp.isfile(metafile):
            with gzip.open(metafile, "rb") as f:
                meta = np.frombuffer(f.read(), dtype="<U32")
            meta = meta.reshape(len(labels), int(len(meta) / len(labels)))

        # support for single-channel image only
        images = None
        images_file = osp.join(self._dataset_dir, osp.basename(path).replace("labels-idx1", "images-idx3"))
        if osp.isfile(images_file):
            with gzip.open(images_file, "rb") as imgpath:
                images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
                if len(meta) == 0 or len(meta[0]) < 2:
                    images = images.reshape(len(labels), MnistPath.IMAGE_SIZE, MnistPath.IMAGE_SIZE)

        pix_num = 0
        for i, annotation in enumerate(labels):
            annotations = []
            label = annotation
            if label != MnistPath.NONE_LABEL:
                annotations.append(Label(label))
                self._ann_types.add(AnnotationType.label)

            image = None
            if images is not None:
                if len(meta) > 0 and len(meta[i]) > 1:
                    h, w = int(meta[i][-2]), int(meta[i][-1])
                    image = images[pix_num : pix_num + h * w].reshape(h, w)
                    pix_num += h * w
                else:
                    image = images[i].reshape(MnistPath.IMAGE_SIZE, MnistPath.IMAGE_SIZE)

            if image is not None:
                image = Image.from_numpy(data=image)

            if len(meta) > 0 and (len(meta[i]) == 1 or len(meta[i]) == 3):
                i = meta[i][0]

            items[i] = DatasetItem(id=i, subset=self._subset, media=image, annotations=annotations)

        return items


class MnistImporter(Importer):
    _FORMAT_EXT = ".gz"

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(
            path,
            cls._FORMAT_EXT,
            "mnist",
            file_filter=lambda p: len(osp.basename(p).split("-")) > 1 and osp.basename(p).split("-")[1] == "labels",
        )

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._FORMAT_EXT]


class MnistExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".png"

    def _apply_impl(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(self._save_dir, exist_ok=True)
        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)

        for subset_name, subset in self._extractor.subsets().items():
            labels = []
            images = np.array([])
            item_ids = {}
            image_sizes = {}
            for item in subset:
                anns = [a.label for a in item.annotations if a.type == AnnotationType.label]
                label = 255
                if anns:
                    label = anns[0]
                labels.append(label)

                if item.id != str(len(labels) - 1):
                    item_ids[len(labels) - 1] = item.id

                if item.media and self._save_media:
                    image = item.media
                    if not image.has_data:
                        image_sizes[len(images) - 1] = [0, 0]
                    else:
                        image = image.data
                        if image.shape[0] != MnistPath.IMAGE_SIZE or image.shape[1] != MnistPath.IMAGE_SIZE:
                            image_sizes[len(labels) - 1] = [image.shape[0], image.shape[1]]
                        images = np.append(images, image.reshape(-1).astype(np.uint8))

            if subset_name == "test":
                labels_file = osp.join(self._save_dir, MnistPath.TEST_LABELS_FILE)
            else:
                labels_file = osp.join(self._save_dir, subset_name + MnistPath.LABELS_FILE)
            self.save_annotations(labels_file, labels)

            if len(images) > 0:
                if subset_name == "test":
                    images_file = osp.join(self._save_dir, MnistPath.TEST_IMAGES_FILE)
                else:
                    images_file = osp.join(self._save_dir, subset_name + MnistPath.IMAGES_FILE)
                self.save_images(images_file, images)

            # it is't in the original format,
            # this is for storng other names and sizes of images
            if item_ids or image_sizes:
                meta = []
                if item_ids and image_sizes:
                    # other names and sizes of images
                    size = [MnistPath.IMAGE_SIZE, MnistPath.IMAGE_SIZE]
                    for i in range(len(labels)):
                        w, h = image_sizes.get(i, size)
                        meta.append([item_ids.get(i, i), w, h])

                elif item_ids:
                    # other names of images
                    for i in range(len(labels)):
                        meta.append([item_ids.get(i, i)])

                elif image_sizes:
                    # other sizes of images
                    size = [MnistPath.IMAGE_SIZE, MnistPath.IMAGE_SIZE]
                    for i in range(len(labels)):
                        meta.append(image_sizes.get(i, size))

                metafile = osp.join(self._save_dir, subset_name + "-meta.gz")
                with gzip.open(metafile, "wb") as f:
                    f.write(np.array(meta, dtype="<U32").tobytes())

        self.save_labels()

    def save_annotations(self, path, data):
        with gzip.open(path, "wb") as f:
            # magic number = 0x0801 (2049, hexadecimal representation)
            # this is used to verify the file with MNIST mark data
            f.write(np.array([0x0801, len(data)], dtype=">i4").tobytes())
            f.write(np.array(data, dtype="uint8").tobytes())

    def save_images(self, path, data):
        with gzip.open(path, "wb") as f:
            # magic number = 0x0803 (2051, hexadecimal representation),
            # this is used to verify the file with MNIST image data
            f.write(np.array([0x0803, len(data), MnistPath.IMAGE_SIZE, MnistPath.IMAGE_SIZE], dtype=">i4").tobytes())
            f.write(np.array(data, dtype="uint8").tobytes())

    def save_labels(self):
        labels_file = osp.join(self._save_dir, "labels.txt")
        with open(labels_file, "w", encoding="utf-8") as f:
            f.writelines(
                l.name + "\n" for l in self._extractor.categories().get(AnnotationType.label, LabelCategories())
            )
