# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import os
import os.path as osp
import pickle  # nosec B403
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.dataset_item_storage import ItemStatus
from datumaro.components.errors import InvalidAnnotationError, MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util import cast
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file
from datumaro.util.pickle_util import PickleLoader


class CifarPath:
    META_10_FILE = "batches.meta"
    META_100_FILE = "meta"
    TRAIN_FILE_PREFIX = "data_batch_"
    USELESS_FILE = "file.txt~"
    IMAGE_SIZE = 32


Cifar10Label = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Support for Python version CIFAR-10/100


class CifarBase(SubsetBase):
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
            subset = osp.splitext(osp.basename(path))[0]

        super().__init__(subset=subset, ctx=ctx)

        self._categories = self._load_categories(osp.dirname(path))
        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        if has_meta_file(path):
            return {AnnotationType.label: LabelCategories.from_iterable(parse_meta_file(path).keys())}

        label_cat = LabelCategories()

        meta_file = osp.join(path, CifarPath.META_10_FILE)
        if not osp.isfile(meta_file):
            meta_file = osp.join(path, CifarPath.META_100_FILE)
        if osp.isfile(meta_file):
            # CIFAR-10:
            # num_cases_per_batch: 1000
            # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer',
            #               'dog', 'frog', 'horse', 'ship', 'truck']
            # num_vis: 3072
            # CIFAR-100:
            # fine_label_names: ['apple', 'aquarium_fish', 'baby', ...]
            # coarse_label_names: ['aquatic_mammals', 'fish', 'flowers', ...]
            with open(meta_file, "rb") as labels_file:
                data = PickleLoader.restricted_load(labels_file)
            labels = data.get("label_names")
            if labels is not None:
                for label in labels:
                    label_cat.add(label)
            else:
                labels = data.get("fine_label_names")
                self._coarse_labels = data.get("coarse_label_names", [])
                if labels is not None:
                    for label in labels:
                        label_cat.add(label)
        else:
            for label in Cifar10Label:
                label_cat.add(label)

        return {AnnotationType.label: label_cat}

    def _load_items(self, path):
        items = {}
        label_cat = self._categories[AnnotationType.label]

        # 'batch_label': 'training batch 1 of 5'
        # 'data': ndarray
        # 'filenames': list
        # CIFAR-10: 'labels': list
        # CIFAR-100: 'fine_labels': list
        #            'coarse_labels': list

        with open(path, "rb") as anno_file:
            annotation_dict = PickleLoader.restricted_load(anno_file)

        labels = annotation_dict.get("labels", [])
        coarse_labels = annotation_dict.get("coarse_labels", [])
        if len(labels) == 0:
            labels = annotation_dict.get("fine_labels", [])
        filenames = annotation_dict.get("filenames", [])
        images_data = annotation_dict.get("data")
        size = annotation_dict.get("image_sizes")

        if len(labels) != len(filenames):
            raise InvalidAnnotationError("The sizes of the arrays 'filenames', 'labels' don't match.")

        if len(images_data) > 0 and len(images_data) != len(filenames):
            raise InvalidAnnotationError("The sizes of the arrays 'data', 'filenames', 'labels' don't match.")

        for i, (filename, label) in enumerate(zip(filenames, labels)):
            item_id = osp.splitext(filename)[0]
            annotations = []
            if label is not None:
                annotations.append(Label(label))
                self._ann_types.add(AnnotationType.label)
                if len(coarse_labels) > 0 and coarse_labels[i] is not None and label_cat[label].parent == "":
                    label_cat[label].parent = self._coarse_labels[coarse_labels[i]]

            image = None
            if len(images_data) > 0:
                image = images_data[i]
                if size is not None and image is not None:
                    image = image.astype(np.uint8).reshape(3, size[i][0], size[i][1])
                    image = np.transpose(image, (1, 2, 0))
                elif image is not None:
                    image = image.astype(np.uint8).reshape(3, CifarPath.IMAGE_SIZE, CifarPath.IMAGE_SIZE)
                    image = np.transpose(image, (1, 2, 0))

            if image is not None:
                image = Image.from_numpy(data=image)

            items[item_id] = DatasetItem(id=item_id, subset=self._subset, media=image, annotations=annotations)
        return items


class CifarImporter(Importer):
    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> Optional[FormatDetectionConfidence]:
        super().detect(context)

        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def find_sources(cls, path: str) -> List[Dict[str, Any]]:
        def _find(path: str, meta_file_name: str) -> List[Dict[str, Any]]:
            # Find dataset root by looking for the meta file
            roots = cls._find_sources_recursive(
                path,
                "",
                "cifar",
                file_filter=lambda p: osp.basename(p) == meta_file_name,
            )

            sources = []
            # Extract subset files from the root path
            for root in roots:
                root_dir = osp.dirname(root["url"])
                sources += cls._find_sources_recursive(
                    root_dir,
                    "",
                    "cifar",
                    # Subset files have no extension in the format, and
                    # should not be the meta file.
                    file_filter=lambda p: not osp.isdir(p)
                    and not osp.splitext(osp.basename(p))[1]
                    and osp.basename(p) != meta_file_name,
                )

            return sources

        sources = []
        sources += _find(path, CifarPath.META_10_FILE)
        sources += _find(path, CifarPath.META_100_FILE)

        return sources

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return list({osp.splitext(p)[1] for p in (CifarPath.META_10_FILE, CifarPath.META_100_FILE)})


class CifarExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".png"

    def _apply_impl(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(self._save_dir, exist_ok=True)

        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)

        label_categories = self._extractor.categories()[AnnotationType.label]
        label_names = []
        coarse_label_names = []
        for label in label_categories:
            label_names.append(label.name)
            if label.parent and (label.parent not in coarse_label_names):
                coarse_label_names.append(label.parent)
        coarse_label_names.sort()

        if coarse_label_names:
            labels_dict = {
                "fine_label_names": label_names,
                "coarse_label_names": coarse_label_names,
            }
            coarse_label_names = OrderedDict((name, i) for i, name in enumerate(coarse_label_names))
            meta_file = osp.join(self._save_dir, CifarPath.META_100_FILE)
        else:
            labels_dict = {"label_names": label_names}
            meta_file = osp.join(self._save_dir, CifarPath.META_10_FILE)
        with open(meta_file, "wb") as f:
            pickle.dump(labels_dict, f)

        for subset_name, subset in self._extractor.subsets().items():
            filenames = []
            labels = []
            coarse_labels = []
            data = []
            image_sizes = {}
            for item in subset:
                filenames.append(self._make_image_filename(item))

                anns = [a for a in item.annotations if a.type == AnnotationType.label]
                if anns:
                    labels.append(anns[0].label)
                    if coarse_label_names:
                        superclass = label_categories[anns[0].label].parent
                        coarse_labels.append(coarse_label_names[superclass])
                else:
                    labels.append(None)
                    coarse_labels.append(None)

                if self._save_media and item.media:
                    image = item.media
                    if not image.has_data:
                        data.append(None)
                    else:
                        image = image.data
                        data.append(np.transpose(image, (2, 0, 1)).reshape(-1).astype(np.uint8))
                        if image.shape[0] != CifarPath.IMAGE_SIZE or image.shape[1] != CifarPath.IMAGE_SIZE:
                            image_sizes[len(data) - 1] = (image.shape[0], image.shape[1])

            annotation_dict = {}

            annotation_dict["filenames"] = filenames
            if labels and (len(labels) == len(coarse_labels)):
                annotation_dict["fine_labels"] = labels
                annotation_dict["coarse_labels"] = coarse_labels
            else:
                annotation_dict["labels"] = labels
            annotation_dict["data"] = np.array(data, dtype=object)

            if image_sizes:
                size = (CifarPath.IMAGE_SIZE, CifarPath.IMAGE_SIZE)
                # 'image_sizes' isn't included in the standard format,
                # needed for different image sizes
                annotation_dict["image_sizes"] = [image_sizes.get(p, size) for p in range(len(data))]

            batch_label = None
            if subset_name.startswith(CifarPath.TRAIN_FILE_PREFIX):
                num = subset_name[len(CifarPath.TRAIN_FILE_PREFIX) :]
                if cast(num, int) is not None:
                    batch_label = "training batch %s of 5" % num
            elif subset_name == "test":
                batch_label = "testing batch 1 of 1"

            if batch_label:
                annotation_dict["batch_label"] = batch_label

            annotation_file = osp.join(self._save_dir, subset_name)

            if self._patch and subset_name in self._patch.updated_subsets and not annotation_dict["filenames"]:
                if osp.isfile(annotation_file):
                    # Remove subsets that became empty
                    os.remove(annotation_file)
                continue

            with open(annotation_file, "wb") as labels_file:
                pickle.dump(annotation_dict, labels_file)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        for subset in patch.updated_subsets:
            conv = cls(dataset.get_subset(subset), save_dir=save_dir, **kwargs)
            conv._patch = patch
            conv.apply()

        for subset, status in patch.updated_subsets.items():
            if status != ItemStatus.removed:
                continue

            subset_file = osp.join(save_dir, subset)
            if osp.isfile(subset_file):
                os.remove(subset_file)
