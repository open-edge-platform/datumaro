# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import logging as log
import os
from pathlib import Path
from typing import List, Optional, Union

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer, with_subset_dirs
from datumaro.components.media import Image
from datumaro.util.definitions import SUBSET_NAME_BLACKLIST, SUBSET_NAME_WHITELIST
from datumaro.util.image import IMAGE_EXTENSIONS, find_images
from datumaro.util.os_util import walk


class ImagenetPath:
    IMAGE_DIR_NO_LABEL = "no_label"
    SEP_TOKEN = ":"


class ImagenetBase(SubsetBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
        min_depth: Optional[int] = None,
        max_depth: Optional[int] = None,
    ):
        if not Path(path).is_dir():
            raise NotADirectoryError(errno.ENOTDIR, "Can't find dataset directory", path)
        super().__init__(subset=subset, ctx=ctx)
        self._max_depth = min_depth
        self._min_depth = max_depth
        self._categories = self._load_categories(path)
        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        label_cat = LabelCategories()
        path = Path(path)
        for dirname in sorted(d for d in path.rglob("*") if d.is_dir()):
            dirname = dirname.relative_to(path)
            level = len(dirname.parts)
            if str(dirname) != ImagenetPath.IMAGE_DIR_NO_LABEL:
                parent = None
                if level > 1:
                    parent = str(dirname.parents[0])
                    if not any([g.name == parent for g in label_cat.label_groups]):
                        label_cat.add_label_group(parent, [str(dirname.name)], group_type=0)
                    else:
                        g = next(x for x in label_cat.label_groups if x.name == parent)
                        g.labels.append(str(dirname.name))
                label_cat.add(str(dirname), parent)

        return {AnnotationType.label: label_cat}

    def _load_items(self, path):
        items = {}

        for image_path in find_images(
            path, recursive=True, max_depth=self._max_depth, min_depth=self._min_depth
        ):
            label = str(Path(image_path).parent.relative_to(path))
            if label == ".":  # image is located in the root directory
                label = ImagenetPath.IMAGE_DIR_NO_LABEL
            image_name = Path(image_path).stem
            item_id = str(label) + ImagenetPath.SEP_TOKEN + image_name
            item = items.get(item_id)
            try:
                if item is None:
                    item = DatasetItem(
                        id=item_id, subset=self._subset, media=Image.from_file(path=image_path)
                    )
                    items[item_id] = item
            except Exception as e:
                self._ctx.error_policy.report_item_error(e, item_id=(item_id, self._subset))
            annotations = item.annotations

            if label != ImagenetPath.IMAGE_DIR_NO_LABEL:
                try:
                    label = self._categories[AnnotationType.label].find(label)[0]
                    annotations.append(Label(label=label))
                    self._ann_types.add(AnnotationType.label)
                except Exception as e:
                    self._ctx.error_policy.report_annotation_error(
                        e, item_id=(item_id, self._subset)
                    )

        return items

    @property
    def is_stream(self) -> bool:
        return True


class ImagenetImporter(Importer):
    """
        Multi-level version of ImagenetImporter.
        For example, it imports the following directory structure.

    .. code-block:: text

        root
        ├── label_0
        │   ├── label_0_1
        │   │   └── img1.jpg
        │   └── label_0_2
        │       └── img2.jpg
        └── label_1
            └── img3.jpg

    """

    _MIN_DEPTH = None
    _MAX_DEPTH = None
    _FORMAT = ImagenetBase.NAME
    DETECT_CONFIDENCE = FormatDetectionConfidence.EXTREME_LOW

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        # Images must not be under a directory whose name is blacklisted.
        for dname, dirnames, filenames in os.walk(context.root_path):
            if dname in SUBSET_NAME_WHITELIST:
                context.fail(
                    f"Following directory names are not permitted: {SUBSET_NAME_WHITELIST}"
                )
            rel_dname = Path(dname).relative_to(context.root_path)
            level = len(rel_dname.parts)
            if cls._MIN_DEPTH is not None and level < cls._MIN_DEPTH and filenames:
                context.fail("Found files out of the directory level bounds.")
            if cls._MAX_DEPTH is not None and level > cls._MAX_DEPTH and filenames:
                context.fail("Found files out of the directory level bounds.")
            dpath = Path(context.root_path) / rel_dname
            if dpath.is_dir():
                if str(rel_dname).lower() in SUBSET_NAME_BLACKLIST:
                    context.fail(
                        f"{dname} is found in {context.root_path}. "
                        "However, Images must not be under a directory whose name is blacklisted "
                        f"(SUBSET_NAME_BLACKLIST={SUBSET_NAME_BLACKLIST})."
                    )

        return super().detect(context)

    @classmethod
    def contains_only_images(cls, path: Union[str, Path]):
        for _, dirnames, filenames in walk(path, cls._MAX_DEPTH, cls._MIN_DEPTH):
            if filenames:
                for filename in filenames:
                    if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                        return False
            elif not dirnames:
                return False
        return True

    @classmethod
    def find_sources(cls, path):
        if not Path(path).is_dir():
            return []

        return [{"url": path, "format": cls._FORMAT}] if cls.contains_only_images(path) else []

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return list(IMAGE_EXTENSIONS)

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("--path", required=True)
        parser.add_argument("--subset")

        return parser


@with_subset_dirs
class ImagenetWithSubsetDirsImporter(ImagenetImporter):
    """Multi-level image directory structure importer.
    Example:

    .. code-block::

        root
        ├── train
        │   ├── label_0
        │   │   ├── label_0_1
        │   │   │   └── img1.jpg
        │   │   └── label_0_2
        │   │       └── img2.jpg
        │   └── label_1
        │       └── img3.jpg
        ├── val
        │   ├── label_0
        │   │   ├── label_0_1
        │   │   │   └── img1.jpg
        │   │   └── label_0_2
        │   │       └── img2.jpg
        │   └── label_1
        │       └── img3.jpg
        └── test
            │   ├── label_0
            │   ├── label_0_1
            │   │   └── img1.jpg
            │   └── label_0_2
            │       └── img2.jpg
            └── label_1
                └── img3.jpg
    """


class ImagenetExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".jpg"
    USE_SUBSET_DIRS = False

    def _apply_impl(self):
        def _get_name(item: DatasetItem) -> str:
            id_parts = item.id.split(ImagenetPath.SEP_TOKEN)

            if len(id_parts) == 1:
                # e.g. item.id = my_img_1
                return item.id
            else:
                # e.g. item.id = label_1:my_img_1
                return "_".join(id_parts[1:])  # ":" is not allowed in windows

        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        if 1 < len(self._extractor.subsets()) and not self.USE_SUBSET_DIRS:
            log.warning(
                f"There are more than one subset in the dataset ({len(self._extractor.subsets())}). "
                "However, ImageNet format exports all dataset items into the same directory. "
                "Therefore, subset information will be lost. To prevent it, please use ImagenetWithSubsetDirsExporter. "
                'For example, dataset.export("<path/to/output>", format="imagenet_with_subset_dirs").'
            )

        root_dir = Path(self._save_dir)
        extractor = self._extractor
        labels = {}
        for item in self._extractor:
            file_name = _get_name(item)
            labels = set(p.label for p in item.annotations if p.type == AnnotationType.label)

            for label in labels:
                label_name = extractor.categories()[AnnotationType.label][label].name
                self._save_image(
                    item,
                    subdir=root_dir / item.subset / label_name
                    if self.USE_SUBSET_DIRS
                    else root_dir / label_name,
                    name=file_name,
                )

            if not labels:
                self._save_image(
                    item,
                    subdir=root_dir / item.subset / ImagenetPath.IMAGE_DIR_NO_LABEL
                    if self.USE_SUBSET_DIRS
                    else root_dir / ImagenetPath.IMAGE_DIR_NO_LABEL,
                    name=file_name,
                )


class ImagenetWithSubsetDirsExporter(ImagenetExporter):
    USE_SUBSET_DIRS = True
