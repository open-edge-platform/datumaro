# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import csv
import errno
import os
import os.path as osp
from typing import Dict, List, Optional, Union

from defusedxml import ElementTree

from datumaro.components.annotation import Annotation, AnnotationType, Bbox, Label, LabelCategories, RotatedBbox
from datumaro.components.dataset import DatasetItem
from datumaro.components.dataset_base import SubsetBase
from datumaro.components.errors import InvalidAnnotationError, UndeclaredLabelError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image, ImageFromFile
from datumaro.plugins.data_formats.coco.base import _CocoBase
from datumaro.plugins.data_formats.coco.format import CocoImporterType, CocoTask
from datumaro.plugins.data_formats.voc.base import VocBase
from datumaro.plugins.data_formats.voc.format import VocImporterType, VocTask
from datumaro.plugins.data_formats.yolo.base import YoloUltralyticsBase
from datumaro.util import parse_json_file
from datumaro.util.image import IMAGE_EXTENSIONS, ImageMeta
from datumaro.util.os_util import find_files


class RoboflowCocoBase(_CocoBase):
    """
    Parses Roboflow COCO annotations written in the following format:
    https://cocodataset.org/#format-data
    """

    def __init__(
        self,
        path,
        *,
        subset: Optional[str] = None,
        stream: bool = False,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(
            path,
            task=CocoTask.instances,
            coco_importer_type=CocoImporterType.roboflow,
            subset=subset,
            stream=stream,
            ctx=ctx,
        )


class RoboflowVocBase(VocBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(
            path,
            task=VocTask.voc_detection,
            voc_importer_type=VocImporterType.roboflow,
            subset=subset,
            ctx=ctx,
        )

    def _load_categories(self, path):
        cats = set()
        ann_files = [file for file in os.listdir(path) if file.endswith(".xml")]
        for ann_file in ann_files:
            xml_file = osp.join(path, ann_file)

            root = ElementTree.parse(xml_file).getroot()

            if root.tag != "annotation":
                continue

            for object_elem in root.iterfind("object"):
                cat_name = self._parse_field(object_elem, "name")
                cats.add(cat_name)

        label_categories = LabelCategories()
        for _, cat in enumerate(sorted(cats)):
            label_categories.add(cat)

        categories = {AnnotationType.label: label_categories}

        return categories

    def _load_subset_list(self, path):
        return [os.path.splitext(file)[0] for file in os.listdir(path) if file.endswith(".xml")]


class RoboflowYoloBase(YoloUltralyticsBase):
    def __init__(
        self,
        config_path: str,
        image_info: Union[None, str, ImageMeta] = None,
        urls: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(config_path, image_info, urls, **kwargs)

    def _get_rootpath(self, config_path: str) -> str:
        return osp.dirname(config_path)

    def _load_img_files(self, rootpath: str) -> Dict:
        return {
            self._get_fname(img_file): img_file
            for img_file in find_files(rootpath, IMAGE_EXTENSIONS, recursive=True, max_depth=2)
            if osp.split(osp.relpath(osp.dirname(img_file), rootpath))[0] == self._subset
        }


class RoboflowYoloObbBase(RoboflowYoloBase):
    def __init__(
        self,
        config_path: str,
        image_info: Union[None, str, ImageMeta] = None,
        urls: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(config_path, image_info, urls, **kwargs)

    def _parse_annotations(
        self,
        anno_path: str,
        image: ImageFromFile,
        *,
        label_categories: LabelCategories,
    ) -> List[Annotation]:
        lines = []
        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

        annotations = []
        for idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 10:
                raise InvalidAnnotationError(
                    f"Unexpected field count {len(parts)} in the bbox description. "
                    "Expected 10 fields (x1, y1, x2, y2, x3, y3, x4, y4, label, 0)."
                )
            x1, y1, x2, y2, x3, y3, x4, y4, label_name = parts[:-1]

            label_name = self._parse_field(label_name, str, "bbox label name")
            label_id = label_categories.find(label_name)[0]

            if label_id is None:
                raise UndeclaredLabelError(str(label_id))

            x1 = self._parse_field(x1, float, "x1")
            y1 = self._parse_field(y1, float, "y1")
            x2 = self._parse_field(x2, float, "x2")
            y2 = self._parse_field(y2, float, "y2")
            x3 = self._parse_field(x3, float, "x3")
            y3 = self._parse_field(y3, float, "y3")
            x4 = self._parse_field(x4, float, "x4")
            y4 = self._parse_field(y4, float, "y4")
            annotations.append(
                RotatedBbox.from_rectangle(
                    points=[(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                    label=label_id,
                    id=idx,
                    group=idx,
                )
            )

        return annotations


class RoboflowCreateMlBase(SubsetBase):
    """
    Parses Roboflow CreateML annotations written in the following format:
    https://cocodataset.org/#format-data
    """

    def __init__(
        self,
        path,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isfile(path):
            raise FileNotFoundError(errno.ENOENT, "Can't find JSON file", path)
        self._path = path

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
        super().__init__(subset=subset, ctx=ctx)

        json_data = parse_json_file(path)
        self._categories = self._load_categories(json_data)
        self._items = self._load_items(json_data)

    def _load_categories(self, json_data):
        cats = set()
        for anns in json_data:
            for ann in anns["annotations"]:
                cats.add(ann["label"])

        label_categories = LabelCategories()
        for cat in sorted(cats):
            label_categories.add(cat)

        categories = {AnnotationType.label: label_categories}

        return categories

    def _load_items(self, json_data):
        items = {}
        for anns in self._ctx.progress_reporter.iter(json_data, desc=f"Parsing boxes in '{self._subset}'"):
            annotations = []
            for ann_id, ann in enumerate(anns["annotations"]):
                label_id, _ = self._categories[AnnotationType.label].find(ann["label"])
                if label_id is None:
                    raise UndeclaredLabelError(ann["label"])

                x = ann["coordinates"]["x"]
                y = ann["coordinates"]["y"]
                w = ann["coordinates"]["width"]
                h = ann["coordinates"]["height"]
                annotations.append(Bbox(x, y, w, h, label=label_id, id=ann_id, group=ann_id))
                self._ann_types.add(AnnotationType.bbox)

            img_id = osp.splitext(anns["image"])[0]
            items[img_id] = DatasetItem(
                id=img_id,
                subset=self._subset,
                media=Image.from_file(path=osp.join(osp.dirname(self._path), anns["image"])),
                annotations=annotations,
            )

        return items.values()


class RoboflowMulticlassBase(SubsetBase):
    """
    Parses Roboflow Multiclass annotations written in the following format:
    https://cocodataset.org/#format-data
    """

    def __init__(
        self,
        path,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isfile(path):
            raise FileNotFoundError(errno.ENOENT, "Can't find CSV file", path)
        self._path = path

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
        super().__init__(subset=subset, ctx=ctx)

        self._label_mapping = {}
        self._categories = self._load_categories(path)
        self._items = self._load_items(path)

    def _load_categories(self, path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cats = [label.strip() for label in reader.fieldnames[1:]]

        label_categories = LabelCategories()
        for idx, cat in enumerate(sorted(cats)):
            label_categories.add(cat)
            self._label_mapping[cat] = idx

        categories = {AnnotationType.label: label_categories}

        return categories

    def _load_items(self, path):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for anns in csv.DictReader(f):
                img_id = anns.get("filename", None)
                img_id = osp.splitext(img_id)[0] if img_id else None
                idx = 0
                annotations = []
                for key, val in anns.items():
                    if key.strip() not in self._label_mapping:
                        continue
                    if int(val) == 1:
                        annotations.append(Label(label=self._label_mapping[key.strip()], id=idx, group=idx))
                        self._ann_types.add(AnnotationType.label)
                        idx += 1

                items.append(
                    DatasetItem(
                        id=img_id,
                        subset=self._subset,
                        media=Image.from_file(path=osp.join(osp.dirname(self._path), anns["filename"])),
                        annotations=annotations,
                    )
                )

        return items
