# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import os
import os.path as osp
import re
import warnings
from typing import Dict, Optional, Type, TypeVar, Union

import numpy as np
import pandas as pd
from defusedxml import ElementTree

from datumaro.components.annotation import AnnotationType, Bbox, ExtractedMask, Label, LabelCategories, MaskCategories
from datumaro.components.dataset import DatasetItem
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetBase, SubsetBase
from datumaro.components.errors import InvalidAnnotationError, InvalidFieldError, MissingFieldError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image, ImageFromFile
from datumaro.plugins.data_formats.coco.base import CocoInstancesBase
from datumaro.plugins.data_formats.coco.format import CocoTask
from datumaro.plugins.data_formats.coco.page_mapper import COCOPageMapper
from datumaro.util import parse_json_file
from datumaro.util.image import IMAGE_EXTENSIONS, lazy_image

T = TypeVar("T")


class KaggleImageCsvBase(DatasetBase):
    def __init__(
        self,
        path: str,
        ann_file: str,
        columns: Dict[str, str],
        *,
        subset: Optional[str] = DEFAULT_SUBSET_NAME,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(ctx=ctx)

        self._subset = subset
        self._path = path

        if "media" not in columns:
            raise MissingFieldError("media")

        self._label_cat = LabelCategories()
        self._items = self._load_items(ann_file, columns)
        self._categories = {AnnotationType.label: self._label_cat}

    def _get_media_path(self, media_name: str):
        media_path = osp.join(self._path, media_name)
        if osp.exists(media_path):
            return media_path

        for ext in IMAGE_EXTENSIONS:
            media_path_with_ext = media_path + ext
            if osp.exists(media_path_with_ext):
                return media_path_with_ext

        return None

    def _parse_bbox_coords(self, bbox_str):
        coords = re.findall(r"[-+]?\d*\.\d+|\d+", bbox_str)
        if len(coords) != 4:
            raise ValueError("Bounding box coordinates must have exactly 4 values.")

        # expected to output [x1, y1, x2, y2]
        return [float(coord.strip()) for coord in coords]

    def _load_annotations(self, datas: list, indices: Dict[str, Union[int, Dict[str, int]]], bbox_flag: bool):
        if "label" in indices:
            label_indices = indices["label"]
            if isinstance(label_indices, dict):
                labels = []
                list_values = datas[1:]
                index_to_label = {v: k for k, v in label_indices.items()}
                present_labels = [index_to_label[i + 1] for i, value in enumerate(list_values) if value == "1"]

                for label_name in present_labels:
                    label, cat = self._label_cat.find(label_name)
                    if not cat:
                        self._label_cat.add(label_name)
                        label, _ = self._label_cat.find(label_name)
                    labels.append(Label(label=label))
            else:
                label_name = str(datas[indices["label"]])
                label, cat = self._label_cat.find(label_name)
                if not cat:
                    self._label_cat.add(label_name)
                    label, _ = self._label_cat.find(label_name)
        else:
            _, cat = self._label_cat.find("object")
            if not cat:
                self._label_cat.add("object")
            label = 0

        if "label" in indices and not bbox_flag:
            label_indices = indices["label"]
            if isinstance(label_indices, dict):
                return labels
            return Label(label=label)

        if bbox_flag:
            if "bbox" in indices:
                coords = self._parse_bbox_coords(datas[indices["bbox"]])
                return Bbox(
                    label=label,
                    x=coords[0],
                    y=coords[1],
                    w=coords[2] - coords[0],
                    h=coords[3] - coords[1],
                )
            if "width" in indices and "height" in indices:
                return Bbox(
                    label=label,
                    x=float(datas[indices["x1"]]),
                    y=float(datas[indices["y1"]]),
                    w=float(datas[indices["width"]]),
                    h=float(datas[indices["height"]]),
                )
            if "x2" in indices and "y2" in indices:
                return Bbox(
                    label=label,
                    x=float(datas[indices["x1"]]),
                    y=float(datas[indices["y1"]]),
                    w=float(datas[indices["x2"]]) - float(datas[indices["x1"]]),
                    h=float(datas[indices["y2"]]) - float(datas[indices["y1"]]),
                )

    def _load_items(self, ann_file: str, columns: Dict[str, Union[str, list]]):
        df = pd.read_csv(ann_file, header=None, on_bad_lines="skip")
        df_fields = list(df.iloc[0])

        indices = {"media": df_fields.index(columns["media"])}
        if "label" in columns:
            label_columns = columns["label"]
            if isinstance(label_columns, list):
                indices_label = {}
                for label in label_columns:
                    indices_label[label] = df_fields.index(label)
                indices.update({"label": indices_label})
            else:
                indices.update({"label": df_fields.index(label_columns)})

        bbox_flag = False
        bbox_index = columns.get("bbox")
        if bbox_index:
            bbox_flag = True
            bbox_indices = {"x1", "x2", "y1", "y2", "width", "height"}
            if isinstance(bbox_index, str):
                indices["bbox"] = df_fields.index(bbox_index)
            elif isinstance(bbox_index, dict):
                indices.update({key: df_fields.index(bbox_index[key]) for key in bbox_indices if bbox_index.get(key)})
            if not ({"x1", "x2", "y1", "y2"} <= bbox_indices or {"x1", "y1", "width", "height"} <= bbox_indices):
                warnings.warn("Insufficient box coordinate is given for importing bounding boxes.")
                bbox_flag = False

        items = dict()
        for _, row in df.iloc[1:].iterrows():  # Skip header row
            data_info = list(row)

            media_name = data_info[indices["media"]]
            item_id = osp.splitext(media_name)[0]

            media_path = self._get_media_path(media_name)
            if not media_path or not osp.exists(media_path):
                warnings.warn(
                    f"'{media_path}' is not existed in the directory, "
                    f"so we skip to create an dataset item according to {row}."
                )
                continue

            ann = self._load_annotations(data_info, indices, bbox_flag)
            if isinstance(ann, list):
                for label in ann:
                    self._ann_types.add(label.type)
                if item_id in items:
                    for label in ann:
                        items[item_id].annotations.append(label)
                else:
                    items[item_id] = DatasetItem(
                        id=item_id,
                        subset=self._subset,
                        media=Image.from_file(path=media_path),
                        annotations=ann,
                    )
            else:
                self._ann_types.add(ann.type)
                if item_id in items:
                    items[item_id].annotations.append(ann)
                else:
                    items[item_id] = DatasetItem(
                        id=item_id,
                        subset=self._subset,
                        media=Image.from_file(path=media_path),
                        annotations=[ann],
                    )
        return items.values()

    def categories(self):
        return self._categories

    def __iter__(self):
        yield from self._items

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("--path", required=True)
        parser.add_argument("--ann_file", required=True)
        parser.add_argument("--columns", required=True, type=json.loads)

        return parser


class KaggleImageTxtBase(KaggleImageCsvBase):
    def __init__(
        self,
        path: str,
        ann_file: str,
        columns: Dict[str, int],
        *,
        subset: Optional[str] = DEFAULT_SUBSET_NAME,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(path=path, ann_file=ann_file, columns=columns, subset=subset, ctx=ctx)

    def _load_items(self, ann_file: str, columns: Dict[str, Union[int, Dict]]):
        bbox_flag = False
        if "bbox" in columns:
            bbox_flag = True
            bbox_columns = columns.pop("bbox")
            if isinstance(bbox_columns, dict):
                if not (
                    all(item in bbox_columns for item in ["x1", "x2", "y1", "y2"])
                    or all(item in bbox_columns for item in ["x1", "y1", "width", "height"])
                ):
                    warnings.warn("Insufficient box coordinate is given for importing bounding boxes.")
                    bbox_flag = False
                columns.update(bbox_columns)

        items = dict()
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                line = re.split(r"\s|,", line)

                media_name = line[columns["media"]]
                item_id = osp.splitext(media_name)[0]

                media_path = self._get_media_path(media_name)
                if not media_path or not osp.exists(media_path):
                    warnings.warn(
                        f"'{media_path}' is not existed in the directory, "
                        f"so we skip to create an dataset item according to {line}."
                    )
                    continue

                ann = self._load_annotations(line, columns, bbox_flag)
                self._ann_types.add(ann.type)
                if item_id in items:
                    items[item_id].annotations.append(ann)
                else:
                    items[item_id] = DatasetItem(
                        id=item_id,
                        subset=self._subset,
                        media=Image.from_file(path=media_path),
                        annotations=[ann],
                    )

        return items.values()


class KaggleImageMaskBase(DatasetBase):
    def __init__(
        self,
        path: str,
        mask_path: str,
        labelmap_file: Optional[str] = None,
        *,
        subset: Optional[str] = DEFAULT_SUBSET_NAME,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(ctx=ctx)

        self._subset = subset

        self._path = path
        self._mask_path = mask_path

        self._label_ids = []
        self._categories = self._load_categories(labelmap_file)
        self._items = self._load_items()

    def _load_categories(self, label_map_file: Optional[str]):
        label_map = dict()
        if not label_map_file:
            label_map["background"] = (0, 0, 0)
            label_map["object"] = (255, 255, 255)
        else:
            df = pd.read_csv(label_map_file)
            for _, row in df.iterrows():
                name = row[0]
                color = tuple([int(c) for c in row[1:]])
                label_map[name] = color

        label_categories = LabelCategories()
        for label in label_map:
            label_categories.add(label)

        categories = {}
        categories[AnnotationType.label] = label_categories

        colormap = {}
        for label_name, label_color in label_map.items():
            label_id = label_categories.find(label_name)[0]
            colormap[label_id] = label_color
            self._label_ids.append(label_id)

        categories[AnnotationType.mask] = MaskCategories(colormap)

        return categories

    def _load_items(self):
        items = []
        for media_name in sorted(os.listdir(self._path)):
            id = osp.splitext(media_name)[0]

            anns = []
            for mask_name in os.listdir(self._mask_path):
                if id in mask_name:
                    index_mask = lazy_image(path=osp.join(self._mask_path, mask_name), dtype=np.int32)
                    for label_id in self._label_ids:
                        anns.append(
                            ExtractedMask(
                                index_mask=index_mask,
                                index=label_id,
                                label=label_id,
                            )
                        )
                        self._ann_types.add(AnnotationType.mask)

            items.append(
                DatasetItem(
                    id=id,
                    subset=self._subset,
                    media=Image.from_file(path=osp.join(self._path, media_name)),
                    annotations=anns,
                )
            )

        return items

    def categories(self):
        return self._categories

    def __iter__(self):
        yield from self._items

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("--path", required=True)
        parser.add_argument("--mask_path", required=True)
        parser.add_argument("--labelmap_file")

        return parser


class KaggleVocBase(SubsetBase):
    ann_extensions = ".xml"

    def __init__(
        self,
        path: str,
        ann_path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(subset=subset, ctx=ctx)

        self._label_cat = LabelCategories()
        self._items = []
        self._size = None

        for img_filename in sorted(os.listdir(path)):
            if not img_filename.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                continue
            item_id = osp.splitext(img_filename)[0]

            img_file = osp.join(path, img_filename)
            ann_file = osp.join(ann_path, item_id + self.ann_extensions)

            annotations = self._parse_annotations(img_file, ann_file) if osp.isfile(ann_file) else []
            for ann in annotations:
                self._ann_types.add(ann.type)

            media = Image.from_file(path=img_file, size=self._size)

            self._items.append(
                DatasetItem(
                    id=item_id,
                    subset=self._subset,
                    media=media,
                    annotations=annotations,
                )
            )
        self._categories = {AnnotationType.label: self._label_cat}

    def _parse_annotations(self, img_file: str, ann_file: str):
        root_elem = ElementTree.parse(ann_file).getroot()
        if root_elem.tag != "annotation":
            raise MissingFieldError("annotation")

        height = self._parse_field(root_elem, "size/height", int, required=False)
        width = self._parse_field(root_elem, "size/width", int, required=False)
        if height and width:
            self._size = (height, width)

        annotations = []
        for obj_id, object_elem in enumerate(root_elem.iterfind("object")):
            label_name = self._parse_field(object_elem, "name", str, required=True)

            bbox_elem = object_elem.find("bndbox")
            if not bbox_elem:
                raise MissingFieldError("bndbox")

            xmin = self._parse_field(bbox_elem, "xmin", float)
            xmax = self._parse_field(bbox_elem, "xmax", float)
            ymin = self._parse_field(bbox_elem, "ymin", float)
            ymax = self._parse_field(bbox_elem, "ymax", float)

            label_id, cat = self._label_cat.find(label_name)
            if not cat:
                self._label_cat.add(label_name)
                label_id, _ = self._label_cat.find(label_name)

            annotations.append(Bbox(id=obj_id, label=label_id, x=xmin, y=ymin, w=xmax - xmin, h=ymax - ymin))

        return annotations

    @staticmethod
    def _parse_field(root, xpath: str, cls: Type[T] = str, required: bool = True) -> Optional[T]:
        elem = root.find(xpath)
        if elem is None:
            if required:
                raise MissingFieldError(xpath)
            return None

        if cls is str:
            return elem.text

        try:
            return cls(elem.text)
        except Exception as e:
            raise InvalidFieldError(xpath) from e

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("--path", required=True)
        parser.add_argument("--ann_path", required=True)
        parser.add_argument("--subset")

        return parser


class KaggleYoloBase(KaggleVocBase, SubsetBase):
    ann_extensions = ".txt"

    def __init__(
        self,
        path: str,
        ann_path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(path=path, ann_path=ann_path, subset=subset, ctx=ctx)

    def _parse_annotations(self, img_file: str, ann_file: str):
        image = ImageFromFile(path=img_file)
        image_height, image_width = image.size

        lines = []
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

        annotations = []
        for obj_id, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                raise InvalidAnnotationError(
                    f"Unexpected field count {len(parts)} in the bbox description. "
                    "Expected 5 fields (label, xc, yc, w, h)."
                )
            label_name, xc, yc, w, h = parts
            xc = float(xc)
            yc = float(yc)
            w = float(w)
            h = float(h)
            x = (xc - w * 0.5) * image_width
            y = (yc - h * 0.5) * image_height
            w *= image_width
            h *= image_height

            label_id, cat = self._label_cat.find(label_name)
            if not cat:
                self._label_cat.add(label_name)
                label_id, _ = self._label_cat.find(label_name)
            label_id, _ = self._label_cat.find(label_name)

            annotations.append(Bbox(id=obj_id, label=label_id, x=x, y=y, w=w, h=h))

        return annotations


class KaggleCocoBase(CocoInstancesBase, SubsetBase):
    def __init__(
        self,
        path: str,
        ann_file: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
        stream: bool = False,
    ):
        SubsetBase.__init__(self, subset=subset, ctx=ctx)

        self._rootpath = path
        self._images_dir = path
        self._path = ann_file
        self._task = CocoTask.instances
        self._merge_instance_polygons = False

        keep_original_category_ids = False

        self._stream = stream
        if not stream:
            self._page_mapper = None  # No use in case of stream = False

            json_data = parse_json_file(ann_file)

            self._load_categories(
                json_data,
                keep_original_ids=keep_original_category_ids,
            )

            self._items = self._load_items(json_data)

            del json_data
        else:
            self._page_mapper = COCOPageMapper(ann_file)

            categories_data = self._page_mapper.stream_parse_categories_data()

            self._load_categories(
                {"categories": categories_data},
                keep_original_ids=keep_original_category_ids,
            )

            self._length = None

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("--path", required=True)
        parser.add_argument("--ann_file", required=True)
        parser.add_argument("--subset")

        return parser
