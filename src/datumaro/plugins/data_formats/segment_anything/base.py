# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from glob import glob
from inspect import isclass
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

from datumaro.components.annotation import Bbox, RleMask
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import (
    DatasetImportError,
    InvalidAnnotationError,
    InvalidFieldTypeError,
    MissingFieldError,
)
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image
from datumaro.util import NOTSET, parse_json_file

T = TypeVar("T")


def parse_field(
    ann: Dict[str, Any],
    key: str,
    cls: Union[Type[T], Tuple[Type, ...]],
    default: Any = NOTSET,
) -> Any:
    value = ann.get(key, NOTSET)
    if value is NOTSET:
        if default is not NOTSET:
            return default
        raise MissingFieldError(key)
    if not isinstance(value, cls):
        cls = (cls,) if isclass(cls) else cls
        raise InvalidFieldTypeError(key, actual=str(type(value)), expected=tuple(str(t) for t in cls))
    return value


class SegmentAnythingBase(SubsetBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isdir(path):
            raise DatasetImportError(f"path {path} must be directory.")
        self._path = path

        super().__init__(subset=subset, ctx=ctx)
        self._items = self._load_items()

    def _load_items(self):
        pbar = self._ctx.progress_reporter
        items = []
        for annotation_file in pbar.iter(
            glob(osp.join(self._path, "*.json")),
            desc=f"Parsing data in {osp.basename(self._path)}",
        ):
            image_id = None
            annotations = []
            item_kwargs = {
                "id": None,
                "subset": self._subset,
                "media": None,
                "annotations": [],
                "attributes": {},
            }

            try:
                contents = parse_json_file(annotation_file)
                image_info = contents["image"]
                annotations = contents["annotations"]

                image_id = parse_field(image_info, "image_id", int)
                item_kwargs["attributes"]["id"] = image_id

                image_size = (
                    parse_field(image_info, "height", int, default=None),
                    parse_field(image_info, "width", int, default=None),
                )
                if any(i is None for i in image_size):
                    image_size = None
                file_name = parse_field(image_info, "file_name", str)

                item_kwargs["id"] = osp.splitext(file_name)[0]
                item_kwargs["media"] = Image.from_file(path=osp.join(self._path, file_name), size=image_size)
            except Exception as e:
                self._ctx.error_policy.report_item_error(e, item_id=(image_id, self._subset))

            try:
                for annotation in annotations:
                    anno_id = parse_field(annotation, "id", int)
                    attributes = {
                        "predicted_iou": parse_field(
                            annotation,
                            "predicted_iou",
                            float,
                            0.0,
                        ),
                        "stability_score": parse_field(
                            annotation,
                            "stability_score",
                            float,
                            0.0,
                        ),
                        "point_coords": parse_field(
                            annotation,
                            "point_coords",
                            list,
                            [[]],
                        ),
                        "crop_box": parse_field(annotation, "crop_box", list, []),
                    }

                    group = anno_id  # make sure all tasks' annotations are merged

                    segmentation = parse_field(annotation, "segmentation", dict, None)
                    if segmentation is None:
                        raise InvalidAnnotationError("'segmentation' label is not found.")
                    item_kwargs["annotations"].append(
                        RleMask(
                            rle=segmentation,
                            label=None,
                            id=anno_id,
                            attributes=attributes,
                            group=group,
                        )
                    )

                    bbox = parse_field(annotation, "bbox", list, None)
                    if bbox is None:
                        bbox = item_kwargs["annotations"][-1].get_bbox().tolist()

                    if len(bbox) > 0:
                        if len(bbox) != 4:
                            raise InvalidAnnotationError(f"Bbox has wrong value count {len(bbox)}. Expected 4 values.")
                        x, y, w, h = bbox
                        item_kwargs["annotations"].append(
                            Bbox(
                                x,
                                y,
                                w,
                                h,
                                label=None,
                                id=anno_id,
                                attributes=attributes,
                                group=group,
                            )
                        )
            except Exception as e:
                self._ctx.error_policy.report_annotation_error(e, item_id=(image_id, self._subset))

            try:
                items.append(DatasetItem(**item_kwargs))
                for ann in item_kwargs["annotations"]:
                    self._ann_types.add(ann.type)
            except Exception as e:
                self._ctx.error_policy.report_item_error(e, item_id=(image_id, self._subset))

        return items
