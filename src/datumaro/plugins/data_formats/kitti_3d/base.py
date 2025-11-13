# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import logging
import os
import os.path as osp
from typing import List, Optional, Type, TypeVar

from datumaro.components.annotation import AnnotationType, Bbox
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import InvalidAnnotationError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image
from datumaro.util.image import find_images

from .format import Kitti3DLabelMap, Kitti3dPath, make_kitti3d_categories

T = TypeVar("T")


class Kitti3dBase(SubsetBase):
    # https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        assert osp.isdir(path), path

        self._path = path

        if not subset:
            folder_path = path.rsplit(Kitti3dPath.LABEL_DIR, 1)[0]
            img_dir = osp.join(folder_path, Kitti3dPath.IMAGE_DIR)
            if any(os.path.isdir(os.path.join(img_dir, item)) for item in os.listdir(img_dir)):
                subset = osp.split(path)[-1]
                self._path = folder_path
        super().__init__(subset=subset, ctx=ctx)

        self._categories = make_kitti3d_categories(Kitti3DLabelMap)
        self._items = self._load_items()

    def _load_items(self) -> List[DatasetItem]:
        items = []

        image_dir = osp.join(self._path, Kitti3dPath.IMAGE_DIR)
        image_path_by_id = {
            osp.split(osp.splitext(osp.relpath(p, image_dir))[0])[-1]: p for p in find_images(image_dir, recursive=True)
        }

        if self._subset == "default":
            ann_dir = osp.join(self._path, Kitti3dPath.LABEL_DIR)
        else:
            ann_dir = osp.join(self._path, Kitti3dPath.LABEL_DIR, self._subset)

        label_categories = self._categories[AnnotationType.label]

        for labels_path in sorted(glob.glob(osp.join(ann_dir, "**", "*.txt"), recursive=True)):
            item_id = osp.splitext(osp.relpath(labels_path, ann_dir))[0]
            anns = []

            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except IOError as e:
                logging.error(f"Error reading file {labels_path}: {e}")
                continue

            for line_idx, line in enumerate(lines):
                line = line.split()
                if len(line) not in [15, 16]:
                    logging.warning(f"Unexpected line length {len(line)} in file {labels_path} at line {line_idx + 1}")
                    continue

                label_name = line[0]
                label_id = label_categories.find(label_name)[0]
                if label_id is None:
                    label_id = label_categories.add(label_name)

                try:
                    x1 = self._parse_field(line[4], float, "bbox left-top x")
                    y1 = self._parse_field(line[5], float, "bbox left-top y")
                    x2 = self._parse_field(line[6], float, "bbox right-bottom x")
                    y2 = self._parse_field(line[7], float, "bbox right-bottom y")

                    attributes = {
                        "truncated": self._parse_field(line[1], float, "truncated"),
                        "occluded": self._parse_field(line[2], int, "occluded"),
                        "alpha": self._parse_field(line[3], float, "alpha"),
                        "dimensions": [
                            self._parse_field(line[8], float, "height (in meters)"),
                            self._parse_field(line[9], float, "width (in meters)"),
                            self._parse_field(line[10], float, "length (in meters)"),
                        ],
                        "location": [
                            self._parse_field(line[11], float, "x (in meters)"),
                            self._parse_field(line[12], float, "y (in meters)"),
                            self._parse_field(line[13], float, "z (in meters)"),
                        ],
                        "rotation_y": self._parse_field(line[14], float, "rotation_y"),
                    }
                except ValueError as e:
                    logging.error(f"Error parsing line {line_idx + 1} in file {labels_path}: {e}")
                    continue

                anns.append(
                    Bbox(
                        x=x1,
                        y=y1,
                        w=x2 - x1,
                        h=y2 - y1,
                        id=line_idx,
                        attributes=attributes,
                        label=label_id,
                    )
                )
                self._ann_types.add(AnnotationType.bbox)

            image = image_path_by_id.pop(item_id, None)
            if image:
                image = Image.from_file(path=image)

            if self._subset == "default":
                calib_path = osp.join(self._path, Kitti3dPath.CALIB_DIR, item_id + ".txt")
            else:
                calib_path = osp.join(self._path, Kitti3dPath.CALIB_DIR, self._subset, item_id + ".txt")
            items.append(
                DatasetItem(
                    id=item_id,
                    subset=self._subset,
                    media=image,
                    attributes={"calib_path": calib_path},
                    annotations=anns,
                )
            )

        return items

    def _parse_field(self, value: str, desired_type: Type[T], field_name: str) -> T:
        try:
            return desired_type(value)
        except Exception as e:
            raise InvalidAnnotationError(f"Can't parse {field_name} from '{value}'. Expected {desired_type}") from e
