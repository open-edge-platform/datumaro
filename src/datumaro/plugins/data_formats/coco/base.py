# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import logging as log
import os.path as osp
from inspect import isclass
from typing import Any, Dict, Iterator, Optional, Tuple, Type, TypeVar, Union, overload

import pycocotools.mask as mask_utils
from attrs import define

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    CompiledMask,
    Label,
    LabelCategories,
    Mask,
    Points,
    PointsCategories,
    Polygon,
    RleMask,
)
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetItem, SubsetBase
from datumaro.components.errors import (
    DatasetImportError,
    InvalidAnnotationError,
    InvalidFieldTypeError,
    MissingFieldError,
    UndeclaredLabelError,
)
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image
from datumaro.util import NOTSET, parse_json_file, take_by
from datumaro.util.image import lazy_image, load_image
from datumaro.util.mask_tools import bgr2index
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file

from .format import CocoImporterType, CocoPath, CocoTask
from .page_mapper import COCOPageMapper

T = TypeVar("T")


class DirPathExtracter:
    @staticmethod
    def find_rootpath(path: str) -> str:
        """Find root path from annotation json file path."""
        path = osp.abspath(path)
        if osp.dirname(path).endswith(CocoPath.ANNOTATIONS_DIR):
            return path.rsplit(CocoPath.ANNOTATIONS_DIR, maxsplit=1)[0]
        raise DatasetImportError(
            f"Annotation path ({path}) should be under the directory which is named {CocoPath.ANNOTATIONS_DIR}. "
            "If not, Datumaro fails to find the root path for this dataset. "
            "Please follow this instruction, https://github.com/cocodataset/cocoapi/blob/master/README.txt"
        )

    @staticmethod
    def find_images_dir(rootpath: str, subset: str) -> str:
        """Find images directory from the root path."""

        if rootpath and osp.isdir(osp.join(rootpath, CocoPath.IMAGES_DIR)):
            images_dir = osp.join(rootpath, CocoPath.IMAGES_DIR)
            if osp.isdir(osp.join(images_dir, subset or DEFAULT_SUBSET_NAME)):
                images_dir = osp.join(images_dir, subset or DEFAULT_SUBSET_NAME)
            return images_dir

        raise DatasetImportError(
            f"We found the rootpath ({rootpath}) for this dataset. "
            f"However, there should exist a directory for images as {osp.join(rootpath, CocoPath.IMAGES_DIR)}. "
            "If not, Datumaro fails to find the image directory path. "
            "Please follow this instruction, https://github.com/cocodataset/cocoapi/blob/master/README.txt"
        )


class RoboflowDirPathExtracter(DirPathExtracter):
    @staticmethod
    def find_rootpath(path: str) -> str:
        path, _ = osp.split(path)
        path, _ = osp.split(path)
        return path

    @staticmethod
    def find_images_dir(rootpath: str, subset: str) -> str:
        return osp.join(rootpath, subset)


class MmdetDirPathExtracter(DirPathExtracter):
    @staticmethod
    def find_rootpath(path: str) -> str:
        """Find root path from annotation json file path."""
        path = osp.abspath(path)
        if osp.dirname(path).endswith(CocoPath.ANNOTATIONS_DIR):
            return path.rsplit(CocoPath.ANNOTATIONS_DIR, maxsplit=1)[0]
        raise DatasetImportError(
            f"Annotation path ({path}) should be under the directory which is named {CocoPath.ANNOTATIONS_DIR}. "
            "If not, Datumaro fails to find the root path for this dataset. "
            "Please follow this instruction, https://github.com/cocodataset/cocoapi/blob/master/README.txt"
        )

    @staticmethod
    def find_images_dir(rootpath: str, subset: str) -> str:
        """Find images directory from the root path."""
        return osp.join(rootpath, subset)


class _CocoBase(SubsetBase):
    """
    Parses COCO annotations written in the following format:
    https://cocodataset.org/#format-data
    """

    def __init__(
        self,
        path,
        task,
        *,
        merge_instance_polygons: bool = False,
        keep_original_category_ids: bool = False,
        coco_importer_type: CocoImporterType = CocoImporterType.default,
        subset: Optional[str] = None,
        stream: bool = False,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isfile(path):
            raise FileNotFoundError(errno.ENOENT, "Can't find JSON file", path)
        self._path = path

        if not subset:
            parts = osp.splitext(osp.basename(path))[0].split(task.name + "_", maxsplit=1)
            subset = parts[1] if len(parts) == 2 else None
        super().__init__(subset=subset, ctx=ctx)

        if coco_importer_type == CocoImporterType.default:
            self._rootpath = DirPathExtracter.find_rootpath(path)
            self._images_dir = DirPathExtracter.find_images_dir(self._rootpath, subset)
        elif coco_importer_type == CocoImporterType.roboflow:
            self._rootpath = RoboflowDirPathExtracter.find_rootpath(path)
            self._images_dir = RoboflowDirPathExtracter.find_images_dir(self._rootpath, subset)
        elif coco_importer_type == CocoImporterType.mmdet:
            self._rootpath = MmdetDirPathExtracter.find_rootpath(path)
            self._images_dir = MmdetDirPathExtracter.find_images_dir(self._rootpath, subset)
        else:
            raise DatasetImportError(f"Not supported type: {coco_importer_type}")

        self._task = task

        self._merge_instance_polygons = merge_instance_polygons

        self._label_map = {}  # coco_id -> dm_id
        if self._task == CocoTask.panoptic:
            self._mask_dir = osp.splitext(path)[0]
        else:
            self._mask_dir = None

        self._stream = stream
        if not stream:
            self._page_mapper = None  # No use in case of stream = False

            json_data = parse_json_file(path)

            self._load_categories(
                json_data,
                keep_original_ids=keep_original_category_ids,
            )

            self._items = self._load_items(json_data)

            del json_data
        else:
            self._page_mapper = COCOPageMapper(path)

            categories_data = self._page_mapper.stream_parse_categories_data()

            self._load_categories(
                {"categories": categories_data},
                keep_original_ids=keep_original_category_ids,
            )

            self._length = None

    def __len__(self) -> int:
        if self.is_stream:
            if self._length is None:
                # Before we actually iterate over the items, we use the length of item page map.
                # It can be different with the actual length,
                # because there is a possiblity that an item cannot be parsed properly.
                return len(self._page_mapper)
            return self._length
        else:
            return len(self._items)

    def __iter__(self) -> Iterator[DatasetItem]:
        if self.is_stream:
            yield from self._stream_items()
        else:
            yield from self._items.values()

    def _load_categories(self, json_data, *, keep_original_ids):
        self._categories = {}

        if has_meta_file(self._rootpath):
            labels = parse_meta_file(self._rootpath).keys()
            self._categories = {AnnotationType.label: LabelCategories.from_iterable(labels)}
            # 0 is reserved for no class
            self._label_map = {i + 1: i for i in range(len(labels))}
        elif self._task in [
            CocoTask.instances,
            CocoTask.labels,
            CocoTask.person_keypoints,
            CocoTask.stuff,
            CocoTask.panoptic,
        ]:
            self._load_label_categories(
                self._parse_field(json_data, "categories", list),
                keep_original_ids=keep_original_ids,
            )

            if self._task == CocoTask.person_keypoints:
                self._load_person_kp_categories(self._parse_field(json_data, "categories", list))

        # informs users if 0 is found as category id sicne 0 is reserved for no class
        found = [
            self._categories[AnnotationType.label][label_id].name
            for cat_id, label_id in self._label_map.items()
            if cat_id == 0
            and self._categories[AnnotationType.label][label_id].name.lower() != "background"
        ]
        if found:
            category_name = found[0]
            log.warning(
                "Category id of '0' is reserved for no class (background) but "
                f"category named '{category_name}' with id of '0' is found in {self._path}. "
                "Please be warned that annotations with category id of '0' would have `None` as label. "
                "(https://open-edge-platform.github.io/datumaro/latest/docs/explanation/formats/coco.html#import-coco-dataset)"
            )

    def _load_label_categories(self, json_cat, *, keep_original_ids):
        categories = LabelCategories()
        label_map = {}

        cats = sorted(
            (
                {
                    "id": self._parse_field(c, "id", int),
                    "name": self._parse_field(c, "name", str),
                    "supercategory": c.get("supercategory"),
                }
                for c in json_cat
            ),
            key=lambda cat: cat["id"],
        )

        if keep_original_ids:
            for cat in cats:
                label_map[cat["id"]] = cat["id"]

                while len(categories) < cat["id"]:
                    categories.add(f"class-{len(categories)}")

                categories.add(cat["name"], parent=cat.get("supercategory"))
        else:
            for idx, cat in enumerate(cats):
                label_map[cat["id"]] = idx
                categories.add(cat["name"], parent=cat.get("supercategory"))

        self._categories[AnnotationType.label] = categories
        self._label_map = label_map

    def _load_person_kp_categories(self, json_cat):
        categories = PointsCategories()
        for cat in json_cat:
            label_id = self._label_map[self._parse_field(cat, "id", int)]
            categories.add(
                label_id,
                labels=self._parse_field(cat, "keypoints", list),
                joints=self._parse_field(cat, "skeleton", list),
            )

        self._categories[AnnotationType.points] = categories

    def _stream_items(self) -> Iterator[DatasetItem]:
        pbars = self._ctx.progress_reporter
        length = 0

        for img_info, ann_infos in pbars.iter(
            self._page_mapper,
            desc=f"Importing '{self._subset}'",
        ):
            parsed = self._parse_item(img_info)
            if parsed is None:
                continue

            _, item = parsed

            for ann_info in ann_infos:
                self._parse_anns(img_info, ann_info, item)

            yield item
            length += 1
            for ann in item.annotations:
                self._ann_types.add(ann.type)

        self._length = length

    def _parse_anns(self, img_info, ann_info, item):
        try:
            if self._task is not CocoTask.panoptic:
                self._load_annotations(ann_info, img_info, parsed_annotations=item.annotations)
            else:
                self._load_panoptic_ann(ann_info, parsed_annotations=item.annotations)
        except Exception as e:
            self._ctx.error_policy.report_annotation_error(
                e, item_id=(ann_info.get("id", None), self._subset)
            )

    def _load_items(self, json_data):
        pbar = self._ctx.progress_reporter

        def _gen_ann(info_lists):
            while info_lists:
                yield info_lists.pop()

        items = {}
        img_infos = {}
        img_lists = self._parse_field(json_data, "images", list)
        for img_info in _gen_ann(img_lists):
            parsed = self._parse_item(img_info)
            if parsed is None:
                continue

            img_id, item = parsed

            # Store item (DatasetItem) and img_info (Dict) to the integer key dictionary
            items[img_id] = item
            img_infos[img_id] = img_info

        ann_lists = self._parse_field(json_data, "annotations", list)
        for ann_info in pbar.iter(
            _gen_ann(ann_lists),
            desc=f"Importing '{self._subset}'",
            total=len(ann_lists),
        ):
            try:
                img_id = self._parse_field(ann_info, "image_id", int)
                if img_id not in img_infos:
                    log.warn(f"Unknown image id '{img_id}'")
                    continue

                # Retrieve item (DatasetItem) and img_info (Dict) from the integer key dictionary
                item = items[img_id]
                img_info = img_infos[img_id]
                self._parse_anns(img_info, ann_info, item)
                for ann in item.annotations:
                    self._ann_types.add(ann.type)
            except Exception as e:
                self._ctx.error_policy.report_annotation_error(
                    e, item_id=(ann_info.get("id", None), self._subset)
                )

        return items

    def _parse_item(self, img_info: Dict[str, Any]) -> Optional[Tuple[int, DatasetItem]]:
        try:
            img_id = self._parse_field(img_info, "id", int)

            if img_info.get("height") and img_info.get("width"):
                image_size = (
                    self._parse_field(img_info, "height", int),
                    self._parse_field(img_info, "width", int),
                )
            else:
                image_size = None

            file_name = self._parse_field(img_info, "file_name", str)
            return img_id, DatasetItem(
                id=osp.splitext(file_name)[0],
                subset=self._subset,
                media=Image.from_file(path=osp.join(self._images_dir, file_name), size=image_size),
                annotations=[],
                attributes={"id": img_id},
            )
        except Exception as e:
            self._ctx.error_policy.report_item_error(
                e, item_id=(img_info.get("id", None), self._subset)
            )

    def _load_panoptic_ann(self, ann, parsed_annotations=None):
        if parsed_annotations is None:
            parsed_annotations = []

        # For the panoptic task, each annotation struct is a per-image
        # annotation rather than a per-object annotation.
        mask_path = osp.join(self._mask_dir, self._parse_field(ann, "file_name", str))
        mask = lazy_image(mask_path, loader=self._load_pan_mask)
        mask = CompiledMask(instance_mask=mask)
        for segm_info in self._parse_field(ann, "segments_info", list):
            cat_id = self._get_label_id(segm_info)
            segm_id = self._parse_field(segm_info, "id", int)
            attributes = {"is_crowd": bool(self._parse_field(segm_info, "iscrowd", int))}
            parsed_annotations.append(
                Mask(
                    image=mask.lazy_extract(segm_id),
                    label=cat_id,
                    id=segm_id,
                    group=segm_id,
                    attributes=attributes,
                )
            )

        return parsed_annotations

    @staticmethod
    def _load_pan_mask(path):
        mask = load_image(path)
        mask = bgr2index(mask)
        return mask

    @define
    class _lazy_merged_mask:
        segmentation: Any
        h: int
        w: int

        def __call__(self):
            rles = mask_utils.frPyObjects(self.segmentation, self.h, self.w)
            return mask_utils.merge(rles)

    def _get_label_id(self, ann):
        cat_id = self._parse_field(ann, "category_id", int)
        if not cat_id:
            return None

        label_id = self._label_map.get(cat_id)
        if label_id is None:
            raise UndeclaredLabelError(str(cat_id))
        return label_id

    @overload
    def _parse_field(self, ann: Dict[str, Any], key: str, cls: Type[T]) -> T:
        ...

    @overload
    def _parse_field(self, ann: Dict[str, Any], key: str, cls: Tuple[Type, ...]) -> Any:
        ...

    def _parse_field(
        self, ann: Dict[str, Any], key: str, cls: Union[Type[T], Tuple[Type, ...]]
    ) -> Any:
        value = ann.get(key, NOTSET)
        if value is NOTSET:
            raise MissingFieldError(key)
        elif not isinstance(value, cls):
            cls = (cls,) if isclass(cls) else cls
            raise InvalidFieldTypeError(
                key, actual=str(type(value)), expected=tuple(str(t) for t in cls)
            )
        return value

    def _load_annotations(self, ann, image_info=None, parsed_annotations=None):
        if parsed_annotations is None:
            parsed_annotations = []

        ann_id = self._parse_field(ann, "id", int)

        attributes = ann.get("attributes", {})
        if "score" in ann:
            attributes["score"] = self._parse_field(ann, "score", (int, float))

        group = ann_id  # make sure all tasks' annotations are merged

        if (
            self._task is CocoTask.instances
            or self._task is CocoTask.person_keypoints
            or self._task is CocoTask.stuff
        ):
            label_id = self._get_label_id(ann)

            attributes["is_crowd"] = bool(self._parse_field(ann, "iscrowd", int))

            if self._task is CocoTask.person_keypoints:
                keypoints = self._parse_field(ann, "keypoints", list)
                if len(keypoints) % 3 != 0:
                    raise InvalidAnnotationError(
                        f"Keypoints have invalid value count {len(keypoints)}, "
                        "which is not divisible by 3. Expected (x, y, visibility) triplets."
                    )

                points = []
                visibility = []
                for x, y, v in take_by(keypoints, 3):
                    points.append(x)
                    points.append(y)
                    visibility.append(v)

                parsed_annotations.append(
                    Points(
                        points,
                        visibility,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                    )
                )

            try:
                segmentation = self._parse_field(ann, "segmentation", (list, dict))
            except MissingFieldError as e:
                log.warn(str(e))
                segmentation = None

            if segmentation and segmentation != [[]]:
                rle = None

                if isinstance(segmentation, list):
                    if not self._merge_instance_polygons:
                        # polygon - a single object can consist of multiple parts
                        for polygon_points in segmentation:
                            if len(polygon_points) % 2 != 0:
                                raise InvalidAnnotationError(
                                    f"Polygon has invalid value count {len(polygon_points)}, "
                                    "which is not divisible by 2."
                                )
                            elif len(polygon_points) < 6:
                                raise InvalidAnnotationError(
                                    f"Polygon has invalid value count {len(polygon_points)}. "
                                    "Expected at least 3 (x, y) pairs."
                                )

                            parsed_annotations.append(
                                Polygon(
                                    points=polygon_points,
                                    label=label_id,
                                    id=ann_id,
                                    attributes=attributes,
                                    group=group,
                                )
                            )
                    else:
                        # merge all parts into a single mask RLE
                        img_h = self._parse_field(image_info, "height", int)
                        img_w = self._parse_field(image_info, "width", int)
                        rle = self._lazy_merged_mask(segmentation, img_h, img_w)
                elif isinstance(segmentation["counts"], list):
                    # uncompressed RLE
                    img_h = self._parse_field(image_info, "height", int)
                    img_w = self._parse_field(image_info, "width", int)

                    mask_size = self._parse_field(segmentation, "size", list)
                    if len(mask_size) != 2:
                        raise InvalidAnnotationError(
                            f"Mask size has wrong value count {len(mask_size)}. Expected 2 values."
                        )
                    mask_h, mask_w = mask_size

                    if not ((img_h == mask_h) and (img_w == mask_w)):
                        raise InvalidAnnotationError(
                            "Mask #%s does not match image size: %s vs. %s"
                            % (ann_id, (mask_h, mask_w), (img_h, img_w))
                        )
                    rle = self._lazy_merged_mask([segmentation], mask_h, mask_w)
                else:
                    # compressed RLE
                    rle = segmentation

                if rle:
                    parsed_annotations.append(
                        RleMask(
                            rle=rle, label=label_id, id=ann_id, attributes=attributes, group=group
                        )
                    )

            bbox = self._parse_field(ann, "bbox", list)
            if bbox and len(bbox) > 0:
                if len(bbox) != 4:
                    raise InvalidAnnotationError(
                        f"Bbox has wrong value count {len(bbox)}. Expected 4 values."
                    )
                x, y, w, h = bbox
                parsed_annotations.append(
                    Bbox(
                        x,
                        y,
                        w,
                        h,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                    )
                )

        elif self._task is CocoTask.labels:
            label_id = self._get_label_id(ann)
            parsed_annotations.append(
                Label(label=label_id, id=ann_id, attributes=attributes, group=group)
            )
        elif self._task is CocoTask.captions:
            caption = self._parse_field(ann, "caption", str)
            parsed_annotations.append(
                Caption(caption, id=ann_id, attributes=attributes, group=group)
            )
        else:
            raise NotImplementedError()

        return parsed_annotations

    @property
    def is_stream(self) -> bool:
        return self._stream

    def get_dataset_item(self, item_key: int) -> Optional[DatasetItem]:
        if self.is_stream:
            img_info = self._page_mapper.get_item_dict(item_key)
            ann_infos = self._page_mapper.get_anns_dict(item_key)

            parsed = self._parse_item(img_info)
            if parsed is None:
                return None

            _, item = parsed

            for ann_info in ann_infos:
                self._parse_anns(img_info, ann_info, item)

            return item
        else:
            return self._items[item_key]

    def iter_item_ids(self) -> Iterator[int]:
        if self.is_stream:
            return self._page_mapper.iter_item_ids()
        else:
            return self._items.keys()


class CocoImageInfoBase(_CocoBase):
    def __init__(self, path, **kwargs):
        kwargs["task"] = CocoTask.image_info
        super().__init__(path, **kwargs)


class CocoCaptionsBase(_CocoBase):
    def __init__(self, path, **kwargs):
        kwargs["task"] = CocoTask.captions
        super().__init__(path, **kwargs)


class CocoInstancesBase(_CocoBase):
    def __init__(self, path, **kwargs):
        kwargs["task"] = CocoTask.instances
        super().__init__(path, **kwargs)


class CocoPersonKeypointsBase(_CocoBase):
    def __init__(self, path, **kwargs):
        kwargs["task"] = CocoTask.person_keypoints
        super().__init__(path, **kwargs)


class CocoLabelsBase(_CocoBase):
    def __init__(self, path, **kwargs):
        kwargs["task"] = CocoTask.labels
        super().__init__(path, **kwargs)


class CocoPanopticBase(_CocoBase):
    def __init__(self, path, **kwargs):
        kwargs["task"] = CocoTask.panoptic
        super().__init__(path, **kwargs)


class CocoStuffBase(_CocoBase):
    def __init__(self, path, **kwargs):
        kwargs["task"] = CocoTask.stuff
        super().__init__(path, **kwargs)
