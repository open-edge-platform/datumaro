# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Callable, Optional

# Disable B410: import_lxml - the library is used for writing
from lxml import etree as ET  # nosec

from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    Caption,
    Ellipse,
    Label,
    Mask,
    Points,
    Polygon,
    PolyLine,
)
from datumaro.components.media import Image
from datumaro.components.transformer import ItemTransform
from datumaro.util.deprecation import deprecated

if TYPE_CHECKING:
    from datumaro.components.dataset_base import CategoriesInfo, DatasetItem, IDataset

__all__ = [
    "XPathDatasetFilter",
    "XPathAnnotationsFilter",
    "UserFunctionDatasetFilter",
    "UserFunctionAnnotationsFilter",
]


@deprecated(deprecated_version="1.11", removed_version="1.12")
class DatasetItemEncoder:
    @classmethod
    def encode(
        cls, item: DatasetItem, categories: Optional[CategoriesInfo] = None
    ) -> ET.ElementBase:
        item_elem = ET.Element("item")
        ET.SubElement(item_elem, "id").text = str(item.id)
        ET.SubElement(item_elem, "subset").text = str(item.subset)

        image = item.media
        if isinstance(image, Image):
            item_elem.append(cls.encode_image(image))

        for ann in item.annotations:
            item_elem.append(cls.encode_annotation(ann, categories))

        return item_elem

    @classmethod
    def encode_image(cls, image: Image) -> ET.ElementBase:
        image_elem = ET.Element("image")

        size = image.size
        if size is not None:
            h, w = str(size[0]), str(size[1])
        else:
            h = "unknown"
            w = h
        ET.SubElement(image_elem, "height").text = h
        ET.SubElement(image_elem, "width").text = w

        ET.SubElement(image_elem, "has_data").text = "%d" % int(image.has_data)
        if hasattr(image, "path"):
            ET.SubElement(image_elem, "path").text = image.path

        return image_elem

    @classmethod
    def encode_annotation_base(cls, annotation: Annotation) -> ET.ElementBase:
        assert isinstance(annotation, Annotation)
        ann_elem = ET.Element("annotation")
        ET.SubElement(ann_elem, "id").text = str(annotation.id)
        ET.SubElement(ann_elem, "type").text = str(annotation.type.name)

        for k, v in annotation.attributes.items():
            if k.isdigit():
                k = "_" + k
            ET.SubElement(ann_elem, k.replace(" ", "-")).text = str(v)

        ET.SubElement(ann_elem, "group").text = str(annotation.group)

        return ann_elem

    @staticmethod
    def _get_label(label_id: Optional[int], categories: Optional[CategoriesInfo]) -> str:
        label = ""
        if label_id is None:
            return ""
        if categories is not None:
            label_cat = categories.get(AnnotationType.label)
            if label_cat is not None:
                label = label_cat.items[label_id].name
        return label

    @classmethod
    def encode_label_object(
        cls, obj: Label, categories: Optional[CategoriesInfo]
    ) -> ET.ElementBase:
        ann_elem = cls.encode_annotation_base(obj)

        ET.SubElement(ann_elem, "label").text = str(cls._get_label(obj.label, categories))
        ET.SubElement(ann_elem, "label_id").text = str(obj.label)

        return ann_elem

    @classmethod
    def encode_mask_object(cls, obj: Mask, categories: Optional[CategoriesInfo]) -> ET.ElementBase:
        ann_elem = cls.encode_annotation_base(obj)

        ET.SubElement(ann_elem, "label").text = str(cls._get_label(obj.label, categories))
        ET.SubElement(ann_elem, "label_id").text = str(obj.label)

        return ann_elem

    @classmethod
    def encode_bbox_object(cls, obj: Bbox, categories: Optional[CategoriesInfo]) -> ET.ElementBase:
        ann_elem = cls.encode_annotation_base(obj)

        ET.SubElement(ann_elem, "label").text = str(cls._get_label(obj.label, categories))
        ET.SubElement(ann_elem, "label_id").text = str(obj.label)
        ET.SubElement(ann_elem, "x").text = str(obj.x)
        ET.SubElement(ann_elem, "y").text = str(obj.y)
        ET.SubElement(ann_elem, "w").text = str(obj.w)
        ET.SubElement(ann_elem, "h").text = str(obj.h)
        ET.SubElement(ann_elem, "area").text = str(obj.get_area())

        return ann_elem

    @classmethod
    def encode_points_object(
        cls, obj: Points, categories: Optional[CategoriesInfo]
    ) -> ET.ElementBase:
        ann_elem = cls.encode_annotation_base(obj)

        ET.SubElement(ann_elem, "label").text = str(cls._get_label(obj.label, categories))
        ET.SubElement(ann_elem, "label_id").text = str(obj.label)

        x, y, w, h = obj.get_bbox()
        area = w * h
        bbox_elem = ET.SubElement(ann_elem, "bbox")
        ET.SubElement(bbox_elem, "x").text = str(x)
        ET.SubElement(bbox_elem, "y").text = str(y)
        ET.SubElement(bbox_elem, "w").text = str(w)
        ET.SubElement(bbox_elem, "h").text = str(h)
        ET.SubElement(bbox_elem, "area").text = str(area)

        points = obj.points
        for i in range(0, len(points), 2):
            point_elem = ET.SubElement(ann_elem, "point")
            ET.SubElement(point_elem, "x").text = str(points[i])
            ET.SubElement(point_elem, "y").text = str(points[i + 1])
            ET.SubElement(point_elem, "visible").text = str(obj.visibility[i // 2].name)

        return ann_elem

    @classmethod
    def encode_polygon_object(
        cls, obj: Polygon, categories: Optional[CategoriesInfo]
    ) -> ET.ElementBase:
        ann_elem = cls.encode_annotation_base(obj)

        ET.SubElement(ann_elem, "label").text = str(cls._get_label(obj.label, categories))
        ET.SubElement(ann_elem, "label_id").text = str(obj.label)

        x, y, w, h = obj.get_bbox()
        area = w * h
        bbox_elem = ET.SubElement(ann_elem, "bbox")
        ET.SubElement(bbox_elem, "x").text = str(x)
        ET.SubElement(bbox_elem, "y").text = str(y)
        ET.SubElement(bbox_elem, "w").text = str(w)
        ET.SubElement(bbox_elem, "h").text = str(h)
        ET.SubElement(bbox_elem, "area").text = str(area)

        points = obj.points
        for i in range(0, len(points), 2):
            point_elem = ET.SubElement(ann_elem, "point")
            ET.SubElement(point_elem, "x").text = str(points[i])
            ET.SubElement(point_elem, "y").text = str(points[i + 1])

        return ann_elem

    @classmethod
    def encode_polyline_object(
        cls, obj: PolyLine, categories: Optional[CategoriesInfo]
    ) -> ET.ElementBase:
        ann_elem = cls.encode_annotation_base(obj)

        ET.SubElement(ann_elem, "label").text = str(cls._get_label(obj.label, categories))
        ET.SubElement(ann_elem, "label_id").text = str(obj.label)

        x, y, w, h = obj.get_bbox()
        area = w * h
        bbox_elem = ET.SubElement(ann_elem, "bbox")
        ET.SubElement(bbox_elem, "x").text = str(x)
        ET.SubElement(bbox_elem, "y").text = str(y)
        ET.SubElement(bbox_elem, "w").text = str(w)
        ET.SubElement(bbox_elem, "h").text = str(h)
        ET.SubElement(bbox_elem, "area").text = str(area)

        points = obj.points
        for i in range(0, len(points), 2):
            point_elem = ET.SubElement(ann_elem, "point")
            ET.SubElement(point_elem, "x").text = str(points[i])
            ET.SubElement(point_elem, "y").text = str(points[i + 1])

        return ann_elem

    @classmethod
    def encode_caption_object(cls, obj: Caption) -> ET.ElementBase:
        ann_elem = cls.encode_annotation_base(obj)

        ET.SubElement(ann_elem, "caption").text = str(obj.caption)

        return ann_elem

    @classmethod
    def encode_ellipse_object(
        cls, obj: Ellipse, categories: Optional[CategoriesInfo]
    ) -> ET.ElementBase:
        ann_elem = cls.encode_annotation_base(obj)

        ET.SubElement(ann_elem, "label").text = str(cls._get_label(obj.label, categories))
        ET.SubElement(ann_elem, "label_id").text = str(obj.label)

        ET.SubElement(ann_elem, "x1").text = str(obj.x1)
        ET.SubElement(ann_elem, "y1").text = str(obj.y1)
        ET.SubElement(ann_elem, "x2").text = str(obj.x2)
        ET.SubElement(ann_elem, "y2").text = str(obj.y2)
        ET.SubElement(ann_elem, "area").text = str(obj.get_area())

        return ann_elem

    @classmethod
    def encode_annotation(
        cls, o: Annotation, categories: Optional[CategoriesInfo] = None
    ) -> ET.ElementBase:
        if isinstance(o, Label):
            return cls.encode_label_object(o, categories)
        if isinstance(o, Mask):
            return cls.encode_mask_object(o, categories)
        if isinstance(o, Bbox):
            return cls.encode_bbox_object(o, categories)
        if isinstance(o, Points):
            return cls.encode_points_object(o, categories)
        if isinstance(o, PolyLine):
            return cls.encode_polyline_object(o, categories)
        if isinstance(o, Polygon):
            return cls.encode_polygon_object(o, categories)
        if isinstance(o, Caption):
            return cls.encode_caption_object(o)
        if isinstance(o, Ellipse):
            return cls.encode_ellipse_object(o, categories)

        raise NotImplementedError("Unexpected annotation object passed: %s" % o)

    @staticmethod
    def to_string(encoded_item: ET.ElementBase) -> str:
        return ET.tostring(encoded_item, encoding="unicode", pretty_print=True)


@deprecated(deprecated_version="1.11", removed_version="1.12")
class XPathDatasetFilter(ItemTransform):
    def __init__(self, extractor: IDataset, xpath: str) -> None:
        super().__init__(extractor)

        try:
            xpath_eval = ET.XPath(xpath)
        except Exception:
            log.error("Failed to create XPath from expression '%s'", xpath)
            raise

        # Return true -> filter out an item
        self._f = lambda item: bool(
            xpath_eval(DatasetItemEncoder.encode(item, extractor.categories()))
        )

    def transform_item(self, item: DatasetItem) -> Optional[DatasetItem]:
        if not self._f(item):
            return None
        return item


@deprecated(deprecated_version="1.11", removed_version="1.12")
class XPathAnnotationsFilter(ItemTransform):
    def __init__(self, extractor: IDataset, xpath: str, remove_empty: bool = False) -> None:
        super().__init__(extractor)

        try:
            xpath_eval = ET.XPath(xpath)
        except Exception:
            log.error("Failed to create XPath from expression '%s'", xpath)
            raise

        self._filter = xpath_eval

        self._remove_empty = remove_empty

    def transform_item(self, item: DatasetItem) -> Optional[DatasetItem]:
        if self._filter is None:
            return item

        encoded = DatasetItemEncoder.encode(item, self._extractor.categories())
        filtered = self._filter(encoded)
        filtered = [elem for elem in filtered if elem.tag == "annotation"]

        encoded = encoded.findall("annotation")
        annotations = [item.annotations[encoded.index(e)] for e in filtered]

        if self._remove_empty and len(annotations) == 0:
            return None
        return self.wrap_item(item, annotations=annotations)


class UserFunctionDatasetFilter(ItemTransform):
    """Filter dataset items using a user-provided Python function.

    Parameters:
        extractor: Datumaro `Dataset` to filter.
        filter_func: A Python callable that takes a `DatasetItem` as its input and
            returns a boolean. If the return value is True, that `DatasetItem` will be retained.
            Otherwise, it is removed.

    Example:
        This is an example of filtering dataset items with images larger than 1024 pixels::

        from datumaro.components.media import Image

        def filter_func(item: DatasetItem) -> bool:
            h, w = item.media_as(Image).size
            return h > 1024 or w > 1024

        filtered = UserFunctionDatasetFilter(
            extractor=dataset, filter_func=filter_func)
        # No items with an image height or width greater than 1024
        filtered_items = [item for item in filtered]
    """

    def __init__(self, extractor: IDataset, filter_func: Callable[[DatasetItem], bool]):
        super().__init__(extractor)

        self._filter_func = filter_func

    def transform_item(self, item: DatasetItem) -> Optional[DatasetItem]:
        return item if self._filter_func(item) else None


class UserFunctionAnnotationsFilter(ItemTransform):
    """Filter annotations using a user-provided Python function.

    Parameters:
        extractor: Datumaro `Dataset` to filter.
        filter_func: A Python callable that takes `DatasetItem` and `Annotation` as its inputs
            and returns a boolean. If the return value is True, the `Annotation` will be retained.
            Otherwise, it is removed.
        remove_empty: If True, `DatasetItem` without any annotations is removed
            after filtering its annotations. Otherwise, do not filter `DatasetItem`.

    Example:
        This is an example of removing bounding boxes sized greater than 50% of the image size::

        from datumaro.components.media import Image
        from datumaro.components.annotation import Annotation, Bbox

        def filter_func(item: DatasetItem, ann: Annotation) -> bool:
            # If the annotation is not a Bbox, do not filter
            if not isinstance(ann, Bbox):
                return False

            h, w = item.media_as(Image).size
            image_size = h * w
            bbox_size = ann.h * ann.w

            # Accept Bboxes smaller than 50% of the image size
            return bbox_size < 0.5 * image_size

        filtered = UserFunctionAnnotationsFilter(
            extractor=dataset, filter_func=filter_func)
        # No bounding boxes with a size greater than 50% of their image
        filtered_items = [item for item in filtered]
    """

    def __init__(
        self,
        extractor: IDataset,
        filter_func: Callable[[DatasetItem, Annotation], bool],
        remove_empty: bool = False,
    ):
        super().__init__(extractor)

        self._filter_func = filter_func
        self._remove_empty = remove_empty

    def transform_item(self, item: DatasetItem) -> Optional[DatasetItem]:
        filtered_anns = [ann for ann in item.annotations if self._filter_func(item, ann)]

        if self._remove_empty and not filtered_anns:
            return None
        return self.wrap_item(item, annotations=filtered_anns)
