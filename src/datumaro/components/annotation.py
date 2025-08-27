# Copyright (C) 2021-2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from enum import IntEnum
from functools import partial
from itertools import zip_longest
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import attr
import cv2
import numpy as np
import shapely.geometry as sg
from attr import asdict, attrs, field
from typing_extensions import Literal

from datumaro.components.media import Image
from datumaro.util.attrs_util import default_if_none, not_empty
from datumaro.util.points_util import normalize_points


class AnnotationType(IntEnum):
    unknown = 0
    label = 1
    mask = 2
    points = 3
    polygon = 4
    polyline = 5
    bbox = 6
    caption = 7
    cuboid_3d = 8
    super_resolution_annotation = 9
    depth_annotation = 10
    ellipse = 11
    tabular = 13
    rotated_bbox = 14
    cuboid_2d = 15


COORDINATE_ROUNDING_DIGITS = 2
CHECK_POLYGON_EQ_EPSILONE = 1e-7
NO_GROUP = 0
NO_OBJECT_ID = -1


@attrs(slots=True, kw_only=True, order=False)
class Annotation:
    """
    A base annotation class.

    Derived classes must define the '_type' class variable with a value
    from the AnnotationType enum.
    """

    # Describes an identifier of the annotation
    # Is not required to be unique within DatasetItem annotations or dataset
    id: int = field(default=0, validator=default_if_none(int))

    # Arbitrary annotation-specific attributes. Typically, includes
    # metainfo and properties that are not covered by other fields.
    # If possible, try to limit value types of values by the simple
    # builtin types (int, float, bool, str) to increase compatibility with
    # different formats.
    # There are some established names for common attributes like:
    # - "occluded" (bool)
    # - "visible" (bool)
    # Possible dataset attributes can be described in Categories.attributes.
    attributes: Dict[str, Any] = field(factory=dict, validator=default_if_none(dict))

    # Annotations can be grouped, which means they describe parts of a
    # single object. The value of 0 means there is no group.
    group: int = field(default=NO_GROUP, validator=default_if_none(int))

    # obeject identifier over the multiple items
    # e.g.) in a video, person 'A' could be annotated on the multiple frame images
    #   the user could assign >=0 value as id of person 'A'.
    object_id: int = field(default=NO_OBJECT_ID, validator=default_if_none(int))

    _type = AnnotationType.unknown

    @property
    def type(self) -> AnnotationType:
        return self._type  # must be set in subclasses

    def as_dict(self) -> Dict[str, Any]:
        "Returns a dictionary { field_name: value }"
        return asdict(self)

    def wrap(self, **kwargs):
        "Returns a modified copy of the object"
        return attr.evolve(self, **kwargs)


@attrs(slots=True, kw_only=True, order=False)
class Categories:
    """
    A base class for annotation metainfo. It is supposed to include
    dataset-wide metainfo like available labels, label colors,
    label attributes etc.
    """

    # Describes the list of possible annotation-type specific attributes
    # in a dataset.
    attributes: Set[str] = field(factory=set, validator=default_if_none(set), eq=False)


class GroupType(IntEnum):
    EXCLUSIVE = 0
    INCLUSIVE = 1
    RESTRICTED = 2

    def to_str(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, text: str) -> GroupType:
        try:
            return cls[text.upper()]
        except KeyError:
            raise ValueError(f"Invalid GroupType: {text}")


@attrs(slots=True, order=False)
class LabelCategories(Categories):
    @attrs(slots=True, order=False)
    class Category:
        name: str = field(converter=str, validator=not_empty)
        parent: str = field(default="", validator=default_if_none(str))
        attributes: Set[str] = field(factory=set, validator=default_if_none(set))

    @attrs(slots=True, order=False)
    class LabelGroup:
        name: str = field(converter=str, validator=not_empty)
        labels: List[str] = field(default=[], validator=default_if_none(list))
        group_type: GroupType = field(
            default=GroupType.EXCLUSIVE, validator=default_if_none(GroupType)
        )

    items: List[str] = field(factory=list, validator=default_if_none(list))
    label_groups: List[LabelGroup] = field(factory=list, validator=default_if_none(list))
    _indices: Dict[str, int] = field(factory=dict, init=False, eq=False)

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[
            Union[
                str,
                Tuple[str],
                Tuple[str, str],
                Tuple[str, str, List[str]],
            ]
        ],
    ) -> LabelCategories:
        """
        Creates a LabelCategories from iterable.

        Args:
            iterable: This iterable object can be:

                - a list of str - will be interpreted as list of Category names
                - a list of positional arguments - will generate Categories
                  with these arguments

        Returns: a LabelCategories object
        """

        temp_categories = cls()

        for category in iterable:
            if isinstance(category, str):
                category = [category]
            temp_categories.add(*category)

        return temp_categories

    def __attrs_post_init__(self):
        self._reindex()

    def _reindex(self):
        indices = {}
        for index, item in enumerate(self.items):
            assert item.name not in self._indices
            indices[item.name] = index
        self._indices = indices

    def add(
        self,
        name: str,
        parent: Optional[str] = None,
        attributes: Optional[Set[str]] = None,
    ) -> int:
        assert name
        assert name not in self._indices, name

        index = len(self.items)
        self.items.append(self.Category(name, parent, attributes))
        self._indices[name] = index
        return index

    def add_label_group(
        self,
        name: str,
        labels: List[str],
        group_type: GroupType,
    ) -> int:
        assert name

        index = len(self.label_groups)
        self.label_groups.append(self.LabelGroup(name, labels, group_type))
        return index

    def find(self, name: str) -> Tuple[Optional[int], Optional[Category]]:
        index = self._indices.get(name)
        if index is not None:
            return index, self.items[index]
        return index, None

    def __getitem__(self, idx: int) -> Category:
        return self.items[idx]

    def __contains__(self, value: Union[int, str]) -> bool:
        if isinstance(value, str):
            return self.find(value)[1] is not None
        else:
            return 0 <= value and value < len(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Category]:
        return iter(self.items)


@attrs(slots=True, order=False)
class Label(Annotation):
    _type = AnnotationType.label
    label: int = field(converter=int)


RgbColor = Tuple[int, int, int]

Colormap = Dict[int, RgbColor]
"""Represents { index -> color } mapping for segmentation masks"""


@attrs(slots=True, eq=False, order=False)
class MaskCategories(Categories):
    """
    Describes a color map for segmentation masks.
    """

    @classmethod
    def generate(cls, size: int = 255, include_background: bool = True) -> MaskCategories:
        """
        Generates MaskCategories with the specified size.

        If include_background is True, the result will include the item
            "0: (0, 0, 0)", which is typically used as a background color.
        """
        from datumaro.util.mask_tools import generate_colormap

        return cls(generate_colormap(size, include_background=include_background))

    colormap: Colormap = field(factory=dict, validator=default_if_none(dict))
    _inverse_colormap: Optional[Dict[RgbColor, int]] = field(
        default=None, validator=attr.validators.optional(dict)
    )

    @property
    def inverse_colormap(self) -> Dict[RgbColor, int]:
        from datumaro.util.mask_tools import invert_colormap

        if self._inverse_colormap is None:
            if self.colormap is not None:
                self._inverse_colormap = invert_colormap(self.colormap)
        return self._inverse_colormap

    def __contains__(self, idx: int) -> bool:
        return idx in self.colormap

    def __getitem__(self, idx: int) -> RgbColor:
        return self.colormap[idx]

    def __len__(self) -> int:
        return len(self.colormap)

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return False
        for label_id, my_color in self.colormap.items():
            other_color = other.colormap.get(label_id)
            if not np.array_equal(my_color, other_color):
                return False
        return True


BinaryMaskImage = np.ndarray  # 2d array of type bool
BinaryMaskImageCallable = Callable[[], BinaryMaskImage]
IndexMaskImage = np.ndarray  # 2d array of type int
IndexMaskImageCallable = Callable[[], IndexMaskImage]


@attrs(slots=True, eq=False, order=False)
class Mask(Annotation):
    """
    Represents a 2d single-instance binary segmentation mask.
    """

    _type = AnnotationType.mask
    _image: Union[BinaryMaskImage, BinaryMaskImageCallable] = field()
    label: Optional[int] = field(
        converter=attr.converters.optional(int), default=None, kw_only=True
    )
    z_order: int = field(default=0, validator=default_if_none(int), kw_only=True)

    def __attrs_post_init__(self):
        if isinstance(self._image, np.ndarray):
            self._image = self._image.astype(bool)

    @property
    def image(self) -> BinaryMaskImage:
        image = self._image
        if callable(image):
            image = image()
        return image

    def as_class_mask(
        self,
        label_id: Optional[int] = None,
        ignore_index: int = 0,
        dtype: Optional[np.dtype] = None,
    ) -> IndexMaskImage:
        """Produces a class index mask based on the binary mask.

        Args:
            label_id: Scalar value to represent the class index of the mask.
                If not specified, `self.label` will be used. Defaults to None.
            ignore_index: Scalar value to fill in the zeros in the binary mask.
                Defaults to 0.
            dtype: Data type for the resulting mask. If not specified,
                it will be inferred from the provided `label_id` to hold its value.
                For example, if `label_id=255`, the inferred dtype will be `np.uint8`.
                Defaults to None.

        Returns:
            IndexMaskImage: Class index mask generated from the binary mask.
        """
        if label_id is None:
            label_id = self.label
        from datumaro.util.mask_tools import make_index_mask

        return make_index_mask(self.image, index=label_id, ignore_index=ignore_index, dtype=dtype)

    def as_instance_mask(
        self,
        instance_id: int,
        ignore_index: int = 0,
        dtype: Optional[np.dtype] = None,
    ) -> IndexMaskImage:
        """Produces an instance index mask based on the binary mask.

        Args:
            instance_id: Scalar value to represent the instance id.
            ignore_index: Scalar value to fill in the zeros in the binary mask.
                Defaults to 0.
            dtype: Data type for the resulting mask. If not specified,
                it will be inferred from the provided `label_id` to hold its value.
                For example, if `label_id=255`, the inferred dtype will be `np.uint8`.
                Defaults to None.

        Returns:
            IndexMaskImage: Instance index mask generated from the binary mask.
        """
        from datumaro.util.mask_tools import make_index_mask

        return make_index_mask(
            self.image, index=instance_id, ignore_index=ignore_index, dtype=dtype
        )

    def get_area(self) -> int:
        return np.count_nonzero(self.image)

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """
        Computes the bounding box of the mask.

        Returns: [x, y, w, h]
        """
        from datumaro.util.mask_tools import find_mask_bbox

        return find_mask_bbox(self.image)

    def paint(self, colormap: Colormap) -> np.ndarray:
        """
        Applies a colormap to the mask and produces the resulting image.
        """
        from datumaro.util.mask_tools import paint_mask

        return paint_mask(self.as_class_mask(), colormap)

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return False
        return (
            (self.label == other.label)
            and (self.z_order == other.z_order)
            and (np.array_equal(self.image, other.image))
        )


@attrs(slots=True, eq=False, order=False)
class RleMask(Mask):
    """
    An RLE-encoded instance segmentation mask.
    """

    _rle = field()  # uses pycocotools RLE representation

    _image = field(init=False, default=None)

    @property
    def image(self) -> BinaryMaskImage:
        return self._decode(self.rle)

    @property
    def rle(self):
        rle = self._rle
        if callable(rle):
            rle = rle()
        return rle

    @staticmethod
    def _decode(rle):
        from pycocotools import mask as mask_utils

        return mask_utils.decode(rle)

    def get_area(self) -> int:
        from pycocotools import mask as mask_utils

        return mask_utils.area(self.rle)

    def get_bbox(self) -> Tuple[int, int, int, int]:
        from pycocotools import mask as mask_utils

        return mask_utils.toBbox(self.rle)

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return super().__eq__(other)
        return self.rle == other.rle


@attrs(slots=True, eq=False, order=False)
class ExtractedMask(Mask):
    """Mask annotation (binary mask) extracted from an index mask (integer 2D Numpy array).

    This class can extract a binary mask with given index mask and index value.
    The advantage of this class is that we can create multiple binary mask but they share a single index mask source.

    Attributes:
        index_mask: Integer 2D Numpy array. Its pixel can indicate a label id (class) or an instance id.
        index: Integer value to extract a binary mask from the given index mask.

    Examples:
        This example demonstrates how to create an `ExtractedMask` from a synthetic index mask,
        which denotes a semantic segmentation mask with binary values such as 0 for background
        and 1 for foreground.

        >>> import numpy as np
        >>> from datumaro.components.annotation import ExtractedMask
        >>>
        >>> index_mask = np.random.randint(low=0, high=2, size=(10, 10), dtype=np.uint8)
        >>> mask1 = ExtractedMask(index_mask=index_mask, index=0, label=0)  # 0 for background
        >>> mask2 = ExtractedMask(index_mask=index_mask, index=1, label=1)  # 1 for foreground
        >>> np.unique(mask1.image).tolist()  # `image` property create a binary mask
        np.array([0, 1])
        >>> mask1.index_mask == mask2.index_mask  # They share the same source
        True
    """

    index_mask: Union[IndexMaskImage, IndexMaskImageCallable] = field()
    index: int = field()

    _image: None = field(init=False, default=None)

    @property
    def image(self) -> BinaryMaskImage:
        index_mask = self.index_mask() if callable(self.index_mask) else self.index_mask
        return index_mask == self.index


CompiledMaskImage = np.ndarray  # 2d of integers (of different precision)


class CompiledMask:
    """
    Represents class- and instance- segmentation masks with
    all the instances (opposed to single-instance masks).
    """

    @staticmethod
    def from_instance_masks(
        instance_masks: Iterable[Mask],
        instance_ids: Optional[Iterable[int]] = None,
        instance_labels: Optional[Iterable[int]] = None,
    ) -> CompiledMask:
        """
        Joins instance masks into a single mask. Masks are sorted by
        z_order (ascending) prior to merging.

        Parameters:
            instance_ids: Instance id values for the produced instance mask.
                By default, mask positions are used.
            instance_labels: Instance label id values for the produced class
                mask. By default, mask labels are used.
        """

        from datumaro.util.mask_tools import make_index_mask

        instance_ids = instance_ids or []
        instance_labels = instance_labels or []
        masks = sorted(
            zip_longest(instance_masks, instance_ids, instance_labels), key=lambda m: m[0].z_order
        )

        max_index = len(masks) + 1
        index_dtype = np.min_scalar_type(max_index)

        masks = (
            (m, 1 + i, id if id is not None else 1 + i, label if label is not None else m.label)
            for i, (m, id, label) in enumerate(masks)
        )

        # This optimized version is supposed for:
        # 1. Avoiding memory explosion on materialization of all masks
        # 2. Optimizing mask materialization calls (RLE decoding)
        # 3. Optimizing intermediate mask memory use
        #
        # Basically, a mask can be quite large (e.g. 10k x 10k @ int32 etc.),
        # so we can only afford having just few copies in
        # memory simultaneously.

        it = iter(masks)

        instance_map = [0]
        class_map = [0]

        m, idx, instance_id, class_id = next(it)
        if not class_id:
            idx = 0
        index_mask = make_index_mask(m.image, idx, dtype=index_dtype)
        instance_map.append(instance_id)
        class_map.append(class_id)

        for m, idx, instance_id, class_id in it:
            if not class_id:
                idx = 0
            index_mask = np.where(m.image, idx, index_mask)
            instance_map.append(instance_id)
            class_map.append(class_id)

        # Generate compiled masks

        if np.array_equal(instance_map, range(max_index)):
            merged_instance_mask = index_mask
        else:
            merged_instance_mask = np.array(instance_map, dtype=np.min_scalar_type(instance_map))[
                index_mask
            ]

        merged_class_mask = np.array(class_map, dtype=np.min_scalar_type(class_map))[index_mask]

        return __class__(class_mask=merged_class_mask, instance_mask=merged_instance_mask)

    def __init__(
        self,
        class_mask: Union[None, CompiledMaskImage, Callable[[], CompiledMaskImage]] = None,
        instance_mask: Union[None, CompiledMaskImage, Callable[[], CompiledMaskImage]] = None,
    ):
        self._class_mask = class_mask
        self._instance_mask = instance_mask

    @staticmethod
    def _get_image(image):
        if callable(image):
            return image()
        return image

    @property
    def class_mask(self) -> Optional[CompiledMaskImage]:
        return self._get_image(self._class_mask)

    @property
    def instance_mask(self) -> Optional[CompiledMaskImage]:
        return self._get_image(self._instance_mask)

    @property
    def instance_count(self) -> int:
        return int(self.instance_mask.max())

    def get_instance_labels(self) -> Dict[int, int]:
        """
        Matches the class and instance masks.

        Returns: { instance id: class id }
        """

        class_shift = 16
        m = (self.class_mask.astype(np.uint32) << class_shift) + self.instance_mask.astype(
            np.uint32
        )
        keys = np.unique(m)
        instance_labels = {
            int(k & ((1 << class_shift) - 1)): int(k >> class_shift)
            for k in keys
            if k & ((1 << class_shift) - 1) != 0
        }
        return instance_labels

    def extract(self, instance_id: int) -> IndexMaskImage:
        """
        Extracts a single-instance mask from the compiled mask.
        """

        return self.instance_mask == instance_id

    def lazy_extract(self, instance_id: int) -> Callable[[], IndexMaskImage]:
        return partial(self.extract, instance_id)


@attrs(slots=True, order=False)
class Shape(Annotation):
    """
    Base class for shape annotations. This class defines the common attributes and methods
    for different types of shape annotations.

    Attributes:
        points (List[float]): List of float values representing the coordinates of the shape.
        label (Optional[int]): Optional label ID for the shape. Default is None.
        z_order (int): Z-order of the shape, used to determine the rendering order. Default is 0.

    Methods:
        get_area: Abstract method to calculate the area of the shape.
        as_polygon: Abstract method to convert the shape into a polygon representation.
        get_bbox: Returns the bounding box of the shape as [x, y, w, h].
        get_points: Returns the points of the shape as a list of (x, y) tuples.
    """

    points: List[float] = field(
        converter=lambda x: np.array(x, dtype=np.float32).round(COORDINATE_ROUNDING_DIGITS).tolist()
    )

    label: Optional[int] = field(
        converter=attr.converters.optional(int), default=None, kw_only=True
    )

    z_order: int = field(default=0, validator=default_if_none(int), kw_only=True)

    def get_area(self):
        """
        Calculate the area of the shape.
        """
        raise NotImplementedError()

    def as_polygon(self) -> List[float]:
        """
        Convert the shape into a polygon representation.
        """
        raise NotImplementedError()

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """
        Calculate and return the bounding box of the shape.

        Returns:
            Tuple[float, float, float, float]: The bounding box as [x, y, w, h].
        """

        points = self.points
        if not points:
            return None

        xs = [p for p in points[0::2]]
        ys = [p for p in points[1::2]]
        x0 = min(xs)
        x1 = max(xs)
        y0 = min(ys)
        y1 = max(ys)
        return [x0, y0, x1 - x0, y1 - y0]

    def get_points(self) -> Optional[List[Tuple[float, float]]]:
        """
        Convert and return the points of the shape as a list of (x, y) tuples.

        Returns:
            Optional[List[Tuple[float, float]]]: List of points as (x, y) tuples, or None if no points.
        """
        points = self.points
        if not points:
            return None

        assert len(points) % 2 == 0, "points should have (2 x points) number of float values."

        xs = [p for p in points[0::2]]
        ys = [p for p in points[1::2]]

        return [(x, y) for x, y in zip(xs, ys)]


@attrs(slots=True, order=False)
class PolyLine(Shape):
    """
    PolyLine annotation class. This class represents a polyline shape, which is a series of connected line segments.

    Attributes:
        _type (AnnotationType): The type of annotation, set to `AnnotationType.polyline`.

    Methods:
        as_polygon: Returns the points of the polyline as a polygon.
        get_area: Returns the area of the polyline, which is always 0.
    """

    _type = AnnotationType.polyline

    def as_polygon(self):
        return self.points[:]

    def get_area(self):
        return 0


@attrs(slots=True, init=False, order=False)
class Cuboid3d(Annotation):
    """
    Cuboid3d annotation class. This class represents a 3D cuboid annotation with position, rotation, and scale.

    Attributes:
        _type (AnnotationType): The type of annotation, set to `AnnotationType.cuboid_3d`.
        _points (List[float]): List of float values representing the position, rotation, and scale of the cuboid.
        label (Optional[int]): Optional label ID for the cuboid. Default is None.

    Methods:
        __init__: Initializes the Cuboid3d with position, rotation, and scale.
        position: Property to get and set the position of the cuboid.
        rotation: Property to get and set the rotation of the cuboid.
        scale: Property to get and set the scale of the cuboid.
    """

    _type = AnnotationType.cuboid_3d
    _points: List[float] = field(default=None)
    label: Optional[int] = field(
        converter=attr.converters.optional(int), default=None, kw_only=True
    )

    @_points.validator
    def _points_validator(self, attribute, points):
        """
        Validate and round the points representing the cuboid's position, rotation, and scale.

        Args:
            attribute: The attribute being validated.
            points: The list of float values to validate.
        """
        if points is None:
            points = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        else:
            assert len(points) == 3 + 3 + 3, points
            points = np.around(points, COORDINATE_ROUNDING_DIGITS).tolist()
        self._points = points

    def __init__(self, position, rotation=None, scale=None, **kwargs):
        """
        Initialize the Cuboid3d with position, rotation, and scale.

        Args:
            position (List[float]): List of 3 float values representing the position [x, y, z].
            rotation (List[float], optional): List of 3 float values representing the rotation [rx, ry, rz].
            scale (List[float], optional): List of 3 float values representing the scale [sx, sy, sz].
        """
        assert len(position) == 3, position
        if not rotation:
            rotation = [0] * 3
        if not scale:
            scale = [1] * 3
        kwargs.pop("points", None)
        self.__attrs_init__(points=[*position, *rotation, *scale], **kwargs)

    @property
    def position(self):
        """
        Get the position of the cuboid.

        Returns:
            List[float]: The position [x, y, z] of the cuboid.
        """
        return self._points[0:3]

    @position.setter
    def _set_poistion(self, value):
        """
        Set the position of the cuboid.

        Args:
            value (List[float]): The new position [x, y, z] of the cuboid.
        """
        # TODO: fix the issue with separate coordinate rounding:
        # self.position[0] = 12.345676
        # - the number assigned won't be rounded.
        self.position[:] = np.around(value, COORDINATE_ROUNDING_DIGITS).tolist()

    @property
    def rotation(self):
        """
        Get the rotation of the cuboid.

        Returns:
            List[float]: The rotation [rx, ry, rz] of the cuboid.
        """
        return self._points[3:6]

    @rotation.setter
    def _set_rotation(self, value):
        """
        Set the rotation of the cuboid.

        Args:
            value (List[float]): The new rotation [rx, ry, rz] of the cuboid.
        """
        self.rotation[:] = np.around(value, COORDINATE_ROUNDING_DIGITS).tolist()

    @property
    def scale(self):
        """
        Get the scale of the cuboid.

        Returns:
            List[float]: The scale [sx, sy, sz] of the cuboid.
        """
        return self._points[6:9]

    @scale.setter
    def _set_scale(self, value):
        """
        Set the scale of the cuboid.

        Args:
            value (List[float]): The new scale [sx, sy, sz] of the cuboid.
        """
        self.scale[:] = np.around(value, COORDINATE_ROUNDING_DIGITS).tolist()


@attrs(slots=True, order=False, eq=False)
class Polygon(Shape):
    """
    Polygon annotation class. This class represents a polygon shape defined by a series of points.

    Attributes:
        _type (AnnotationType): The type of annotation, set to `AnnotationType.polygon`.

    Methods:
        __attrs_post_init__: Validates the points to ensure they form a valid polygon.
        get_area: Calculates the area of the polygon using the shoelace formula.
        as_polygon: Returns the points of the polygon.
        __eq__: Compares this polygon with another for equality.
        _get_shoelace_area: Helper method to calculate the area of the polygon using the shoelace formula.
    """

    _type = AnnotationType.polygon

    def __attrs_post_init__(self):
        """
        Validate the points to ensure they form a valid polygon.

        Raises:
            AssertionError: If the number of points is not even or less than 3 pairs of coordinates.
        """
        # keep the message on a single line to produce informative output
        assert len(self.points) % 2 == 0 and 3 <= len(self.points) // 2, (
            "Wrong polygon points: %s" % self.points
        )

    def get_area(self):
        """
        Calculate the area of the polygon using the shoelace formula.

        Returns:
            float: The area of the polygon.
        """
        # import pycocotools.mask as mask_utils

        # x, y, w, h = self.get_bbox()
        # rle = mask_utils.frPyObjects([self.points], y + h, x + w)
        # area = mask_utils.area(rle)[0]
        area = self._get_shoelace_area()
        return area

    def as_polygon(self) -> List[float]:
        """
        Return the points of the polygon.

        Returns:
            List[float]: The points of the polygon.
        """
        return self.points

    def __eq__(self, other):
        """
        Compare this polygon with another for equality.

        Args:
            other: The other polygon to compare with.

        Returns:
            bool: True if the polygons are equal, False otherwise.
        """
        if not isinstance(other, __class__):
            return False
        if (
            not Annotation.__eq__(self, other)
            or self.label != other.label
            or self.z_order != other.z_order
        ):
            return False

        self_points = self.get_points()
        other_points = other.get_points()
        self_polygon = sg.Polygon(self_points)
        other_polygon = sg.Polygon(other_points)
        # if polygon is not valid, compare points
        if not (self_polygon.is_valid and other_polygon.is_valid):
            return self_points == other_points
        inter_area = self_polygon.intersection(other_polygon).area
        return abs(self_polygon.area - inter_area) < CHECK_POLYGON_EQ_EPSILONE

    def _get_shoelace_area(self):
        """
        Calculate the area of the polygon using the shoelace formula.

        Returns:
            float: The area of the polygon.
        """
        points = self.get_points()
        n = len(points)
        # Not a polygon
        if n < 3:
            return 0

        area = 0.0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]  # Next vertex, wrapping around using modulo
            area += x1 * y2 - y1 * x2

        return abs(area) / 2.0


@attrs(slots=True, init=False, order=False)
class Bbox(Shape):
    """
    Bbox annotation class. This class represents a bounding box defined by its top-left corner (x, y)
    and its width and height (w, h).

    Attributes:
        _type (AnnotationType): The type of annotation, set to `AnnotationType.bbox`.

    Methods:
        __init__: Initializes the Bbox with its coordinates and dimensions.
        x: Property to get the x-coordinate of the bounding box.
        y: Property to get the y-coordinate of the bounding box.
        w: Property to get the width of the bounding box.
        h: Property to get the height of the bounding box.
        get_area: Calculates the area of the bounding box.
        get_bbox: Returns the bounding box coordinates and dimensions.
        as_polygon: Returns the bounding box as a list of points forming a polygon.
        iou: Calculates the Intersection over Union (IoU) with another shape.
        wrap: Creates a new Bbox instance with updated attributes.
    """

    _type = AnnotationType.bbox

    def __init__(self, x, y, w, h, *args, **kwargs):
        """
        Initialize the Bbox with its top-left corner (x, y) and its width and height (w, h).

        Args:
            x (float): The x-coordinate of the top-left corner.
            y (float): The y-coordinate of the top-left corner.
            w (float): The width of the bounding box.
            h (float): The height of the bounding box.
        """
        kwargs.pop("points", None)  # comes from wrap()
        self.__attrs_init__([x, y, x + w, y + h], *args, **kwargs)

    @property
    def x(self):
        """
        Get the x-coordinate of the top-left corner of the bounding box.

        Returns:
            float: The x-coordinate of the bounding box.
        """
        return self.points[0]

    @property
    def y(self):
        """
        Get the y-coordinate of the top-left corner of the bounding box.

        Returns:
            float: The y-coordinate of the bounding box.
        """
        return self.points[1]

    @property
    def w(self):
        """
        Get the width of the bounding box.

        Returns:
            float: The width of the bounding box.
        """
        return self.points[2] - self.points[0]

    @property
    def h(self):
        """
        Get the height of the bounding box.

        Returns:
            float: The height of the bounding box.
        """
        return self.points[3] - self.points[1]

    def get_area(self):
        """
        Calculate the area of the bounding box.

        Returns:
            float: The area of the bounding box.
        """
        return self.w * self.h

    def get_bbox(self):
        """
        Get the bounding box coordinates and dimensions.

        Returns:
            List[float]: The bounding box as [x, y, w, h].
        """
        return [self.x, self.y, self.w, self.h]

    def as_polygon(self) -> List[float]:
        """
        Convert the bounding box into a polygon representation.

        Returns:
            List[float]: The bounding box as a polygon.
        """
        x, y, w, h = self.get_bbox()
        return [x, y, x + w, y, x + w, y + h, x, y + h]

    def iou(self, other: Shape) -> Union[float, Literal[-1]]:
        """
        Calculate the Intersection over Union (IoU) with another shape.

        Args:
            other (Shape): The other shape to compare with.

        Returns:
            Union[float, Literal[-1]]: The IoU value or -1 if not applicable.
        """
        from datumaro.util.annotation_util import bbox_iou

        return bbox_iou(self.get_bbox(), other.get_bbox())

    def wrap(item, **kwargs):
        """
        Create a new Bbox instance with updated attributes.

        Args:
            item (Bbox): The original Bbox instance.
            kwargs: Additional attributes to update.

        Returns:
            Bbox: A new Bbox instance with updated attributes.
        """
        d = {"x": item.x, "y": item.y, "w": item.w, "h": item.h}
        d.update(kwargs)
        return attr.evolve(item, **d)


@attrs(slots=True, init=False, order=False)
class RotatedBbox(Shape):
    """
    RotatedBbox annotation class. This class represents a rotated bounding box defined
    by its center (cx, cy), width (w), height (h), and rotation angle (r).

    Attributes:
        _type (AnnotationType): The type of annotation, set to `AnnotationType.rotated_bbox`.

    Methods:
        __init__: Initializes the RotatedBbox with its center, dimensions, and rotation angle.
        from_rectangle: Creates a RotatedBbox from a list of four corner points.
        cx: Property to get the x-coordinate of the center of the bounding box.
        cy: Property to get the y-coordinate of the center of the bounding box.
        w: Property to get the width of the bounding box.
        h: Property to get the height of the bounding box.
        r: Property to get the rotation angle of the bounding box.
        get_area: Calculates the area of the bounding box.
        get_bbox: Returns the bounding box coordinates and dimensions.
        get_rotated_bbox: Returns the rotated bounding box parameters.
        as_polygon: Converts the rotated bounding box into a list of corner points.
        iou: Calculates the Intersection over Union (IoU) with another shape.
        wrap: Creates a new RotatedBbox instance with updated attributes.
    """

    _type = AnnotationType.rotated_bbox

    def __init__(self, cx, cy, w, h, r, *args, **kwargs):
        """
        Initialize the RotatedBbox with its center (cx, cy), width (w), height (h), and rotation angle (r).

        Args:
            cx (float): The x-coordinate of the center.
            cy (float): The y-coordinate of the center.
            w (float): The width of the bounding box.
            h (float): The height of the bounding box.
            r (float): The rotation angle of the bounding box in degrees.
        """
        kwargs.pop("points", None)  # comes from wrap()
        self.__attrs_init__([cx, cy, w, h, r], *args, **kwargs)

    @classmethod
    def from_rectangle(cls, points: List[Tuple[float, float]], *args, **kwargs):
        """
        Create a RotatedBbox from a list of four corner points.

        Args:
            points (List[Tuple[float, float]]): A list of four points defining the rectangle.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            RotatedBbox: A new RotatedBbox instance.
        """
        assert len(points) == 4, "polygon for a rotated bbox should have only 4 coordinates."

        # Calculate rotation angle
        rot = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])

        # Calculate the center of the bounding box
        cx = (points[0][0] + points[2][0]) / 2
        cy = (points[0][1] + points[2][1]) / 2

        # Calculate the width and height
        width = math.sqrt((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2)
        height = math.sqrt((points[2][0] - points[1][0]) ** 2 + (points[2][1] - points[1][1]) ** 2)

        return cls(cx=cx, cy=cy, w=width, h=height, r=math.degrees(rot), *args, **kwargs)

    @property
    def cx(self):
        """
        Get the x-coordinate of the center of the bounding box.

        Returns:
            float: The x-coordinate of the center.
        """
        return self.points[0]

    @property
    def cy(self):
        """
        Get the y-coordinate of the center of the bounding box.

        Returns:
            float: The y-coordinate of the center.
        """
        return self.points[1]

    @property
    def w(self):
        """
        Get the width of the bounding box.

        Returns:
            float: The width of the bounding box.
        """
        return self.points[2]

    @property
    def h(self):
        """
        Get the height of the bounding box.

        Returns:
            float: The height of the bounding box.
        """
        return self.points[3]

    @property
    def r(self):
        """
        Get the rotation angle of the bounding box in degrees.

        Returns:
            float: The rotation angle of the bounding box.
        """
        return self.points[4]

    def get_area(self):
        """
        Calculate the area of the bounding box.

        Returns:
            float: The area of the bounding box.
        """
        return self.w * self.h

    def get_bbox(self):
        """
        Get the bounding box coordinates and dimensions.

        Returns:
            List[float]: The bounding box as [x, y, w, h].
        """
        polygon = self.as_polygon()
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]

        return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

    def get_rotated_bbox(self):
        """
        Get the rotated bounding box parameters.

        Returns:
            List[float]: The rotated bounding box as [cx, cy, w, h, r].
        """
        return [self.cx, self.cy, self.w, self.h, self.r]

    def as_polygon(self) -> List[Tuple[float, float]]:
        """
        Convert the rotated bounding box into a list of corner points.

        Returns:
            List[Tuple[float, float]]: The bounding box as a list of four corner points.
        """

        def _rotate_point(x, y, angle):
            """
            Rotate a point around another point.

            Args:
                x (float): The x-coordinate of the point.
                y (float): The y-coordinate of the point.
                angle (float): The rotation angle in degrees.

            Returns:
                Tuple[float, float]: The rotated point coordinates.
            """
            angle_rad = math.radians(angle)
            cos_theta = math.cos(angle_rad)
            sin_theta = math.sin(angle_rad)
            nx = cos_theta * x - sin_theta * y
            ny = sin_theta * x + cos_theta * y
            return nx, ny

        # Calculate corner points of the rectangle
        corners = [
            (-self.w / 2, -self.h / 2),
            (self.w / 2, -self.h / 2),
            (self.w / 2, self.h / 2),
            (-self.w / 2, self.h / 2),
        ]

        # Rotate each corner point
        rotated_corners = [_rotate_point(p[0], p[1], self.r) for p in corners]

        # Translate the rotated points to the original position
        return [(p[0] + self.cx, p[1] + self.cy) for p in rotated_corners]

    def iou(self, other: Shape) -> Union[float, Literal[-1]]:
        """
        Calculate the Intersection over Union (IoU) with another shape.

        Args:
            other (Shape): The other shape to compare with.

        Returns:
            Union[float, Literal[-1]]: The IoU value or -1 if not applicable.
        """
        from datumaro.util.annotation_util import bbox_iou

        return bbox_iou(self.get_bbox(), other.get_bbox())

    def wrap(item, **kwargs):
        """
        Create a new RotatedBbox instance with updated attributes.

        Args:
            item (RotatedBbox): The original RotatedBbox instance.
            kwargs: Additional attributes to update.

        Returns:
            RotatedBbox: A new RotatedBbox instance with updated attributes.
        """
        d = {"x": item.x, "y": item.y, "w": item.w, "h": item.h, "r": item.r}
        d.update(kwargs)
        return attr.evolve(item, **d)


@attrs(slots=True, init=False, order=False)
class Cuboid2D(Annotation):
    """
    Cuboid2D annotation class. This class represents a 3D bounding box defined by its point coordinates
    in the following way:
    [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7), (x8, y8)].


      2---3
     /|  /|
    1-+-4 |
    | 5 + 6
    |/  |/
    8---7

    Attributes:
        _type (AnnotationType): The type of annotation, set to `AnnotationType.cuboid_2d`.

    Methods:
        __init__: Initializes the Cuboid2D with its coordinates.
        wrap: Creates a new Cuboid2D instance with updated attributes.
    """

    _type = AnnotationType.cuboid_2d
    points = field(default=None)
    label: Optional[int] = field(
        converter=attr.converters.optional(int), default=None, kw_only=True
    )
    z_order: int = field(default=0, validator=default_if_none(int), kw_only=True)
    y_3d: float = field(default=None, validator=default_if_none(float), kw_only=True)

    def __init__(
        self,
        _points: Iterable[Tuple[float, float]],
        *args,
        **kwargs,
    ):
        kwargs.pop("points", None)  # comes from wrap()
        self.__attrs_init__(points=_points, *args, **kwargs)

    @staticmethod
    def _get_plane_equation(points):
        """Calculates coefficients of the plane equation from three points."""
        x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
        x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
        x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = z2 - z1
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = z3 - z1
        a = b1 * c2 - b2 * c1
        b = a2 * c1 - a1 * c2
        c = a1 * b2 - b1 * a2
        d = -a * x1 - b * y1 - c * z1
        return np.array([a, b, c, d])

    @staticmethod
    def _get_denorm(Tr_velo_to_cam_homo):
        """Calculates the denormalized vector perpendicular to the image plane.
        Args:
            Tr_velo_to_cam_homo (np.ndarray): Homogeneous (4x4) LiDAR-to-camera transformation matrix
        Returns:
            np.ndarray: vector"""
        ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        ground_points_lidar = np.concatenate(
            (ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1
        )
        ground_points_cam = np.matmul(Tr_velo_to_cam_homo, ground_points_lidar.T).T
        denorm = -1 * Cuboid2D._get_plane_equation(ground_points_cam)
        return denorm

    @staticmethod
    def _get_3d_points(dim, location, rotation_y, denorm):
        """Get corner points according to the 3D bounding box parameters.

        Args:
            dim (List[float]): The dimensions of the 3D bounding box as [l, w, h].
            location (List[float]): The location of the 3D bounding box as [x, y, z].
            rotation_y (float): The rotation angle around the y-axis.

        Returns:
            np.ndarray: The corner points of the 3D bounding box.
        """

        c, s = np.cos(rotation_y), np.sin(rotation_y)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        l, w, h = dim[2], dim[1], dim[0]
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(R, corners)

        denorm = denorm[:3]
        denorm_norm = denorm / np.sqrt(denorm[0] ** 2 + denorm[1] ** 2 + denorm[2] ** 2)
        ori_denorm = np.array([0.0, -1.0, 0.0])
        theta = -1 * math.acos(np.dot(denorm_norm, ori_denorm))
        n_vector = np.cross(denorm, ori_denorm)
        n_vector_norm = n_vector / np.sqrt(n_vector[0] ** 2 + n_vector[1] ** 2 + n_vector[2] ** 2)
        rotation_matrix, j = cv2.Rodrigues(theta * n_vector_norm)
        corners_3d = np.dot(rotation_matrix, corners_3d)
        corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
        return corners_3d.transpose(1, 0)

    @staticmethod
    def _project_to_2d(pts_3d, P):
        """Project 3D points to 2D image plane.

        Args:
            pts_3d (np.ndarray): The 3D points to project.
            P (np.ndarray): The projection matrix.

        Returns:
            np.ndarray: The 2D points projected to the image
        """
        # Convert to homogeneous coordinates
        pts_3d = pts_3d.T
        pts_3d_homo = np.vstack((pts_3d, np.ones(pts_3d.shape[1])))
        pts_2d = P @ pts_3d_homo
        pts_2d[0, :] = np.divide(pts_2d[0, :], pts_2d[2, :])
        pts_2d[1, :] = np.divide(pts_2d[1, :], pts_2d[2, :])
        pts_2d = pts_2d[:2, :].T

        return pts_2d

    @classmethod
    def from_3d(
        cls,
        dim: np.ndarray,
        location: np.ndarray,
        rotation_y: float,
        P: np.ndarray,
        Tr_velo_to_cam: np.ndarray,
    ) -> Cuboid2D:
        """Creates an instance of Cuboid2D class from 3D bounding box parameters.

        Args:
            dim (np.ndarray): 3 scalars describing length, width and height of a 3D bounding box
            location (np.ndarray): (x, y, z) coordinates of the middle of the top face.
            rotation_y (np.ndarray): rotation along the Y-axis (from -pi to pi)
            P (np.ndarray): Camera-to-Image transformation matrix (3x4)
            Tr_velo_to_cam (np.ndarray): LiDAR-to-Camera transformation matrix (3x4)

        Returns:
            Cuboid2D: Projection points for the given bounding box
        """
        Tr_velo_to_cam_homo = np.eye(4)
        Tr_velo_to_cam_homo[:3, :4] = Tr_velo_to_cam
        denorm = cls._get_denorm(Tr_velo_to_cam_homo)
        pts_3d = cls._get_3d_points(dim, location, rotation_y, denorm)
        y_3d = np.mean(pts_3d[:4, 1])
        pts_2d = cls._project_to_2d(pts_3d, P)

        return cls(list(map(tuple, pts_2d)), y_3d=y_3d)

    def to_3d(self, P_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Reconstructs 3D object Velodyne coordinates (dimensions, location and rotation along the Y-axis)
        from the given Cuboid2D instance.

        Args:
            P_inv (np.ndarray): Pseudo-inverse of Camera-to-Image projection matrix
        Returns:
            tuple: dimensions, location and rotation along the Y-axis
        """
        recon_3d = []
        for idx, coord_2d in enumerate(self.points):
            coord_2d = np.append(coord_2d, 1)
            coord_3d = P_inv @ coord_2d
            if idx < 4:
                coord_3d = coord_3d * self.y_3d / coord_3d[1]
            else:
                coord_3d = coord_3d * recon_3d[idx - 4][0] / coord_3d[0]
            recon_3d.append(coord_3d[:3])
        recon_3d = np.array(recon_3d)

        x = np.mean(recon_3d[:, 0])
        z = np.mean(recon_3d[:, 2])

        yaws = []
        pairs = [(0, 1), (3, 2), (4, 5), (7, 6)]
        for p in pairs:
            delta_x = recon_3d[p[0]][0] - recon_3d[p[1]][0]
            delta_z = recon_3d[p[0]][2] - recon_3d[p[1]][2]
            yaws.append(np.arctan2(delta_x, delta_z))
        yaw = np.mean(yaws)

        widths = []
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        for p in pairs:
            delta_x = np.sqrt(
                (recon_3d[p[0]][0] - recon_3d[p[1]][0]) ** 2
                + (recon_3d[p[0]][2] - recon_3d[p[1]][2]) ** 2
            )
            widths.append(delta_x)
        w = np.mean(widths)

        lengths = []
        pairs = [(1, 2), (0, 3), (5, 6), (4, 7)]
        for p in pairs:
            delta_z = np.sqrt(
                (recon_3d[p[0]][0] - recon_3d[p[1]][0]) ** 2
                + (recon_3d[p[0]][2] - recon_3d[p[1]][2]) ** 2
            )
            lengths.append(delta_z)
        l = np.mean(lengths)

        heights = []
        pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
        for p in pairs:
            delta_y = np.abs(recon_3d[p[0]][1] - recon_3d[p[1]][1])
            heights.append(delta_y)
        h = np.mean(heights)
        return np.array([h, w, l]), np.array([x, self.y_3d, z]), yaw


@attrs(slots=True, order=False)
class PointsCategories(Categories):
    """
    Describes (key-)point metainfo such as point names and joints.
    """

    @attrs(slots=True, order=False)
    class Category:
        # Names for specific points, e.g. eye, hose, mouth etc.
        # These labels are not required to be in LabelCategories
        labels: List[str] = field(factory=list, validator=default_if_none(list))

        # Pairs of connected point indices
        joints: Set[Tuple[int, int]] = field(factory=set, validator=default_if_none(set))

        # Set of default x, y coordinates of the points
        positions: List[float] = field(default=[])

        @positions.validator
        def positions_validator(
            self, attribute: attr.Attribute[list], positions: list[float] | None
        ) -> None:
            """
            Validate a list of point positions in the format [x1, y1, x2, y2, ..., xn, yn].

            To be used as an attrs validator for the positions field of PointsCategories.Category.

            Args:
                attribute (attr.Attribute[list]): The attribute being validated.
                positions (list[float]): A list of point positions.

            Raises:
                ValueError: If the provided value cannot be converted to a list of floats,
                    or if the number of positions is invalid.
            """
            if positions is None:
                positions = []
            else:
                # convert to a list of floats
                try:
                    positions = list(map(float, positions))
                except (TypeError, ValueError):
                    msg = "Cannot convert positions to list of floats. Check your input data."
                    raise ValueError(msg)
                # check if number of coordinates is even
                if len(positions) % 2 != 0:
                    msg = "positions must have an even number of elements"
                    raise ValueError(msg)
                if len(positions) > 0:
                    # check if the number of positions is equal to the number of labels
                    if len(self.labels) > 0 and len(positions) != len(self.labels) * 2:
                        msg = "The number of positions should be equal to the number of labels"
                        raise ValueError(msg)
                    # normalize the positions
                    positions = normalize_points(positions)
            setattr(self, attribute.name, positions)

    items: Dict[int, Category] = field(factory=dict, validator=default_if_none(dict))

    @classmethod
    def from_iterable(
        cls,
        iterable: Union[
            Iterable[Sequence[int, List[str]]],
            Iterable[Sequence[int, List[str], Set[Tuple[int, int]]]],
            Iterable[Sequence[int, List[str], Set[Tuple[int, int]], List[float]]],
        ],
    ) -> PointsCategories:
        """
        Create PointsCategories from an iterable.

        Args:
            iterable: An Iterable with the following elements:

                - a label id
                - a list of positional arguments for Categories

        Returns:
            PointsCategories: PointsCategories object
        """
        temp_categories = cls()

        for args in iterable:
            temp_categories.add(*args)
        return temp_categories

    def add(
        self,
        label_id: int,
        labels: Optional[Iterable[str]] = None,
        joints: Iterable[Tuple[int, int]] = None,
        positions: Iterable[float] = None,
    ):
        if joints is None:
            joints = []
        joints = set(map(tuple, joints))
        self.items[label_id] = self.Category(labels, joints, positions)

    def __contains__(self, idx: int) -> bool:
        return idx in self.items

    def __getitem__(self, idx: int) -> Category:
        return self.items[idx]

    def __len__(self) -> int:
        return len(self.items)


@attrs(slots=True, order=False)
class Points(Shape):
    """
    Represents an ordered set of points.

    Attributes:
        _type (AnnotationType): The type of annotation, set to `AnnotationType.points`.
        visibility (List[IntEnum]): A list indicating the visibility status of each point.

    Nested Class:
        Visibility (IntEnum): Enum representing the visibility state of points. It has three states:
            - absent: Point is absent (0).
            - hidden: Point is hidden (1).
            - visible: Point is visible (2).

    Methods:
        __attrs_post_init__: Validates that the number of points is even.
        get_area: Returns the area covered by the points, always zero.
        get_bbox: Returns the bounding box containing all visible or hidden points.
    """

    class Visibility(IntEnum):
        """
        Enum representing the visibility state of points.

        Attributes:
            absent (int): Point is absent (0).
            hidden (int): Point is hidden (1).
            visible (int): Point is visible (2).
        """

        absent = 0
        hidden = 1
        visible = 2

    _type = AnnotationType.points

    visibility: List[IntEnum] = field(default=None)

    @visibility.validator
    def _visibility_validator(self, attribute, visibility):
        """
        Validates and initializes the visibility list.

        Args:
            attribute: The attribute being validated.
            visibility (List[IntEnum]): A list indicating the visibility status of each point.

        Raises:
            AssertionError: If the length of the visibility list does not match half the length of the points list.
        """
        if visibility is None:
            visibility = [self.Visibility.visible] * (len(self.points) // 2)
        else:
            for i, v in enumerate(visibility):
                if not isinstance(v, self.Visibility):
                    visibility[i] = self.Visibility(v)
        assert len(visibility) == len(self.points) // 2
        self.visibility = visibility

    def __attrs_post_init__(self):
        """
        Validates that the number of points is even after initialization.

        Raises:
            AssertionError: If the number of points is not even.
        """
        assert len(self.points) % 2 == 0, self.points

    def get_area(self):
        """
        Returns the area covered by the points.

        Returns:
            int: Always returns 0.
        """
        return 0

    def get_bbox(self):
        """
        Returns the bounding box containing all visible or hidden points.

        Returns:
            List[float]: The bounding box as [x0, y0, width, height].
        """
        xs = [
            p
            for p, v in zip(self.points[0::2], self.visibility)
            if v != __class__.Visibility.absent
        ]
        ys = [
            p
            for p, v in zip(self.points[1::2], self.visibility)
            if v != __class__.Visibility.absent
        ]
        x0 = min(xs, default=0)
        x1 = max(xs, default=0)
        y0 = min(ys, default=0)
        y1 = max(ys, default=0)
        return [x0, y0, x1 - x0, y1 - y0]


@attrs(slots=True, order=False)
class Caption(Annotation):
    """
    Represents arbitrary text annotations.
    """

    _type = AnnotationType.caption
    caption: str = field(converter=str)


@attrs(slots=True, order=False, eq=False)
class _ImageAnnotation(Annotation):
    image: Image = field()

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return False
        return np.array_equal(self.image, other.image)


@attrs(slots=True, order=False, eq=False)
class SuperResolutionAnnotation(_ImageAnnotation):
    """
    Represents high resolution images.
    """

    _type = AnnotationType.super_resolution_annotation


@attrs(slots=True, order=False, eq=False)
class DepthAnnotation(_ImageAnnotation):
    """
    Represents depth images.
    """

    _type = AnnotationType.depth_annotation


@attrs(slots=True, init=False, order=False)
class Ellipse(Shape):
    """
    Ellipse represents an ellipse that is encapsulated by a rectangle.

    - x1 and y1 represent the top-left coordinate of the encapsulating rectangle
    - x2 and y2 representing the bottom-right coordinate of the encapsulating rectangle

    Parameters
    ----------

    x1: float
        left x coordinate of encapsulating rectangle
    y1: float
        top y coordinate of encapsulating rectangle
    x2: float
        right x coordinate of encapsulating rectangle
    y2: float
        bottom y coordinate of encapsulating rectangle
    """

    _type = AnnotationType.ellipse

    def __init__(self, x1: float, y1: float, x2: float, y2: float, *args, **kwargs):
        kwargs.pop("points", None)  # comes from wrap()
        self.__attrs_init__([x1, y1, x2, y2], *args, **kwargs)

    @property
    def x1(self):
        return self.points[0]

    @property
    def y1(self):
        return self.points[1]

    @property
    def x2(self):
        return self.points[2]

    @property
    def y2(self):
        return self.points[3]

    @property
    def w(self):
        return self.points[2] - self.points[0]

    @property
    def h(self):
        return self.points[3] - self.points[1]

    @property
    def c_x(self):
        return 0.5 * (self.points[0] + self.points[2])

    @property
    def c_y(self):
        return 0.5 * (self.points[1] + self.points[3])

    def get_area(self):
        return 0.25 * np.pi * self.w * self.h

    def get_bbox(self):
        return [self.x1, self.y1, self.w, self.h]

    def get_points(self, num_points: int = 720) -> List[Tuple[float, float]]:
        """
        Return points as a list of tuples, e.g. [(x0, y0), (x1, y1), ...].

        Parameters
        ----------
        num_points: int
            The number of boundary points of the ellipse.
            By default, one point is created for every 1 degree of interior angle (num_points=360).
        """
        points = self.as_polygon(num_points)

        return [(x, y) for x, y in zip(points[0::2], points[1::2])]

    def as_polygon(self, num_points: int = 720) -> List[float]:
        """
        Return a polygon as a list of tuples, e.g. [x0, y0, x1, y1, ...].

        Parameters
        ----------
        num_points: int
            The number of boundary points of the ellipse.
            By default, one point is created for every 1 degree of interior angle (num_points=360).
        """
        theta = np.linspace(0, 2 * np.pi, num=num_points)

        l1 = 0.5 * self.w
        l2 = 0.5 * self.h
        x_points = self.c_x + l1 * np.cos(theta)
        y_points = self.c_y + l2 * np.sin(theta)

        points = []
        for x, y in zip(x_points, y_points):
            points += [x, y]

        return points

    def iou(self, other: Shape) -> Union[float, Literal[-1]]:
        from datumaro.util.annotation_util import bbox_iou

        return bbox_iou(self.get_bbox(), other.get_bbox())

    def wrap(item: Ellipse, **kwargs) -> Ellipse:
        d = {"x1": item.x1, "y1": item.y1, "x2": item.x2, "y2": item.y2}
        d.update(kwargs)
        return attr.evolve(item, **d)


TableDtype = TypeVar("TableDtype", str, int, float)


@attrs(slots=True, order=False, eq=False)
class TabularCategories(Categories):
    """
    Describes tabular data metainfo such as column names and types.
    """

    @attrs(slots=True, order=False, eq=False)
    class Category:
        name: str = field(converter=str, validator=not_empty)
        dtype: Type[TableDtype] = field()
        labels: Set[Union[str, int]] = field(factory=set, validator=default_if_none(set))

        def __eq__(self, other):
            same_name = self.name == other.name
            same_dtype = self.dtype.__name__ == other.dtype.__name__
            same_labels = self.labels == other.labels
            return same_name and same_dtype and same_labels

        def __repr__(self):
            return f"name: {self.name}, dtype: {self.dtype.__name__}, labels: {self.labels}"

    items: List[Category] = field(factory=list, validator=default_if_none(list))
    _indices_by_name: Dict[str, int] = field(factory=dict, init=False, eq=False)

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[
            Union[Tuple[str, Type[TableDtype]], Tuple[str, Type[TableDtype], Set[str]]]
        ],
    ) -> TabularCategories:
        """
        Creates a TabularCategories from iterable.

        Args:
            iterable: a list of (Category name, type) or (Category name, type, set of labels)

        Returns: a TabularCategories object
        """

        temp_categories = cls()

        for category in iterable:
            temp_categories.add(*category)

        return temp_categories

    def add(
        self,
        name: str,
        dtype: Type[TableDtype],
        labels: Optional[Set[str]] = None,
    ) -> int:
        """
        Add a Tabular Category.

        Args:
            name (str): Column name
            dtype (type): Type of the corresponding column. (str, int, or float)
            labels (optional, set(str)): Label values where the column can have.

        Returns:
            int: A index of added category.
        """
        assert name
        assert name not in self._indices_by_name
        assert dtype

        index = len(self.items)
        self.items.append(self.Category(name, dtype, labels))
        self._indices_by_name[name] = index

        return index

    def find(self, name: str) -> Tuple[Optional[int], Optional[Category]]:
        """
        Find Category information for the given column name.

        Args:
            name (str): Column name

        Returns:
            tuple(int, Category): A index and Category information.
        """
        index = self._indices_by_name.get(name)
        return index, self.items[index] if index is not None else None

    def __getitem__(self, index: int) -> Category:
        return self.items[index]

    def __contains__(self, name: str) -> bool:
        return self.find(name)[1] is not None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Category]:
        return iter(self.items)

    def __eq__(self, other) -> bool:
        if not isinstance(other, __class__):
            return False
        return self.items == other.items


@attrs(slots=True, order=False)
class Tabular(Annotation):
    """
    Represents values of target columns in a tabular dataset.
    """

    _type = AnnotationType.tabular
    values: Dict[str, TableDtype] = field(converter=dict)


class Annotations(List[Annotation]):
    """List of `Annotation` equipped with additional utility functions."""

    def get_semantic_seg_mask(
        self, ignore_index: int = 0, dtype: np.dtype = np.uint8
    ) -> np.ndarray:
        """Extract semantic segmentation mask from a collection of Datumaro `Mask`s.

        Args:
            ignore_index: Scalar value to fill in the zeros in each binary mask
                before merging into a semantic segmentation mask. This value is usually used
                to represent a pixel denoting a not-interested region. Defaults to 0.
            dtype: Data type for the resulting mask. Defaults to np.uint8.

        Returns:
            Semantic segmentation mask generated by merging Datumaro `Mask`s.

        Raises:
            ValueError: If there are no mask annotations or if there is an inconsistency in mask sizes.
        """

        masks = [ann for ann in self if isinstance(ann, Mask)]
        # Mask with a lower z_order value will come first
        masks.sort(key=lambda mask: mask.z_order)

        if not masks:
            msg = "There is no mask annotations."
            raise ValueError(msg)

        # Dispatching for better performance
        # If all masks are `ExtractedMask`, share a same source `index_mask`, and
        # there is no label remapping.
        if (
            all(isinstance(mask, ExtractedMask) for mask in masks)
            # and set(id(mask.index_mask) for mask in masks) == 1
            and all(mask.index_mask == next(iter(masks)).index_mask for mask in masks)
            and all(mask.index == mask.label for mask in masks)
        ):
            index_mask = next(iter(masks)).index_mask
            semantic_seg_mask: np.ndarray = index_mask() if callable(index_mask) else index_mask
            if semantic_seg_mask.dtype != dtype:
                semantic_seg_mask = semantic_seg_mask.astype(dtype)

            labels = np.unique(np.array([mask.label for mask in masks]))
            ignore_index_mask = np.isin(semantic_seg_mask, labels, invert=True)

            return np.where(ignore_index_mask, ignore_index, semantic_seg_mask)

        class_masks = [mask.as_class_mask(ignore_index=ignore_index, dtype=dtype) for mask in masks]

        max_h = max([mask.shape[0] for mask in class_masks])
        max_w = max([mask.shape[1] for mask in class_masks])

        semantic_seg_mask = np.full(shape=(max_h, max_w), fill_value=ignore_index, dtype=dtype)

        for class_mask in class_masks:
            if class_mask.shape != semantic_seg_mask.shape:
                msg = f"There is inconsistency in mask size: {class_mask.shape}!={semantic_seg_mask.shape}."
                raise ValueError(msg, class_mask.shape, semantic_seg_mask.shape)

            ignore_index_mask = class_mask == ignore_index
            semantic_seg_mask = np.where(ignore_index_mask, semantic_seg_mask, class_mask)

        return semantic_seg_mask
