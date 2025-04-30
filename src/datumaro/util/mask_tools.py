# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT
import logging as log
from functools import partial
from itertools import chain
from typing import Dict, Optional, Tuple

import numpy as np
from pycocotools import mask as pycocotools_mask

from datumaro._capi import encode
from datumaro.util.image import lazy_image, load_image


def generate_colormap(length=256, *, include_background=True):
    """
    Generates colors using PASCAL VOC algorithm.

    If include_background is True, the result will include the item
    "0: (0, 0, 0)", which is typically used as a background color.
    Otherwise, indices will start from 0, but (0, 0, 0) is not included.

    Returns index -> (R, G, B) mapping.
    """

    def get_bit(number, index):
        return (number >> index) & 1

    colormap = np.zeros((length, 3), dtype=int)

    offset = int(not include_background)
    indices = np.arange(offset, length + offset, dtype=int)

    for j in range(7, -1, -1):
        for c in range(3):
            colormap[:, c] |= get_bit(indices, c) << j
        indices >>= 3

    return {id: tuple(color) for id, color in enumerate(colormap)}


def invert_colormap(colormap):
    return {tuple(a): index for index, a in colormap.items()}


def check_is_mask(mask):
    assert len(mask.shape) in {2, 3}
    if len(mask.shape) == 3:
        assert mask.shape[2] == 1


_default_colormap = generate_colormap()
_default_unpaint_colormap = invert_colormap(_default_colormap)


def unpaint_mask(painted_mask, inverse_colormap=None, default_id=None):
    """
    Convert color mask to index mask

    mask: HWC BGR [0; 255]

    colormap: (R, G, B) -> index
    """
    assert len(painted_mask.shape) == 3
    if inverse_colormap is None:
        inverse_colormap = _default_unpaint_colormap

    if callable(inverse_colormap):
        map_fn = lambda a: inverse_colormap((a >> 16) & 255, (a >> 8) & 255, a & 255)
    else:
        map_fn = lambda a: inverse_colormap.get(((a >> 16) & 255, (a >> 8) & 255, a & 255), None)

    painted_mask = painted_mask.astype(int)
    painted_mask = (
        painted_mask[:, :, 0] + (painted_mask[:, :, 1] << 8) + (painted_mask[:, :, 2] << 16)
    )
    uvals, unpainted_mask = np.unique(painted_mask, return_inverse=True)
    palette = []
    for v in uvals:
        class_id = map_fn(v)
        if class_id is None and default_id is None:
            raise KeyError(f"Undeclared color {((v >> 16) & 255, (v >> 8) & 255, v & 255)}")
        elif class_id is None and default_id is not None:
            class_id = default_id
        palette.append(class_id)
    palette = np.array(palette, dtype=np.min_scalar_type(len(uvals)))
    unpainted_mask = palette[unpainted_mask].reshape(painted_mask.shape[:2])

    return unpainted_mask


def paint_mask(mask, colormap=None):
    """
    Applies colormap to index mask

    mask: HW(C) [0; max_index] mask

    colormap: index -> (R, G, B)
    """
    check_is_mask(mask)

    if colormap is None:
        colormap = _default_colormap
    if callable(colormap):
        map_fn = colormap
    else:
        map_fn = lambda c: colormap.get(c, (-1, -1, -1))
    palette = np.mod([map_fn(c)[::-1] for c in range(256)], 256).astype(np.uint8)

    mask = mask.astype(np.uint8)
    painted_mask = palette[mask].reshape((*mask.shape[:2], 3))
    return painted_mask


def remap_mask(mask, map_fn):
    """
    Changes mask elements from one colormap to another

    # mask: HW(C) [0; max_index] mask
    """
    check_is_mask(mask)

    return np.mod([map_fn(c) for c in range(256)], 256).astype(np.uint8)[mask]


def make_index_mask(
    binary_mask: np.ndarray,
    index: int,
    ignore_index: int = 0,
    dtype: Optional[np.dtype] = None,
):
    """Create an index mask from a binary mask by filling a given index value.

    Args:
        binary_mask: Binary mask to create an index mask.
        index: Scalar value to fill the ones in the binary mask.
        ignore_index: Scalar value to fill in the zeros in the binary mask.
            Defaults to 0.
        dtype: Data type for the resulting mask. If not specified,
                it will be inferred from the provided `index` to hold its value.
                For example, if `index=255`, the inferred dtype will be `np.uint8`.
                Defaults to None.

    Returns:
        np.ndarray: Index mask created from the binary mask.

    Raises:
        ValueError: If dtype is not specified and incompatible scalar types are used for index
            and ignore_index.

    Examples:
        >>> binary_mask = np.eye(2, dtype=np.bool_)
        >>> index_mask = make_index_mask(binary_mask, index=10, ignore_index=255, dtype=np.uint8)
        >>> print(index_mask)
        array([[ 10, 255],
               [255,  10]], dtype=uint8)
    """
    if dtype is None:
        dtype = np.min_scalar_type(index)
        if dtype != np.min_scalar_type(ignore_index):
            msg = (
                "Given dtype is None, "
                "but inferred dtypes from the given index and ignore_index are different each other. "
                "Please mannually set dtype"
            )
            raise ValueError(msg, index, ignore_index)

    flipped_zero_np_scalar = ~np.full(tuple(), fill_value=0, dtype=dtype)

    # NOTE: This dispatching rule is required for a performance boost
    if ignore_index == flipped_zero_np_scalar:
        flipped_index = ~np.full(tuple(), fill_value=index, dtype=dtype)
        return ~(binary_mask * flipped_index)
    elif index < ignore_index:
        diff = ignore_index - index
        mask = ~binary_mask * np.full(tuple(), fill_value=diff, dtype=dtype)
        mask += index
        return mask
    elif index > ignore_index:
        diff = index - ignore_index
        mask = binary_mask * np.full(tuple(), fill_value=diff, dtype=dtype)
        mask += ignore_index
        return mask

    # index == ignore_index
    msg = (
        "index == ignore_index. "
        f"It will create an index mask filling with a single value, index={index}"
    )
    log.warning(msg)
    return np.full_like(binary_mask, fill_value=index, dtype=dtype)


def make_binary_mask(mask):
    if mask.dtype.kind == "b":
        return mask
    return mask.astype(bool)


def bgr2index(img):
    if img.dtype.kind not in {"b", "i", "u"} or img.dtype.itemsize < 4:
        img = img.astype(np.uint32)
    return (img[..., 0] << 16) + (img[..., 1] << 8) + img[..., 2]


def index2bgr(id_map):
    return np.dstack((id_map >> 16, id_map >> 8, id_map)).astype(np.uint8)


def load_mask(path, inverse_colormap=None, default_id=None):
    mask = load_image(path, dtype=np.uint8)
    if inverse_colormap is not None:
        if len(mask.shape) == 3 and mask.shape[2] != 1:
            mask = unpaint_mask(mask, inverse_colormap, default_id)
    return mask


def lazy_mask(path, inverse_colormap=None):
    return lazy_image(path, partial(load_mask, inverse_colormap=inverse_colormap))


def mask_to_rle(binary_mask):
    return encode(binary_mask)


def mask_to_rle_py(binary_mask):
    # walk in row-major order as COCO format specifies
    bounded = binary_mask.ravel(order="F")

    # add borders to sequence
    # find boundary positions for sequences and compute their lengths
    difs = np.diff(bounded, prepend=[1 - bounded[0]], append=[1 - bounded[-1]])
    (counts,) = np.where(difs != 0)

    # start RLE encoding from 0 as COCO format specifies
    if bounded[0] != 0:
        counts = np.diff(counts, prepend=[0])
    else:
        counts = np.diff(counts)

    return {"counts": counts, "size": list(binary_mask.shape)}


def extract_contours(mask):
    """
    Convert an instance mask to polygons

    Args:
        mask: a 2d binary mask
        tolerance: maximum distance from original points of
            a polygon to the approximated ones
        area_threshold: minimal area of generated polygons

    Returns:
        A list of polygons like [[x1,y1, x2,y2 ...], [...]]
    """
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_KCOS
    )

    results = []
    for contour in contours:
        if len(contour) <= 2:
            continue

        contour = contour.reshape((-1, 2))

        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))  # make polygon closed
        results.append(contour.flatten().clip(0))  # [x0, y0, ...]

    return results


def mask_to_polygons(mask, area_threshold=1):
    """
    Convert an instance mask to polygons

    Args:
        mask: a 2d binary mask
        tolerance: maximum distance from original points of
            a polygon to the approximated ones
        area_threshold: minimal area of generated polygons

    Returns:
        A list of polygons like [[x1,y1, x2,y2 ...], [...]]
    """

    contours = extract_contours(mask)

    polygons = []
    for contour in contours:
        # Check if the polygon is big enough
        rle = pycocotools_mask.frPyObjects([contour], mask.shape[0], mask.shape[1])
        area = sum(pycocotools_mask.area(rle))
        if area_threshold <= area:
            polygons.append(contour)
    return polygons


def mask_to_bboxes(mask):
    """
    Convert an instance mask to bboxes

    Args:
        mask: a 2d binary mask

    Returns:
        A list of bboxes like [[x1,x2,y1,y2], [...]]
    """

    contours = extract_contours(mask)

    bboxes = []
    for contour in contours:
        x1, x2 = min(contour[0::2]), max(contour[0::2])
        y1, y2 = min(contour[1::2]), max(contour[1::2])

        bboxes.append([x1, x2, y1, y2])

    return bboxes


def crop_covered_segments(
    segments,
    width,
    height,
    iou_threshold=0.0,
    ratio_tolerance=0.001,
    area_threshold=1,
    return_masks=False,
):
    """
    Find all segments occluded by others and crop them to the visible part only.
    Input segments are expected to be sorted from background to foreground.

    Args:
        segments: 1d list of segment RLEs (in COCO format)
        width: width of the image
        height: height of the image
        iou_threshold: IoU threshold for objects to be counted as intersected
            By default is set to 0 to process any intersected objects
        ratio_tolerance: an IoU "handicap" value for a situation
            when an object is (almost) fully covered by another one and we
            don't want make a "hole" in the background object
        area_threshold: minimal area of included segments

    Returns:
        A list of input segments' parts (in the same order as input):

        .. code-block::

            [
                [[x1,y1, x2,y2 ...], ...], # input segment #0 parts
                mask1, # input segment #1 mask (if source segment is mask)
                [], # when source segment is too small
                ...
            ]
    """
    segments = [[s] for s in segments]
    input_rles = [pycocotools_mask.frPyObjects(s, height, width) for s in segments]

    for i, rle_bottom in enumerate(input_rles):
        area_bottom = sum(pycocotools_mask.area(rle_bottom))
        if area_bottom < area_threshold:
            segments[i] = [] if not return_masks else None
            continue

        rles_top = []
        for j in range(i + 1, len(input_rles)):
            rle_top = input_rles[j]
            iou = sum(pycocotools_mask.iou(rle_bottom, rle_top, [0]))[0]

            if iou <= iou_threshold:
                continue

            area_top = sum(pycocotools_mask.area(rle_top))
            area_ratio = area_top / area_bottom

            # If a segment is fully inside another one, skip this segment
            if abs(area_ratio - iou) < ratio_tolerance:
                continue

            # Check if the bottom segment is fully covered by the top one.
            # There is a mistake in the annotation, keep the background one
            if abs(1 / area_ratio - iou) < ratio_tolerance:
                rles_top = []
                break

            rles_top += rle_top

        if not rles_top and not isinstance(segments[i][0], dict) and not return_masks:
            continue

        rle_bottom = rle_bottom[0]
        bottom_mask = pycocotools_mask.decode(rle_bottom).astype(np.uint8)

        if rles_top:
            rle_top = pycocotools_mask.merge(rles_top)
            top_mask = pycocotools_mask.decode(rle_top).astype(np.uint8)

            bottom_mask -= top_mask
            bottom_mask[bottom_mask != 1] = 0

        if not return_masks and not isinstance(segments[i][0], dict):
            segments[i] = mask_to_polygons(bottom_mask, area_threshold=area_threshold)
        else:
            segments[i] = bottom_mask

    return segments


def rles_to_mask(rles, width, height):
    rles = pycocotools_mask.frPyObjects(rles, height, width)
    rles = pycocotools_mask.merge(rles)
    mask = pycocotools_mask.decode(rles)
    return mask


def rle_to_mask(rle_uncompressed: Dict[str, np.ndarray]) -> np.ndarray:
    """Decode the uncompressed RLE string to the binary mask (2D np.ndarray)

    The uncompressed RLE string can be obtained by
    the datumaro.util.mask_tools.mask_to_rle() function
    """
    resulting_mask = pycocotools_mask.frPyObjects(rle_uncompressed, *rle_uncompressed["size"])
    resulting_mask = pycocotools_mask.decode(resulting_mask)
    return resulting_mask


def find_mask_bbox(mask) -> Tuple[int, int, int, int]:
    cols = np.any(mask, axis=0)
    rows = np.any(mask, axis=1)
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0, y1 = np.where(rows)[0][[0, -1]]
    return (x0, y0, x1 - x0, y1 - y0)


def merge_masks(masks, start=None):
    """
    Merges masks into one, mask order is responsible for z order.
    To avoid memory explosion on mask materialization, consider passing
    a generator.

    Inputs: a sequence of index masks or (binary mask, index) pairs

    Outputs: an index mask
    """
    if start is not None:
        masks = chain([start], masks)

    it = iter(masks)

    try:
        merged_mask = next(it)
        if isinstance(merged_mask, tuple) and len(merged_mask) == 2:
            merged_mask = merged_mask[0] * merged_mask[1]
    except StopIteration:
        return None

    for m in it:
        if isinstance(m, tuple) and len(m) == 2:
            merged_mask = np.where(m[0], m[1], merged_mask)
        else:
            merged_mask = np.where(m, m, merged_mask)

    return merged_mask
