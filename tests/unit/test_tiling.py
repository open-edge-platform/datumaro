from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Dict, List
from unittest import TestCase

import numpy as np
import shapely.geometry as sg

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    Cuboid3d,
    DepthAnnotation,
    Label,
    Mask,
    Points,
    Polygon,
    PolyLine,
    SuperResolutionAnnotation,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image
from datumaro.plugins.tiling import Tile
from datumaro.plugins.tiling.util import xywh_to_x1y1x2y2

from tests.utils.test_utils import compare_datasets


class _TestBase:
    n_items = 2
    n_tiles = 2
    height = 16
    width = 8

    default_attrs = {
        "attributes": {"dummy": "dummy"},
        "group": 10,
    }

    def get_id(self, row, col):
        return self.n_tiles * row + col

    @property
    def tile_height(self):
        return self.height // self.n_tiles

    @property
    def tile_width(self):
        return self.width // self.n_tiles

    @property
    def default_label_attrs(self):
        attrs = deepcopy(self.default_attrs)
        attrs["label"] = 10
        return attrs

    @property
    def default_shape_attrs(self):
        attrs = deepcopy(self.default_label_attrs)
        attrs["z_order"] = 10
        return attrs

    @property
    def source_dataset_label(self) -> Dataset:
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[Label(id=idx, **self.default_label_attrs)],
                )
                for idx in range(self.n_items)
            ]
        )

    @property
    def source_dataset_caption(self) -> Dataset:
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[Caption(id=idx, caption=f"caption_{idx}", **self.default_attrs)],
                )
                for idx in range(self.n_items)
            ]
        )

    @property
    def source_dataset_bbox(self) -> Dataset:
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[
                        Bbox(
                            x=self.tile_width * col,
                            y=self.tile_height * row,
                            w=self.tile_width,
                            h=self.tile_height,
                            id=self.get_id(row, col),
                            **self.default_shape_attrs,
                        )
                        for row in range(self.n_tiles)
                        for col in range(self.n_tiles)
                    ],
                )
                for idx in range(self.n_items)
            ]
        )

    @property
    def source_dataset_polygon(self) -> Dataset:
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[
                        Polygon(
                            Bbox(
                                x=self.tile_width * col,
                                y=self.tile_height * row,
                                w=self.tile_width,
                                h=self.tile_height,
                            ).as_polygon(),
                            id=self.get_id(row, col),
                            **self.default_shape_attrs,
                        )
                        for row in range(self.n_tiles)
                        for col in range(self.n_tiles)
                    ],
                )
                for idx in range(self.n_items)
            ]
        )

    @property
    def source_dataset_points(self) -> Dataset:
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[
                        Points(
                            Bbox(
                                x=self.tile_width * col,
                                y=self.tile_height * row,
                                w=self.tile_width,
                                h=self.tile_height,
                            ).as_polygon(),
                            **self.default_shape_attrs,
                        )
                        for row in range(self.n_tiles)
                        for col in range(self.n_tiles)
                    ],
                )
                for idx in range(self.n_items)
            ]
        )

    @property
    def source_dataset_polyline(self) -> Dataset:
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[
                        PolyLine(
                            Bbox(
                                x=self.tile_width * col,
                                y=self.tile_height * row,
                                w=self.tile_width,
                                h=self.tile_height,
                            ).as_polygon(),
                            **self.default_shape_attrs,
                        )
                        for row in range(self.n_tiles)
                        for col in range(self.n_tiles)
                    ],
                )
                for idx in range(self.n_items)
            ]
        )

    @property
    def source_dataset_mask(self) -> Dataset:
        mask_tile = np.zeros([self.tile_height, self.tile_width])
        n_pixels = min(self.tile_height, self.tile_width)
        for i in range(n_pixels):
            mask_tile[i, i] = 1
        mask = np.tile(mask_tile, (self.n_tiles, self.n_tiles))

        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[
                        Mask(
                            mask,
                            **self.default_shape_attrs,
                        )
                    ],
                )
                for idx in range(self.n_items)
            ]
        )

    @property
    def source_dataset_depth_annotation(self) -> Dataset:
        depth_tile = np.zeros([self.tile_height, self.tile_width])
        n_pixels = min(self.tile_height, self.tile_width)
        for i in range(n_pixels):
            depth_tile[i, i] = 1
        depth = np.tile(depth_tile, (self.n_tiles, self.n_tiles))

        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[
                        DepthAnnotation(
                            depth,
                            **self.default_attrs,
                        )
                    ],
                )
                for idx in range(self.n_items)
            ]
        )

    @property
    def source_dataset_cuboid3d(self) -> Dataset:
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[Cuboid3d(position=(0, 0, 0), **self.default_attrs)],
                )
                for idx in range(self.n_items)
            ]
        )

    @property
    def source_dataset_super_resolution_annotation(self):
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[
                        SuperResolutionAnnotation(
                            image=np.zeros((self.height, self.width, 3)), **self.default_attrs
                        )
                    ],
                )
                for idx in range(self.n_items)
            ]
        )


class TileTest(_TestBase, TestCase):
    def _test_common(
        self, transformed: List[DatasetItem], attrs_to_test: Dict, ann_type: AnnotationType
    ):
        expected_size = (self.height // self.n_tiles, self.width // self.n_tiles)

        unique_ids = set()
        ann_counts = defaultdict(int)

        for item in transformed:
            unique_ids.add((item.id, item.subset))

            assert item.media.size == expected_size
            assert item.media.data.shape[:2] == expected_size

            for ann in item.annotations:
                if ann.type == ann_type:
                    ann_counts[item.id] += 1

                for k, v in attrs_to_test.items():
                    assert getattr(ann, k) == v

        assert len(unique_ids) == self.n_tiles * self.n_tiles * self.n_items

        for cnt in ann_counts.values():
            assert cnt == 1

    def test_overlap(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[Label(id=idx, **self.default_label_attrs)],
                )
                for idx in range(self.n_items)
            ]
        )
        p_overlap = 0.5
        transformed = source.transform(
            Tile,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.5, 0.5),
            threshold_drop_ann=0.5,
        )

        multiplier = 1.0 + 0.5 * p_overlap
        expected_size = (
            int(multiplier * self.height // self.n_tiles),
            int(multiplier * self.width // self.n_tiles),
        )

        for item in transformed:
            assert item.media.size == expected_size
            assert item.media.data.shape[:2] == expected_size

    def test_label(self):
        source = self.source_dataset_label

        transformed = source.transform(
            Tile,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0.5,
        )

        self._test_common(transformed, self.default_label_attrs, AnnotationType.label)

        expected_label_counts = {idx: self.n_tiles * self.n_tiles for idx in range(self.n_items)}
        label_counts = defaultdict(lambda: 0)

        for item in transformed:
            for ann in item.annotations:
                if ann.type == AnnotationType.label:
                    label_counts[ann.id] += 1

        assert label_counts == expected_label_counts

    def test_caption(self):
        source = self.source_dataset_caption

        transformed = source.transform(
            Tile,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0.5,
        )

        self._test_common(transformed, self.default_attrs, AnnotationType.caption)

        expected_label_counts = {
            f"caption_{idx}": self.n_tiles * self.n_tiles for idx in range(self.n_items)
        }
        caption_counts = defaultdict(lambda: 0)

        for item in transformed:
            for ann in item.annotations:
                if ann.type == AnnotationType.caption:
                    caption_counts[ann.caption] += 1

        assert caption_counts == expected_label_counts

    def test_bbox(self):
        source = self.source_dataset_bbox

        transformed = source.transform(
            Tile,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0.5,
        )

        self._test_common(transformed, self.default_shape_attrs, AnnotationType.bbox)

        # For each tiled item, we created a Bbox which has the same size as the tiled image.
        for item in transformed:
            for ann in item.annotations:
                assert ann.x == 0
                assert ann.y == 0
                assert ann.w == self.tile_width
                assert ann.h == self.tile_height

    def test_polygon(self):
        source = self.source_dataset_polygon

        transformed = source.transform(
            Tile,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0.5,
        )

        self._test_common(transformed, self.default_shape_attrs, AnnotationType.polygon)

        expected_points = Polygon(
            Bbox(0, 0, self.tile_width, self.tile_height).as_polygon()
        ).get_points()
        expected_polygon = sg.Polygon(expected_points)

        # For each tiled item, we created a Polygon which has the same size as the tiled image.
        for item in transformed:
            for ann in item.annotations:
                actual_polygon = sg.Polygon(ann.get_points())

                inter_area = actual_polygon.intersection(expected_polygon).area
                union_area = actual_polygon.area + expected_polygon.area - inter_area

                iou = inter_area / union_area
                assert iou == 1.0

    def test_points(self):
        source = self.source_dataset_points

        transformed = source.transform(
            Tile,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0.5,
        )

        self._test_common(transformed, self.default_shape_attrs, AnnotationType.points)

        expected_points = Polygon(
            Bbox(0, 0, self.tile_width, self.tile_height).as_polygon()
        ).get_points()

        # For each tiled item, we created a Points covered by the tiled image.
        for item in transformed:
            for ann in item.annotations:
                for a_p, e_p in zip(ann.get_points(), expected_points):
                    assert a_p == e_p

    def test_polyline(self):
        source = self.source_dataset_polyline

        transformed = source.transform(
            Tile,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0.5,
        )

        self._test_common(transformed, self.default_shape_attrs, AnnotationType.polyline)

        expected_points = Polygon(
            Bbox(0, 0, self.tile_width, self.tile_height).as_polygon()
        ).get_points()

        # For each tiled item, we created a Points covered by the tiled image.
        for item in transformed:
            for ann in item.annotations:
                for a_p, e_p in zip(ann.get_points(), expected_points):
                    assert a_p == e_p

    def test_mask(self):
        n_pixels = min(self.tile_height, self.tile_width)
        source = self.source_dataset_mask

        transformed = source.transform(
            Tile,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0.5,
        )

        self._test_common(transformed, self.default_shape_attrs, AnnotationType.mask)

        # For each tiled item, we created a Mask which has n_pixels number of 1s.
        for item in transformed:
            for ann in item.annotations:
                assert ann.image.astype(np.int32).sum() == n_pixels

    def test_depth_annotation(self):
        n_pixels = min(self.tile_height, self.tile_width)
        source = self.source_dataset_depth_annotation

        transformed = source.transform(
            Tile,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0.5,
        )

        self._test_common(transformed, self.default_attrs, AnnotationType.depth_annotation)

        # For each tiled item, we created a Depth map which has n_pixels number of 1 distances.
        for item in transformed:
            for ann in item.annotations:
                assert ann.image.astype(np.int32).sum() == n_pixels

    def test_cuboid3d_annotation(self):
        source = self.source_dataset_cuboid3d

        transformed = source.transform(
            Tile,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0.5,
        )

        # Do not support this annotation type.
        with self.assertRaises(DatumaroError):
            self._test_common(transformed, self.default_attrs, AnnotationType.cuboid_3d)

    def test_super_resolution_annotation(self):
        source = self.source_dataset_super_resolution_annotation

        transformed = source.transform(
            Tile,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0.5,
        )

        # Do not support this annotation type.
        with self.assertRaises(DatumaroError):
            self._test_common(
                transformed, self.default_attrs, AnnotationType.super_resolution_annotation
            )

    def _create_sticking_out_box(self, row: int, col: int) -> Bbox:
        return Bbox(
            x=self.tile_width * (col + 0.5),
            y=self.tile_height * (row + 0.5),
            w=self.tile_width,
            h=self.tile_height,
        )

    def test_drop_annotations(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[
                        Points(
                            self._create_sticking_out_box(row, col).as_polygon(),
                            **self.default_shape_attrs,
                        )
                        for row in range(self.n_tiles)
                        for col in range(self.n_tiles)
                    ]
                    + [
                        PolyLine(
                            self._create_sticking_out_box(row, col).as_polygon(),
                            **self.default_shape_attrs,
                        )
                        for row in range(self.n_tiles)
                        for col in range(self.n_tiles)
                    ],
                )
                for idx in range(self.n_items)
            ]
        )

        transformed = source.transform(
            Tile,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0.5,
        )

        # TileTransform drops all Points and PolyLines
        # because parts of them are maded to be sticking out of the tiled image.
        for item in transformed:
            assert len(item.annotations) == 0

    def test_crop_annotations(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(
                    id=idx,
                    media=Image.from_numpy(data=np.zeros((self.height, self.width, 3))),
                    annotations=[
                        self._create_sticking_out_box(row, col)
                        for row in range(self.n_tiles)
                        for col in range(self.n_tiles)
                    ]
                    + [
                        Polygon(self._create_sticking_out_box(row, col).as_polygon())
                        for row in range(self.n_tiles)
                        for col in range(self.n_tiles)
                    ],
                )
                for idx in range(self.n_items)
            ]
        )

        # Set threshold=0.5. All annotations must be dropped.
        dropped = Tile(
            source,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0.5,
        )

        for item in dropped:
            assert len(item.annotations) == 0

        # Set threshold=0. All annotations must be accepted.
        accepted = Tile(
            source,
            grid_size=(self.n_tiles, self.n_tiles),
            overlap=(0.0, 0.0),
            threshold_drop_ann=0,
        )

        tile_roi_polygon = sg.Polygon(
            Polygon(Bbox(0, 0, self.tile_width, self.tile_height).as_polygon()).get_points()
        )

        for item in accepted:
            assert len(item.annotations) >= 2

            for ann in item.annotations:
                if ann.type == AnnotationType.bbox:
                    actual_polygon = sg.box(*xywh_to_x1y1x2y2(*ann.get_bbox()))
                elif ann.type == AnnotationType.polygon:
                    actual_polygon = sg.Polygon(ann.get_points())
                else:
                    raise RuntimeError()

                # There should be no protrusion.
                assert tile_roi_polygon.covers(actual_polygon)


class MergeTileTest(_TestBase, TestCase):
    def test_tile_and_merge_tile(self):
        for ann_type in [
            "label",
            "caption",
            "bbox",
            "polygon",
            "points",
            "polyline",
            "mask",
            "depth_annotation",
        ]:
            source = getattr(self, f"source_dataset_{ann_type}")
            transformed = (
                getattr(self, f"source_dataset_{ann_type}")
                .transform(
                    "tile",
                    grid_size=(self.n_tiles, self.n_tiles),
                    overlap=(0.0, 0.0),
                    threshold_drop_ann=0.5,
                )
                .transform("merge_tile")
            )
            compare_datasets(self, transformed, source, require_media=True)
