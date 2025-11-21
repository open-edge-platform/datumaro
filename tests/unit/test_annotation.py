# Copyright (C) 2020-2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import List
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from shapely import Polygon as ShapelyPolygon

from datumaro.components.annotation import (
    Annotations,
    Cuboid2D,
    Ellipse,
    ExtractedMask,
    Mask,
    PointsCategories,
    RotatedBbox,
)
from datumaro.util.image import lazy_image
from datumaro.util.points_util import normalize_points


class EllipseTest:
    @pytest.fixture
    def fxt_ellipses(self) -> List[Ellipse]:
        np.random.seed(3003)
        size = 5
        x1x2 = size * np.random.random([10, 2])
        y1y2 = size * np.random.random([10, 2])

        x1x2.sort(axis=1)
        y1y2.sort(axis=1)

        return [Ellipse(x1, y1, x2, y2) for (x1, x2), (y1, y2) in zip(x1x2, y1y2)]

    def test_get_points(self, fxt_ellipses: List[Ellipse]):
        for ellipse in fxt_ellipses:
            analytical_area = ellipse.get_area()
            numerical_area = ShapelyPolygon(ellipse.get_points(num_points=360 * 10)).area
            assert np.abs(analytical_area - numerical_area) < 1e-6


class RotatedBboxTest:
    @pytest.fixture
    def fxt_rot_bbox(self):
        coords = np.random.randint(0, 180, size=(5,), dtype=np.uint8)
        return RotatedBbox(coords[0], coords[1], coords[2], coords[3], coords[4])

    def test_create_polygon(self, fxt_rot_bbox):
        polygon = fxt_rot_bbox.as_polygon()

        expected = RotatedBbox.from_rectangle(polygon)
        assert fxt_rot_bbox == expected


@pytest.fixture
def fxt_index_mask():
    return np.random.randint(0, 10, size=(10, 10))


@pytest.fixture
def fxt_index_mask_file(fxt_index_mask, tmpdir):
    fpath = Path(tmpdir, "mask.png")
    cv2.imwrite(str(fpath), fxt_index_mask)
    yield fpath


class ExtractedMaskTest:
    def test_extracted_mask(self, fxt_index_mask, fxt_index_mask_file):
        index_mask = lazy_image(path=str(fxt_index_mask_file), dtype=np.uint8)
        for index in range(10):
            mask = ExtractedMask(index_mask=index_mask, index=index)
            assert np.allclose(mask.image, (fxt_index_mask == index))


class AnnotationsTest:
    @pytest.mark.parametrize("dtype", [np.uint8, np.int32])
    def test_get_semantic_seg_mask_extracted_mask(self, fxt_index_mask_file, fxt_index_mask, dtype):
        index_mask = lazy_image(path=str(fxt_index_mask_file), dtype=np.uint8)
        annotations = Annotations(ExtractedMask(index_mask=index_mask, index=index, label=index) for index in range(10))
        with patch("datumaro.components.annotation.Mask.as_class_mask") as mock_as_class_mask:
            semantic_seg_mask = annotations.get_semantic_seg_mask(ignore_index=255, dtype=dtype)

        assert np.allclose(semantic_seg_mask, fxt_index_mask)
        # It should directly look up index_mask and there is no calling as_class_mask()
        mock_as_class_mask.assert_not_called()

    @pytest.mark.parametrize("dtype", [np.uint8, np.int32])
    def test_get_semantic_seg_mask_extracted_mask_remapping_label(self, fxt_index_mask_file, fxt_index_mask, dtype):
        index_mask = lazy_image(path=str(fxt_index_mask_file), dtype=np.uint8)
        annotations = Annotations(
            ExtractedMask(
                index_mask=index_mask,
                index=index,
                label=index % 5,  # Remapping label
            )
            for index in range(10)
        )
        semantic_seg_mask = annotations.get_semantic_seg_mask(ignore_index=255, dtype=dtype)

        # fxt_index_mask % 5 is label-remapped ground truth
        assert np.allclose(semantic_seg_mask, fxt_index_mask % 5)

    @pytest.mark.parametrize("dtype", [np.uint8, np.int32])
    def test_get_semantic_seg_mask_binary_mask(self, fxt_index_mask, dtype):
        annotations = Annotations(
            Mask(
                image=fxt_index_mask == index,
                label=index,
            )
            for index in range(10)
        )
        semantic_seg_mask = annotations.get_semantic_seg_mask(ignore_index=255, dtype=dtype)

        assert np.allclose(semantic_seg_mask, fxt_index_mask)


class PointsCategoriesTest:
    @pytest.mark.parametrize(
        "positions, expected",
        [
            (
                [2, 3, 4, 6, 3, 5],
                [0.0, 0.0, 0.666667, 1.0, 0.333333, 0.666667],
            ),  # basic functionality
            ([1, 1, 1, 1, 1, 1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # all points are the same
            ([1, 1, 3, 1, 5, 1], [0.0, 0.0, 0.5, 0.0, 1.0, 0.0]),  # points form horizontal line
            ([1, 1, 1, 3, 1, 5], [0.0, 0.0, 0.0, 0.5, 0.0, 1.0]),  # points form vertical line
            (
                [-2, -3, -4, -6, -3, -5],
                [0.666667, 1.0, 0.0, 0.0, 0.333333, 0.333333],
            ),  # negative coords
            (
                [1000, 2000, 4000, 6000, 3000, 5000],
                [0.0, 0.0, 0.75, 1.0, 0.50, 0.75],
            ),  # large range
            (
                [0.001, 0.002, 0.004, 0.006, 0.003, 0.005],
                [0.0, 0.0, 0.75, 1.0, 0.50, 0.75],
            ),  # small range
            ([2, 3], [0.0, 0.0]),  # single point
        ],
    )
    def test_normalize_positions(self, positions, expected):
        result = normalize_points(positions)
        assert np.allclose(result, expected), f"Expected {expected}, got {result}"

    class PointsPositionsValidatorTest:
        """Tests for the validator of the `positions` field in PointsCategories.Category."""

        @staticmethod
        def test_empty_positions_list():
            """Test that an empty list of positions is allowed."""
            obj = PointsCategories.Category(positions=[])
            assert obj.positions == []  # Should allow empty list

        @staticmethod
        def test_empty_positions_tuple():
            """Test that an empty list of positions is allowed."""
            obj = PointsCategories.Category(positions=())
            assert obj.positions == []  # Should allow empty list

        @staticmethod
        def test_none_positions():
            """Test that None is allowed and converted to an empty list."""
            obj = PointsCategories.Category(positions=None)
            assert obj.positions == []  # Should allow None and convert to empty list

        @staticmethod
        def test_valid_positions():
            """Test that valid positions are allowed."""
            labels = ["p1", "p2"]
            positions = [1.0, 2.0, 3.0, 4.0]
            obj = PointsCategories.Category(labels=labels, positions=positions)
            assert obj

        @staticmethod
        def test_type_not_list():
            """Test that a non-list type for positions raises an error."""
            with pytest.raises(ValueError, match="Cannot convert positions to list of floats"):
                PointsCategories.Category(positions=56)

        @staticmethod
        def test_coordinates_as_string():
            """Test that the coordinates may be represented as a string."""
            labels = ["p1", "p2"]
            positions = ["1", "2", "3", "4"]
            obj = PointsCategories.Category(labels=labels, positions=positions)
            assert obj

        @staticmethod
        def test_non_numeric_elements():
            """Test that passing non-numeric elements raises an error."""
            labels = ["p1", "p2"]
            positions = [1.0, 2.0, 3.0, "not_a_number"]
            with pytest.raises(ValueError, match="Cannot convert positions to list of floats"):
                PointsCategories.Category(labels=labels, positions=positions)

        @staticmethod
        def test_uneven_number_of_elements():
            """Test that an uneven number of elements raises an error."""
            with pytest.raises(ValueError, match="positions must have an even number of elements"):
                PointsCategories.Category(positions=[1.0, 2.0, 3.0])

        @staticmethod
        def test_num_positions_not_same_as_num_labels():
            """Test that the number of positions must match the number of labels."""
            labels = ["p1", "p2", "p3"]  # 3 labels
            positions = [1.0, 2.0, 3.0, 4.0]  # 2 positions
            with pytest.raises(ValueError, match="number of positions should be equal to the number of labels"):
                PointsCategories.Category(labels=labels, positions=positions)

        @staticmethod
        def test_positions_allowed_with_empty_labels():
            """Test that the the number of coordinates check is skipped when labels is empty."""
            labels = []  # no labels
            positions = [1.0, 2.0, 3.0, 4.0]  # 2 positions
            obj = PointsCategories.Category(labels=labels, positions=positions)
            assert obj

        @staticmethod
        def test_labels_allowed_with_empty_positions():
            """Test that the the number of coordinates check is skipped when positions is empty."""
            labels = ["p1", "p2", "p3"]  # 3 labels
            positions = []  # no positions
            obj = PointsCategories.Category(labels=labels, positions=positions)
            assert obj


class Cuboid2DTest:
    @pytest.fixture
    def fxt_cuboid_2d(self):
        return Cuboid2D(
            [
                (684.172, 237.810),
                (750.486, 237.673),
                (803.791, 256.313),
                (714.712, 256.542),
                (684.035, 174.227),
                (750.263, 174.203),
                (803.399, 170.990),
                (714.476, 171.015),
            ],
            y_3d=1.49,
        )

    @pytest.fixture
    def fxt_kitti_data(self):
        dimensions = np.array([1.49, 1.56, 4.34])
        location = np.array([2.51, 1.49, 14.75])
        rot_y = -1.59

        return dimensions, location, rot_y

    @pytest.fixture
    def fxt_P2(self):
        return np.array(
            [
                [7.215377000000e02, 0.000000000000e00, 6.095593000000e02, 4.485728000000e01],
                [0.000000000000e00, 7.215377000000e02, 1.728540000000e02, 2.163791000000e-01],
                [0.000000000000e00, 0.000000000000e00, 1.000000000000e00, 2.745884000000e-03],
            ]
        )

    @pytest.fixture
    def fxt_velo_to_cam(self):
        return np.array(
            [
                [7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03],
                [1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02],
                [9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01],
            ]
        )

    def test_create_from_3d(self, fxt_kitti_data, fxt_cuboid_2d, fxt_P2, fxt_velo_to_cam):
        actual = Cuboid2D.from_3d(
            dim=fxt_kitti_data[0],
            location=fxt_kitti_data[1],
            rotation_y=fxt_kitti_data[2],
            P=fxt_P2,
            Tr_velo_to_cam=fxt_velo_to_cam,
        )

        assert np.allclose(actual.points, fxt_cuboid_2d.points, atol=1e-3)

    def test_to_3d(self, fxt_kitti_data, fxt_cuboid_2d, fxt_P2):
        P_inv = np.linalg.pinv(fxt_P2)
        actual = fxt_cuboid_2d.to_3d(P_inv)
        for act, exp in zip(actual, fxt_kitti_data):
            assert np.allclose(act, exp, atol=2)
