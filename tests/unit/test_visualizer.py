# Copyright (C) 2022-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from unittest import TestCase, mock

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from datumaro.components.annotation import (
    Bbox,
    Caption,
    DepthAnnotation,
    Ellipse,
    HashKey,
    Label,
    Mask,
    Points,
    Polygon,
    PolyLine,
    SuperResolutionAnnotation,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.components.visualizer import Visualizer

from ..requirements import Requirements, mark_requirement


@dataclass
class GridSizeTestCase:
    infer_grid_size: tuple
    expected_grid_size: tuple


class VisualizerTestBase:
    DEFAULT_GRID_SIZE_TEST_CASES = [
        GridSizeTestCase((None, None), (3, 2)),
        GridSizeTestCase((3, None), (3, 2)),
        GridSizeTestCase((None, 2), (3, 2)),
        GridSizeTestCase((3, 2), (3, 2)),
        GridSizeTestCase((None, 5), (1, 5)),
        GridSizeTestCase((5, 1), (5, 1)),
        GridSizeTestCase((5, 1), (5, 1)),
    ]

    def _test_vis_one_sample(self, func_name: str, check_z_order: bool = True):
        visualizer = Visualizer(self.dataset)

        with mock.patch(
            f"datumaro.components.visualizer.Visualizer.{func_name}",
            wraps=getattr(visualizer, func_name),
        ) as mocked:
            # Call by (id, subset) pair
            for item in self.items:
                fig = visualizer.vis_one_sample(item.id, self.subset)

                # Check count
                assert mocked.call_count == len(item.annotations)

                # Check z-order
                if check_z_order:
                    called_z_order = [call[0][0].z_order for call in mocked.call_args_list]
                    assert sorted(called_z_order) == called_z_order

                self.assertIsInstance(fig, Figure)
                mocked.reset_mock()

            # Call by item
            for item in self.items:
                fig = visualizer.vis_one_sample(item)

                # Check count
                assert mocked.call_count == len(item.annotations)

                # Check z-order
                if check_z_order:
                    called_z_order = [call[0][0].z_order for call in mocked.call_args_list]
                    assert sorted(called_z_order) == called_z_order

                self.assertIsInstance(fig, Figure)
                mocked.reset_mock()

        # Unknown id
        with self.assertRaises(Exception):
            visualizer.vis_one_sample("unknown", self.subset)

        # Unknown subset
        for item in self.items:
            with self.assertRaises(Exception):
                visualizer.vis_one_sample(item.id, "unknown")

    def _test_vis_gallery(self, test_cases: GridSizeTestCase):
        visualizer = Visualizer(self.dataset)

        # Too small nrows and ncols
        ids = [item.id for item in self.items]

        cnt = 1
        while cnt * cnt < len(ids):
            cnt += 1

        with self.assertRaises(Exception):
            small_grid_size = (cnt - 1, cnt - 1)
            visualizer.vis_gallery(ids, self.subset, grid_size=small_grid_size)

        # Infer grid size for 5 items
        def _check(infer_grid_size, expected_grid_size):
            expected_nrows, expected_ncols = expected_grid_size

            # Call by ids and subsets
            fig = visualizer.vis_gallery(ids, self.subset, grid_size=infer_grid_size)
            self.assertIsInstance(fig, Figure)
            grid_spec = fig.axes[0].get_gridspec()
            self.assertEqual(grid_spec.nrows, expected_nrows)
            self.assertEqual(grid_spec.ncols, expected_ncols)

            # Call by items
            fig = visualizer.vis_gallery(self.items, grid_size=infer_grid_size)
            self.assertIsInstance(fig, Figure)
            grid_spec = fig.axes[0].get_gridspec()
            self.assertEqual(grid_spec.nrows, expected_nrows)
            self.assertEqual(grid_spec.ncols, expected_ncols)

        for test_case in test_cases:
            _check(test_case.infer_grid_size, test_case.expected_grid_size)


class TestCaseClosePltFigure(TestCase):
    def tearDown(self) -> None:
        plt.close("all")
        return super().tearDown()


class LabelVisualizerTest(TestCaseClosePltFigure, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image.from_numpy(data=np.ones((4, 6, 3))),
                annotations=[
                    Label(
                        label=label_idx,
                        id=img_idx * img_idx + label_idx,
                        group=1,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_one_sample(self):
        self._test_vis_one_sample("_draw_label", check_z_order=False)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        self._test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class PointsVisualizerTest(TestCaseClosePltFigure, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image.from_numpy(data=np.ones((4, 6, 3))),
                annotations=[
                    Points(
                        np.random.randint(0, 6, size=10 * 2),
                        label=label_idx,
                        id=img_idx * img_idx + label_idx,
                        group=1,
                        z_order=label_idx,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_one_sample(self):
        self._test_vis_one_sample("_draw_points", check_z_order=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        self._test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class MaskVisualizerTest(TestCaseClosePltFigure, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image.from_numpy(data=np.ones((4, 6, 3))),
                annotations=[
                    Mask(
                        image=np.random.randint(0, 2, size=(4, 6)),
                        label=label_idx,
                        id=img_idx * img_idx + label_idx,
                        group=1,
                        z_order=label_idx,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_one_sample(self):
        self._test_vis_one_sample("_draw_mask", check_z_order=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        self._test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class PolygonVisualizerTest(TestCaseClosePltFigure, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image.from_numpy(data=np.ones((4, 6, 3))),
                annotations=[
                    Polygon(
                        np.random.randint(0, 6, size=10 * 2),
                        label=label_idx,
                        id=img_idx * img_idx + label_idx,
                        group=1,
                        z_order=label_idx,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_one_sample(self):
        self._test_vis_one_sample("_draw_polygon", check_z_order=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        self._test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class PolyLineVisualizerTest(TestCaseClosePltFigure, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image.from_numpy(data=np.ones((4, 6, 3))),
                annotations=[
                    PolyLine(
                        np.random.randint(0, 6, size=10 * 2),
                        label=label_idx,
                        id=img_idx * img_idx + label_idx,
                        group=1,
                        z_order=label_idx,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_one_sample(self):
        self._test_vis_one_sample("_draw_polygon", check_z_order=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        self._test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class BboxVisualizerTest(TestCaseClosePltFigure, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image.from_numpy(data=np.ones((4, 6, 3))),
                annotations=[
                    Bbox(
                        0,
                        0,
                        1,
                        1,
                        label=label_idx,
                        id=img_idx * img_idx + label_idx,
                        group=1,
                        z_order=label_idx,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_one_sample(self):
        self._test_vis_one_sample("_draw_bbox")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        self._test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class CaptionVisualizerTest(TestCaseClosePltFigure, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image.from_numpy(data=np.ones((4, 6, 3))),
                annotations=[
                    Caption(
                        caption="".join([str(label_idx)] * 10),
                        id=img_idx * img_idx + label_idx,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_one_sample(self):
        self._test_vis_one_sample("_draw_caption", check_z_order=False)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        self._test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class SuperResolutionVisualizerTest(TestCaseClosePltFigure, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image.from_numpy(data=np.ones((4, 6, 3))),
                annotations=[
                    SuperResolutionAnnotation(
                        Image.from_numpy(data=np.ones((8, 12, 3))),
                        id=img_idx,
                        attributes={},
                    )
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_one_sample(self):
        self._test_vis_one_sample("_draw_super_resolution_annotation", check_z_order=False)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        self._test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class DepthVisualizerTest(TestCaseClosePltFigure, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image.from_numpy(data=np.ones((4, 6, 3))),
                annotations=[
                    DepthAnnotation(
                        Image.from_numpy(data=np.ones((4, 6, 3))),
                        id=img_idx,
                        attributes={},
                    )
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_one_sample(self):
        self._test_vis_one_sample("_draw_depth_annotation", check_z_order=False)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        self._test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class EllipseVisualizerTest(TestCaseClosePltFigure, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image.from_numpy(data=np.ones((4, 6, 3))),
                annotations=[
                    Ellipse(
                        0,
                        0,
                        1,
                        1,
                        label=label_idx,
                        id=img_idx * img_idx + label_idx,
                        group=1,
                        z_order=label_idx,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_one_sample(self):
        self._test_vis_one_sample("_draw_ellipse")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        self._test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class UnsupportedTypeTest(LabelVisualizerTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        for item in cls.dataset:
            item.annotations.append(HashKey(np.ones(96).astype(np.uint8)))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_one_sample(self):
        self._test_vis_one_sample("_draw", check_z_order=False)
