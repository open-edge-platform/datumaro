import os
import os.path as osp
from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Label,
    LabelCategories,
    Mask,
    Points,
    Polygon,
    PolyLine,
    RotatedBbox,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.cvat.base import CvatImporter
from datumaro.plugins.data_formats.cvat.exporter import CvatExporter
from datumaro.plugins.transforms import ProjectLabels
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, check_save_and_load, compare_datasets

DUMMY_IMAGE_DATASET_DIRS = [
    get_test_asset_path("cvat_dataset", "for_images", export_type) for export_type in ["export_project", "export_task"]
]

DUMMY_VIDEO_DATASET_DIR = get_test_asset_path("cvat_dataset", "for_video")

DUMMY_VIDEO_DATASET_FILE = get_test_asset_path("cvat_dataset", "test.mp4")

EXPECTED_IMAGE_DATASETS = [
    # export_project
    Dataset.from_iterable(
        [
            DatasetItem(
                id="img0",
                subset=subset,
                media=Image.from_numpy(data=np.ones((8, 8, 3))),
                annotations=[
                    Bbox(
                        0,
                        2,
                        4,
                        2,
                        label=0,
                        z_order=1,
                        attributes={
                            "occluded": True,
                            "a1": True,
                            "a2": "v3",
                            "a3": "0003",
                            "a4": 2.4,
                        },
                    ),
                    PolyLine([1, 2, 3, 4, 5, 6, 7, 8], label=1, attributes={"occluded": False}),
                ],
                attributes={"frame": 0},
            )
            for subset in ["Test", "Train", "Validation"]
        ]
        + [
            DatasetItem(
                id="img1",
                subset=subset,
                media=Image.from_numpy(data=np.ones((10, 10, 3))),
                annotations=[
                    Polygon([1, 2, 3, 4, 6, 5], label=0, z_order=1, attributes={"occluded": False}),
                    Points([1, 2, 3, 4, 5, 6], label=1, z_order=2, attributes={"occluded": False}),
                ],
                attributes={"frame": 1},
            )
            for subset in ["Test", "Train", "Validation"]
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                [
                    ["label1", "", {"a1", "a2", "a3", "a4"}],
                    ["label2"],
                ]
            )
        },
    ),
    # export_task
    Dataset.from_iterable(
        [
            DatasetItem(
                id="img0",
                subset="train",
                media=Image.from_numpy(data=np.ones((8, 8, 3))),
                annotations=[
                    Bbox(
                        0,
                        2,
                        4,
                        2,
                        label=0,
                        z_order=1,
                        attributes={
                            "occluded": True,
                            "a1": True,
                            "a2": "v3",
                            "a3": "0003",
                            "a4": 2.4,
                        },
                    ),
                    PolyLine([1, 2, 3, 4, 5, 6, 7, 8], attributes={"occluded": False}),
                ],
                attributes={"frame": 0},
            ),
            DatasetItem(
                id="img1",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 10, 3))),
                annotations=[
                    Polygon([1, 2, 3, 4, 6, 5], z_order=1, attributes={"occluded": False}),
                    Points([1, 2, 3, 4, 5, 6], label=1, z_order=2, attributes={"occluded": False}),
                ],
                attributes={"frame": 1},
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                [
                    ["label1", "", {"a1", "a2", "a3", "a4"}],
                    ["label2"],
                ]
            )
        },
    ),
]


class CvatImporterTest(TestCase):
    def test_can_detect_image(self):
        for dataset_dir in DUMMY_IMAGE_DATASET_DIRS:
            detected_formats = Environment().detect_dataset(dataset_dir)
            self.assertEqual([CvatImporter.NAME], detected_formats)

    def test_can_detect_video(self):
        detected_formats = Environment().detect_dataset(DUMMY_VIDEO_DATASET_DIR)
        self.assertEqual([CvatImporter.NAME], detected_formats)

    def test_can_load_image(self):
        for expected_dataset, dataset_dir in zip(EXPECTED_IMAGE_DATASETS, DUMMY_IMAGE_DATASET_DIRS):
            parsed_dataset = Dataset.import_from(dataset_dir, "cvat")

            compare_datasets(self, expected_dataset, parsed_dataset, require_media=True)

    def test_can_load_video(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="frame_000010",
                    subset="annotations",
                    media=Image.from_numpy(data=255 * np.ones((20, 25, 3))),
                    annotations=[
                        Bbox(
                            3,
                            4,
                            7,
                            1,
                            label=2,
                            id=0,
                            attributes={
                                "occluded": True,
                                "outside": False,
                                "keyframe": True,
                                "track_id": 0,
                            },
                        ),
                        Points(
                            [21.95, 8.00, 2.55, 15.09, 2.23, 3.16],
                            label=0,
                            id=1,
                            attributes={
                                "occluded": False,
                                "outside": False,
                                "keyframe": True,
                                "track_id": 1,
                                "hgl": "hgkf",
                            },
                        ),
                    ],
                    attributes={"frame": 10},
                ),
                DatasetItem(
                    id="frame_000013",
                    subset="annotations",
                    media=Image.from_numpy(data=255 * np.ones((20, 25, 3))),
                    annotations=[
                        Bbox(
                            7,
                            6,
                            7,
                            2,
                            label=2,
                            id=0,
                            attributes={
                                "occluded": False,
                                "outside": True,
                                "keyframe": True,
                                "track_id": 0,
                            },
                        ),
                        Points(
                            [21.95, 8.00, 9.55, 15.09, 5.23, 1.16],
                            label=0,
                            id=1,
                            attributes={
                                "occluded": False,
                                "outside": True,
                                "keyframe": True,
                                "track_id": 1,
                                "hgl": "jk",
                            },
                        ),
                        PolyLine(
                            [7.85, 13.88, 3.50, 6.67, 15.90, 2.00, 13.31, 7.21],
                            label=2,
                            id=2,
                            attributes={
                                "occluded": False,
                                "outside": False,
                                "keyframe": True,
                                "track_id": 2,
                            },
                        ),
                    ],
                    attributes={"frame": 13},
                ),
                DatasetItem(
                    id="frame_000016",
                    subset="annotations",
                    media=Image.from_numpy(data=255 * np.ones((20, 25, 3))),
                    annotations=[
                        Bbox(
                            8,
                            7,
                            6,
                            10,
                            label=2,
                            id=0,
                            attributes={
                                "occluded": False,
                                "outside": True,
                                "keyframe": True,
                                "track_id": 0,
                            },
                        ),
                        PolyLine(
                            [7.85, 13.88, 3.50, 6.67, 15.90, 2.00, 13.31, 7.21],
                            label=2,
                            id=2,
                            attributes={
                                "occluded": False,
                                "outside": True,
                                "keyframe": True,
                                "track_id": 2,
                            },
                        ),
                    ],
                    attributes={"frame": 16},
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable([["klhg", "", {"hgl"}], ["z U k"], ["II"]]),
            },
        )

        parsed_dataset = Dataset.import_from(DUMMY_VIDEO_DATASET_DIR, "cvat")

        compare_datasets(self, expected_dataset, parsed_dataset, require_media=True)


class CvatExporterTest(TestCase):
    def _test_save_and_load(
        self, source_dataset, converter, test_dir, target_dataset=None, importer_args=None, **kwargs
    ):
        return check_save_and_load(
            self,
            source_dataset,
            converter,
            test_dir,
            importer="cvat",
            target_dataset=target_dataset,
            importer_args=importer_args,
            **kwargs,
        )

    def test_can_save_and_load(self):
        src_label_cat = LabelCategories(attributes={"occluded", "common"})
        for i in range(10):
            src_label_cat.add(str(i))
        src_label_cat.items[2].attributes.update(["a1", "a2", "empty"])

        mask = np.random.randint(0, 2, size=(5, 10), dtype=np.uint8)
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    subset="s1",
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            label=1,
                            group=4,
                            attributes={"occluded": True, "common": "t"},
                        ),
                        Points(
                            [1, 1, 3, 2, 2, 3],
                            label=2,
                            attributes={"a1": "x", "a2": 42, "empty": "", "unknown": "bar"},
                        ),
                        Label(1),
                        Label(2, attributes={"a1": "y", "a2": 44}),
                    ],
                ),
                DatasetItem(
                    id=1,
                    subset="s1",
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        PolyLine([0, 0, 4, 0, 4, 4], label=3, id=4, group=4),
                        Bbox(5, 0, 1, 9, label=3, id=4, group=4),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="s2",
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            z_order=1,
                            label=3,
                            group=4,
                            attributes={"occluded": False},
                        ),
                        PolyLine([5, 0, 9, 0, 5, 5]),  # will be skipped as no label
                    ],
                ),
                DatasetItem(
                    id=3,
                    subset="s3",
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[Mask(image=mask, z_order=1, label=3, group=4)],
                ),
                DatasetItem(
                    id=4,
                    subset="s4",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    annotations=[
                        RotatedBbox(1, 10, 5, 5, 45, label=0, attributes={"occluded": False}),
                    ],
                ),
            ],
            categories={AnnotationType.label: src_label_cat},
        )

        target_label_cat = LabelCategories(attributes={"occluded"})  # unable to represent a common attribute
        for i in range(10):
            target_label_cat.add(str(i), attributes={"common"})
        target_label_cat.items[2].attributes.update(["a1", "a2", "empty", "common"])
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    subset="s1",
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            label=1,
                            group=4,
                            attributes={"occluded": True, "common": "t"},
                        ),
                        Points(
                            [1, 1, 3, 2, 2, 3],
                            label=2,
                            attributes={"occluded": False, "empty": "", "a1": "x", "a2": "42"},
                        ),
                        Label(1),
                        Label(2, attributes={"a1": "y", "a2": "44"}),
                    ],
                    attributes={"frame": 0},
                ),
                DatasetItem(
                    id=1,
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    subset="s1",
                    annotations=[
                        PolyLine([0, 0, 4, 0, 4, 4], label=3, group=4, attributes={"occluded": False}),
                        Bbox(5, 0, 1, 9, label=3, group=4, attributes={"occluded": False}),
                    ],
                    attributes={"frame": 1},
                ),
                DatasetItem(
                    id=2,
                    subset="s2",
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Polygon(
                            [0, 0, 4, 0, 4, 4],
                            z_order=1,
                            label=3,
                            group=4,
                            attributes={"occluded": False},
                        ),
                    ],
                    attributes={"frame": 0},
                ),
                DatasetItem(
                    id=3,
                    subset="s3",
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Mask(
                            image=mask,
                            z_order=1,
                            label=3,
                            group=4,
                            attributes={"occluded": False},
                        )
                    ],
                    attributes={"frame": 0},
                ),
                DatasetItem(
                    id=4,
                    subset="s4",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    annotations=[
                        RotatedBbox(1, 10, 5, 5, 45, label=0, attributes={"occluded": False}),
                    ],
                    attributes={"frame": 0},
                ),
            ],
            categories={AnnotationType.label: target_label_cat},
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(CvatExporter.convert, save_media=True),
                test_dir,
                target_dataset=target_dataset,
                require_media=True,
            )

    def test_can_allow_undeclared_attrs(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Label(0, attributes={"x": 4, "y": 2}),
                        Bbox(1, 2, 3, 4, label=0, attributes={"x": 1, "y": 1}),
                    ],
                ),
            ],
            categories=[("a", "", {"x"})],
        )

        target_label_cat = LabelCategories(attributes={"occluded"})
        target_label_cat.add("a", attributes={"x"})
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    media=Image.from_numpy(data=np.zeros((5, 10, 3))),
                    annotations=[
                        Label(0, attributes={"x": "4", "y": "2"}),
                        Bbox(1, 2, 3, 4, label=0, attributes={"x": "1", "y": "1", "occluded": False}),
                    ],
                    attributes={"frame": 0},
                ),
            ],
            categories={AnnotationType.label: target_label_cat},
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(CvatExporter.convert, allow_undeclared_attrs=True, save_media=True),
                test_dir,
                target_dataset=target_dataset,
                require_media=True,
            )

    def test_relative_paths(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.ones((4, 2, 3)))),
                DatasetItem(id="subdir1/1", media=Image.from_numpy(data=np.ones((2, 6, 3)))),
                DatasetItem(id="subdir2/1", media=Image.from_numpy(data=np.ones((5, 4, 3)))),
            ],
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image.from_numpy(data=np.ones((4, 2, 3))), attributes={"frame": 0}),
                DatasetItem(
                    id="subdir1/1",
                    media=Image.from_numpy(data=np.ones((2, 6, 3))),
                    attributes={"frame": 1},
                ),
                DatasetItem(
                    id="subdir2/1",
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    attributes={"frame": 2},
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(CvatExporter.convert, save_media=True),
                test_dir,
                target_dataset=target_dataset,
                require_media=True,
            )

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        label_categories = LabelCategories(attributes={"occluded"})
        for i in range(10):
            label_categories.add(str(i))
        label_categories.items[2].attributes.update(["a1", "a2", "empty"])

        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    subset="s1",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    annotations=[
                        Label(1),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: label_categories,
            },
        )

        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    subset="s1",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    annotations=[
                        Label(1),
                    ],
                    attributes={"frame": 0},
                ),
            ],
            categories={
                AnnotationType.label: label_categories,
            },
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(CvatExporter.convert, save_media=True),
                test_dir,
                target_dataset=target_dataset,
                require_media=True,
            )

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    "q/1",
                    media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".jpeg"),
                    attributes={"frame": 1},
                ),
                DatasetItem(
                    "a/b/c/2",
                    media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp"),
                    attributes={"frame": 2},
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected,
                partial(CvatExporter.convert, save_media=True),
                test_dir,
                require_media=True,
            )
            self.assertTrue(osp.isfile(osp.join(test_dir, "images", "q", "1.jpeg")))
            self.assertTrue(osp.isfile(osp.join(test_dir, "images", "a", "b", "c", "2.bmp")))

    def test_preserve_frame_ids(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="some/name1",
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    attributes={"frame": 40},
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected_dataset,
                partial(CvatExporter.convert, save_media=True),
                test_dir,
                require_media=True,
            )

    def test_reindex(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="some/name1",
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    attributes={"frame": 40},
                ),
            ],
        )

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="some/name1",
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    attributes={"frame": 0},
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(CvatExporter.convert, save_media=True, reindex=True),
                test_dir,
                target_dataset=expected_dataset,
                require_media=True,
            )

    def test_inplace_save_writes_only_updated_data(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="a", media=Image.from_numpy(data=np.ones((3, 2, 3)))),
                DatasetItem(2, subset="a", media=Image.from_numpy(data=np.ones((3, 2, 3)))),
                DatasetItem(2, subset="b", media=Image.from_numpy(data=np.ones((3, 2, 3)))),
            ],
            categories=[],
        )

        with TestDir() as path:
            # generate initial dataset
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(1, subset="a", media=Image.from_numpy(data=np.ones((3, 2, 3)))),
                    DatasetItem(2, subset="b", media=Image.from_numpy(data=np.ones((3, 2, 3)))),
                    DatasetItem(3, subset="c", media=Image.from_numpy(data=np.ones((3, 2, 3)))),
                ],
            )
            dataset.export(path, "cvat", save_media=True)

            dataset.put(DatasetItem(2, subset="a", media=Image.from_numpy(data=np.ones((3, 2, 3)))))
            dataset.remove(3, "c")
            dataset.save(save_media=True)

            self.assertEqual({"a.xml", "b.xml", "images"}, set(os.listdir(path)))
            self.assertEqual({"1.jpg", "2.jpg"}, set(os.listdir(osp.join(path, "images"))))
            compare_datasets(
                self,
                expected,
                Dataset.import_from(path, "cvat"),
                require_media=True,
                ignored_attrs={"frame"},
            )

    def test_video_export(self):
        dataset = Dataset.import_from(
            DUMMY_VIDEO_DATASET_FILE,
            "video_frames",
            start_frame=0,
            end_frame=20,
            name_pattern="frame_%06d",
        )

        dataset.transform(ProjectLabels, dst_labels=["label1", "label2"])

        for frame, item in enumerate(dataset):
            item.annotations = [
                Bbox(
                    3,
                    4,
                    7,
                    1,
                    label=1,
                    id=0,
                    attributes={
                        "occluded": True,
                        "outside": False,
                        "keyframe": True,
                        "track_id": 0,
                    },
                ),
                Points(
                    [21.95, 8.00, 2.55, 15.09, 2.23, 3.16],
                    label=0,
                    id=1,
                    attributes={
                        "occluded": False,
                        "outside": False,
                        "keyframe": True,
                        "track_id": 1,
                    },
                ),
            ]
            item.attributes = {"frame": frame}

        with TestDir() as path:
            dataset.export(
                format="cvat",
                save_dir=path,
                save_media=True,
                use_track=True,
            )
            dataset_re = Dataset.import_from(path, "cvat")

            self.assertEqual(sorted(dataset.subsets()), sorted(dataset_re.subsets()))
            self.assertEqual(len(dataset), len(dataset_re))
