import os
import os.path as osp
from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Cuboid3d, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image, PointCloud
from datumaro.plugins.data_formats.kitti_raw.base import KittiRawImporter
from datumaro.plugins.data_formats.kitti_raw.exporter import KittiRawExporter
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import Dimensions, TestDir, check_save_and_load, compare_datasets_3d

DUMMY_DATASET_DIR = get_test_asset_path("kitti_dataset", "kitti_raw")


class KittiRawImporterTest(TestCase):
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([KittiRawImporter.NAME], detected_formats)

    def test_can_load(self):
        pcd1 = osp.join(DUMMY_DATASET_DIR, "velodyne_points", "data", "0000000000.pcd")
        pcd2 = osp.join(DUMMY_DATASET_DIR, "velodyne_points", "data", "0000000001.pcd")
        pcd3 = osp.join(DUMMY_DATASET_DIR, "velodyne_points", "data", "0000000002.pcd")

        image1 = Image.from_file(path=osp.join(DUMMY_DATASET_DIR, "IMAGE_00", "data", "0000000000.png"))
        image2 = Image.from_file(path=osp.join(DUMMY_DATASET_DIR, "IMAGE_00", "data", "0000000001.png"))
        image3 = Image.from_file(path=osp.join(DUMMY_DATASET_DIR, "IMAGE_00", "data", "0000000002.png"))

        expected_label_cat = LabelCategories(attributes={"occluded"})
        expected_label_cat.add("bus")
        expected_label_cat.add("car")
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0000000000",
                    annotations=[
                        Cuboid3d(
                            position=[1, 2, 3],
                            scale=[-3.62, 7.95, -1.03],
                            label=1,
                            attributes={"occluded": False, "track_id": 1},
                        ),
                        Cuboid3d(
                            position=[1, 1, 0],
                            scale=[23.01, 8.34, -0.76],
                            label=0,
                            attributes={"occluded": False, "track_id": 2},
                        ),
                    ],
                    media=PointCloud.from_file(path=pcd1, extra_images=[image1]),
                    attributes={"frame": 0},
                ),
                DatasetItem(
                    id="0000000001",
                    annotations=[
                        Cuboid3d(
                            position=[0, 1, 0],
                            scale=[23.01, 8.34, -0.76],
                            rotation=[1, 1, 3],
                            label=0,
                            attributes={"occluded": True, "track_id": 2},
                        )
                    ],
                    media=PointCloud.from_file(path=pcd2, extra_images=[image2]),
                    attributes={"frame": 1},
                ),
                DatasetItem(
                    id="0000000002",
                    annotations=[
                        Cuboid3d(
                            position=[1, 2, 3],
                            scale=[13.54, -9.41, 0.24],
                            label=1,
                            attributes={"occluded": False, "track_id": 3},
                        )
                    ],
                    media=PointCloud.from_file(path=pcd3, extra_images=[image3]),
                    attributes={"frame": 2},
                ),
            ],
            categories={AnnotationType.label: expected_label_cat},
            media_type=PointCloud,
        )

        parsed_dataset = Dataset.import_from(DUMMY_DATASET_DIR, "kitti_raw")

        compare_datasets_3d(self, expected_dataset, parsed_dataset, require_point_cloud=True)


class KittiRawExporterTest(TestCase):
    pcd1 = osp.abspath(osp.join(DUMMY_DATASET_DIR, "velodyne_points", "data", "0000000000.pcd"))
    pcd2 = osp.abspath(osp.join(DUMMY_DATASET_DIR, "velodyne_points", "data", "0000000001.pcd"))
    pcd3 = osp.abspath(osp.join(DUMMY_DATASET_DIR, "velodyne_points", "data", "0000000002.pcd"))

    image1 = Image.from_file(path=osp.abspath(osp.join(DUMMY_DATASET_DIR, "IMAGE_00", "data", "0000000000.png")))
    image2 = Image.from_file(path=osp.abspath(osp.join(DUMMY_DATASET_DIR, "IMAGE_00", "data", "0000000001.png")))
    image3 = Image.from_file(path=osp.abspath(osp.join(DUMMY_DATASET_DIR, "IMAGE_00", "data", "0000000002.png")))

    def _test_save_and_load(
        self, source_dataset, converter, test_dir, target_dataset=None, importer_args=None, **kwargs
    ):
        kwargs.setdefault("dimension", Dimensions.dim_3d)
        return check_save_and_load(
            self,
            source_dataset,
            converter,
            test_dir,
            importer="kitti_raw",
            target_dataset=target_dataset,
            importer_args=importer_args,
            **kwargs,
        )

    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0000000000",
                    annotations=[
                        Cuboid3d(
                            position=[13.54, -9.41, 0.24],
                            label=0,
                            attributes={"occluded": False, "track_id": 1},
                        ),
                        Cuboid3d(
                            position=[3.4, -2.11, 4.4],
                            label=1,
                            attributes={"occluded": True, "track_id": 2},
                        ),
                    ],
                    media=PointCloud.from_file(path=self.pcd1, extra_images=[self.image1]),
                    attributes={"frame": 0},
                ),
                DatasetItem(
                    id="0000000001",
                    annotations=[
                        Cuboid3d(position=[1.4, 2.1, 1.4], label=1, attributes={"track_id": 2}),
                        Cuboid3d(
                            position=[11.4, -0.1, 4.2],
                            scale=[1, 2, 2],
                            label=0,
                            attributes={"track_id": 3},
                        ),
                    ],
                ),
                DatasetItem(
                    id="0000000002",
                    annotations=[
                        Cuboid3d(
                            position=[0.4, -1, 2.24],
                            scale=[1, 2, 2],
                            label=0,
                            attributes={"track_id": 3},
                        ),
                    ],
                    media=PointCloud.from_file(path=self.pcd3),
                    attributes={"frame": 2},
                ),
                DatasetItem(
                    id="0000000003",
                    annotations=[
                        Cuboid3d(
                            position=[0.4, -1, 3.24],
                            scale=[1, 2, 2],
                            label=0,
                            attributes={"track_id": 3},
                        ),
                    ],
                    media=PointCloud.from_bytes(
                        data=b"3" * 5,
                        extra_images=[
                            Image.from_numpy(data=np.ones([5, 5, 3]), ext=".png"),
                        ],
                    ),
                    attributes={"frame": 3},
                ),
            ],
            categories=["cat", "dog"],
            media_type=PointCloud,
        )

        with TestDir() as test_dir:
            target_label_cat = LabelCategories(attributes={"occluded"})
            target_label_cat.add("cat")
            target_label_cat.add("dog")

            target_dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id="0000000000",
                        annotations=[
                            Cuboid3d(
                                position=[13.54, -9.41, 0.24],
                                label=0,
                                attributes={"occluded": False, "track_id": 1},
                            ),
                            Cuboid3d(
                                position=[3.4, -2.11, 4.4],
                                label=1,
                                attributes={"occluded": True, "track_id": 2},
                            ),
                        ],
                        media=PointCloud.from_file(
                            path=osp.join(test_dir, "velodyne_points", "data", "0000000000.pcd"),
                            extra_images=[
                                Image.from_file(path=osp.join(test_dir, "image_00", "data", "0000000000.png"))
                            ],
                        ),
                        attributes={"frame": 0},
                    ),
                    DatasetItem(
                        id="0000000001",
                        annotations=[
                            Cuboid3d(
                                position=[1.4, 2.1, 1.4],
                                label=1,
                                attributes={"occluded": False, "track_id": 2},
                            ),
                            Cuboid3d(
                                position=[11.4, -0.1, 4.2],
                                scale=[1, 2, 2],
                                label=0,
                                attributes={"occluded": False, "track_id": 3},
                            ),
                        ],
                        attributes={"frame": 1},
                    ),
                    DatasetItem(
                        id="0000000002",
                        annotations=[
                            Cuboid3d(
                                position=[0.4, -1, 2.24],
                                scale=[1, 2, 2],
                                label=0,
                                attributes={"occluded": False, "track_id": 3},
                            ),
                        ],
                        media=PointCloud.from_file(
                            path=osp.join(test_dir, "velodyne_points", "data", "0000000002.pcd")
                        ),
                        attributes={"frame": 2},
                    ),
                    DatasetItem(
                        id="0000000003",
                        annotations=[
                            Cuboid3d(
                                position=[0.4, -1, 3.24],
                                scale=[1, 2, 2],
                                label=0,
                                attributes={"occluded": False, "track_id": 3},
                            ),
                        ],
                        media=PointCloud.from_file(
                            path=osp.join(test_dir, "velodyne_points", "data", "0000000003.pcd"),
                            extra_images=[
                                Image.from_file(
                                    path=osp.join(test_dir, "image_00", "data", "0000000003.png"),
                                )
                            ],
                        ),
                        attributes={"frame": 3},
                    ),
                ],
                categories={AnnotationType.label: target_label_cat},
                media_type=PointCloud,
            )

            self._test_save_and_load(
                source_dataset,
                partial(KittiRawExporter.convert, save_media=True),
                test_dir,
                target_dataset=target_dataset,
                require_point_cloud=True,
            )

    def test_preserve_frame_ids(self):
        source_dataset = Dataset.from_iterable(
            [DatasetItem(id="abc", attributes={"frame": 40})],
            categories=[],
            media_type=PointCloud,
        )

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset, KittiRawExporter.convert, test_dir)

    def test_reindex_frames(self):
        source_dataset = Dataset.from_iterable(
            [DatasetItem(id="abc")],
            categories=[],
            media_type=PointCloud,
        )

        expected_dataset = Dataset.from_iterable(
            [DatasetItem(id="abc", attributes={"frame": 0})],
            categories=[],
            media_type=PointCloud,
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(KittiRawExporter.convert, reindex=True),
                test_dir,
                target_dataset=expected_dataset,
            )

    def test_requires_track_id(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="abc",
                    annotations=[
                        Cuboid3d(position=[0.4, -1, 2.24], label=0),
                    ],
                )
            ],
            categories=["dog"],
            media_type=PointCloud,
        )

        with TestDir() as test_dir, self.assertRaisesRegex(Exception, "track_id"):
            KittiRawExporter.convert(source_dataset, test_dir)

    def test_reindex_allows_single_annotations(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="abc",
                    annotations=[
                        Cuboid3d(position=[0.4, -1, 2.24], label=0),
                    ],
                )
            ],
            categories=["dog"],
            media_type=PointCloud,
        )

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="abc",
                    annotations=[
                        Cuboid3d(
                            position=[0.4, -1, 2.24],
                            label=0,
                            attributes={"track_id": 1, "occluded": False},
                        ),
                    ],
                    attributes={"frame": 0},
                )
            ],
            categories=["dog"],
            media_type=PointCloud,
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(KittiRawExporter.convert, reindex=True),
                test_dir,
                target_dataset=expected_dataset,
            )

    def test_can_save_and_load_attributes(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0000000000",
                    annotations=[
                        Cuboid3d(
                            position=[13.54, -9.41, 0.24],
                            label=0,
                            attributes={"track_id": 1, "occluded": True, "a": "w", "b": 5},
                        )
                    ],
                    attributes={"frame": 0},
                )
            ],
            categories=["cat"],
            media_type=PointCloud,
        )

        target_label_cat = LabelCategories(attributes={"occluded"})
        target_label_cat.add("cat", attributes=["a", "b"])
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0000000000",
                    annotations=[
                        Cuboid3d(
                            position=[13.54, -9.41, 0.24],
                            label=0,
                            attributes={"track_id": 1, "occluded": True, "a": "w", "b": 5},
                        )
                    ],
                    attributes={"frame": 0},
                )
            ],
            categories={AnnotationType.label: target_label_cat},
            media_type=PointCloud,
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(KittiRawExporter.convert, allow_attrs=True),
                test_dir,
                target_dataset=target_dataset,
            )

    def test_can_discard_attributes(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0000000000",
                    annotations=[
                        Cuboid3d(
                            position=[13.54, -9.41, 0.24],
                            label=0,
                            attributes={"track_id": 1, "a": "w", "b": 5},
                        )
                    ],
                    attributes={"frame": 0},
                )
            ],
            categories=["cat"],
            media_type=PointCloud,
        )

        target_label_cat = LabelCategories(attributes={"occluded"})
        target_label_cat.add("cat")
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0000000000",
                    annotations=[
                        Cuboid3d(
                            position=[13.54, -9.41, 0.24],
                            label=0,
                            attributes={"track_id": 1, "occluded": False},
                        )
                    ],
                    attributes={"frame": 0},
                )
            ],
            categories={AnnotationType.label: target_label_cat},
            media_type=PointCloud,
        )

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset, KittiRawExporter.convert, test_dir, target_dataset=target_dataset)

    def test_can_save_and_load_without_annotations(self):
        source_dataset = Dataset.from_iterable(
            [DatasetItem(id="0000000000", attributes={"frame": 0})],
            categories=[],
            media_type=PointCloud,
        )

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset, KittiRawExporter.convert, test_dir)

    def test_can_save_and_load_arbitrary_paths(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/d",
                    annotations=[Cuboid3d(position=[1, 2, 3], label=0, attributes={"track_id": 1})],
                    media=PointCloud.from_file(path=self.pcd1, extra_images=[self.image1]),
                    attributes={"frame": 3},
                ),
            ],
            categories=["cat"],
            media_type=PointCloud,
        )

        with TestDir() as test_dir:
            target_label_cat = LabelCategories(attributes={"occluded"})
            target_label_cat.add("cat")
            target_dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id="a/d",
                        annotations=[
                            Cuboid3d(
                                position=[1, 2, 3],
                                label=0,
                                attributes={"track_id": 1, "occluded": False},
                            )
                        ],
                        media=PointCloud.from_file(
                            path=osp.join(test_dir, "velodyne_points", "data", "a", "d.pcd"),
                            extra_images=[
                                Image.from_file(path=osp.join(test_dir, "image_00", "data", "a", "d.png")),
                            ],
                        ),
                        attributes={"frame": 3},
                    ),
                ],
                categories={AnnotationType.label: target_label_cat},
                media_type=PointCloud,
            )

            self._test_save_and_load(
                source_dataset,
                partial(KittiRawExporter.convert, save_media=True),
                test_dir,
                target_dataset=target_dataset,
                require_point_cloud=True,
            )
            self.assertTrue(osp.isfile(osp.join(test_dir, "image_00", "data", "a", "d.png")))

    def test_can_save_and_load_multiple_related_images(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a/d",
                    annotations=[Cuboid3d(position=[1, 2, 3], label=0, attributes={"track_id": 1})],
                    media=PointCloud.from_file(path=self.pcd1, extra_images=[self.image1, self.image2, self.image3]),
                    attributes={"frame": 3},
                ),
            ],
            categories=["cat"],
            media_type=PointCloud,
        )

        with TestDir() as test_dir:
            target_label_cat = LabelCategories(attributes={"occluded"})
            target_label_cat.add("cat")
            target_dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id="a/d",
                        annotations=[
                            Cuboid3d(
                                position=[1, 2, 3],
                                label=0,
                                attributes={"track_id": 1, "occluded": False},
                            )
                        ],
                        media=PointCloud.from_file(
                            path=osp.join(test_dir, "velodyne_points", "data", "a", "d.pcd"),
                            extra_images=[
                                Image.from_file(path=osp.join(test_dir, "image_00", "data", "a", "d.png")),
                                Image.from_file(path=osp.join(test_dir, "image_01", "data", "a", "d.png")),
                                Image.from_file(path=osp.join(test_dir, "image_02", "data", "a", "d.png")),
                            ],
                        ),
                        attributes={"frame": 3},
                    ),
                ],
                categories={AnnotationType.label: target_label_cat},
                media_type=PointCloud,
            )

            self._test_save_and_load(
                source_dataset,
                partial(KittiRawExporter.convert, save_media=True),
                test_dir,
                target_dataset=target_dataset,
                require_point_cloud=True,
            )
            self.assertTrue(osp.isfile(osp.join(test_dir, "image_00", "data", "a", "d.png")))
            self.assertTrue(osp.isfile(osp.join(test_dir, "image_01", "data", "a", "d.png")))
            self.assertTrue(osp.isfile(osp.join(test_dir, "image_02", "data", "a", "d.png")))

    def test_inplace_save_writes_only_updated_data(self):
        with TestDir() as path:
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id="frame1",
                        annotations=[Cuboid3d(position=[3.5, 9.8, 0.3], label=0, attributes={"track_id": 1})],
                        media=PointCloud.from_file(path=self.pcd1, extra_images=[self.image1]),
                        attributes={"frame": 0},
                    )
                ],
                categories=["car", "bus"],
                media_type=PointCloud,
            )
            dataset.export(path, "kitti_raw", save_media=True)

            dataset.put(
                DatasetItem(
                    "frame2",
                    annotations=[Cuboid3d(position=[1, 2, 0], label=1, attributes={"track_id": 1})],
                    media=PointCloud.from_file(path=self.pcd2, extra_images=[self.image2]),
                    attributes={"frame": 1},
                )
            )
            dataset.remove("frame1")
            dataset.save(save_media=True)

            self.assertEqual({"frame2.png"}, set(os.listdir(osp.join(path, "image_00", "data"))))
            self.assertEqual({"frame2.pcd"}, set(os.listdir(osp.join(path, "velodyne_points", "data"))))

    def test_can_save_and_load_with_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0000000000",
                    annotations=[
                        Cuboid3d(
                            position=[13.54, -9.41, 0.24],
                            label=0,
                            attributes={"occluded": False, "track_id": 1},
                        )
                    ],
                    media=PointCloud.from_file(path=self.pcd1, extra_images=[self.image1]),
                    attributes={"frame": 0},
                ),
                DatasetItem(
                    id="0000000001",
                    annotations=[Cuboid3d(position=[1.4, 2.1, 1.4], label=1, attributes={"track_id": 2})],
                ),
            ],
            categories=["cat", "dog"],
            media_type=PointCloud,
        )

        with TestDir() as test_dir:
            target_label_cat = LabelCategories(attributes={"occluded"})
            target_label_cat.add("cat")
            target_label_cat.add("dog")

            target_dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id="0000000000",
                        annotations=[
                            Cuboid3d(
                                position=[13.54, -9.41, 0.24],
                                label=0,
                                attributes={"occluded": False, "track_id": 1},
                            )
                        ],
                        media=PointCloud.from_file(
                            path=osp.join(test_dir, "velodyne_points", "data", "0000000000.pcd"),
                            extra_images=[
                                Image.from_file(path=osp.join(test_dir, "image_00", "data", "0000000000.png"))
                            ],
                        ),
                        attributes={"frame": 0},
                    ),
                    DatasetItem(
                        id="0000000001",
                        annotations=[
                            Cuboid3d(
                                position=[1.4, 2.1, 1.4],
                                label=1,
                                attributes={"occluded": False, "track_id": 2},
                            )
                        ],
                        attributes={"frame": 1},
                    ),
                ],
                categories={AnnotationType.label: target_label_cat},
                media_type=PointCloud,
            )

            self._test_save_and_load(
                source_dataset,
                partial(KittiRawExporter.convert, save_media=True, save_dataset_meta=True),
                test_dir,
                target_dataset=target_dataset,
                require_point_cloud=True,
            )
            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))
