import os.path as osp
from unittest import TestCase

from datumaro.components.annotation import AnnotationType, Cuboid3d, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, PointCloud

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets_3d
from tests.utils.test_utils import run_datum as run

DUMMY_DATASET_DIR = get_test_asset_path(
    "kitti_dataset",
    "kitti_raw",
)


class KittiRawIntegrationScenarios(TestCase):
    def test_can_convert_to_kitti_raw(self):
        with TestDir() as test_dir:
            export_dir = osp.join(test_dir, "export_dir")
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
                        media=PointCloud.from_file(
                            path=osp.join(export_dir, "ds0", "pointcloud", "0000000000.pcd"),
                            extra_images=[
                                Image.from_file(
                                    path=osp.join(
                                        export_dir,
                                        "ds0",
                                        "related_images",
                                        "0000000000_pcd",
                                        "0000000000.png",
                                    )
                                )
                            ],
                        ),
                        attributes={"frame": 0, "description": ""},
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
                        media=PointCloud.from_file(
                            path=osp.join(export_dir, "ds0", "pointcloud", "0000000001.pcd"),
                            extra_images=[
                                Image.from_file(
                                    path=osp.join(
                                        export_dir,
                                        "ds0",
                                        "related_images",
                                        "0000000001_pcd",
                                        "0000000001.png",
                                    )
                                )
                            ],
                        ),
                        attributes={"frame": 1, "description": ""},
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
                        media=PointCloud.from_file(
                            path=osp.join(export_dir, "ds0", "pointcloud", "0000000002.pcd"),
                            extra_images=[
                                Image.from_file(
                                    path=osp.join(
                                        export_dir,
                                        "ds0",
                                        "related_images",
                                        "0000000002_pcd",
                                        "0000000002.png",
                                    )
                                )
                            ],
                        ),
                        attributes={"frame": 2, "description": ""},
                    ),
                ],
                categories={AnnotationType.label: expected_label_cat},
                media_type=PointCloud,
            )

            run(
                self,
                "convert",
                "-if",
                "kitti_raw",
                "-i",
                DUMMY_DATASET_DIR,
                "-f",
                "sly_pointcloud",
                "-o",
                export_dir,
                "--",
                "--save-media",
            )

            parsed_dataset = Dataset.import_from(export_dir, format="sly_pointcloud")
            compare_datasets_3d(self, expected_dataset, parsed_dataset, require_point_cloud=True)
