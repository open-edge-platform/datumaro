import os.path as osp
from unittest import TestCase

from datumaro.components.annotation import AnnotationType, Cuboid3d, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, PointCloud
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets_3d
from tests.utils.test_utils import run_datum as run

DUMMY_DATASET_DIR = get_test_asset_path("sly_pointcloud_dataset")


class SlyPointCloudIntegrationScenarios(TestCase):
    def test_can_convert_to_kitti_raw(self):
        with TestDir() as test_dir:
            export_dir = osp.join(test_dir, "export_dir")
            expected_label_cat = LabelCategories(attributes={"occluded"})
            expected_label_cat.add("bus", attributes={"tag1", "tag3"})
            expected_label_cat.add("car", attributes={"tag1", "tag3"})
            expected_dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id="frame1",
                        annotations=[
                            Cuboid3d(
                                label=1,
                                position=[0.47, 0.23, 0.79],
                                scale=[0.01, 0.01, 0.01],
                                attributes={
                                    "track_id": 2,
                                    "tag1": "fd",
                                    "tag3": "4s",
                                    "occluded": False,
                                },
                            ),
                            Cuboid3d(
                                label=1,
                                position=[0.36, 0.64, 0.93],
                                scale=[0.01, 0.01, 0.01],
                                attributes={
                                    "track_id": 3,
                                    "tag1": "v12",
                                    "tag3": "",
                                    "occluded": False,
                                },
                            ),
                        ],
                        media=PointCloud.from_file(
                            path=osp.join(export_dir, "velodyne_points", "data", "frame1.pcd"),
                            extra_images=[Image.from_file(path=osp.join(export_dir, "image_00", "data", "frame1.png"))],
                        ),
                        attributes={"frame": 0},
                    ),
                    DatasetItem(
                        id="frame2",
                        annotations=[
                            Cuboid3d(
                                label=0,
                                position=[0.59, 14.41, -0.61],
                                attributes={
                                    "track_id": 1,
                                    "tag1": "",
                                    "tag3": "",
                                    "occluded": False,
                                },
                            )
                        ],
                        media=PointCloud.from_file(
                            path=osp.join(export_dir, "velodyne_points", "data", "frame2.pcd"),
                            extra_images=[Image.from_file(path=osp.join(export_dir, "image_00", "data", "frame2.png"))],
                        ),
                        attributes={"frame": 1},
                    ),
                ],
                categories={AnnotationType.label: expected_label_cat},
                media_type=PointCloud,
            )

            run(
                self,
                "convert",
                "-if",
                "sly_pointcloud",
                "-i",
                DUMMY_DATASET_DIR,
                "-f",
                "kitti_raw",
                "-o",
                export_dir,
                "--",
                "--save-media",
                "--allow-attrs",
            )

            parsed_dataset = Dataset.import_from(export_dir, format="kitti_raw")
            compare_datasets_3d(self, expected_dataset, parsed_dataset, require_point_cloud=True)
