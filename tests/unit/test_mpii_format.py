from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories, Points, PointsCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.mpii.mpii_mat import MPII_POINTS_JOINTS, MPII_POINTS_LABELS, MpiiImporter
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

DUMMY_DATASET_DIR = get_test_asset_path("mpii_dataset")


class MpiiImporterTest(TestCase):
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="000000001",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Points(
                            [
                                620.0,
                                394.0,
                                616.0,
                                269.0,
                                573.0,
                                185.0,
                                647.0,
                                188.0,
                                661.0,
                                221.0,
                                656.0,
                                231.0,
                                610.0,
                                187.0,
                                647.0,
                                176.0,
                                637.02,
                                189.818,
                                695.98,
                                108.182,
                                606.0,
                                217.0,
                                553.0,
                                161.0,
                                601.0,
                                167.0,
                                692.0,
                                185.0,
                                693.0,
                                240.0,
                                688.0,
                                313.0,
                            ],
                            [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            attributes={"center": [594.000, 257.000], "scale": 3.021},
                            label=0,
                            group=1,
                        ),
                        Bbox(615, 218.65, 288.4, 286.95, label=0, group=1),
                    ],
                ),
                DatasetItem(
                    id="000000002",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Points(
                            [
                                650.0,
                                424.0,
                                646.0,
                                309.0,
                                603.0,
                                215.0,
                                677.0,
                                218.0,
                                691.0,
                                251.0,
                                686.0,
                                261.0,
                                640.0,
                                217.0,
                                677.0,
                                216.0,
                                667.02,
                                219.818,
                                725.98,
                                138.182,
                                636.0,
                                247.0,
                                583.0,
                                191.0,
                                631.0,
                                197.0,
                                722.0,
                                215.0,
                                723.0,
                                270.0,
                                718.0,
                                343.0,
                            ],
                            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            attributes={"center": [624.0, 287.0], "scale": 3.7},
                            label=0,
                            group=1,
                        ),
                        Bbox(101.1, 33.3, 113.9, 81.4, label=0, group=1),
                    ],
                ),
                DatasetItem(
                    id="000000003",
                    media=Image.from_numpy(data=np.ones((5, 5, 3))),
                    annotations=[
                        Points(
                            [
                                590.0,
                                364.0,
                                586.0,
                                239.0,
                                533.0,
                                155.0,
                                617.0,
                                158.0,
                                631.0,
                                191.0,
                                626.0,
                                201.0,
                                580.0,
                                157.0,
                                617.0,
                                146.0,
                                607.02,
                                159.818,
                                645.98,
                                68.182,
                                576.0,
                                187.0,
                                532.0,
                                131.0,
                                571.0,
                                137.0,
                                662.0,
                                155.0,
                                663.0,
                                210.0,
                                658.0,
                                283.0,
                            ],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                            attributes={"center": [564.0, 227.0], "scale": 3.2},
                            label=0,
                            group=1,
                        ),
                        Bbox(313.3, 512.43, 220.7, 121.57, label=0, group=1),
                        Points(
                            [
                                490.0,
                                264.0,
                                486.0,
                                139.0,
                                433.0,
                                55.0,
                                517.0,
                                58.0,
                                531.0,
                                91.0,
                                526.0,
                                101.0,
                                480.0,
                                57.0,
                                517.0,
                                46.0,
                                507.02,
                                59.818,
                                545.98,
                                8.182,
                                476.0,
                                87.0,
                                432.0,
                                31.0,
                                471.0,
                                37.0,
                                562.0,
                                55.0,
                                563.0,
                                110.0,
                                558.0,
                                183.0,
                            ],
                            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                            attributes={"center": [464.0, 127.0], "scale": 2.65},
                            label=0,
                            group=2,
                        ),
                        Bbox(213.3, 412.43, 120.7, 21.57, label=0, group=2),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["human"]),
                AnnotationType.points: PointsCategories.from_iterable([(0, MPII_POINTS_LABELS, MPII_POINTS_JOINTS)]),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "mpii")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([MpiiImporter.NAME], detected_formats)
