import pickle  # nosec B403
from copy import deepcopy
from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.contexts.importer import ImportErrorPolicy
from datumaro.components.dataset import Dataset, StreamDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.imagenet import (
    ImagenetExporter,
    ImagenetImporter,
    ImagenetWithSubsetDirsExporter,
    ImagenetWithSubsetDirsImporter,
)

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import (
    TestCaseHelper,
    TestDir,
    compare_datasets,
    compare_datasets_strict,
)


@pytest.fixture
def fxt_standard():
    source = Dataset.from_iterable(
        [
            DatasetItem(
                id="label_0:1",
                media=Image.from_numpy(data=np.ones((8, 8, 3))),
                annotations=[Label(0)],
            ),
            DatasetItem(
                id="label_1:2",
                media=Image.from_numpy(data=np.ones((10, 10, 3))),
                annotations=[Label(1)],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                "label_" + str(label) for label in range(2)
            ),
        },
    )
    expected = deepcopy(source)
    return source, expected


@pytest.fixture
def fxt_multiple_labels():
    source = Dataset.from_iterable(
        [
            DatasetItem(
                id="1",
                media=Image.from_numpy(data=np.ones((8, 8, 3))),
                annotations=[Label(0), Label(1)],
            ),
            DatasetItem(id="2", media=Image.from_numpy(data=np.ones((8, 8, 3)))),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                "label_" + str(label) for label in range(2)
            ),
        },
    )

    expected = Dataset.from_iterable(
        [
            DatasetItem(
                id="label_0:1",
                media=Image.from_numpy(data=np.ones((8, 8, 3))),
                annotations=[Label(0)],
            ),
            DatasetItem(
                id="label_1:1",
                media=Image.from_numpy(data=np.ones((8, 8, 3))),
                annotations=[Label(1)],
            ),
            DatasetItem(id="no_label:2", media=Image.from_numpy(data=np.ones((8, 8, 3)))),
        ],
        categories=["label_0", "label_1"],
    )
    return source, expected


@pytest.fixture()
def fxt_cyrillic_and_spaces_in_filename():
    source = Dataset.from_iterable(
        [
            DatasetItem(
                id="label_0:кириллица с пробелом",
                media=Image.from_numpy(data=np.ones((8, 8, 3))),
                annotations=[Label(0)],
            ),
        ],
        categories=["label_0"],
    )
    expected = deepcopy(source)
    return source, expected


@pytest.fixture()
def fxt_arbitrary_extension():
    source = Dataset.from_iterable(
        [
            DatasetItem(id="no_label:a", media=Image.from_numpy(data=np.zeros((4, 3, 3)))),
            DatasetItem(id="no_label:b", media=Image.from_numpy(data=np.zeros((3, 4, 3)))),
        ],
        categories=[],
    )
    expected = deepcopy(source)
    return source, expected


class ImagenetFormatTest:
    helper = TestCaseHelper()
    exporter = ImagenetExporter

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "fxt_test_case",
        [
            "fxt_standard",
            "fxt_multiple_labels",
            "fxt_cyrillic_and_spaces_in_filename",
            "fxt_arbitrary_extension",
        ],
        indirect=True,
    )
    def test_can_save_and_load(self, fxt_test_case):
        source, expected = fxt_test_case

        with TestDir() as test_dir:
            self.exporter.convert(source, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, self.exporter.NAME)
            compare_datasets(self.helper, expected, parsed_dataset, require_media=True)


class ImagenetWithSubsetDirsFormatTest(ImagenetFormatTest):
    helper = TestCaseHelper()
    exporter = ImagenetWithSubsetDirsExporter

    @pytest.fixture
    def fxt_test_case_with_subsets(self, request):
        fxt_name = request.param
        source, expected = request.getfixturevalue(fxt_name)

        _to_subsets = lambda dataset: Dataset.from_extractors(
            *[
                deepcopy(dataset).transform("map_subsets", mapping={"default": subset})
                for subset in ["train", "val", "test"]
            ]
        )
        return _to_subsets(source), _to_subsets(expected)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "fxt_test_case_with_subsets",
        [
            "fxt_standard",
            "fxt_multiple_labels",
            "fxt_cyrillic_and_spaces_in_filename",
            "fxt_arbitrary_extension",
        ],
        indirect=True,
    )
    def test_can_save_and_load(self, fxt_test_case_with_subsets):
        super().test_can_save_and_load(fxt_test_case_with_subsets)


class ImagenetImporterTest:
    DUMMY_DATASET_DIR = get_test_asset_path("imagenet_dataset")
    IMPORTER_NAME = ImagenetImporter.NAME

    def _create_expected_dataset(self):
        label_categories = LabelCategories.from_iterable(
            ("label_0", "label_1", f"{Path('label_1', 'label_1_1')}")
        )
        label_categories[-1].parent = "label_1"
        label_categories.add_label_group(name="label_1", labels=["label_1_1"], group_type=0)

        return Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0:label_0_1",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="no_label:label_0_2",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                ),
                DatasetItem(
                    id=f"{Path('label_1', 'label_1_1')}:label_1_1",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(2)],
                ),
                DatasetItem(
                    id="label_1:label_1_1",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[Label(1)],
                ),
            ],
            categories={AnnotationType.label: label_categories},
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_can_import(self, dataset_cls, is_stream, helper_tc):
        expected_dataset = self._create_expected_dataset()
        dataset = dataset_cls.import_from(
            self.DUMMY_DATASET_DIR, self.IMPORTER_NAME, error_policy=ImportErrorPolicy()
        )
        assert dataset.is_stream == is_stream

        compare_datasets(helper_tc, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_imagenet(self):
        detected_formats = Environment().detect_dataset(self.DUMMY_DATASET_DIR)
        assert [self.IMPORTER_NAME] == detected_formats

    @mark_requirement(Requirements.DATUM_673)
    def test_can_pickle(self, helper_tc):
        source = Dataset.import_from(self.DUMMY_DATASET_DIR, format=self.IMPORTER_NAME)

        parsed = pickle.loads(pickle.dumps(source))  # nosec

        compare_datasets_strict(helper_tc, source, parsed)


class ImagenetWithSubsetDirsImporterTest(ImagenetImporterTest):
    DUMMY_DATASET_DIR = get_test_asset_path("imagenet_subsets_dataset")
    IMPORTER_NAME = ImagenetWithSubsetDirsImporter.NAME

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_can_import(self, dataset_cls, is_stream, helper_tc):
        dataset = dataset_cls.import_from(
            self.DUMMY_DATASET_DIR, self.IMPORTER_NAME, error_policy=ImportErrorPolicy()
        )
        assert dataset.is_stream == is_stream

        for subset_name, subset in dataset.subsets().items():
            expected_dataset = self._create_expected_dataset().transform(
                "map_subsets", mapping={"default": subset_name}
            )
            compare_datasets(helper_tc, expected_dataset, subset, require_media=True)
