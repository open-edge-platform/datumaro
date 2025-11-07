# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import pickle  # nosec B403

import numpy as np
import pytest

from datumaro.components.annotation import Bbox
from datumaro.components.dataset import Dataset, StreamDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import (
    AnnotationImportError,
    DatasetExportError,
    DatasetImportError,
    InvalidAnnotationError,
    ItemImportError,
    UndeclaredLabelError,
)
from datumaro.components.media import Image
from datumaro.plugins.data_formats.yolo.base import YoloStrictBase
from datumaro.plugins.data_formats.yolo.exporter import YoloExporter
from datumaro.util.image import save_image
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets, compare_datasets_strict


class YoloExportertTest:
    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_can_save_and_load(self, dataset_cls, is_stream, test_dir, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2, id=0, group=0),
                        Bbox(0, 1, 2, 3, label=4, id=1, group=1),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2, id=0, group=0),
                        Bbox(3, 3, 2, 3, label=4, id=1, group=1),
                        Bbox(2, 1, 2, 3, label=4, id=2, group=2),
                    ],
                ),
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=2, id=0, group=0),
                        Bbox(0, 2, 3, 2, label=5, id=1, group=1),
                        Bbox(0, 2, 4, 2, label=6, id=2, group=2),
                        Bbox(0, 7, 3, 2, label=7, id=3, group=3),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        YoloExporter.convert(source_dataset, test_dir, save_media=True, stream=is_stream)
        parsed_dataset = dataset_cls.import_from(test_dir, "yolo")
        assert parsed_dataset.is_stream == is_stream

        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_can_save_dataset_with_image_info(self, dataset_cls, is_stream, test_dir, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_file(path="1.jpg", size=(10, 15)),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2, id=0, group=0),
                        Bbox(3, 3, 2, 3, label=4, id=1, group=1),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        YoloExporter.convert(source_dataset, test_dir, stream=is_stream)

        save_image(osp.join(test_dir, "obj_train_data", "1.jpg"), np.ones((10, 15, 3)))  # put the image for dataset
        parsed_dataset = dataset_cls.import_from(test_dir, "yolo")
        assert parsed_dataset.is_stream == is_stream

        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_can_load_dataset_with_exact_image_info(self, dataset_cls, is_stream, test_dir, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_file(path="1.jpg", size=(10, 15)),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2, id=0, group=0),
                        Bbox(3, 3, 2, 3, label=4, id=1, group=1),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        YoloExporter.convert(source_dataset, test_dir, stream=is_stream)

        parsed_dataset = dataset_cls.import_from(test_dir, "yolo", image_info={"1": (10, 15)})
        assert parsed_dataset.is_stream == is_stream

        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self, dataset_cls, is_stream, test_dir, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2, id=0, group=0),
                        Bbox(0, 1, 2, 3, label=4, id=1, group=1),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        YoloExporter.convert(source_dataset, test_dir, save_media=True, stream=is_stream)
        parsed_dataset = dataset_cls.import_from(test_dir, "yolo")
        assert parsed_dataset.is_stream == is_stream

        compare_datasets(helper_tc, source_dataset, parsed_dataset, require_media=True)

    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    @pytest.mark.parametrize("save_media", [True, False])
    def test_relative_paths(self, dataset_cls, is_stream, save_media, test_dir, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="train", media=Image.from_numpy(data=np.ones((4, 2, 3)))),
                DatasetItem(id="subdir1/1", subset="train", media=Image.from_numpy(data=np.ones((2, 6, 3)))),
                DatasetItem(id="subdir2/1", subset="train", media=Image.from_numpy(data=np.ones((5, 4, 3)))),
            ],
            categories=[],
        )

        YoloExporter.convert(source_dataset, test_dir, save_media=save_media, stream=is_stream)
        parsed_dataset = dataset_cls.import_from(test_dir, "yolo")
        assert parsed_dataset.is_stream == is_stream

        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_can_save_and_load_image_with_arbitrary_extension(self, dataset_cls, is_stream, test_dir, helper_tc):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "q/1",
                    subset="train",
                    media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".JPEG"),
                ),
                DatasetItem(
                    "a/b/c/2",
                    subset="valid",
                    media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp"),
                ),
            ],
            categories=[],
        )

        YoloExporter.convert(dataset, test_dir, save_media=True, stream=is_stream)
        parsed_dataset = dataset_cls.import_from(test_dir, "yolo")
        assert parsed_dataset.is_stream == is_stream

        compare_datasets(helper_tc, dataset, parsed_dataset, require_media=True)

    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_inplace_save_writes_only_updated_data(self, dataset_cls, is_stream, test_dir, helper_tc):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="train", media=Image.from_numpy(data=np.ones((2, 4, 3)))),
                DatasetItem(2, subset="train", media=Image.from_numpy(data=np.ones((3, 2, 3)))),
            ],
            categories=[],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, subset="train", media=Image.from_numpy(data=np.ones((2, 4, 3)))),
                DatasetItem(2, subset="train", media=Image.from_file(path="2.jpg", size=(3, 2))),
                DatasetItem(3, subset="valid", media=Image.from_numpy(data=np.ones((2, 2, 3)))),
            ],
            categories=[],
        )
        dataset.export(test_dir, "yolo", save_media=True, stream=is_stream)

        dataset.put(DatasetItem(2, subset="train", media=Image.from_numpy(data=np.ones((3, 2, 3)))))
        dataset.remove(3, "valid")
        dataset.save(save_media=True, stream=is_stream)

        assert {"1.txt", "2.txt", "1.jpg", "2.jpg"} == set(os.listdir(osp.join(test_dir, "obj_train_data")))
        assert set() == set(os.listdir(osp.join(test_dir, "obj_valid_data")))
        actual = dataset_cls.import_from(test_dir, "yolo")
        assert actual.is_stream == is_stream
        compare_datasets(helper_tc, expected, actual, require_media=True)

    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_can_save_and_load_with_meta_file(self, dataset_cls, is_stream, test_dir, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2, id=0, group=0),
                        Bbox(0, 1, 2, 3, label=4, id=1, group=1),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2, id=0, group=0),
                        Bbox(3, 3, 2, 3, label=4, id=1, group=1),
                        Bbox(2, 1, 2, 3, label=4, id=2, group=2),
                    ],
                ),
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=2, id=0, group=0),
                        Bbox(0, 2, 3, 2, label=5, id=1, group=1),
                        Bbox(0, 2, 4, 2, label=6, id=2, group=2),
                        Bbox(0, 7, 3, 2, label=7, id=3, group=3),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        YoloExporter.convert(source_dataset, test_dir, save_media=True, save_dataset_meta=True, stream=is_stream)
        parsed_dataset = dataset_cls.import_from(test_dir, "yolo")
        assert parsed_dataset.is_stream == is_stream

        assert osp.isfile(osp.join(test_dir, "dataset_meta.json"))
        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_can_save_and_load_with_custom_subset_name(self, dataset_cls, is_stream, test_dir, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="anything",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=2, id=0, group=0),
                        Bbox(0, 2, 3, 2, label=5, id=1, group=1),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        YoloExporter.convert(source_dataset, test_dir, save_media=True, stream=is_stream)
        parsed_dataset = dataset_cls.import_from(test_dir, "yolo")
        assert parsed_dataset.is_stream == is_stream

        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    def test_cant_save_with_reserved_subset_name(self):
        for subset in ["backup", "classes"]:
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id=3,
                        subset=subset,
                        media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    ),
                ],
                categories=["a"],
            )

            with TestDir() as test_dir:
                with pytest.raises(DatasetExportError, match=f"Can't export '{subset}' subset"):
                    YoloExporter.convert(dataset, test_dir)

    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_can_save_and_load_without_path_prefix(self, dataset_cls, is_stream, test_dir, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image.from_numpy(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=1),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        YoloExporter.convert(source_dataset, test_dir, save_media=True, add_path_prefix=False, stream=is_stream)
        parsed_dataset = dataset_cls.import_from(test_dir, "yolo")
        assert parsed_dataset.is_stream == is_stream

        with open(osp.join(test_dir, "obj.data"), "r") as f:
            lines = f.readlines()
            assert "valid = valid.txt\n" in lines

        with open(osp.join(test_dir, "valid.txt"), "r") as f:
            lines = f.readlines()
            assert "obj_valid_data/3.jpg\n" in lines

        compare_datasets(helper_tc, source_dataset, parsed_dataset)


DUMMY_DATASET_DIR = get_test_asset_path("yolo_dataset", "strict")


class YoloImporterTest:
    @pytest.mark.parametrize("dataset_cls, is_stream", [(Dataset, False), (StreamDataset, True)])
    def test_can_pickle(self, dataset_cls, is_stream, helper_tc):
        source = dataset_cls.import_from(DUMMY_DATASET_DIR, format="yolo")
        assert source.is_stream == is_stream

        parsed = pickle.loads(pickle.dumps(source))  # nosec

        compare_datasets_strict(helper_tc, source, parsed)


class YoloStrictBaseTest:
    @pytest.fixture
    def fxt_dataset(self, test_dir: str) -> Dataset:
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "a",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((5, 10, 3))),
                    annotations=[Bbox(1, 1, 2, 4, label=0)],
                )
            ],
            categories=["test"],
        )
        dataset.export(test_dir, "yolo", save_media=True)

        return dataset

    def test_can_parse(self, fxt_dataset, test_dir, helper_tc):
        expected = fxt_dataset

        actual = Dataset.import_from(test_dir, "yolo")
        compare_datasets(helper_tc, expected, actual)

    def test_can_report_invalid_data_file(self, test_dir):
        with pytest.raises(DatasetImportError, match="Can't read dataset descriptor file"):
            YoloStrictBase(test_dir)

    def test_can_report_invalid_ann_line_format(self, fxt_dataset, test_dir):
        with open(osp.join(test_dir, "obj_train_data", "a.txt"), "w") as f:
            f.write("1 2 3\n")

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, "yolo").init_cache()
        assert isinstance(capture.value.__cause__, InvalidAnnotationError)
        assert "Unexpected field count" in str(capture.value.__cause__)

    def test_can_report_invalid_label(self, fxt_dataset, test_dir):
        with open(osp.join(test_dir, "obj_train_data", "a.txt"), "w") as f:
            f.write("10 0.5 0.5 0.5 0.5\n")

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, "yolo").init_cache()
        assert isinstance(capture.value.__cause__, UndeclaredLabelError)
        assert capture.value.__cause__.id == "10"

    @pytest.mark.parametrize(
        "field, field_name",
        [
            (1, "bbox center x"),
            (2, "bbox center y"),
            (3, "bbox width"),
            (4, "bbox height"),
        ],
    )
    def test_can_report_invalid_field_type(self, field, field_name, fxt_dataset, test_dir):
        with open(osp.join(test_dir, "obj_train_data", "a.txt"), "w") as f:
            values = [0, 0.5, 0.5, 0.5, 0.5]
            values[field] = "a"
            f.write(" ".join(str(v) for v in values))

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, "yolo").init_cache()
        assert isinstance(capture.value.__cause__, InvalidAnnotationError)
        assert field_name in str(capture.value.__cause__)

    def test_can_report_missing_ann_file(self, fxt_dataset, test_dir):
        os.remove(osp.join(test_dir, "obj_train_data", "a.txt"))

        with pytest.raises(ItemImportError) as capture:
            Dataset.import_from(test_dir, "yolo").init_cache()
        assert isinstance(capture.value.__cause__, FileNotFoundError)

    def test_can_report_missing_image_info(self, fxt_dataset, test_dir):
        os.remove(osp.join(test_dir, "obj_train_data", "a.jpg"))

        with pytest.raises(ItemImportError) as capture:
            Dataset.import_from(test_dir, "yolo").init_cache()
        assert isinstance(capture.value.__cause__, DatasetImportError)
        assert "Can't find image info" in str(capture.value.__cause__)

    def test_can_report_missing_subset_info(self, fxt_dataset, test_dir):
        os.remove(osp.join(test_dir, "train.txt"))

        with pytest.raises(InvalidAnnotationError, match="subset list file"):
            try:
                Dataset.import_from(test_dir, "yolo").init_cache()
            except Exception as e:
                if isinstance(e, DatasetImportError) and e.__cause__:
                    raise e.__cause__
                raise
