# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging
import os
import os.path as osp
import pickle
from typing import Callable, List, Sequence  # nosec B403
from unittest import TestCase, mock

import numpy as np
import pytest

from datumaro.components.annotation import (
    Annotation,
    Annotations,
    AnnotationType,
    Bbox,
    Caption,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    Polygon,
    PolyLine,
)
from datumaro.components.dataset import DEFAULT_FORMAT, Dataset, eager_mode
from datumaro.components.dataset_base import (
    DEFAULT_SUBSET_NAME,
    DatasetBase,
    DatasetItem,
    SubsetBase,
)
from datumaro.components.dataset_item_storage import ItemStatus
from datumaro.components.environment import Environment
from datumaro.components.errors import (
    ConflictingCategoriesError,
    DatasetNotFoundError,
    MediaTypeError,
    MismatchingAttributesError,
    MismatchingImageInfoError,
    MismatchingMediaPathError,
    MultipleFormatsMatchError,
    NoMatchingFormatsError,
    RepeatedItemError,
    UnknownFormatError,
)
from datumaro.components.exporter import Exporter
from datumaro.components.filter import (
    DatasetItemEncoder,
    XPathAnnotationsFilter,
    XPathDatasetFilter,
)
from datumaro.components.importer import FailingImportErrorPolicy, ImportErrorPolicy
from datumaro.components.launcher import Launcher
from datumaro.components.media import Image, MediaElement, Video
from datumaro.components.merge.intersect_merge import IntersectMerge
from datumaro.components.progress_reporting import NullProgressReporter, ProgressReporter
from datumaro.components.registry import DatasetBaseRegistry, ImporterRegistry
from datumaro.components.transformer import ItemTransform, Transform
from datumaro.plugins.transforms import ProjectInfos, RemapLabels

from ..requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir, compare_datasets, compare_datasets_strict


class DatasetTest(TestCase):
    def build_default_environment(
        self,
        importers_override: Callable[[Environment], List] = None,
        extractors_override: Callable[[Environment], List] = None,
    ) -> Environment:
        env = Environment(use_lazy_import=False)
        default_importers = [env.importers[DEFAULT_FORMAT]]
        default_extractors = [env.extractors[DEFAULT_FORMAT]]
        importers = importers_override(env) if importers_override is not None else default_importers
        extractors = (
            extractors_override(env) if extractors_override is not None else default_extractors
        )
        env._importers = ImporterRegistry()
        env._importers.batch_register(importers)
        env._extractors = DatasetBaseRegistry()
        env._extractors.batch_register(extractors)
        return env

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_create_from_extractors(self):
        class SrcExtractor1(DatasetBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id=1,
                            subset="train",
                            annotations=[
                                Bbox(1, 2, 3, 4),
                                Label(4),
                            ],
                        ),
                        DatasetItem(
                            id=1,
                            subset="val",
                            annotations=[
                                Label(4),
                            ],
                        ),
                    ]
                )

        class SrcExtractor2(DatasetBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id=1,
                            subset="val",
                            annotations=[
                                Label(5),
                            ],
                        ),
                    ]
                )

        class DstExtractor(DatasetBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id=1,
                            subset="train",
                            annotations=[
                                Bbox(1, 2, 3, 4),
                                Label(4),
                            ],
                        ),
                        DatasetItem(
                            id=1,
                            subset="val",
                            annotations=[
                                Label(4),
                                Label(5),
                            ],
                        ),
                    ]
                )

        dataset = Dataset.from_extractors(SrcExtractor1(), SrcExtractor2())

        compare_datasets(self, DstExtractor(), dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_create_from_iterable(self):
        class TestExtractor(DatasetBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id=1,
                            subset="train",
                            annotations=[
                                Bbox(1, 2, 3, 4, label=2),
                                Label(4),
                            ],
                        ),
                        DatasetItem(
                            id=1,
                            subset="val",
                            annotations=[
                                Label(3),
                            ],
                        ),
                    ]
                )

            def categories(self):
                return {
                    AnnotationType.label: LabelCategories.from_iterable(["a", "b", "c", "d", "e"])
                }

        actual = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    annotations=[
                        Bbox(1, 2, 3, 4, label=2),
                        Label(4),
                    ],
                ),
                DatasetItem(
                    id=1,
                    subset="val",
                    annotations=[
                        Label(3),
                    ],
                ),
            ],
            categories=["a", "b", "c", "d", "e"],
        )

        compare_datasets(self, TestExtractor(), actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_join_datasets_with_empty_categories(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                        Bbox(1, 2, 3, 4),
                        Caption("hello world"),
                    ],
                )
            ],
            categories=["a"],
        )

        src1 = Dataset.from_iterable(
            [DatasetItem(1, annotations=[Bbox(1, 2, 3, 4, label=None)])],
            categories=[],
        )

        src2 = Dataset.from_iterable(
            [DatasetItem(1, annotations=[Label(0)])],
            categories=["a"],
        )

        src3 = Dataset.from_iterable([DatasetItem(1, annotations=[Caption("hello world")])])

        actual = Dataset.from_extractors(src1, src2, src3)

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, annotations=[Label(2)]),
            ],
            categories=["a", "b", "c"],
        )

        with TestDir() as test_dir:
            source_dataset.save(test_dir)

            loaded_dataset = Dataset.load(test_dir)

            compare_datasets(self, source_dataset, loaded_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        env = self.build_default_environment()
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, annotations=[Label(2)]),
            ],
            categories=["a", "b", "c"],
        )

        with TestDir() as test_dir:
            dataset.save(test_dir)

            detected_format = Dataset.detect(test_dir, env=env)

            self.assertEqual(DEFAULT_FORMAT, detected_format)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_with_nested_folder(self):
        env = self.build_default_environment()

        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, annotations=[Label(2)]),
            ],
            categories=["a", "b", "c"],
        )

        with TestDir() as test_dir:
            dataset_path = osp.join(test_dir, "a")
            dataset.save(dataset_path)

            detected_format = Dataset.detect(test_dir, env=env)

            self.assertEqual(DEFAULT_FORMAT, detected_format)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_with_nested_folder_and_multiply_matches(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1, media=Image.from_numpy(data=np.ones((3, 3, 3))), annotations=[Label(2)]
                ),
            ],
            categories=["a", "b", "c"],
        )

        with TestDir() as test_dir:
            dataset_path = osp.join(test_dir, "a", "b")
            dataset.export(dataset_path, "coco_labels", save_media=True)

            detected_format = Dataset.detect(test_dir, depth=2)

            self.assertEqual("coco_labels", detected_format)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cannot_detect_for_non_existent_path(self):
        with TestDir() as test_dir:
            dataset_path = osp.join(test_dir, "a")

            with self.assertRaises(FileNotFoundError):
                Dataset.detect(dataset_path)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_and_import(self):
        env = self.build_default_environment()

        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, annotations=[Label(2)]),
            ],
            categories=["a", "b", "c"],
        )

        with TestDir() as test_dir:
            source_dataset.save(test_dir)

            imported_dataset = Dataset.import_from(test_dir, env=env)

            self.assertEqual(imported_dataset.data_path, test_dir)
            self.assertEqual(imported_dataset.format, DEFAULT_FORMAT)
            compare_datasets(self, source_dataset, imported_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_report_no_dataset_found(self):
        env = self.build_default_environment()

        with TestDir() as test_dir, self.assertRaises(DatasetNotFoundError):
            Dataset.import_from(test_dir, DEFAULT_FORMAT, env=env)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_report_multiple_formats_match(self):
        env = self.build_default_environment()
        env.importers.register("coco_instances", env.importers[DEFAULT_FORMAT])
        env.extractors.register("coco_instances", env.extractors[DEFAULT_FORMAT])

        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, annotations=[Label(2)]),
            ],
            categories=["a", "b", "c"],
        )

        with TestDir() as test_dir:
            source_dataset.save(test_dir)

            with self.assertRaises(MultipleFormatsMatchError):
                Dataset.import_from(test_dir, env=env)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_report_no_matching_formats(self):
        env = self.build_default_environment(lambda _: [], lambda _: [])

        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, annotations=[Label(2)]),
            ],
            categories=["a", "b", "c"],
        )

        with TestDir() as test_dir:
            source_dataset.save(test_dir)

            with self.assertRaises(NoMatchingFormatsError):
                Dataset.import_from(test_dir, env=env)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_report_unknown_format_requested(self):
        env = self.build_default_environment(lambda _: [], lambda _: [])

        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, annotations=[Label(2)]),
            ],
            categories=["a", "b", "c"],
        )

        with TestDir() as test_dir:
            source_dataset.save(test_dir)

            with self.assertRaises(UnknownFormatError):
                Dataset.import_from(test_dir, format="custom", env=env)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_export_by_string_format_name(self):
        env = Environment()
        env.exporters._items = {"qq": env.exporters[DEFAULT_FORMAT]}

        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, annotations=[Label(2)]),
            ],
            categories=["a", "b", "c"],
            env=env,
        )

        with TestDir() as test_dir:
            dataset.export(format="qq", save_dir=test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_remember_export_options(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, media=Image.from_numpy(data=np.ones((1, 2, 3)))),
            ],
            categories=["a"],
        )

        with TestDir() as test_dir:
            dataset.save(test_dir, save_media=True)
            dataset.put(dataset.get(1))  # mark the item modified for patching

            image_path = osp.join(test_dir, "images", "default", "1.jpg")
            os.remove(image_path)

            dataset.save(test_dir)

            self.assertEqual({"save_media": True}, dataset.options)
            self.assertTrue(osp.isfile(image_path))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_compute_length_when_created_from_scratch(self):
        dataset = Dataset(media_type=MediaElement)

        dataset.put(DatasetItem(1))
        dataset.put(DatasetItem(2))
        dataset.put(DatasetItem(3))
        dataset.remove(1)

        self.assertEqual(2, len(dataset))
        self.assertEqual(2, len(dataset.get_subset(DEFAULT_SUBSET_NAME)))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_compute_length_when_created_from_extractor(self):
        class TestExtractor(DatasetBase):
            def __iter__(self):
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                ]

        dataset = Dataset.from_extractors(TestExtractor())

        self.assertEqual(3, len(dataset))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_compute_length_when_created_from_sequence(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1),
                DatasetItem(2),
                DatasetItem(3),
            ]
        )

        self.assertEqual(3, len(dataset))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_transform_by_string_name(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(id=1, attributes={"qq": 1}),
            ]
        )

        class TestTransform(ItemTransform):
            def transform_item(self, item):
                return self.wrap_item(item, attributes={"qq": 1})

        env = Environment()
        env.transforms.register("qq", TestTransform)

        dataset = Dataset.from_iterable([DatasetItem(id=1)], env=env)

        actual = dataset.transform("qq")

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_transform(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(id=1, attributes={"qq": 1}),
            ]
        )

        class TestTransform(ItemTransform):
            def transform_item(self, item):
                return self.wrap_item(item, attributes={"qq": 1})

        dataset = Dataset.from_iterable([DatasetItem(id=1)])

        actual = dataset.transform(TestTransform)

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_join_annotations(self):
        a = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    annotations=[
                        Label(1, id=3),
                        Label(2, attributes={"x": 1}),
                    ],
                    attributes={"x": 1, "y": 2},
                )
            ],
            categories=["a", "b", "c", "d"],
        )

        b = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    annotations=[
                        Label(2, attributes={"x": 1}),
                        Label(3, id=4),
                    ],
                    attributes={"z": 3, "y": 2},
                )
            ],
            categories=["a", "b", "c", "d"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    annotations=[
                        Label(1, id=3),
                        Label(2, attributes={"x": 1}),
                        Label(3, id=4),
                    ],
                    attributes={"x": 1, "y": 2, "z": 3},
                )
            ],
            categories=["a", "b", "c", "d"],
        )

        merged = Dataset.from_extractors(a, b)

        compare_datasets(self, expected, merged)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_join_different_categories(self):
        s1 = Dataset.from_iterable([], categories=["a", "b"])
        s2 = Dataset.from_iterable([], categories=["b", "a"])

        with self.assertRaises(ConflictingCategoriesError):
            Dataset.from_extractors(s1, s2)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_join_different_image_info(self):
        s1 = Dataset.from_iterable(
            [DatasetItem(1, media=Image.from_file(path="1.png", size=(2, 4)))]
        )
        s2 = Dataset.from_iterable(
            [DatasetItem(1, media=Image.from_file(path="1.png", size=(4, 2)))]
        )

        with self.assertRaises(MismatchingImageInfoError):
            Dataset.from_extractors(s1, s2)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_join_different_images(self):
        s1 = Dataset.from_iterable([DatasetItem(1, media=Image.from_file(path="1.png"))])
        s2 = Dataset.from_iterable([DatasetItem(1, media=Image.from_file(path="2.png"))])

        with self.assertRaises(MismatchingMediaPathError):
            Dataset.from_extractors(s1, s2)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_join_different_attrs(self):
        s1 = Dataset.from_iterable([DatasetItem(1, attributes={"x": 1})])
        s2 = Dataset.from_iterable([DatasetItem(1, attributes={"x": 2})])

        with self.assertRaises(MismatchingAttributesError):
            Dataset.from_extractors(s1, s2)

    @mark_requirement(Requirements.DATUM_GENERIC_MEDIA)
    def test_cant_join_different_media_types(self):
        s1 = Dataset.from_iterable([], media_type=Video)
        s2 = Dataset.from_iterable([], media_type=Image)

        with self.assertRaises(MediaTypeError):
            Dataset.from_extractors(s1, s2)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_join_datasets(self):
        s1 = Dataset.from_iterable([DatasetItem(0), DatasetItem(1)])
        s2 = Dataset.from_iterable([DatasetItem(1), DatasetItem(2)])
        expected = Dataset.from_iterable([DatasetItem(0), DatasetItem(1), DatasetItem(2)])

        actual = Dataset.from_extractors(s1, s2)

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_track_modifications_on_addition(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1),
                DatasetItem(2),
            ]
        )

        self.assertFalse(dataset.is_modified)

        dataset.put(DatasetItem(3, subset="a"))

        self.assertTrue(dataset.is_modified)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_track_modifications_on_removal(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1),
                DatasetItem(2),
            ]
        )

        self.assertFalse(dataset.is_modified)

        dataset.remove(1)

        self.assertTrue(dataset.is_modified)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_create_patch(self):
        expected = Dataset.from_iterable([DatasetItem(2), DatasetItem(3, subset="a")])

        dataset = Dataset.from_iterable(
            [
                DatasetItem(1),
                DatasetItem(2),
            ]
        )
        dataset.put(DatasetItem(2))
        dataset.put(DatasetItem(3, subset="a"))
        dataset.remove(1)

        patch = dataset.get_patch()

        self.assertEqual(
            {
                ("1", DEFAULT_SUBSET_NAME): ItemStatus.removed,
                ("2", DEFAULT_SUBSET_NAME): ItemStatus.added,
                ("3", "a"): ItemStatus.added,
            },
            patch.updated_items,
        )

        self.assertEqual(
            {
                "default": ItemStatus.modified,
                "a": ItemStatus.modified,
            },
            patch.updated_subsets,
        )

        self.assertEqual(2, len(patch.data))
        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(dataset.get(2), patch.data.get(2))
        self.assertEqual(dataset.get(3, "a"), patch.data.get(3, "a"))

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_create_patch_when_cached(self):
        expected = Dataset.from_iterable([DatasetItem(2), DatasetItem(3, subset="a")])

        dataset = Dataset.from_iterable(
            [
                DatasetItem(1),
                DatasetItem(2),
            ]
        )
        dataset.init_cache()
        dataset.put(DatasetItem(2))
        dataset.put(DatasetItem(3, subset="a"))
        dataset.remove(1)

        patch = dataset.get_patch()

        self.assertEqual(
            {
                ("1", DEFAULT_SUBSET_NAME): ItemStatus.removed,
                # Item was not changed from the original one.
                # TODO: add item comparison and remove this line
                ("2", DEFAULT_SUBSET_NAME): ItemStatus.modified,
                ("3", "a"): ItemStatus.added,
            },
            patch.updated_items,
        )

        self.assertEqual(
            {
                "default": ItemStatus.modified,
                "a": ItemStatus.modified,
            },
            patch.updated_subsets,
        )

        self.assertEqual(2, len(patch.data))
        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(dataset.get(2), patch.data.get(2))
        self.assertEqual(dataset.get(3, "a"), patch.data.get(3, "a"))

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_BUG_257)
    def test_can_create_patch_when_transforms_mixed(self):
        expected = Dataset.from_iterable([DatasetItem(2), DatasetItem(3, subset="a")])

        dataset = Dataset.from_iterable(
            [
                DatasetItem(1),
                DatasetItem(2),
            ]
        )

        class Remove1(Transform):
            def __iter__(self):
                for item in self._extractor:
                    if item.id != "1":
                        yield item

        class Add3(Transform):
            def __iter__(self):
                for item in self._extractor:
                    if item.id == "2":
                        yield item
                yield DatasetItem(3, subset="a")

        dataset.transform(Remove1)
        dataset.transform(Add3)

        patch = dataset.get_patch()

        self.assertEqual(
            {
                ("1", DEFAULT_SUBSET_NAME): ItemStatus.removed,
                ("2", DEFAULT_SUBSET_NAME): ItemStatus.modified,
                ("3", "a"): ItemStatus.added,
            },
            patch.updated_items,
        )

        self.assertEqual(
            {
                "default": ItemStatus.modified,
                "a": ItemStatus.modified,
            },
            patch.updated_subsets,
        )

        self.assertEqual(2, len(patch.data))
        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(dataset.get(2), patch.data.get(2))
        self.assertEqual(dataset.get(3, "a"), patch.data.get(3, "a"))

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_BUG_257)
    def test_can_create_patch_when_transforms_chained(self):
        expected = Dataset.from_iterable([DatasetItem(2), DatasetItem(3, subset="a")])

        class TestExtractor(DatasetBase):
            iter_called = 0

            def __iter__(self):
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                ]

                __class__.iter_called += 1

        class Remove1(Transform):
            iter_called = 0

            def __iter__(self):
                for item in self._extractor:
                    if item.id != "1":
                        yield item

                __class__.iter_called += 1

        class Add3(Transform):
            iter_called = 0

            def __iter__(self):
                yield from self._extractor
                yield DatasetItem(3, subset="a")

                __class__.iter_called += 1

        dataset = Dataset.from_extractors(TestExtractor())
        dataset.transform(Remove1)
        dataset.transform(Add3)

        patch = dataset.get_patch()

        self.assertEqual(
            {
                ("1", DEFAULT_SUBSET_NAME): ItemStatus.removed,
                ("2", DEFAULT_SUBSET_NAME): ItemStatus.modified,
                ("3", "a"): ItemStatus.added,
            },
            patch.updated_items,
        )

        self.assertEqual(
            {
                "default": ItemStatus.modified,
                "a": ItemStatus.modified,
            },
            patch.updated_subsets,
        )

        self.assertEqual(2, len(patch.data))
        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(dataset.get(2), patch.data.get(2))
        self.assertEqual(dataset.get(3, "a"), patch.data.get(3, "a"))

        self.assertEqual(TestExtractor.iter_called, 2)  # 1 for items, 1 for list
        self.assertEqual(Remove1.iter_called, 1)
        self.assertEqual(Add3.iter_called, 1)

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_BUG_257)
    def test_can_create_patch_when_transforms_intermixed_with_direct_ops(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(3, subset="a"),
                DatasetItem(4),
                DatasetItem(5),
            ],
        )

        class TestExtractor(DatasetBase):
            iter_called = 0

            def __iter__(self):
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                ]

                __class__.iter_called += 1

        class Remove1(Transform):
            iter_called = 0

            def __iter__(self):
                for item in self._extractor:
                    if item.id != "1":
                        yield item

                __class__.iter_called += 1

        class Add3(Transform):
            iter_called = 0

            def __iter__(self):
                yield from self._extractor
                yield DatasetItem(3, subset="a")

                __class__.iter_called += 1

        dataset = Dataset.from_extractors(TestExtractor())
        dataset.init_cache()
        dataset.put(DatasetItem(4))
        dataset.transform(Remove1)
        dataset.put(DatasetItem(5))
        dataset.remove(2)
        dataset.transform(Add3)

        patch = dataset.get_patch()

        self.assertEqual(
            {
                ("1", DEFAULT_SUBSET_NAME): ItemStatus.removed,
                ("2", DEFAULT_SUBSET_NAME): ItemStatus.removed,
                ("3", "a"): ItemStatus.added,
                ("4", DEFAULT_SUBSET_NAME): ItemStatus.added,
                ("5", DEFAULT_SUBSET_NAME): ItemStatus.added,
            },
            patch.updated_items,
        )

        self.assertEqual(
            {
                "default": ItemStatus.modified,
                "a": ItemStatus.modified,
            },
            patch.updated_subsets,
        )

        self.assertEqual(3, len(patch.data))

        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(None, patch.data.get(2))
        self.assertEqual(dataset.get(3, "a"), patch.data.get(3, "a"))
        self.assertEqual(dataset.get(4), patch.data.get(4))
        self.assertEqual(dataset.get(5), patch.data.get(5))

        self.assertEqual(TestExtractor.iter_called, 1)
        self.assertEqual(Remove1.iter_called, 1)
        self.assertEqual(Add3.iter_called, 1)

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_BUG_257)
    def test_can_create_patch_when_local_transforms_stacked(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(4),
                DatasetItem(5),
            ],
        )

        class TestExtractor(DatasetBase):
            iter_called = 0

            def __iter__(self):
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                ]

                __class__.iter_called += 1

        class ShiftIds(ItemTransform):
            def transform_item(self, item):
                return item.wrap(id=int(item.id) + 1)

        dataset = Dataset.from_extractors(TestExtractor())
        dataset.remove(2)
        dataset.transform(ShiftIds)
        dataset.transform(ShiftIds)
        dataset.transform(ShiftIds)
        dataset.put(DatasetItem(5))

        patch = dataset.get_patch()

        self.assertEqual(
            {
                ("1", DEFAULT_SUBSET_NAME): ItemStatus.removed,
                ("2", DEFAULT_SUBSET_NAME): ItemStatus.removed,
                ("4", DEFAULT_SUBSET_NAME): ItemStatus.added,
                ("5", DEFAULT_SUBSET_NAME): ItemStatus.added,
            },
            patch.updated_items,
        )

        self.assertEqual(
            {
                "default": ItemStatus.modified,
            },
            patch.updated_subsets,
        )

        self.assertEqual(2, len(patch.data))

        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(None, patch.data.get(2))
        self.assertEqual(None, patch.data.get(3))
        self.assertEqual(dataset.get(4), patch.data.get(4))
        self.assertEqual(dataset.get(5), patch.data.get(5))

        self.assertEqual(TestExtractor.iter_called, 1)

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_BUG_257)
    def test_can_create_patch_when_transforms_chained_and_source_cached(self):
        expected = Dataset.from_iterable([DatasetItem(2), DatasetItem(3, subset="a")])

        class TestExtractor(DatasetBase):
            iter_called = 0

            def __iter__(self):
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                ]

                __class__.iter_called += 1

        class Remove1(Transform):
            iter_called = 0

            def __iter__(self):
                for item in self._extractor:
                    if item.id != "1":
                        yield item

                __class__.iter_called += 1

        class Add3(Transform):
            iter_called = 0

            def __iter__(self):
                yield from self._extractor
                yield DatasetItem(3, subset="a")

                __class__.iter_called += 1

        dataset = Dataset.from_extractors(TestExtractor())
        dataset.init_cache()
        dataset.transform(Remove1)
        dataset.transform(Add3)

        patch = dataset.get_patch()

        self.assertEqual(
            {
                ("1", DEFAULT_SUBSET_NAME): ItemStatus.removed,
                ("2", DEFAULT_SUBSET_NAME): ItemStatus.modified,  # TODO: remove this
                ("3", "a"): ItemStatus.added,
            },
            patch.updated_items,
        )

        self.assertEqual(
            {
                "default": ItemStatus.modified,
                "a": ItemStatus.modified,
            },
            patch.updated_subsets,
        )

        self.assertEqual(2, len(patch.data))
        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(dataset.get(2), patch.data.get(2))
        self.assertEqual(dataset.get(3, "a"), patch.data.get(3, "a"))

        self.assertEqual(TestExtractor.iter_called, 1)  # 1 for items and list
        self.assertEqual(Remove1.iter_called, 1)
        self.assertEqual(Add3.iter_called, 1)

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_do_lazy_put_and_remove(self):
        iter_called = False

        class TestExtractor(DatasetBase):
            def __iter__(self):
                nonlocal iter_called
                iter_called = True
                return iter(
                    [
                        DatasetItem(1),
                        DatasetItem(2),
                    ]
                )

        dataset = Dataset.from_extractors(TestExtractor())

        self.assertFalse(dataset.is_cache_initialized)

        dataset.put(DatasetItem(3))
        dataset.remove(DatasetItem(1))

        self.assertFalse(dataset.is_cache_initialized)
        self.assertFalse(iter_called)

        dataset.init_cache()

        self.assertTrue(dataset.is_cache_initialized)
        self.assertTrue(iter_called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_put(self):
        dataset = Dataset(media_type=MediaElement)

        dataset.put(DatasetItem(1))

        self.assertTrue((1, "") in dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_do_lazy_get_on_updated_item(self):
        iter_called = False

        class TestExtractor(DatasetBase):
            def __iter__(self):
                nonlocal iter_called
                iter_called = True
                return iter(
                    [
                        DatasetItem(1),
                        DatasetItem(2),
                    ]
                )

        dataset = Dataset.from_extractors(TestExtractor())

        dataset.put(DatasetItem(2))

        self.assertTrue((2, "") in dataset)
        self.assertFalse(iter_called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_switch_eager_and_lazy_with_cm_global(self):
        iter_called = False

        class TestExtractor(DatasetBase):
            def __iter__(self):
                nonlocal iter_called
                iter_called = True
                return iter(
                    [
                        DatasetItem(1),
                        DatasetItem(2),
                    ]
                )

        with eager_mode():
            Dataset.from_extractors(TestExtractor())

        self.assertTrue(iter_called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_switch_eager_and_lazy_with_cm_local(self):
        iter_called = False

        class TestExtractor(DatasetBase):
            def __iter__(self):
                nonlocal iter_called
                iter_called = True
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                    DatasetItem(4),
                ]

        dataset = Dataset.from_extractors(TestExtractor())

        with eager_mode(dataset=dataset):
            dataset.select(lambda item: int(item.id) < 3)
            dataset.select(lambda item: int(item.id) < 2)

        self.assertTrue(iter_called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_do_lazy_select(self):
        iter_called = 0

        class TestExtractor(DatasetBase):
            def __iter__(self):
                nonlocal iter_called
                iter_called += 1
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                    DatasetItem(4),
                ]

        dataset = Dataset.from_extractors(TestExtractor())

        dataset.select(lambda item: int(item.id) < 3)
        dataset.select(lambda item: int(item.id) < 2)

        self.assertEqual(iter_called, 0)

        self.assertEqual(1, len(dataset))

        self.assertEqual(iter_called, 1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_chain_lazy_transforms(self):
        iter_called = 0

        class TestExtractor(DatasetBase):
            def __iter__(self):
                nonlocal iter_called
                iter_called += 1
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                    DatasetItem(4),
                ]

        dataset = Dataset.from_extractors(TestExtractor())

        class TestTransform(ItemTransform):
            def transform_item(self, item):
                return self.wrap_item(item, id=int(item.id) + 1)

        dataset.transform(TestTransform)
        dataset.transform(TestTransform)

        self.assertEqual(iter_called, 0)

        self.assertEqual(4, len(dataset))
        self.assertEqual(3, int(min(int(item.id) for item in dataset)))

        self.assertEqual(iter_called, 1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_get_len_after_local_transforms(self):
        iter_called = 0

        class TestExtractor(DatasetBase):
            def __iter__(self):
                nonlocal iter_called
                iter_called += 1
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                    DatasetItem(4),
                ]

        dataset = Dataset.from_extractors(TestExtractor())

        class TestTransform(ItemTransform):
            def transform_item(self, item):
                return self.wrap_item(item, id=int(item.id) + 1)

        dataset.transform(TestTransform)
        dataset.transform(TestTransform)

        self.assertEqual(iter_called, 0)

        self.assertEqual(4, len(dataset))

        self.assertEqual(iter_called, 1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_get_len_after_nonlocal_transforms(self):
        iter_called = 0

        class TestExtractor(DatasetBase):
            def __iter__(self):
                nonlocal iter_called
                iter_called += 1
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                    DatasetItem(4),
                ]

        dataset = Dataset.from_extractors(TestExtractor())

        class TestTransform(Transform):
            def __iter__(self):
                for item in self._extractor:
                    yield self.wrap_item(item, id=int(item.id) + 1)

        dataset.transform(TestTransform)
        dataset.transform(TestTransform)

        self.assertEqual(iter_called, 0)

        self.assertEqual(4, len(dataset))

        self.assertEqual(iter_called, 2)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_get_subsets_after_local_transforms(self):
        iter_called = 0

        class TestExtractor(DatasetBase):
            def __iter__(self):
                nonlocal iter_called
                iter_called += 1
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                    DatasetItem(4),
                ]

        dataset = Dataset.from_extractors(TestExtractor())

        class TestTransform(ItemTransform):
            def transform_item(self, item):
                return self.wrap_item(item, id=int(item.id) + 1, subset="a")

        dataset.transform(TestTransform)
        dataset.transform(TestTransform)

        self.assertEqual(iter_called, 0)

        self.assertEqual({"a"}, set(dataset.subsets()))

        self.assertEqual(iter_called, 1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_get_subsets_after_nonlocal_transforms(self):
        iter_called = 0

        class TestExtractor(DatasetBase):
            def __iter__(self):
                nonlocal iter_called
                iter_called += 1
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                    DatasetItem(4),
                ]

        dataset = Dataset.from_extractors(TestExtractor())

        class TestTransform(Transform):
            def __iter__(self):
                for item in self._extractor:
                    yield self.wrap_item(item, id=int(item.id) + 1, subset="a")

        dataset.transform(TestTransform)
        dataset.transform(TestTransform)

        self.assertEqual(iter_called, 0)

        self.assertEqual({"a"}, set(dataset.subsets()))

        self.assertEqual(iter_called, 2)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_raises_when_repeated_items_in_source(self):
        dataset = Dataset.from_iterable([DatasetItem(0), DatasetItem(0)])

        with self.assertRaises(RepeatedItemError):
            dataset.init_cache()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_check_item_existence(self):
        dataset = Dataset.from_iterable([DatasetItem(0, subset="a"), DatasetItem(1)])

        self.assertTrue(DatasetItem(0, subset="a") in dataset)
        self.assertFalse(DatasetItem(0, subset="b") in dataset)
        self.assertTrue((0, "a") in dataset)
        self.assertFalse((0, "b") in dataset)
        self.assertTrue(1 in dataset)
        self.assertFalse(0 in dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_put_with_id_override(self):
        dataset = Dataset.from_iterable([])

        dataset.put(DatasetItem(0, subset="a"), id=2, subset="b")

        self.assertTrue((2, "b") in dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_compute_cache_with_empty_source(self):
        dataset = Dataset.from_iterable([])
        dataset.put(DatasetItem(2))

        dataset.init_cache()

        self.assertTrue(2 in dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_do_partial_caching_in_get_when_default(self):
        iter_called = 0

        class TestExtractor(DatasetBase):
            def __iter__(self):
                nonlocal iter_called
                iter_called += 1
                return iter(
                    [
                        DatasetItem(1),
                        DatasetItem(2),
                        DatasetItem(3),
                        DatasetItem(4),
                    ]
                )

        dataset = Dataset.from_extractors(TestExtractor())

        dataset.get(3)
        dataset.get(4)

        self.assertEqual(1, iter_called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_do_partial_caching_in_get_when_redefined(self):
        iter_called = 0
        get_called = 0

        class TestExtractor(DatasetBase):
            def __iter__(self):
                nonlocal iter_called
                iter_called += 1
                return iter(
                    [
                        DatasetItem(1),
                        DatasetItem(2),
                        DatasetItem(3),
                        DatasetItem(4),
                    ]
                )

            def get(self, id, subset=None):
                nonlocal get_called
                get_called += 1
                return DatasetItem(id, subset=subset)

        dataset = Dataset.from_extractors(TestExtractor())

        dataset.get(3)
        dataset.get(4)

        self.assertEqual(0, iter_called)
        self.assertEqual(2, get_called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_binds_on_save(self):
        dataset = Dataset.from_iterable([DatasetItem(1)])

        self.assertFalse(dataset.is_bound)

        with TestDir() as test_dir:
            dataset.save(test_dir)

            self.assertTrue(dataset.is_bound)
            self.assertEqual(dataset.data_path, test_dir)
            self.assertEqual(dataset.format, DEFAULT_FORMAT)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_flushes_changes_on_save(self):
        dataset = Dataset.from_iterable([])
        dataset.put(DatasetItem(1))

        self.assertTrue(dataset.is_modified)

        with TestDir() as test_dir:
            dataset.save(test_dir)

            self.assertFalse(dataset.is_modified)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_does_not_load_images_on_saving(self):
        # Issue https://github.com/open-edge-platform/datumaro/issues/177
        # Missing image metadata (size etc.) can lead to image loading on
        # dataset save without image saving

        called = False

        def test_loader():
            nonlocal called
            called = True

        dataset = Dataset.from_iterable([DatasetItem(1, media=Image.from_numpy(data=test_loader))])

        with TestDir() as test_dir:
            dataset.save(test_dir)

        self.assertFalse(called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_transform_labels(self):
        expected = Dataset.from_iterable([], categories=["c", "b"])
        dataset = Dataset.from_iterable([], categories=["a", "b"])

        actual = dataset.transform("remap_labels", mapping={"a": "c"})

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_run_model(self):
        dataset = Dataset.from_iterable(
            [DatasetItem(i, media=Image.from_numpy(data=np.ones((i, i, 3)))) for i in range(5)],
            categories=["label"],
        )

        batch_size = 3

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    i,
                    media=Image.from_numpy(data=np.ones((i, i, 3))),
                    annotations=[Label(0, attributes={"idx": i % batch_size, "data": i})],
                )
                for i in range(5)
            ],
            categories=["label"],
        )

        calls = 0

        class TestLauncher(Launcher):
            def launch(self, batch: Sequence[DatasetItem]) -> List[List[Annotation]]:
                nonlocal calls
                calls += 1

                return [
                    [Label(0, attributes={"idx": i, "data": inp.media.data.shape[0]})]
                    for i, inp in enumerate(batch)
                ]

        model = TestLauncher()

        actual = dataset.run_model(model, batch_size=batch_size)

        compare_datasets(self, expected, actual, require_media=True)
        self.assertEqual(2, calls)

    @mark_requirement(Requirements.DATUM_BUG_259)
    def test_can_filter_items(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=0, subset="train"),
                DatasetItem(id=1, subset="test"),
            ]
        )

        dataset.filter("/item[id > 0]")

        self.assertEqual(1, len(dataset))

    @mark_requirement(Requirements.DATUM_BUG_257)
    def test_filter_registers_changes(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=0, subset="train"),
                DatasetItem(id=1, subset="test"),
            ]
        )

        dataset.filter("/item[id > 0]")

        self.assertEqual(
            {
                ("0", "train"): ItemStatus.removed,
                ("1", "test"): ItemStatus.modified,  # TODO: remove this line
            },
            dataset.get_patch().updated_items,
        )

    @mark_requirement(Requirements.DATUM_BUG_259)
    def test_can_filter_annotations(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=0, subset="train", annotations=[Label(0), Label(1)]),
                DatasetItem(id=1, subset="val", annotations=[Label(2)]),
                DatasetItem(id=2, subset="test", annotations=[Label(0), Label(2)]),
            ],
            categories=["a", "b", "c"],
        )

        dataset.filter('/item/annotation[label = "c"]', filter_annotations=True, remove_empty=True)

        self.assertEqual(2, len(dataset))

    @mark_requirement(Requirements.DATUM_BUG_259)
    def test_can_filter_items_in_merged_dataset(self):
        dataset = Dataset.from_extractors(
            Dataset.from_iterable([DatasetItem(id=0, subset="train")]),
            Dataset.from_iterable([DatasetItem(id=1, subset="test")]),
        )

        dataset.filter("/item[id > 0]")

        self.assertEqual(1, len(dataset))

    @mark_requirement(Requirements.DATUM_BUG_259)
    def test_can_filter_annotations_in_merged_dataset(self):
        dataset = Dataset.from_extractors(
            Dataset.from_iterable(
                [
                    DatasetItem(id=0, subset="train", annotations=[Label(0)]),
                ],
                categories=["a", "b", "c"],
            ),
            Dataset.from_iterable(
                [
                    DatasetItem(id=1, subset="val", annotations=[Label(1)]),
                ],
                categories=["a", "b", "c"],
            ),
            Dataset.from_iterable(
                [
                    DatasetItem(id=2, subset="test", annotations=[Label(2)]),
                ],
                categories=["a", "b", "c"],
            ),
        )

        dataset.filter('/item/annotation[label = "c"]', filter_annotations=True, remove_empty=True)

        self.assertEqual(1, len(dataset))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        class CustomExporter(Exporter):
            DEFAULT_IMAGE_EXT = ".jpg"

            def _apply_impl(self):
                assert osp.isdir(self._save_dir)

                for item in self._extractor:
                    name = f"{item.subset}_{item.id}"
                    with open(osp.join(self._save_dir, name + ".txt"), "w") as f:
                        f.write("\n")

                    if self._save_media and item.media and item.media.has_data:
                        self._save_image(item, name=name)

        env = Environment()
        env.exporters._items = {"test": CustomExporter}

        with TestDir() as path:
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(1, subset="train", media=Image.from_numpy(data=np.ones((2, 4, 3)))),
                    DatasetItem(
                        2, subset="train", media=Image.from_file(path="2.jpg", size=(3, 2))
                    ),
                    DatasetItem(3, subset="valid", media=Image.from_numpy(data=np.ones((2, 2, 3)))),
                ],
                categories=[],
                env=env,
            )
            dataset.export(path, "test", save_media=True)

            dataset.put(
                DatasetItem(2, subset="train", media=Image.from_numpy(data=np.ones((3, 2, 3))))
            )
            dataset.remove(3, "valid")
            dataset.save(save_media=True)

            self.assertEqual(
                {"train_1.txt", "train_1.jpg", "train_2.txt", "train_2.jpg"}, set(os.listdir(path))
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_update_overwrites_matching_items(self):
        patch = Dataset.from_iterable(
            [DatasetItem(id=1, annotations=[Bbox(1, 2, 3, 4, label=1)])],
            categories=["a", "b"],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, annotations=[Bbox(2, 2, 1, 1, label=0)]),
                DatasetItem(id=2, annotations=[Bbox(1, 1, 1, 1, label=1)]),
            ],
            categories=["a", "b"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(id=1, annotations=[Bbox(1, 2, 3, 4, label=1)]),
                DatasetItem(id=2, annotations=[Bbox(1, 1, 1, 1, label=1)]),
            ],
            categories=["a", "b"],
        )

        dataset.update(patch)

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_update_can_reorder_labels(self):
        patch = Dataset.from_iterable(
            [DatasetItem(id=1, annotations=[Bbox(1, 2, 3, 4, label=1)])],
            categories=["b", "a"],
        )

        dataset = Dataset.from_iterable(
            [DatasetItem(id=1, annotations=[Bbox(2, 2, 1, 1, label=0)])],
            categories=["a", "b"],
        )

        # Note that label id and categories are changed
        expected = Dataset.from_iterable(
            [DatasetItem(id=1, annotations=[Bbox(1, 2, 3, 4, label=0)])],
            categories=["a", "b"],
        )

        dataset.update(patch)

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_update_can_project_labels(self):
        dataset = Dataset.from_iterable(
            [
                # Must be overridden
                DatasetItem(
                    id=100,
                    annotations=[
                        Bbox(1, 2, 3, 3, label=0),
                    ],
                ),
                # Must be kept
                DatasetItem(id=1, annotations=[Bbox(1, 2, 3, 4, label=1)]),
            ],
            categories=["a", "b"],
        )

        patch = Dataset.from_iterable(
            [
                # Must override
                DatasetItem(
                    id=100,
                    annotations=[
                        Bbox(1, 2, 3, 4, label=0),  # Label must be remapped
                        Bbox(5, 6, 2, 3, label=1),  # Label must be remapped
                        Bbox(2, 2, 2, 3, label=2),  # Will be dropped due to label
                    ],
                ),
                # Must be added
                DatasetItem(
                    id=2, annotations=[Bbox(1, 2, 3, 2, label=1)]
                ),  # Label must be remapped
            ],
            categories=["b", "a", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    annotations=[
                        Bbox(1, 2, 3, 4, label=1),
                        Bbox(5, 6, 2, 3, label=0),
                    ],
                ),
                DatasetItem(id=1, annotations=[Bbox(1, 2, 3, 4, label=1)]),
                DatasetItem(id=2, annotations=[Bbox(1, 2, 3, 2, label=0)]),
            ],
            categories=["a", "b"],
        )

        dataset.update(patch)

        compare_datasets(self, expected, dataset, ignored_attrs="*")

    @mark_requirement(Requirements.DATUM_PROGRESS_REPORTING)
    def test_progress_reporter_implies_eager_mode(self):
        class TestExtractor(SubsetBase):
            def __init__(self, url, **kwargs):
                super().__init__(**kwargs)

            def __iter__(self):
                yield DatasetItem("1")

        env = Environment()
        env.importers._items.clear()
        env.extractors._items["test"] = TestExtractor

        dataset = Dataset.import_from("", "test", env=env, progress_reporter=NullProgressReporter())

        self.assertTrue(dataset.is_cache_initialized)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_error_reporter_implies_eager_mode(self):
        class TestExtractor(SubsetBase):
            def __init__(self, url, **kwargs):
                super().__init__(**kwargs)

            def __iter__(self):
                yield DatasetItem("1")

        env = Environment()
        env.importers._items.clear()
        env.extractors._items["test"] = TestExtractor

        dataset = Dataset.import_from("", "test", env=env, error_policy=FailingImportErrorPolicy())

        self.assertTrue(dataset.is_cache_initialized)

    @mark_requirement(Requirements.DATUM_PROGRESS_REPORTING)
    def test_can_report_progress_from_extractor(self):
        class TestExtractor(SubsetBase):
            def __init__(self, url, **kwargs):
                super().__init__(**kwargs)

            def __iter__(self):
                list(self._ctx.progress_reporter.iter([None] * 5, desc="loading images"))
                yield from []

        class TestProgressReporter(ProgressReporter):
            def split(self, count):
                return (self,) * count

        progress_reporter = TestProgressReporter()
        period_mock = mock.PropertyMock(return_value=0.1)
        type(progress_reporter).period = period_mock
        progress_reporter.start = mock.MagicMock()
        progress_reporter.report_status = mock.MagicMock()
        progress_reporter.finish = mock.MagicMock()

        env = Environment()
        env.importers._items.clear()
        env.extractors._items["test"] = TestExtractor

        Dataset.import_from("", "test", env=env, progress_reporter=progress_reporter)

        period_mock.assert_called_once()
        progress_reporter.start.assert_called_once()
        progress_reporter.report_status.assert_called()
        progress_reporter.finish.assert_called_once()

    @mark_requirement(Requirements.DATUM_PROGRESS_REPORTING)
    def test_can_report_progress_from_extractor_multiple_pbars(self):
        class TestExtractor(SubsetBase):
            def __init__(self, url, **kwargs):
                super().__init__(**kwargs, media_type=MediaElement)

            def __iter__(self):
                pbars = self._ctx.progress_reporter.split(2)
                list(pbars[0].iter([None] * 5))
                list(pbars[1].iter([None] * 5))

                yield from []

        class TestProgressReporter(ProgressReporter):
            def init(self, *args, **kwargs):
                pass

            def split(self, count):
                return tuple(TestProgressReporter() for _ in range(count))

            iter = mock.MagicMock()

        progress_reporter = TestProgressReporter()
        progress_reporter.split = mock.MagicMock(
            wraps=lambda count: TestProgressReporter.split(progress_reporter, count)
        )

        env = Environment()
        env.importers._items.clear()
        env.extractors._items["test"] = TestExtractor

        Dataset.import_from("", "test", env=env, progress_reporter=progress_reporter)

        progress_reporter.split.assert_called_once()

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_errors_from_extractor(self):
        class TestExtractor(SubsetBase):
            def __init__(self, url, **kwargs):
                super().__init__(**kwargs, media_type=MediaElement)

            def __iter__(self):
                class TestError(Exception):
                    pass

                self._ctx.error_policy.report_item_error(TestError(), item_id=("0", "a"))
                self._ctx.error_policy.report_annotation_error(TestError(), item_id=("0", "a"))
                yield from []

        env = Environment()
        env.importers._items.clear()
        env.extractors._items["test"] = TestExtractor

        class TestErrorPolicy(ImportErrorPolicy):
            pass

        error_policy = TestErrorPolicy()
        error_policy.report_item_error = mock.MagicMock()
        error_policy.report_annotation_error = mock.MagicMock()

        Dataset.import_from("", "test", env=env, error_policy=error_policy)

        error_policy.report_item_error.assert_called()
        error_policy.report_annotation_error.assert_called()

    @mark_requirement(Requirements.DATUM_PROGRESS_REPORTING)
    def test_can_report_progress_from_exporter(self):
        class TestExporter(Exporter):
            DEFAULT_IMAGE_EXT = ".jpg"

            def _apply_impl(self):
                list(self._ctx.progress_reporter.iter([None] * 5, desc="loading images"))

        class TestProgressReporter(ProgressReporter):
            pass

        progress_reporter = TestProgressReporter()
        period_mock = mock.PropertyMock(return_value=0.1)
        type(progress_reporter).period = period_mock
        progress_reporter.start = mock.MagicMock()
        progress_reporter.report_status = mock.MagicMock()
        progress_reporter.finish = mock.MagicMock()

        with TestDir() as test_dir:
            Dataset(media_type=MediaElement).export(
                test_dir, TestExporter, progress_reporter=progress_reporter
            )

        period_mock.assert_called_once()
        progress_reporter.start.assert_called_once()
        progress_reporter.report_status.assert_called()
        progress_reporter.finish.assert_called()

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_errors_from_exporter(self):
        class TestExporter(Exporter):
            DEFAULT_IMAGE_EXT = ".jpg"

            def _apply_impl(self):
                class TestError(Exception):
                    pass

                self._ctx.error_policy.report_item_error(TestError(), item_id=("0", "a"))
                self._ctx.error_policy.report_annotation_error(TestError(), item_id=("0", "a"))

        class TestErrorPolicy(ImportErrorPolicy):
            pass

        error_policy = TestErrorPolicy()
        error_policy.report_item_error = mock.MagicMock()
        error_policy.report_annotation_error = mock.MagicMock()

        with TestDir() as test_dir:
            Dataset(media_type=MediaElement).export(
                test_dir, TestExporter, error_policy=error_policy
            )

        error_policy.report_item_error.assert_called()
        error_policy.report_annotation_error.assert_called()

    @mark_requirement(Requirements.DATUM_673)
    def test_can_pickle(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="subset",
                    media=Image.from_numpy(data=np.ones((5, 4, 3))),
                    annotations=[
                        Label(0, attributes={"a1": 1, "a2": "2"}, id=1, group=2),
                        Caption("hello", id=1, group=5),
                        Label(2, id=3, group=2, attributes={"x": 1, "y": "2"}),
                        Bbox(1, 2, 3, 4, label=4, id=4, attributes={"a": 1.0}),
                        Points([1, 2, 2, 0, 1, 1], label=0, id=5, group=6),
                        Mask(label=3, id=5, image=np.ones((2, 3))),
                        PolyLine([1, 2, 3, 4, 5, 6, 7, 8], id=11),
                        Polygon([1, 2, 3, 4, 5, 6, 7, 8]),
                    ],
                )
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a", "b"]),
                AnnotationType.mask: MaskCategories.generate(2),
            },
        )
        source.init_cache()

        parsed = pickle.loads(pickle.dumps(source))  # nosec

        compare_datasets_strict(self, source, parsed)

    @mark_requirement(Requirements.DATUM_GENERIC_MEDIA)
    def test_can_specify_media_type_in_ctor(self):
        dataset = Dataset.from_iterable(
            [DatasetItem(id=1, media=Image.from_numpy(data=np.ones((5, 4, 3))))], media_type=Video
        )

        self.assertTrue(dataset.media_type() is Video)

    @mark_requirement(Requirements.DATUM_GENERIC_MEDIA)
    def test_cant_put_item_with_mismatching_media_type(self):
        dataset = Dataset(media_type=Video)

        with self.assertRaises(MediaTypeError):
            dataset.put(DatasetItem(id=1, media=Image.from_numpy(data=np.ones((5, 4, 3)))))

    @mark_requirement(Requirements.DATUM_GENERIC_MEDIA)
    def test_cant_change_media_type_with_transform(self):
        class TestTransform(Transform):
            def media_type(self):
                return Image

        dataset = Dataset(media_type=Video)

        with self.assertRaises(MediaTypeError):
            dataset.transform(TestTransform)
            dataset.init_cache()

    @mark_requirement(Requirements.DATUM_GENERIC_MEDIA)
    def test_can_get_media_type_from_extractor(self):
        class TestExtractor(DatasetBase):
            def __init__(self, **kwargs):
                super().__init__(media_type=Video, **kwargs)

        dataset = Dataset.from_extractors(TestExtractor())

        self.assertTrue(dataset.media_type() is Video)

    @mark_requirement(Requirements.DATUM_GENERIC_MEDIA)
    def test_can_check_media_type_on_caching(self):
        dataset = Dataset.from_iterable(
            [DatasetItem(id=1, media=Image.from_numpy(data=np.ones((5, 4, 3))))], media_type=Video
        )

        with self.assertRaises(MediaTypeError):
            dataset.init_cache()

    @mark_requirement(Requirements.DATUM_GENERIC_MEDIA)
    def test_get_label_cat_names(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 6, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=1),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )
        self.assertEqual(dataset.get_label_cat_names(), ["a", "b", "c"])

    def test_index_access(self):
        dataset = Dataset.from_iterable(DatasetItem(id) for id in range(5))
        self.assertEqual(dataset[2].id, "2")

        dataset.remove(2)
        self.assertEqual(dataset[2].id, "3")

        dataset.put(DatasetItem(2))
        self.assertEqual(dataset[4].id, "2")
        self.assertRaises(IndexError, lambda: dataset[len(dataset)])

    def test_index_access_tile(self):
        dataset = Dataset.from_iterable(
            DatasetItem(
                id,
                media=Image.from_numpy(data=np.ones((224, 224, 3))),
                annotations=[
                    Bbox(1, 2, 3, 4, label=1),
                ],
            )
            for id in range(5)
        )
        length = len(dataset)
        n_rows, n_cols = (4, 4)
        dataset.transform("tile", grid_size=(n_rows, n_cols), overlap=(0, 0), threshold_drop_ann=0)
        tiled_length = length * n_rows * n_cols
        for i in range(length):
            for row in range(n_rows):
                for col in range(n_cols):
                    idx = i * n_rows * n_cols + row * n_cols + col
                    _id = f"{i}_tile_{row * n_cols + col}"
                    self.assertEqual(dataset[idx].id, _id)
        self.assertRaises(IndexError, lambda: dataset[tiled_length])

        dataset.transform("merge_tile")
        for i in range(length):
            self.assertEqual(dataset[i].id, str(i))
        self.assertRaises(IndexError, lambda: dataset[length])


class DatasetItemTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_ctor_requires_id(self):
        with pytest.raises(Exception):
            # pylint: disable=no-value-for-parameter
            DatasetItem()
            # pylint: enable=no-value-for-parameter

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"id": 0, "media": None},
            {"id": 0, "media": Image.from_file(path="path.jpg"), "annotations": []},
            {
                "id": 0,
                "media": Image.from_numpy(data=np.array([1, 2, 3])),
                "annotations": Annotations(),
            },
            {
                "id": 0,
                "media": Image.from_numpy(data=lambda f: np.array([1, 2, 3])),
                "annotations": [Label(label=0), Label(label=1)],
            },
            {
                "id": 0,
                "media": Image.from_numpy(data=np.array([1, 2, 3])),
                "annotations": [Label(label=0), Label(label=1)],
            },
        ],
    )
    def test_ctors_with_image(self, kwargs):
        item = DatasetItem(**kwargs)
        assert isinstance(item.annotations, Annotations)
        assert item.annotations == kwargs.get("annotations", [])


class DatasetFilterTest(TestCase):
    @staticmethod
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_item_representations():
        item = DatasetItem(
            id=1,
            subset="subset",
            media=Image.from_numpy(data=np.ones((5, 4, 3))),
            annotations=[
                Label(0, attributes={"a1": 1, "a2": "2"}, id=1, group=2),
                Caption("hello", id=1),
                Caption("world", group=5),
                Label(2, id=3, attributes={"x": 1, "y": "2"}),
                Bbox(1, 2, 3, 4, label=4, id=4, attributes={"a": 1.0}),
                Bbox(5, 6, 7, 8, id=5, group=5),
                Points([1, 2, 2, 0, 1, 1], label=0, id=5),
                Mask(id=5, image=np.ones((3, 2))),
                Mask(label=3, id=5, image=np.ones((2, 3))),
                PolyLine([1, 2, 3, 4, 5, 6, 7, 8], id=11),
                Polygon([1, 2, 3, 4, 5, 6, 7, 8]),
            ],
        )

        encoded = DatasetItemEncoder.encode(item)
        DatasetItemEncoder.to_string(encoded)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_item_filter_can_be_applied(self):
        class TestExtractor(DatasetBase):
            def __iter__(self):
                for i in range(4):
                    yield DatasetItem(id=i, subset="train")

        extractor = TestExtractor()

        filtered = XPathDatasetFilter(extractor, "/item[id > 1]")

        self.assertEqual(2, len(filtered))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_annotations_filter_can_be_applied(self):
        class SrcExtractor(DatasetBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(id=0),
                        DatasetItem(
                            id=1,
                            annotations=[
                                Label(0),
                                Label(1),
                            ],
                        ),
                        DatasetItem(
                            id=2,
                            annotations=[
                                Label(0),
                                Label(2),
                            ],
                        ),
                    ]
                )

        class DstExtractor(DatasetBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(id=0),
                        DatasetItem(
                            id=1,
                            annotations=[
                                Label(0),
                            ],
                        ),
                        DatasetItem(
                            id=2,
                            annotations=[
                                Label(0),
                            ],
                        ),
                    ]
                )

        extractor = SrcExtractor()

        filtered = XPathAnnotationsFilter(extractor, "/item/annotation[label_id = 0]")

        self.assertListEqual(list(filtered), list(DstExtractor()))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_annotations_filter_can_remove_empty_items(self):
        source = Dataset.from_iterable(
            [
                DatasetItem(id=0),
                DatasetItem(
                    id=1,
                    annotations=[
                        Label(0),
                        Label(1),
                    ],
                ),
                DatasetItem(
                    id=2,
                    annotations=[
                        Label(0),
                        Label(2),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(id=2, annotations=[Label(2)]),
            ],
            categories=["a", "b", "c"],
        )

        filtered = XPathAnnotationsFilter(
            source, "/item/annotation[label_id = 2]", remove_empty=True
        )

        compare_datasets(self, expected, filtered)


class DatasetTransformTest:
    @pytest.mark.parametrize("mode", [True, False])
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_stack_transform(self, mode: bool, caplog: pytest.LogCaptureFixture):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    f"item_{i}",
                    media=Image.from_numpy(np.ones([10, 10, 3], dtype=np.uint8)),
                    annotations=[Label(label=i % 2)],
                )
                for i in range(10)
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["cat", "dog"]),
            },
        )

        with eager_mode(mode, dataset), caplog.at_level(logging.ERROR):
            # Call a malformed transform by giving wrong argument name
            dataset.transform("random_split", wrong_argument_name=[("train", 0.67), ("val", 0.33)])

            # The previous malformed transform should be not stacked,
            # so that the following valid transform should successfully be executed.
            try:
                dataset = dataset.transform("random_split", splits=[("train", 0.67), ("val", 0.33)])
                for _ in dataset:
                    continue
            except Exception:
                raise pytest.fail("It should be successfully be executed.")

            for record in caplog.get_records("call"):
                assert "Automatically drop" in record.getMessage()

    @pytest.mark.parametrize(
        "expr_or_filter_func",
        ["/item[id=0]", lambda item: str(item.id) == "0"],
        ids=["xpath", "pyfunc"],
    )
    def test_can_filter_items(self, expr_or_filter_func, helper_tc):
        expected = Dataset.from_iterable(
            [DatasetItem(0, subset="train")], categories=["cat", "dog"]
        )

        dataset = Dataset.from_iterable(
            [DatasetItem(0, subset="train"), DatasetItem(1, subset="train")],
            categories=["cat", "dog"],
        )

        actual = dataset.filter(expr_or_filter_func)

        compare_datasets(helper_tc, expected, actual)

    @pytest.mark.parametrize(
        "expr_or_filter_func",
        ["/item/annotation[id=1]", lambda item, ann: str(ann.id) == "1"],
        ids=["xpath", "pyfunc"],
    )
    def test_can_filter_annotations(self, expr_or_filter_func, helper_tc):
        expected = Dataset.from_iterable(
            [DatasetItem(0, subset="train", annotations=[Label(0, id=1)])],
            categories=["cat", "dog"],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    subset="train",
                    annotations=[
                        Label(0, id=0),
                        Label(0, id=1),
                    ],
                ),
                DatasetItem(1, subset="train"),
            ],
            categories=["cat", "dog"],
        )

        actual = dataset.filter(expr_or_filter_func, filter_annotations=True, remove_empty=True)

        compare_datasets(helper_tc, expected, actual)


@pytest.fixture
def fxt_test_case():
    return TestCase()


@pytest.fixture
def fxt_sample_dataset_factory():
    def sample_dataset_factory(
        items=None,
        infos=None,
        categories=None,
    ):
        if items is None:
            items = [DatasetItem(0, subset="train"), DatasetItem(1, subset="train")]
        if infos is None:
            infos = {}
        if categories is None:
            categories = ["cat", "dog"]

        dataset = Dataset.from_iterable(
            items,
            infos=infos,
            categories=categories,
        )
        return dataset

    return sample_dataset_factory


@pytest.fixture
def fxt_sample_infos():
    infos_1 = {"info 1": 1}
    infos_2 = {"info 2": "meta-info 2"}
    infos = {}
    infos.update(infos_1)
    infos.update(infos_2)

    return infos_1, infos_2, infos


class DatasetInfosTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_dataset_infos(self, fxt_test_case, fxt_sample_dataset_factory, fxt_sample_infos):
        _, _, infos = fxt_sample_infos
        dataset = fxt_sample_dataset_factory(infos=infos)
        fxt_test_case.assertEqual(dataset.infos(), infos)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_dataset_infos_exact_merge(
        self, fxt_test_case, fxt_sample_dataset_factory, fxt_sample_infos
    ):
        infos_1, infos_2, infos = fxt_sample_infos

        dataset_1 = fxt_sample_dataset_factory(infos=infos_1)
        dataset_2 = fxt_sample_dataset_factory(infos=infos_2)

        dataset = Dataset.from_extractors(dataset_1, dataset_2)

        fxt_test_case.assertEqual(dataset.infos(), infos)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_dataset_infos_intersect_merge(
        self, fxt_test_case, fxt_sample_dataset_factory, fxt_sample_infos
    ):
        infos_1, infos_2, infos = fxt_sample_infos

        dataset_1 = fxt_sample_dataset_factory(infos=infos_1)
        dataset_2 = fxt_sample_dataset_factory(infos=infos_2)

        merger = IntersectMerge()
        dataset = merger(dataset_1, dataset_2)

        fxt_test_case.assertEqual(dataset.infos(), infos)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_dataset_infos_transform(
        self, fxt_test_case, fxt_sample_dataset_factory, fxt_sample_infos, is_eager
    ):
        infos_1, infos_2, infos = fxt_sample_infos
        with eager_mode(is_eager):
            dataset = fxt_sample_dataset_factory(infos=infos_1)

            dataset.transform(ProjectInfos, dst_infos=infos_2, overwrite=False)
            fxt_test_case.assertEqual(dataset.infos(), infos)

            dataset.transform(ProjectInfos, dst_infos=infos_2, overwrite=True)
            fxt_test_case.assertEqual(dataset.infos(), infos_2)

            dataset.transform(
                RemapLabels, mapping={"car": "apple", "cat": "banana", "dog": "cinnamon"}
            )
            fxt_test_case.assertEqual(dataset.infos(), infos_2)
