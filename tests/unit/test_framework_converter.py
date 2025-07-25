# Copyright (C) 2023-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import tempfile
from typing import Any, Dict
from unittest import TestCase, skipIf

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    Label,
    LabelCategories,
    Mask,
    Polygon,
    Tabular,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, Table, TableRow
from datumaro.plugins.framework_converter import (
    TASK_ANN_TYPE,
    DmTfDataset,
    DmTorchDataset,
    FrameworkConverter,
    FrameworkConverterFactory,
    _MultiFrameworkDataset,
)

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path

try:
    import torch
    from torchvision import datasets, transforms
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True

try:
    import tensorflow as tf
except ImportError:
    TF_AVAILABLE = False
else:
    TF_AVAILABLE = True


@pytest.fixture
def fxt_dataset():
    label_cat = LabelCategories.from_iterable([f"label_{i}" for i in range(4)])
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="0",
                subset="train",
                annotations=[
                    Label(0),
                    Label(1),
                    Label(2),
                    Bbox(0, 0, 2, 2, label=0, attributes={"occluded": True}),
                    Bbox(2, 2, 4, 4, label=1, attributes={"occluded": False}),
                    Polygon([0, 0, 0, 2, 2, 2, 2, 0], label=0, attributes={"occluded": True}),
                    Polygon([2, 2, 2, 4, 4, 4, 4, 4], label=1, attributes={"occluded": True}),
                    Mask(
                        image=np.array([[0, 0, 0, 1, 1]] * 5),
                        label=1,
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="1",
                subset="train",
                annotations=[
                    Label(1),
                    Label(3),
                    Bbox(1, 1, 2, 2, label=1, attributes={"occluded": True}),
                    Bbox(0, 0, 1, 1, label=2, attributes={"occluded": False}),
                    Bbox(2, 2, 4, 4, label=4, attributes={"occluded": True}),
                    Polygon([1, 1, 1, 2, 2, 2, 2, 2], label=1, attributes={"occluded": True}),
                    Mask(
                        image=np.array([[1, 1, 0, 0, 0]] * 5),
                        label=1,
                    ),
                    Mask(
                        image=np.array([[0, 0, 1, 1, 0]] * 5),
                        label=2,
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 0, 1]] * 5),
                        label=3,
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="2",
                subset="val",
                annotations=[
                    Label(2),
                    Label(3),
                    Bbox(0, 0, 1, 1, label=1, attributes={"occluded": False}),
                    Bbox(1, 1, 2, 2, label=2, attributes={"occluded": False}),
                    Bbox(2, 2, 4, 4, label=3, attributes={"occluded": True}),
                    Polygon([0, 0, 1, 0, 1, 1, 0, 1], label=2, attributes={"occluded": False}),
                    Mask(
                        image=np.array([[0, 1, 0, 0, 0]] * 5),
                        label=2,
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 0, 1]] * 5),
                        label=3,
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="3",
                subset="val",
                annotations=[
                    Label(3),
                    Bbox(1, 1, 4, 4, label=3, attributes={"occluded": True}),
                    Polygon([1, 1, 1, 4, 4, 4, 4, 1], label=3, attributes={"occluded": True}),
                    Mask(
                        image=np.array([[1, 1, 1, 1, 0]] * 5),
                        label=3,
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 0, 1]] * 5),
                        label=2,
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
        ],
        categories={AnnotationType.label: label_cat},
    )


@pytest.fixture
def fxt_text_example():
    return """Datumaro is a framework and CLI tool to build, transform, and analyze datasets.
                a tool to build composite datasets and iterate over them
                a tool to create and maintain datasets
                    Version control of annotations and images
                    Publication (with removal of sensitive information)
                    Editing
                    Joining and splitting
                    Exporting, format changing
                    Image preprocessing
                a dataset storage
                a tool to debug datasets
                A network can be used to generate informative data subsets (e.g., with false-positives) to be analyzed further
            """


@pytest.fixture
def fxt_tabular_label_dataset(fxt_text_example):
    table = Table.from_list(
        [
            {
                "label": 1,
                "text": fxt_text_example,
            }
        ]
    )
    return Dataset.from_iterable(
        [
            DatasetItem(
                id=0,
                subset="train",
                media=TableRow(table=table, index=0),
                annotations=[Label(id=0, attributes={}, group=0, object_id=-1, label=0)],
            )
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                [("label:1", "label"), ("label:2", "label")]
            )
        },
        media_type=TableRow,
    )


@pytest.fixture
def fxt_tabular_caption_dataset():
    table = Table.from_list(
        [
            {
                "source": "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.",
                "target": "Two young, White males are outside near many bushes.",
            }
        ]
    )
    return Dataset.from_iterable(
        [
            DatasetItem(
                id=0,
                subset="train",
                media=TableRow(table=table, index=0),
                annotations=[
                    Caption("target:Two young, White males are outside near many bushes.")
                ],
            )
        ],
        categories={},
        media_type=TableRow,
    )


@pytest.fixture
def data_iter():
    return [(1, "This is a sample text"), (2, "Another sample text")]


@pytest.mark.new
@mark_requirement(Requirements.DATUM_GENERAL_REQ)
class FrameworkConverterFactoryTest(TestCase):
    @skipIf(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    def test_create_converter_torch(self):
        converter = FrameworkConverterFactory.create_converter("torch")
        self.assertEqual(converter, DmTorchDataset)

    @skipIf(not TF_AVAILABLE, reason="Tensorflow is not installed")
    def test_create_converter_tf(self):
        converter = FrameworkConverterFactory.create_converter("tf")
        self.assertEqual(converter, DmTfDataset)

    def test_create_converter_invalid(self):
        with self.assertRaises(ValueError):
            FrameworkConverterFactory.create_converter("invalid_framework")


def _tf_tensor_spec(shape, dtype, name=None):
    return tf.TensorSpec(shape=shape, dtype=dtype, name=name) if TF_AVAILABLE else ""


def _tf_int32():
    return tf.int32 if TF_AVAILABLE else ""


def _tf_float32():
    return tf.float32 if TF_AVAILABLE else ""

def _transforms_to_tensor():
    return transforms.ToTensor() if TORCH_AVAILABLE else ""

@mark_requirement(Requirements.DATUM_GENERAL_REQ)
class MultiframeworkConverterTest:
    @pytest.mark.parametrize(
        "fxt_dataset_type,fxt_subset,fxt_task",
        [
            (
                "fxt_dataset",
                "train",
                "classification",
            ),
            (
                "fxt_dataset",
                "val",
                "multilabel_classification",
            ),
            (
                "fxt_dataset",
                "train",
                "detection",
            ),
            (
                "fxt_dataset",
                "val",
                "instance_segmentation",
            ),
            (
                "fxt_dataset",
                "train",
                "semantic_segmentation",
            ),
            ("fxt_tabular_label_dataset", "train", "tabular"),
        ],
    )
    def test_multi_framework_dataset(
        self, fxt_dataset_type: str, fxt_subset: str, fxt_task: str, request
    ):
        dataset = request.getfixturevalue(fxt_dataset_type)
        dm_multi_framework_dataset = _MultiFrameworkDataset(
            dataset=dataset, subset=fxt_subset, task=fxt_task
        )

        for idx in range(len(dm_multi_framework_dataset)):
            image, label = dm_multi_framework_dataset._gen_item(idx)
            if fxt_task == "tabular":
                image = image()
            assert isinstance(image, (np.ndarray, dict))
            if fxt_task == "classification":
                assert isinstance(label, int)
            elif fxt_task == "multilabel_classification":
                assert isinstance(label, list)
            if fxt_task in ["detection", "instance_segmentation"]:
                assert isinstance(label, list)
            if fxt_task == "semantic_segmentation":
                assert isinstance(label, np.ndarray)
            elif fxt_task == "tabular":
                assert isinstance(label, list)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    @pytest.mark.parametrize(
        "fxt_subset,fxt_task,fxt_convert_kwargs",
        [
            (
                "train",
                "classification",
                {},
            ),
            (
                "val",
                "multilabel_classification",
                {},
            ),
            (
                "train",
                "detection",
                {},
            ),
            (
                "val",
                "instance_segmentation",
                {},
            ),
            (
                "train",
                "semantic_segmentation",
                {},
            ),
            (
                "val",
                "semantic_segmentation",
                {"transform": None, "target_transform": None},
            ),
            (
                "train",
                "semantic_segmentation",
                {"transform": _transforms_to_tensor()},
            ),
        ],
    )
    def test_can_convert_torch_framework(
        self,
        fxt_dataset: Dataset,
        fxt_subset: str,
        fxt_task: str,
        fxt_convert_kwargs: Dict[str, Any],
    ):
        multi_framework_dataset = FrameworkConverter(fxt_dataset, subset=fxt_subset, task=fxt_task)

        dm_torch_dataset = multi_framework_dataset.to_framework(
            framework="torch", **fxt_convert_kwargs
        )

        expected_dataset = fxt_dataset.get_subset(fxt_subset)

        for exp_item, dm_torch_item in zip(expected_dataset, dm_torch_dataset):
            image = exp_item.media.data
            if fxt_task == "classification":
                label = [
                    ann.label for ann in exp_item.annotations if ann.type == TASK_ANN_TYPE[fxt_task]
                ][0]
            elif fxt_task == "multilabel_classification":
                label = [
                    ann.label for ann in exp_item.annotations if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
            elif fxt_task in ["detection", "instance_segmentation"]:
                label = [
                    ann.as_dict()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
            elif fxt_task == "semantic_segmentation":
                masks = [
                    ann.as_class_mask()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
                label = np.sum(masks, axis=0, dtype=np.uint8)
            elif fxt_task == "tabular":
                label = [
                    ann.as_dict()
                    for ann in exp_item.annotations
                    if ann.type in TASK_ANN_TYPE[fxt_task]
                ]
            if fxt_convert_kwargs.get("transform", None):
                actual = dm_torch_item[0].permute(1, 2, 0).mul(255.0).to(torch.uint8).numpy()
                assert np.array_equal(image, actual)
            else:
                assert np.array_equal(image, dm_torch_item[0])

            if fxt_convert_kwargs.get("target_transform", None):
                assert np.array_equal(label, dm_torch_item[1].squeeze(0).numpy())
            else:
                assert np.array_equal(label, dm_torch_item[1])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    def test_can_convert_torch_framework_classification(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 64)),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            torch_dataset = datasets.MNIST(
                root=tmp_dir,
                train=True,
                download=True,
                transform=transform,
            )

            dm_dataset = Dataset.import_from(path=osp.join(tmp_dir, "MNIST"), format="mnist")

            multi_framework_dataset = FrameworkConverter(
                dm_dataset, subset="train", task="classification"
            )
            dm_torch_dataset = multi_framework_dataset.to_framework(
                framework="torch", transform=transform
            )

            assert len(torch_dataset) == len(dm_torch_dataset)
            for torch_item, dm_item in zip(torch_dataset, dm_torch_dataset):
                assert torch.equal(torch_item[0], dm_item[0])
                assert torch_item[1] == dm_item[1]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    def test_can_convert_torch_framework_detection(self):
        DUMMY_DATASET_DIR = get_test_asset_path("coco_dataset")
        format = "coco_instances"

        data_path = osp.join(DUMMY_DATASET_DIR, format)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
            ]
        )

        torch_dataset = datasets.CocoDetection(
            root=osp.join(data_path, "images/train/"),
            annFile=osp.join(data_path, "annotations/instances_train.json"),
            transform=transform,
        )

        dm_dataset = Dataset.import_from(data_path, format)

        multi_framework_dataset = FrameworkConverter(dm_dataset, subset="train", task="detection")
        dm_torch_dataset = multi_framework_dataset.to_framework(
            framework="torch", transform=transform
        )

        for torch_item, dm_item in zip(torch_dataset, dm_torch_dataset):
            assert torch.equal(torch_item[0], dm_item[0])
            for torch_ann, dm_ann in zip(torch_item[1], dm_item[1]):
                assert torch_ann["id"] == dm_ann["id"]
                assert torch_ann["category_id"] == dm_ann["label"] + 1
                # torch: (x, y, w, h), while dm: (x1, y1, x2, y2)
                x1, y1, x2, y2 = dm_ann["points"]
                assert torch_ann["bbox"] == [x1, y1, x2 - x1, y2 - y1]
                assert torch_ann["iscrowd"] == dm_ann["attributes"]["is_crowd"]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    def test_can_convert_torch_framework_tabular_label(
        self, fxt_tabular_label_dataset, fxt_text_example
    ):
        class DummyTabularDataset(Dataset):
            def __init__(self, data, tokenizer, transform=None):
                self.data = data
                self.transform = transform
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                label, text = self.data[idx]
                token_ids = self.tokenizer(text)

                if self.transform:
                    token_ids = self.transform(token_ids)

                return torch.tensor(token_ids, dtype=torch.long), torch.tensor(
                    label, dtype=torch.long
                )

        # Prepare data and tokenizer
        first_item = (
            1,
            fxt_text_example,
        )
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.train_from_iterator(first_item[1], trainer)

        def apply_tokenizer(text):
            return tokenizer.encode(text).ids

        # Create torch dataset
        torch_dataset = DummyTabularDataset([first_item], apply_tokenizer)

        # Convert to dm_torch_dataset
        dm_dataset = fxt_tabular_label_dataset
        multi_framework_dataset = FrameworkConverter(dm_dataset, subset="train", task="tabular")
        dm_torch_dataset = multi_framework_dataset.to_framework(
            framework="torch", target={"input": "text"}, tokenizer=apply_tokenizer
        )

        # Verify equality of items in torch_dataset and dm_torch_dataset
        label_indices = dm_dataset.categories().get(AnnotationType.label)._indices
        torch_item = torch_dataset[0]
        dm_item = dm_torch_dataset[0]
        assert torch.equal(torch_item[0], dm_item[0]), "Token IDs do not match"

        # Extract and compare labels
        torch_item_label = str(torch_item[1].item())
        dm_item_label = list(label_indices.keys())[list(label_indices.values()).index(0)].split(
            ":"
        )[-1]
        assert torch_item_label == dm_item_label, "Labels do not match"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    def test_can_convert_torch_framework_tabular_caption(self, fxt_tabular_caption_dataset):
        class Multi30kDataset(Dataset):
            def __init__(self, dataset, src_tokenizer, tgt_tokenizer):
                self.dataset = list(dataset)
                self.src_tokenizer = src_tokenizer
                self.tgt_tokenizer = tgt_tokenizer

            def __len__(self):
                return len(self.dataset)

            def _data_process(self, text, tokenizer):
                token_ids = tokenizer(text)
                return torch.tensor(token_ids, dtype=torch.long)

            def __getitem__(self, idx):
                src, tgt = self.dataset[idx]
                src_tensor = self._data_process(src, self.src_tokenizer)
                tgt_tensor = self._data_process(tgt, self.tgt_tokenizer)
                return src_tensor, tgt_tensor

        # Prepare data and tokenizer
        # First item of Multi30k
        first_item = (
            "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.",
            "Two young, White males are outside near many bushes.",
        )

        def build_single_vocab(item):
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
            tokenizer.train_from_iterator(item, trainer)

            def encode(text):
                return tokenizer.encode(text).ids

            return encode

        # Build vocabularies
        src_tokenizer = build_single_vocab(first_item[0])
        tgt_tokenizer = build_single_vocab(first_item[1])

        # Create torch dataset
        torch_dataset = Multi30kDataset(iter([first_item]), src_tokenizer, tgt_tokenizer)

        # Convert to dm_torch_dataset
        dm_dataset = fxt_tabular_caption_dataset
        multi_framework_dataset = FrameworkConverter(dm_dataset, subset="train", task="tabular")
        dm_torch_dataset = multi_framework_dataset.to_framework(
            framework="torch",
            target={"input": "source", "output": "target"},
            tokenizer=(src_tokenizer, tgt_tokenizer),
        )

        # Verify equality of items in torch_dataset and dm_torch_dataset
        torch_item = torch_dataset[0]
        dm_item = dm_torch_dataset[0]

        assert torch.equal(torch_item[0], dm_item[0]), "Token IDs for de do not match"
        assert torch.equal(torch_item[1], dm_item[1]), "Token IDs for en do not match"

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @pytest.mark.parametrize(
        "fxt_subset,fxt_task,fxt_convert_kwargs",
        [
            (
                "train",
                "classification",
                {
                    "output_signature": {
                        "image": _tf_tensor_spec(shape=(None, None, None), dtype=_tf_int32()),
                        "label": _tf_tensor_spec(shape=(), dtype=_tf_int32()),
                    }
                },
            ),
            (
                "val",
                "multilabel_classification",
                {
                    "output_signature": {
                        "image": _tf_tensor_spec(shape=(None, None, None), dtype=_tf_int32()),
                        "label": _tf_tensor_spec(shape=(None,), dtype=_tf_int32()),
                    }
                },
            ),
            (
                "train",
                "detection",
                {
                    "output_signature": {
                        "image": _tf_tensor_spec(shape=(None, None, None), dtype=_tf_int32()),
                        "bbox": _tf_tensor_spec(shape=(None, 4), dtype=_tf_float32(), name="points"),
                        "category_id": _tf_tensor_spec(shape=(None,), dtype=_tf_int32(), name="label"),
                    }
                },
            ),
            (
                "val",
                "instance_segmentation",
                {
                    "output_signature": {
                        "image": _tf_tensor_spec(shape=(None, None, None), dtype=_tf_int32()),
                        "polygon": _tf_tensor_spec(
                            shape=(None, None), dtype=_tf_float32(), name="points"
                        ),
                        "category_id": _tf_tensor_spec(shape=(None,), dtype=_tf_int32(), name="label"),
                    }
                },
            ),
            (
                "train",
                "semantic_segmentation",
                {
                    "output_signature": {
                        "image": _tf_tensor_spec(shape=(None, None, None), dtype=_tf_int32()),
                        "label": _tf_tensor_spec(shape=(None, None), dtype=_tf_int32()),
                    }
                },
            ),
        ],
    )
    def test_can_convert_tf_framework(
        self,
        fxt_dataset: Dataset,
        fxt_subset: str,
        fxt_task: str,
        fxt_convert_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        multi_framework_dataset = FrameworkConverter(fxt_dataset, subset=fxt_subset, task=fxt_task)

        dm_tf_dataset = multi_framework_dataset.to_framework(framework="tf", **fxt_convert_kwargs)

        expected_dataset = fxt_dataset.get_subset(fxt_subset)

        for exp_item, tf_item in zip(expected_dataset, dm_tf_dataset.create()):
            image = exp_item.media.data
            if fxt_task == "classification":
                label = exp_item.annotations[0].label
            if fxt_task == "multilabel_classification":
                label = [
                    ann.label for ann in exp_item.annotations if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
            elif fxt_task in ["detection", "instance_segmentation"]:
                label = [
                    ann.as_dict()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
            elif fxt_task == "semantic_segmentation":
                masks = [
                    ann.as_class_mask()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
                label = np.sum(masks, axis=0, dtype=np.uint8)

            assert np.array_equal(image, tf_item["image"])

            if fxt_task == "classification":
                assert label == tf_item["label"]

            if fxt_task == "multilabel_classification":
                assert np.array_equal(label, tf_item["label"])

            elif fxt_task == "detection":
                bboxes = [p["points"] for p in label]
                labels = [p["label"] for p in label]

                assert np.array_equal(bboxes, tf_item["bbox"].numpy())
                assert np.array_equal(labels, tf_item["category_id"].numpy())

            elif fxt_task == "instance_segmentation":
                polygons = [p["points"] for p in label]
                labels = [p["label"] for p in label]

                assert np.array_equal(polygons, tf_item["polygon"].numpy())
                assert np.array_equal(labels, tf_item["category_id"].numpy())

            elif fxt_task == "semantic_segmentation":
                assert np.array_equal(label, tf_item["label"])

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @pytest.mark.parametrize(
        "fxt_subset,fxt_task,fxt_output_signature",
        [
            (
                "train",
                "classification",
                {
                    "image": _tf_tensor_spec(shape=(None, None, None), dtype=_tf_int32()),
                    "label": _tf_tensor_spec(shape=(), dtype=_tf_int32()),
                },
            ),
            (
                "val",
                "detection",
                {
                    "image": _tf_tensor_spec(shape=(None, None, None), dtype=_tf_int32()),
                    "bbox": _tf_tensor_spec(shape=(None, 4), dtype=_tf_float32(), name="points"),
                    "category_id": _tf_tensor_spec(shape=(None,), dtype=_tf_int32(), name="label"),
                },
            ),
        ],
    )
    def test_tf_get_rawitem(
        self, fxt_dataset: Dataset, fxt_subset: str, fxt_task: str, fxt_output_signature: dict
    ):
        dm_tf_dataset = DmTfDataset(
            dataset=fxt_dataset,
            subset=fxt_subset,
            task=fxt_task,
            output_signature=fxt_output_signature,
        )

        expected_dataset = fxt_dataset.get_subset(fxt_subset)

        for idx, exp_item in enumerate(expected_dataset):
            image = exp_item.media.data
            if fxt_task == "classification":
                label = exp_item.annotations[0].label
            elif fxt_task == "detection":
                bboxes = [
                    ann.as_dict()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
                label = []
                for key, spec in fxt_output_signature.items():
                    if key == "image":
                        continue
                    label += [tf.convert_to_tensor([bbox.get(spec.name, None) for bbox in bboxes])]

            tf_item = dm_tf_dataset._get_rawitem(idx)

            assert np.array_equal(image, tf_item[0])
            if fxt_task == "classification":
                assert label == tf_item[1]
            elif fxt_task == "detection":
                for label_types in range(len(label)):
                    assert np.array_equal(label[label_types], tf_item[label_types + 1])

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @pytest.mark.parametrize(
        "fxt_subset,fxt_task,fxt_output_signature",
        [
            (
                "train",
                "classification",
                {
                    "image": _tf_tensor_spec(shape=(None, None, None), dtype=_tf_int32()),
                    "label": _tf_tensor_spec(shape=(), dtype=_tf_int32()),
                },
            ),
            (
                "val",
                "detection",
                {
                    "image": _tf_tensor_spec(shape=(None, None, None), dtype=_tf_int32()),
                    "bbox": _tf_tensor_spec(shape=(None, 4), dtype=_tf_float32(), name="points"),
                    "category_id": _tf_tensor_spec(shape=(None,), dtype=_tf_int32(), name="label"),
                },
            ),
        ],
    )
    def test_tf_process_item(
        self, fxt_dataset: Dataset, fxt_subset: str, fxt_task: str, fxt_output_signature: dict
    ):
        dm_tf_dataset = DmTfDataset(
            dataset=fxt_dataset,
            subset=fxt_subset,
            task=fxt_task,
            output_signature=fxt_output_signature,
        )

        expected_dataset = fxt_dataset.get_subset(fxt_subset)

        for idx, exp_item in enumerate(expected_dataset):
            image = exp_item.media.data
            if fxt_task == "classification":
                label = exp_item.annotations[0].label
            elif fxt_task == "detection":
                bboxes = [
                    ann.as_dict()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
                label = []
                for key, spec in fxt_output_signature.items():
                    if key == "image":
                        continue
                    label += [tf.convert_to_tensor([bbox.get(spec.name, None) for bbox in bboxes])]

            tf_item = dm_tf_dataset._process_item(idx)

            assert np.array_equal(image, tf_item["image"])
            if fxt_task == "classification":
                assert label == tf_item["label"]
            elif fxt_task == "detection":
                assert np.array_equal(label[0], tf_item["bbox"])
                assert np.array_equal(label[1], tf_item["category_id"])

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    def test_tf_dataset_repeat(self, fxt_dataset: Dataset):
        output_signature = {
            "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
            "label": tf.TensorSpec(shape=(), dtype=tf.int32),
        }

        dm_tf_dataset = DmTfDataset(
            dataset=fxt_dataset,
            subset="train",
            task="classification",
            output_signature=output_signature,
        )
        original_dataset = dm_tf_dataset.create()
        repeated_dataset = dm_tf_dataset.repeat(count=5)

        n_dataset = len(list(original_dataset))

        for idx, item in enumerate(repeated_dataset):
            assert np.array_equal(item["image"], list(original_dataset)[idx % n_dataset]["image"])
            assert np.array_equal(item["label"], list(original_dataset)[idx % n_dataset]["label"])

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    def test_tf_dataset_batch(self, fxt_dataset: Dataset):
        output_signature = {
            "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
            "label": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        }

        dm_tf_dataset = DmTfDataset(
            dataset=fxt_dataset,
            subset="val",
            task="semantic_segmentation",
            output_signature=output_signature,
        )
        original_dataset = dm_tf_dataset.create()
        batched_dataset = dm_tf_dataset.batch(batch_size=2)

        for idx, item in enumerate(batched_dataset):
            assert np.array_equal(item["image"][0], list(original_dataset)[idx]["image"])
            assert np.array_equal(item["image"][1], list(original_dataset)[idx + 1]["image"])
            assert np.array_equal(item["label"][0], list(original_dataset)[idx]["label"])
            assert np.array_equal(item["label"][1], list(original_dataset)[idx + 1]["label"])

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    def test_can_convert_tf_framework_classification(self):
        output_signature = {
            "image": tf.TensorSpec(shape=(28, 28), dtype=tf.uint8),
            "label": tf.TensorSpec(shape=(), dtype=tf.uint8),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            tf_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

            keras_data_dir = osp.expanduser("~/.keras/datasets")
            dm_dataset = Dataset.import_from(
                path=osp.join(keras_data_dir, "fashion-mnist"), format="mnist"
            )

            multi_framework_dataset = FrameworkConverter(
                dm_dataset, subset="test", task="classification"
            )
            dm_tf_dataset = multi_framework_dataset.to_framework(
                framework="tf", output_signature=output_signature
            )

            epoch, batch_size = 1, 16
            for tf_item, dm_item in zip(
                tf_dataset.repeat(epoch).batch(batch_size),
                dm_tf_dataset.repeat(epoch).batch(batch_size),
            ):
                assert tf.reduce_all(tf_item[0] == dm_item["image"])
                assert tf.reduce_all(tf_item[1] == dm_item["label"])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    def test_missing_tokenizer(self, fxt_tabular_label_dataset, fxt_text_example):
        # Convert to dm_torch_dataset
        dm_dataset = fxt_tabular_label_dataset
        multi_framework_dataset = FrameworkConverter(dm_dataset, subset="train", task="tabular")

        with pytest.raises(ValueError):
            dm_torch_dataset = multi_framework_dataset.to_framework(
                framework="torch", target={"input": "text"}
            )
