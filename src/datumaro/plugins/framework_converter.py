# Copyright (C) 2023-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from typing import Callable, Optional

import numpy as np

import datumaro.util.mask_tools as mask_tools
from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset import Dataset

TASK_ANN_TYPE = {
    "classification": AnnotationType.label,
    "multilabel_classification": AnnotationType.label,
    "detection": AnnotationType.bbox,
    "instance_segmentation": AnnotationType.polygon,
    "semantic_segmentation": AnnotationType.mask,
    "tabular": [AnnotationType.label, AnnotationType.caption],
}


class FrameworkConverterFactory:
    @staticmethod
    def create_converter(framework):
        if framework == "torch":
            return DmTorchDataset
        elif framework == "tf":
            return DmTfDataset
        else:
            raise ValueError("Unsupported framework")


class FrameworkConverter:
    def __init__(self, dataset, subset, task):
        self._dataset = dataset
        self._subset = subset
        self._task = task

    def to_framework(self, framework, **kwargs):
        converter_cls = FrameworkConverterFactory.create_converter(framework)
        return converter_cls(
            dataset=self._dataset,
            subset=self._subset,
            task=self._task,
            **kwargs,
        )


class _MultiFrameworkDataset:
    def __init__(
        self,
        dataset: Dataset,
        subset: str,
        task: str,
    ):
        self.dataset = dataset.get_subset(subset)
        self.task = task
        self.subset = subset
        self._ids = [item.id for item in self.dataset]

    def __len__(self) -> int:
        return len(self.dataset)

    def _gen_item(self, idx: int):
        item = self.dataset.get(id=self._ids[idx], subset=self.subset)
        image = item.media.data

        if self.task == "classification":
            label = [ann.label for ann in item.annotations if ann.type == TASK_ANN_TYPE[self.task]]
            if len(label) > 1:
                log.warning(
                    "Item %s has multiple labels and we choose the first one by default. "
                    "Please choose task=multilabel_classification for allowing this",
                    item.id,
                )
            label = label[0]
        elif self.task == "multilabel_classification":
            label = [ann.label for ann in item.annotations if ann.type == TASK_ANN_TYPE[self.task]]
        elif self.task in ["detection", "instance_segmentation"]:
            label = [
                ann.as_dict() for ann in item.annotations if ann.type == TASK_ANN_TYPE[self.task]
            ]
        elif self.task == "semantic_segmentation":
            masks = [
                (ann.image, ann.label)
                for ann in item.annotations
                if ann.type == TASK_ANN_TYPE[self.task]
            ]
            label = mask_tools.merge_masks((mask, label_id) for mask, label_id in masks)
        elif self.task == "tabular":
            label = [
                ann.as_dict() for ann in item.annotations if ann.type in TASK_ANN_TYPE[self.task]
            ]
        return image, label


try:
    import torch

    class DmTorchDataset(_MultiFrameworkDataset, torch.utils.data.Dataset):
        """
        Create a PyTorch dataset for a specific task given a dataset and subset.

        :param tokenizer: Callable converting a string into a series of tokens.
                          The output can either be token IDs (integers) or token strings (text).
                          If the later, the vocab parameter must also be provided.
        :param vocab: Callable converting a list of token IDs into a list of token strings
        """

        def __init__(
            self,
            dataset: Dataset,
            subset: str,
            task: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            target: Optional[str] = None,
            tokenizer: Optional[tuple[Callable, Callable]] = None,
            vocab: Optional[tuple[Callable, Callable]] = None,
        ):
            super().__init__(dataset=dataset, subset=subset, task=task)

            self.transform = transform
            self.target_transform = target_transform

            if self.task == "tabular":
                if not isinstance(target, dict):
                    raise ValueError(
                        "Target should be a dictionary with 'input' and 'output' keys."
                    )
                self.input_target = target.get("input")
                self.output_target = target.get("output")
                if not self.input_target:
                    raise ValueError(
                        "Please provide target column for tabular task which is used for input"
                    )

                if not (tokenizer):
                    raise ValueError("A tokenizer must be provided for tabular task")
                self.tokenizer = tokenizer
                self.vocab = vocab

        def __getitem__(self, idx):
            image, label = self._gen_item(idx)

            if self.task == "tabular":
                try:
                    text = image[self.input_target]
                except TypeError:
                    text = image()[self.input_target]

                if self.output_target:
                    src_tokenizer, tgt_tokenizer = self.tokenizer
                    src_vocab, tgt_vocab = self.vocab if self.vocab else None, None
                    src_tokens = src_tokenizer(text)
                    src_token_ids = src_vocab(src_tokens) if src_vocab else src_tokens

                    label_text = label[0]["caption"].split(f"{self.output_target}:")[-1]
                    tgt_tokens = tgt_tokenizer(label_text)
                    tgt_token_ids = tgt_vocab(tgt_tokens) if tgt_vocab else tgt_tokens

                    return torch.tensor(src_token_ids, dtype=torch.long), torch.tensor(
                        tgt_token_ids, dtype=torch.long
                    )
                else:
                    tokens = self.tokenizer(text)
                    token_ids = self.vocab(tokens) if self.vocab else tokens
                    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(
                        label[0]["label"], dtype=torch.long
                    )

            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)

            if self.transform:
                # Use image.copy() because PyTorch does not support non-writable tensors.
                image = self.transform(image.copy())

            if self.target_transform:
                label = self.target_transform(label)

            return image, label

except ImportError:

    class DmTorchDataset:
        def __init__(self):
            raise ImportError("PyTorch package not found. Cannot convert to PyTorch dataset.")


try:
    import tensorflow as tf

    class DmTfDataset(_MultiFrameworkDataset):
        def __init__(
            self,
            dataset: Dataset,
            subset: str,
            task: str,
            output_signature: Optional[tuple] = None,
        ):
            super().__init__(dataset=dataset, subset=subset, task=task)
            self.output_signature = output_signature

        def _get_rawitem(self, item_id):
            item_id = item_id.numpy() if isinstance(item_id, tf.Tensor) else item_id
            image, label = self._gen_item(item_id)

            if len(self.output_signature.keys()) == 2:
                return image, label

            outputs = []
            for key, spec in self.output_signature.items():
                if key == "image":
                    outputs += [image]
                else:
                    outputs += [tf.convert_to_tensor([l.get(spec.name, None) for l in label])]

            return outputs

        def _process_item(self, item_id):
            output_types = [spec.dtype for spec in self.output_signature.values()]
            outputs = tf.py_function(func=self._get_rawitem, inp=[item_id], Tout=output_types)
            output_dict = {}
            for idx, key in enumerate(self.output_signature.keys()):
                output_dict[key] = outputs[idx]

            return output_dict

        def create(self) -> "tf.data.Dataset":
            tf_dataset = tf.data.Dataset.range(len(self.dataset)).map(self._process_item)
            tf_dataset = tf_dataset
            return tf_dataset

        def repeat(self, count=None) -> "tf.data.Dataset":
            return self.create().repeat(count)

        def batch(self, batch_size, drop_remainder=False) -> "tf.data.Dataset":
            return self.create().batch(batch_size, drop_remainder=drop_remainder)

except ImportError:

    class DmTfDataset:
        def __init__(self):
            raise ImportError("Tensorflow package not found. Cannot convert to Tensorflow dataset.")
