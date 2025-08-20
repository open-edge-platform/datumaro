# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Optional, Tuple

import pytest

from datumaro.components.abstracts.model_interpreter import LauncherInputType, ModelPred, PrepInfo
from datumaro.components.annotation import Annotation
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.launcher import Launcher
from datumaro.components.transformer import TabularTransform


class MockLauncher(Launcher):
    def preprocess(self, item: DatasetItem) -> Tuple[LauncherInputType, PrepInfo]:
        return {"item": item}, None

    def infer(self, inputs: LauncherInputType) -> List[ModelPred]:
        return [[Annotation(id=1)] for _ in inputs["item"]]

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        return pred


class TabularTransformTest:
    @pytest.fixture
    def fxt_dataset(self):
        return Dataset.from_iterable(
            [DatasetItem(id=f"item_{i}", annotations=[Annotation(id=0)]) for i in range(10)]
        )

    @pytest.mark.parametrize("batch_size", [1, 10])
    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_tabular_transform(self, fxt_dataset, batch_size, num_workers):
        class MockTabularTransform(TabularTransform):
            def transform_item(self, item: DatasetItem) -> Optional[DatasetItem]:
                # Mock transformation logic
                item.annotations.append(Annotation(id=1))
                return item

        transform = MockTabularTransform(
            extractor=fxt_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        for idx, item in enumerate(transform):
            assert item.id == f"item_{idx}"
            assert item.annotations == [Annotation(id=0), Annotation(id=1)]
