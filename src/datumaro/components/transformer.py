# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT
from multiprocessing.pool import ThreadPool
from typing import Iterator, List, Optional

import numpy as np

from datumaro.components.annotation import Annotation, AnnotationType, LabelCategories
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetBase, DatasetItem, IDataset
from datumaro.components.launcher import Launcher
from datumaro.util import is_method_redefined, take_by
from datumaro.util.multi_procs_util import consumer_generator


class Transform(DatasetBase, CliPlugin):
    """
    A base class for dataset transformations that change dataset items
    or their annotations.
    """

    @staticmethod
    def wrap_item(item, **kwargs):
        return item.wrap(**kwargs)

    def __init__(self, extractor: IDataset):
        super().__init__()

        self._extractor = extractor

    def categories(self):
        return self._extractor.categories()

    def subsets(self):
        if self._subsets is None:
            self._subsets = set(self._extractor.subsets())
        return super().subsets()

    def __len__(self):
        assert self._length in {None, "parent"} or isinstance(self._length, int)
        if (
            self._length is None
            and not is_method_redefined("__iter__", Transform, self)
            or self._length == "parent"
        ):
            self._length = len(self._extractor)
        return super().__len__()

    def media_type(self):
        return self._extractor.media_type()

    def infos(self):
        return self._extractor.infos()


class ItemTransform(Transform):
    def transform_item(self, item: DatasetItem) -> Optional[DatasetItem]:
        """
        Returns a modified copy of the input item.

        Avoid changing and returning the input item, because it can lead to
        unexpected problems. Use wrap_item() or item.wrap() to simplify copying.
        """

        raise NotImplementedError()

    def __iter__(self):
        for item in self._extractor:
            item = self.transform_item(item)
            if item is not None:
                yield item


class TabularTransform(Transform):
    """A transformation class for processing dataset items in batches with optional parallelism.

    This class takes a dataset extractor, batch size, and number of worker threads to process
    dataset items. Depending on the number of workers specified, it can process items either
    sequentially (single-process) or in parallel (multi-process), making it efficient for
    batch transformations.

    Parameters:
        extractor: The dataset extractor to obtain items from.
        batch_size: The batch size for processing items. Default is 1.
        num_workers: The number of worker threads to use for parallel processing.
            Set to 0 for single-process mode. Default is 0.
    """

    def __init__(
        self,
        extractor: IDataset,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super().__init__(extractor)
        self._batch_size = batch_size
        if not (isinstance(num_workers, int) and num_workers >= 0):
            raise ValueError(
                f"num_workers should be a non negative integer, but it is {num_workers}"
            )
        self._num_workers = num_workers

    def __iter__(self) -> Iterator[DatasetItem]:
        if self._num_workers == 0:
            return self._iter_single_proc()
        return self._iter_multi_procs()

    def _iter_multi_procs(self):
        with ThreadPool(processes=self._num_workers) as pool:

            def _producer_gen():
                for batch in take_by(self._extractor, self._batch_size):
                    future = pool.apply_async(
                        func=self._process_batch,
                        args=(batch,),
                    )
                    yield future

            with consumer_generator(producer_generator=_producer_gen()) as consumer_gen:
                for future in consumer_gen:
                    for item in future.get():
                        yield item

    def _iter_single_proc(self) -> Iterator[DatasetItem]:
        for batch in take_by(self._extractor, self._batch_size):
            for item in self._process_batch(batch=batch):
                yield item

    def transform_item(self, item: DatasetItem) -> Optional[DatasetItem]:
        """
        Returns a modified copy of the input item.

        Avoid changing and returning the input item, because it can lead to
        unexpected problems. Use wrap_item() or item.wrap() to simplify copying.
        """

        raise NotImplementedError()

    def _process_batch(
        self,
        batch: List[DatasetItem],
    ) -> List[DatasetItem]:
        results = [self.transform_item(item) for item in batch]

        return results


class ModelTransform(Transform):
    """A transformation class for applying a model's inference to dataset items.

    This class takes an dataset, a launcher, and other optional parameters
    to transform the dataset item from the model outputs by the launcher.
    It can process items using multiple processes if specified, making it suitable for
    parallelized inference tasks.

    Parameters:
        extractor: The dataset extractor to obtain items from.
        launcher: The launcher responsible for model inference.
        batch_size: The batch size for processing items. Default is 1.
        append_annotation: Whether to append inference annotations to existing annotations.
            Default is False.
        num_workers: The number of worker threads to use for parallel inference.
            Set to 0 for single-process mode. Default is 0.
    """

    def __init__(
        self,
        extractor: IDataset,
        launcher: Launcher,
        batch_size: int = 1,
        append_annotation: bool = False,
        num_workers: int = 0,
    ):
        super().__init__(extractor)
        self._launcher = launcher
        self._batch_size = batch_size
        self._append_annotation = append_annotation
        if not (isinstance(num_workers, int) and num_workers >= 0):
            raise ValueError(
                f"num_workers should be a non negative integer, but it is {num_workers}"
            )
        self._num_workers = num_workers

    def __iter__(self) -> Iterator[DatasetItem]:
        if self._num_workers == 0:
            return self._iter_single_proc()
        return self._iter_multi_procs()

    def _iter_multi_procs(self):
        with ThreadPool(processes=self._num_workers) as pool:

            def _producer_gen():
                for batch in take_by(self._extractor, self._batch_size):
                    future = pool.apply_async(
                        func=self._process_batch,
                        args=(batch,),
                    )
                    yield future

            with consumer_generator(producer_generator=_producer_gen()) as consumer_gen:
                for future in consumer_gen:
                    for item in future.get():
                        yield item

    def _iter_single_proc(self) -> Iterator[DatasetItem]:
        for batch in take_by(self._extractor, self._batch_size):
            for item in self._process_batch(batch=batch):
                yield item

    def _process_batch(
        self,
        batch: List[DatasetItem],
    ) -> List[DatasetItem]:
        inference = self._launcher.launch(
            batch=[item for item in batch if self._launcher.type_check(item)]
        )

        for annotations in inference:
            self._check_annotations(annotations)

        return [
            self.wrap_item(
                item,
                annotations=item.annotations + annotations
                if self._append_annotation
                else annotations,
            )
            for item, annotations in zip(batch, inference)
        ]

    def get_subset(self, name):
        subset = self._extractor.get_subset(name)
        return __class__(subset, self._launcher, self._batch_size)

    def infos(self):
        launcher_override = self._launcher.infos()
        if launcher_override is not None:
            return launcher_override
        return self._extractor.infos()

    def categories(self):
        launcher_override = self._launcher.categories()
        if launcher_override is not None:
            return launcher_override
        return self._extractor.categories()

    def transform_item(self, item):
        inputs = np.expand_dims(item.media, axis=0)
        annotations = self._launcher.launch(inputs)[0]
        return self.wrap_item(item, annotations=annotations)

    def _check_annotations(self, annotations: List[Annotation]):
        labels_count = len(self.categories().get(AnnotationType.label, LabelCategories()).items)

        for ann in annotations:
            label = getattr(ann, "label", None)
            if label is None:
                continue

            if label not in range(labels_count):
                raise Exception(
                    "Annotation has unexpected label id %s, "
                    "while there is only %s defined labels." % (label, labels_count)
                )
