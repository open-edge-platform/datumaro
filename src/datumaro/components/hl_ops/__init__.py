# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import inspect
import os
import os.path as osp
import shutil
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Optional, Type, Union, overload

from datumaro.components.comparator import DistanceComparator, EqualityComparator, TableComparator
from datumaro.components.dataset import Dataset, IDataset
from datumaro.components.environment import Environment
from datumaro.components.errors import DatasetError
from datumaro.components.exporter import Exporter
from datumaro.components.filter import (
    UserFunctionAnnotationsFilter,
    UserFunctionDatasetFilter,
    XPathAnnotationsFilter,
    XPathDatasetFilter,
)
from datumaro.components.merge import DEFAULT_MERGE_POLICY, get_merger
from datumaro.components.transformer import Transform
from datumaro.components.validator import TaskType, Validator
from datumaro.util import parse_str_enum_value
from datumaro.util.scope import on_error_do, scoped

if TYPE_CHECKING:
    from datumaro.components.annotation import Annotation
    from datumaro.components.dataset_base import DatasetItem

__all__ = ["HLOps"]


class HLOps:
    """High-level dataset operations for Python API."""

    @staticmethod
    def compare(
        first_dataset: IDataset,
        second_dataset: IDataset,
        report_dir: Optional[str] = None,
        method: str = "table",
        **kwargs,
    ) -> IDataset:
        """
        Compare two datasets and optionally save a comparison report.

        Args:
            first_dataset (IDataset): The first dataset to compare.
            second_dataset (IDataset): The second dataset to compare.
            report_dir (Optional[str], optional): The directory path to save the comparison report. Defaults to None.
            method (str, optional): The comparison method to use. Possible values are "table", "equality", "distance".
                Defaults to "table".
            **kwargs: Additional keyword arguments that can be passed to the comparison method.

        Returns:
            IDataset: The result of the comparison.

        Raises:
            ValueError: If the method is "distance" and report_dir is not specified.

        Example:
            comparator = Comparator()
            result = comparator.compare(first_dataset, second_dataset, report_dir="./comparison_report")
            print(result)
        """
        if method == "table":
            comparator = TableComparator()
            h_table, m_table, l_table, result_dict = comparator.compare_datasets(
                first_dataset, second_dataset
            )
            if report_dir:
                comparator.save_compare_report(h_table, m_table, l_table, result_dict, report_dir)

        elif method == "equality":
            comparator = EqualityComparator(**kwargs)
            output = comparator.compare_datasets(first_dataset, second_dataset)
            if report_dir:
                comparator.save_compare_report(output, report_dir)

        elif method == "distance":
            if not report_dir:
                raise ValueError(
                    "Please specify report_dir to save comparison result for DistanceComparator."
                )
            output_format = kwargs.pop("output_format", "simple")
            comparator = DistanceComparator(**kwargs)
            from datumaro.cli.util.compare import DistanceCompareVisualizer

            with DistanceCompareVisualizer(
                save_dir=report_dir,
                comparator=comparator,
                output_format=output_format,
            ) as visualizer:
                visualizer.save(first_dataset, second_dataset)

        return 0

    @staticmethod
    def transform(
        dataset: IDataset,
        method: Union[str, Type[Transform]],
        *,
        env: Optional[Environment] = None,
        **kwargs,
    ) -> IDataset:
        """
        Applies some function to dataset items.

        Results are computed lazily, if the transform supports this.

        Args:
            dataset: The dataset to be transformed
            method: The transformation to be applied to the dataset.
                If a string is passed, it is treated as a plugin name,
                which is searched for in the environment
                set by the 'env' argument
            env: A plugin collection. If not set, the built-in plugins are used
            **kwargs: Parameters for the transformation

        Returns: a wrapper around the input dataset
        """

        if isinstance(method, str):
            if env is None:
                env = Environment()
            method = env.transforms[method]

        if not (inspect.isclass(method) and issubclass(method, Transform)):
            raise TypeError(f"Unexpected 'method' argument type: {type(method)}")

        produced = method(dataset, **kwargs)

        return Dataset(source=produced, env=env)

    @overload
    @staticmethod
    def filter(
        dataset: IDataset,
        expr: str,
        *,  # pylint: disable=redefined-builtin
        filter_annotations: bool = False,
        remove_empty: bool = False,
    ) -> IDataset:
        """
        Filters out some dataset items or annotations, using a custom filter
        expression.

        Args:
            dataset: The dataset to be filtered
            expr: XPath-formatted filter expression
                (e.g. `/item[subset = 'train']`, `/item/annotation[label = 'cat']`)
            filter_annotations: Indicates if the filter should be
                applied to items or annotations
            remove_empty: When filtering annotations, allows to
                exclude empty items from the resulting dataset

        Returns: a wrapper around the input dataset, which is computed lazily
            during iteration
        """
        ...

    @overload
    @staticmethod
    def filter(
        dataset: IDataset,
        filter_func: Union[
            Callable[[DatasetItem], bool], Callable[[DatasetItem, Annotation], bool]
        ],
        *,  # pylint: disable=redefined-builtin
        filter_annotations: bool = False,
        remove_empty: bool = False,
    ) -> IDataset:
        """
        Filters out some dataset items or annotations, using a user-provided filter
        Python function.

        Results are stored in-place. Modifications are applied lazily.

        Args:
            filter_func: User-provided Python function for filtering
            filter_annotations: Indicates if the filter should be
                applied to items or annotations
            remove_empty: When filtering annotations, allows to
                exclude empty items from the resulting dataset

        Returns: a wrapper around the input dataset, which is computed lazily
            during iteration

        Example:
            - (`filter_annotations=False`) This is an example of filtering
                dataset items with images larger than 1024 pixels::

                from datumaro.components.media import Image

                def filter_func(item: DatasetItem) -> bool:
                    h, w = item.media_as(Image).size
                    return h > 1024 or w > 1024

                filtered = HLOps.filter(
                    dataset=dataset,
                    filter_func=filter_func,
                    filter_annotations=False,
                )
                # No items with an image height or width greater than 1024
                filtered_items = [item for item in filtered]

            - (`filter_annotations=True`) This is an example of removing bounding boxes
                sized greater than 50% of the image size::

                from datumaro.components.media import Image
                from datumaro.components.annotation import Annotation, Bbox

                def filter_func(item: DatasetItem, ann: Annotation) -> bool:
                    # If the annotation is not a Bbox, do not filter
                    if not isinstance(ann, Bbox):
                        return False

                    h, w = item.media_as(Image).size
                    image_size = h * w
                    bbox_size = ann.h * ann.w

                    # Accept Bboxes smaller than 50% of the image size
                    return bbox_size < 0.5 * image_size

                filtered = HLOps.filter(
                    dataset=dataset,
                    filter_func=filter_func,
                    filter_annotations=True,
                )
                # No bounding boxes with a size greater than 50% of their image
                filtered_items = [item for item in filtered]
        """

    def filter(
        dataset: IDataset,
        expr_or_filter_func: Union[
            str, Callable[[DatasetItem], bool], Callable[[DatasetItem, Annotation], bool]
        ],
        *,  # pylint: disable=redefined-builtin
        filter_annotations: bool = False,
        remove_empty: bool = False,
    ):
        if isinstance(expr_or_filter_func, str):
            expr = expr_or_filter_func
            return (
                HLOps.transform(
                    dataset, XPathAnnotationsFilter, xpath=expr, remove_empty=remove_empty
                )
                if filter_annotations
                else HLOps.transform(dataset, XPathDatasetFilter, xpath=expr)
            )
        elif callable(expr_or_filter_func):
            filter_func = expr_or_filter_func
            return (
                HLOps.transform(
                    dataset,
                    UserFunctionAnnotationsFilter,
                    filter_func=filter_func,
                    remove_empty=remove_empty,
                )
                if filter_annotations
                else HLOps.transform(dataset, UserFunctionDatasetFilter, filter_func=filter_func)
            )
        raise TypeError(expr_or_filter_func)

    @staticmethod
    def merge(
        *datasets: Dataset,
        merge_policy: str = DEFAULT_MERGE_POLICY,
        report_path: Optional[str] = None,
        **kwargs,
    ) -> Dataset:
        """
        Merge `datasets` according to `merge_policy`. You have to choose an appropriate `merge_policy`
        for your purpose. The available merge policies are "union", "intersect", and "exact".
        For more details about the merge policies, please refer to :func:`get_merger`.
        """

        merger = get_merger(merge_policy, **kwargs)
        merged = merger(*datasets)
        env = Environment.merge(
            [
                dataset.env
                for dataset in datasets
                if hasattr(dataset, "env")
                # TODO: Sometimes, there is dataset which is not exactly "Dataset",
                # e.g., VocClassificationBase. this should be fixed and every object from
                # Dataset.import_from should have "Dataset" type.
            ]
        )
        if report_path:
            merger.save_merge_report(report_path)
        return Dataset(source=merged, env=env)

    @staticmethod
    @scoped
    def export(
        dataset: IDataset,
        path: str,
        format: Union[str, Type[Exporter]],
        *,
        env: Optional[Environment] = None,
        **kwargs,
    ) -> None:
        """
        Saves the input dataset in some format.

        Args:
            dataset: The dataset to be saved
            path: The output directory
            format: The desired output format for the dataset.
                If a string is passed, it is treated as a plugin name,
                which is searched for in the environment set by the 'env' argument
            env: A plugin collection. If not set, the built-in plugins are used
            **kwargs: Parameters for the export format
        """

        if isinstance(format, str):
            if env is None:
                env = Environment()
            exporter = env.exporters[format]
        else:
            exporter = format

        if not (inspect.isclass(exporter) and issubclass(exporter, Exporter)):
            raise TypeError(f"Unexpected 'format' argument type: {type(exporter)}")

        path = osp.abspath(path)
        if not osp.exists(path):
            on_error_do(shutil.rmtree, path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

        exporter.convert(dataset, save_dir=path, **kwargs)

    @staticmethod
    def validate(
        dataset: IDataset,
        task: Union[str, TaskType],
        *,
        env: Optional[Environment] = None,
        **kwargs,
    ) -> Dict:
        """
        Checks dataset annotations for correctness relatively to a task type.

        Args:
            dataset: The dataset to check
            task: Target task type - classification, detection etc.
            env: A plugin collection. If not set, the built-in plugins are used
            **kwargs: Parameters for the validator

        Returns: a dictionary with validation results
        """

        task = parse_str_enum_value(task, TaskType).name

        if env is None:
            env = Environment()

        validator: Validator = env.validators[task](**kwargs)
        return validator.validate(dataset)

    @staticmethod
    def aggregate(dataset: Dataset, from_subsets: Iterable[str], to_subset: str) -> Dataset:
        subset_names = set(dataset.subsets().keys())

        for subset in from_subsets:
            if subset not in subset_names:
                raise DatasetError(
                    f"{subset} is not found in the subset names ({subset_names}) in the dataset."
                )

        return HLOps.transform(
            dataset, "map_subsets", mapping={subset: to_subset for subset in from_subsets}
        )
