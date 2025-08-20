# Copyright (C) 2020-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
import logging as log
import os
import os.path as osp
import shutil
import warnings
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    overload,
)

from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    LabelCategories,
    TabularCategories,
)
from datumaro.components.config_model import Source
from datumaro.components.dataset_base import (
    DEFAULT_SUBSET_NAME,
    CategoriesInfo,
    DatasetBase,
    DatasetInfo,
    DatasetItem,
    IDataset,
)
from datumaro.components.dataset_item_storage import DatasetItemStorageDatasetView
from datumaro.components.dataset_storage import DatasetPatch, DatasetStorage, StreamDatasetStorage
from datumaro.components.environment import DEFAULT_ENVIRONMENT, Environment
from datumaro.components.errors import (
    DatasetImportError,
    DatumaroError,
    MultipleFormatsMatchError,
    NoMatchingFormatsError,
    StreamedItemError,
    UnknownFormatError,
)
from datumaro.components.exporter import ExportContext, Exporter, ExportErrorPolicy, _ExportFail
from datumaro.components.filter import (
    UserFunctionAnnotationsFilter,
    UserFunctionDatasetFilter,
    XPathAnnotationsFilter,
    XPathDatasetFilter,
)
from datumaro.components.importer import ImportContext, ImportErrorPolicy, _ImportFail
from datumaro.components.media import Image, MediaElement
from datumaro.components.merge import DEFAULT_MERGE_POLICY
from datumaro.components.progress_reporting import NullProgressReporter, ProgressReporter
from datumaro.components.transformer import ItemTransform, Transform
from datumaro.util.log_utils import logging_disabled
from datumaro.util.scope import on_error_do, scoped

DEFAULT_FORMAT = "datumaro"

__all__ = ["Dataset", "eager_mode"]


class DatasetSubset(IDataset):  # non-owning view
    def __init__(self, parent: Dataset, name: str):
        super().__init__()
        self.parent = parent
        self.name = name

    def __iter__(self):
        yield from self.parent._data.get_subset(self.name)

    def __len__(self):
        subset: DatasetItemStorageDatasetView.Subset = self.parent._data.get_subset(self.name)

        return len(subset)

    def put(self, item):
        return self.parent.put(item, subset=self.name)

    def get(self, id, subset=None):
        assert (subset or DEFAULT_SUBSET_NAME) == (self.name or DEFAULT_SUBSET_NAME)
        return self.parent.get(id, subset=self.name)

    def remove(self, id, subset=None):
        assert (subset or DEFAULT_SUBSET_NAME) == (self.name or DEFAULT_SUBSET_NAME)
        return self.parent.remove(id, subset=self.name)

    def get_subset(self, name):
        assert (name or DEFAULT_SUBSET_NAME) == (self.name or DEFAULT_SUBSET_NAME)
        return self

    def subsets(self):
        if (self.name or DEFAULT_SUBSET_NAME) == DEFAULT_SUBSET_NAME:
            return self.parent.subsets()
        return {self.name: self}

    def infos(self):
        return self.parent.infos()

    def categories(self):
        return self.parent.categories()

    def media_type(self):
        return self.parent.media_type()

    def ann_types(self):
        return self.parent.ann_types()

    def get_annotated_items(self):
        return sum(bool(s.annotations) for s in self.parent._data.get_subset(self.name))

    def get_annotations(self):
        annotations_by_type = {t.name: {"count": 0} for t in AnnotationType}
        for item in self.parent._data.get_subset(self.name):
            for ann in item.annotations:
                annotations_by_type[ann.type.name]["count"] += 1
        return sum(t["count"] for t in annotations_by_type.values())

    def as_dataset(self) -> Dataset:
        dataset = Dataset.from_extractors(self, env=self.parent.env)
        dataset._format = self.parent._format
        dataset._source_path = self.parent._source_path
        return dataset


class Dataset(IDataset):
    """
    Represents a dataset, contains metainfo about labels and dataset items.
    Provides iteration and access options to dataset elements.

    By default, all operations are done lazily, it can be changed by
    modifying the `eager` property and by using the `eager_mode`
    context manager.

    Dataset is supposed to have a single media type for its items. If the
    dataset is filled manually or from extractors, and media type does not
    match, an error is raised.
    """

    _global_eager: bool = False
    _stream = False

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[DatasetItem],
        infos: Optional[DatasetInfo] = None,
        categories: Union[CategoriesInfo, List[str], None] = None,
        *,
        env: Optional[Environment] = None,
        media_type: Type[MediaElement] = Image,
        ann_types: Optional[Set[AnnotationType]] = [],
    ) -> Dataset:
        """
        Creates a new dataset from an iterable object producing dataset items -
        a generator, a list etc. It is a convenient way to create and fill
        a custom dataset.

        Parameters:
            iterable: An iterable which returns dataset items
            infos: A dictionary of the dataset specific information
            categories: A simple list of labels or complete information
                about labels. If not specified, an empty list of labels
                is assumed.
            media_type: Media type for the dataset items. If the sequence
                contains items with mismatching media type, an error is
                raised during caching
            env: A context for plugins, which will be used for this dataset.
                If not specified, the builtin plugins will be used.

        Returns:
            dataset: A new dataset with specified contents
        """

        if infos is None:
            infos = {}

        # TODO: remove the default value for media_type
        # https://github.com/open-edge-platform/datumaro/issues/675

        if isinstance(categories, list):
            categories = {AnnotationType.label: LabelCategories.from_iterable(categories)}

        if not categories:
            categories = {}

        class _extractor(DatasetBase):
            def __init__(self):
                super().__init__(
                    length=len(iterable) if hasattr(iterable, "__len__") else None,
                    media_type=media_type,
                    ann_types=ann_types,
                )

            def __iter__(self):
                return iter(iterable)

            def infos(self):
                return infos

            def categories(self):
                return categories

        return cls.from_extractors(_extractor(), env=env)

    @classmethod
    def from_extractors(
        cls,
        *sources: IDataset,
        env: Optional[Environment] = None,
        merge_policy: str = DEFAULT_MERGE_POLICY,
    ) -> Dataset:
        """
        Creates a new dataset from one or several `Extractor`s.

        In case of a single input, creates a lazy wrapper around the input.
        In case of several inputs, merges them and caches the resulting
        dataset.

        Parameters:
            sources: one or many input extractors
            env: A context for plugins, which will be used for this dataset.
                If not specified, the builtin plugins will be used.
            merge_policy: Policy on how to merge multiple datasets.
                Possible options are "exact", "intersect", and "union".

        Returns:
            dataset: A new dataset with contents produced by input extractors
        """

        if len(sources) == 1:
            source = sources[0]
            dataset = cls(source=source, env=env)
        else:
            from datumaro.components.hl_ops import HLOps

            return HLOps.merge(*sources, merge_policy=merge_policy)

        return dataset

    def __init__(
        self,
        source: Optional[IDataset] = None,
        *,
        infos: Optional[DatasetInfo] = None,
        categories: Optional[CategoriesInfo] = None,
        media_type: Optional[Type[MediaElement]] = None,
        ann_types: Optional[Set[AnnotationType]] = None,
        env: Optional[Environment] = None,
    ) -> None:
        super().__init__()

        assert env is None or isinstance(env, Environment), env
        self._env = env

        self.eager = None
        self._data = DatasetStorage(
            source=source,
            infos=infos,
            categories=categories,
            media_type=media_type,
            ann_types=ann_types,
        )
        if self.is_eager:
            self.init_cache()

        self._format = DEFAULT_FORMAT
        self._source_path = None
        self._options = {}

    def __repr__(self) -> str:
        separator = "\t"
        return (
            f"Dataset\n"
            f"\tsize={len(self._data)}\n"
            f"\tsource_path={self._source_path}\n"
            f"\tmedia_type={self.media_type()}\n"
            f"\tann_types={self.ann_types()}\n"
            f"\tannotated_items_count={self.get_annotated_items()}\n"
            f"\tannotations_count={self.get_annotations()}\n"
            f"subsets\n"
            f"\t{separator.join(self.get_subset_info())}"
            f"infos\n"
            f"\t{separator.join(self.get_infos())}"
            f"categories\n"
            f"\t{separator.join(self.get_categories_info())}"
        )

    def define_infos(self, infos: DatasetInfo) -> None:
        self._data.define_infos(infos)

    def define_categories(self, categories: CategoriesInfo) -> None:
        self._data.define_categories(categories)

    def init_cache(self) -> None:
        self._data.init_cache()

    def __iter__(self) -> Iterator[DatasetItem]:
        yield from self._data

    def __len__(self) -> int:
        return len(self._data)

    def get_subset(self, name) -> DatasetSubset:
        return DatasetSubset(self, name)

    def subsets(self) -> Dict[str, DatasetSubset]:
        return {k: self.get_subset(k) for k in self._data.subsets()}

    def infos(self) -> DatasetInfo:
        return self._data.infos()

    def categories(self) -> CategoriesInfo:
        return self._data.categories()

    def media_type(self) -> Type[MediaElement]:
        return self._data.media_type()

    def ann_types(self) -> Set[AnnotationType]:
        return self._data.ann_types()

    def get(self, id: str, subset: Optional[str] = None) -> Optional[DatasetItem]:
        return self._data.get(id, subset)

    def get_annotated_items(self):
        return self._data.get_annotated_items()

    def get_annotations(self):
        return self._data.get_annotations()

    def get_datasetitem_by_path(self, path):
        if self._source_path not in path:
            path = osp.join(self._source_path, path)
        return self._data.get_datasetitem_by_path(path)

    def get_label_cat_names(self):
        return [
            label.name
            for label in self._data.categories().get(AnnotationType.label, LabelCategories())
        ]

    def get_subset_info(self) -> str:
        return (
            f"{subset_name}: # of items={len(self.get_subset(subset_name))}, "
            f"# of annotated items={self.get_subset(subset_name).get_annotated_items()}, "
            f"# of annotations={self.get_subset(subset_name).get_annotations()}\n"
            for subset_name in sorted(self.subsets().keys())
        )

    def get_infos(self) -> Tuple[str]:
        if self.infos() is not None:
            return (f"{k}: {v}\n" for k, v in self.infos().items())
        else:
            return ("\n",)

    def get_categories_info(self) -> Tuple[str]:
        category_dict = {}
        for annotation_type, category in self.categories().items():
            if isinstance(category, LabelCategories):
                category_names = list(category._indices.keys())
                category_dict[annotation_type] = category_names
            if isinstance(category, TabularCategories):
                category_names = list(category._indices_by_name.keys())
                category_dict[annotation_type] = category_names
        return (
            f"{str(annotation_type).split('.')[-1]}: {list(category_dict.get(annotation_type, []))}\n"
            for annotation_type in self.categories().keys()
        )

    def __contains__(self, x: Union[DatasetItem, str, Tuple[str, str]]) -> bool:
        if isinstance(x, DatasetItem):
            x = (x.id, x.subset)
        elif not isinstance(x, (tuple, list)):
            x = [x]
        return self.get(*x) is not None

    def put(
        self, item: DatasetItem, id: Optional[str] = None, subset: Optional[str] = None
    ) -> None:
        overrides = {}
        if id is not None:
            overrides["id"] = id
        if subset is not None:
            overrides["subset"] = subset
        if overrides:
            item = item.wrap(**overrides)

        self._data.put(item)

    def remove(self, id: str, subset: Optional[str] = None) -> None:
        self._data.remove(id, subset)

    @overload
    def filter(
        self,
        expr: str,
        *,
        filter_annotations: bool = False,
        remove_empty: bool = False,
    ) -> Dataset:
        """
        Filters out some dataset items or annotations, using a custom filter
        expression.

        Results are stored in-place. Modifications are applied lazily.

        Args:
            expr: XPath-formatted filter expression
                (e.g. `/item[subset = 'train']`,
                `/item/annotation[label = 'cat']`)
            filter_annotations: Indicates if the filter should be
                applied to items or annotations
            remove_empty: When filtering annotations, allows to
                exclude empty items from the resulting dataset

        Returns: self
        """
        ...

    @overload
    def filter(
        self,
        filter_func: Union[
            Callable[[DatasetItem], bool], Callable[[DatasetItem, Annotation], bool]
        ],
        *,
        filter_annotations: bool = False,
        remove_empty: bool = False,
    ) -> Dataset:
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

        Returns: self

        Example:
            - (`filter_annotations=False`) This is an example of filtering
                dataset items with images larger than 1024 pixels::

                from datumaro.components.media import Image

                def filter_func(item: DatasetItem) -> bool:
                    h, w = item.media_as(Image).size
                    return h > 1024 or w > 1024

                filtered = dataset.filter(filter_func=filter_func, filter_annotations=False)
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

                filtered = dataset.filter(filter_func=filter_func, filter_annotations=True)
                # No bounding boxes with a size greater than 50% of their image
                filtered_items = [item for item in filtered]
        """
        ...

    def filter(
        self,
        expr_or_filter_func: Union[
            str, Callable[[DatasetItem], bool], Callable[[DatasetItem, Annotation], bool]
        ],
        *,
        filter_annotations: bool = False,
        remove_empty: bool = False,
    ) -> Dataset:
        if isinstance(expr_or_filter_func, str):
            expr = expr_or_filter_func
            return (
                self.transform(XPathAnnotationsFilter, xpath=expr, remove_empty=remove_empty)
                if filter_annotations
                else self.transform(XPathDatasetFilter, xpath=expr)
            )
        elif callable(expr_or_filter_func):
            filter_func = expr_or_filter_func
            return (
                self.transform(
                    UserFunctionAnnotationsFilter,
                    filter_func=filter_func,
                    remove_empty=remove_empty,
                )
                if filter_annotations
                else self.transform(UserFunctionDatasetFilter, filter_func=filter_func)
            )
        raise TypeError(expr_or_filter_func)

    def update(self, source: Union[DatasetPatch, IDataset, Iterable[DatasetItem]]) -> Dataset:
        """
        Updates items of the current dataset from another dataset or an
        iterable (the source). Items from the source overwrite matching
        items in the current dataset. Unmatched items are just appended.

        If the source is a DatasetPatch, the removed items in the patch
        will be removed in the current dataset.

        If the source is a dataset, labels are matched. If the labels match,
        but the order is different, the annotation labels will be remapped to
        the current dataset label order during updating.

        Returns: self
        """

        self._data.update(source)
        return self

    def transform(self, method: Union[str, Type[Transform]], **kwargs) -> Dataset:
        """
        Applies some function to dataset items.

        Results are stored in-place. Modifications are applied lazily.
        Transforms are not allowed to change media type of dataset items.

        Args:
            method: The transformation to be applied to the dataset.
                If a string is passed, it is treated as a plugin name,
                which is searched for in the dataset environment.
            **kwargs: Parameters for the transformation

        Returns: self
        """

        if isinstance(method, str):
            method = self.env.transforms[method]

        if not (inspect.isclass(method) and issubclass(method, Transform)):
            raise TypeError("Unexpected 'method' argument type: %s" % type(method))

        self._data.transform(method, **kwargs)
        if self.is_eager:
            self.init_cache()

        return self

    def select(self, pred: Callable[[DatasetItem], bool]) -> Dataset:
        class _DatasetFilter(ItemTransform):
            def transform_item(self, item):
                if pred(item):
                    return item
                return None

        return self.transform(_DatasetFilter)

    @property
    def data_path(self) -> Optional[str]:
        return self._source_path

    @property
    def format(self) -> Optional[str]:
        return self._format

    @property
    def options(self) -> Dict[str, Any]:
        return self._options

    @property
    def is_modified(self) -> bool:
        return self._data.has_updated_items()

    def get_patch(self) -> DatasetPatch:
        return self._data.get_patch()

    @property
    def env(self) -> Environment:
        if self._env is None:
            return DEFAULT_ENVIRONMENT
        return self._env

    @property
    def is_cache_initialized(self) -> bool:
        return self._data.is_cache_initialized()

    @property
    def is_eager(self) -> bool:
        return self.eager if self.eager is not None else self._global_eager

    @property
    def is_bound(self) -> bool:
        return bool(self._source_path) and bool(self._format)

    def bind(
        self, path: str, format: Optional[str] = None, *, options: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Binds the dataset to a speific directory.
        Allows to set default saving parameters.

        The following saves will be done to this directory by default and will
        use the saved parameters.
        """

        self._source_path = path
        self._format = format or DEFAULT_FORMAT
        self._options = options or {}

    def flush_changes(self):
        self._data.flush_changes()

    @scoped
    def export(
        self,
        save_dir: str,
        format: Union[str, Type[Exporter]],
        *,
        progress_reporter: Optional[ProgressReporter] = None,
        error_policy: Optional[ExportErrorPolicy] = None,
        **kwargs,
    ) -> None:
        """
        Saves the dataset in some format.

        Args:
            save_dir: The output directory
            format: The desired output format.
                If a string is passed, it is treated as a plugin name,
                which is searched for in the dataset environment.
            progress_reporter: An object to report progress
            error_policy: An object to report format-related errors
            **kwargs: Parameters for the format
        """

        if not save_dir:
            raise ValueError("Dataset export path is not specified")

        inplace = save_dir == self._source_path and format == self._format

        if isinstance(format, str):
            exporter = self.env.exporters[format]
        else:
            exporter = format

        if not (inspect.isclass(exporter) and issubclass(exporter, Exporter)):
            raise TypeError("Unexpected 'format' argument type: %s" % type(exporter))

        save_dir = osp.abspath(save_dir)
        if not osp.exists(save_dir):
            on_error_do(shutil.rmtree, save_dir, ignore_errors=True)
            inplace = False
        os.makedirs(save_dir, exist_ok=True)

        has_ctx_args = progress_reporter is not None or error_policy is not None

        if not progress_reporter:
            progress_reporter = NullProgressReporter()

        assert "ctx" not in kwargs
        exporter_kwargs = copy(kwargs)
        exporter_kwargs["stream"] = self._stream
        exporter_kwargs["ctx"] = ExportContext(
            progress_reporter=progress_reporter, error_policy=error_policy
        )

        try:
            if not inplace:
                try:
                    exporter.convert(self, save_dir=save_dir, **exporter_kwargs)
                except TypeError as e:
                    # TODO: for backward compatibility. To be removed after 0.3
                    if "unexpected keyword argument 'ctx'" not in str(e):
                        raise

                    if has_ctx_args:
                        warnings.warn(
                            f"It seems that '{format}' exporter "
                            "does not support progress and error reporting, "
                            "It will be disabled in datumaro==1.5.0.",
                            DeprecationWarning,
                        )
                    exporter_kwargs.pop("ctx")

                    exporter.convert(self, save_dir=save_dir, **exporter_kwargs)
            else:
                try:
                    exporter.patch(self, self.get_patch(), save_dir=save_dir, **exporter_kwargs)
                except TypeError as e:
                    # TODO: for backward compatibility. To be removed after 0.3
                    if "unexpected keyword argument 'ctx'" not in str(e):
                        raise

                    if has_ctx_args:
                        warnings.warn(
                            f"It seems that '{format}' exporter "
                            "does not support progress and error reporting, "
                            "It will be disabled in datumaro==1.5.0.",
                            DeprecationWarning,
                        )
                    exporter_kwargs.pop("ctx")

                    exporter.patch(self, self.get_patch(), save_dir=save_dir, **exporter_kwargs)
        except _ExportFail as e:
            raise e.__cause__

        self.bind(save_dir, format, options=copy(kwargs))
        if not self._stream:
            self.flush_changes()

    def save(self, save_dir: Optional[str] = None, **kwargs) -> None:
        options = dict(self._options)
        options.update(kwargs)

        self.export(save_dir or self._source_path, format=self._format, **options)

    @classmethod
    def load(cls, path: str, **kwargs) -> Dataset:
        return cls.import_from(path, format=DEFAULT_FORMAT, **kwargs)

    @classmethod
    def import_from(
        cls,
        path: str,
        format: Optional[str] = None,
        *,
        env: Optional[Environment] = None,
        progress_reporter: Optional[ProgressReporter] = None,
        error_policy: Optional[ImportErrorPolicy] = None,
        **kwargs,
    ) -> Dataset:
        """
        Creates a `Dataset` instance from a dataset on the disk.

        Args:
            path - The input file or directory path
            format - Dataset format.
                If a string is passed, it is treated as a plugin name,
                which is searched for in the `env` plugin context.
                If not set, will try to detect automatically,
                using the `env` plugin context.
            env - A plugin collection. If not set, the built-in plugins are used
            progress_reporter - An object to report progress.
                Implies earger loading.
            error_policy - An object to report format-related errors.
                Implies earger loading.
            **kwargs - Parameters for the format
        """

        if env is None:
            env = DEFAULT_ENVIRONMENT

        if not format:
            format = cls.detect(path, env=env)

        extractor_merger = None
        # TODO: remove importers, put this logic into extractors
        if format in env.importers:
            importer = env.make_importer(format)
            with logging_disabled(log.INFO):
                detected_sources = (
                    importer(path, stream=cls._stream, **kwargs)
                    if importer.can_stream
                    else importer(path, **kwargs)
                )
            extractor_merger = importer.get_extractor_merger()
        elif format in env.extractors:
            detected_sources = [{"url": path, "format": format, "options": kwargs}]
        else:
            raise UnknownFormatError(format)

        # TODO: probably, should not be available in lazy mode, because it
        # becomes unreliable and error-prone. For progress reporting it
        # makes little sense, because loading stage is spread over other
        # operations. Error reporting is going to be unreliable.
        has_ctx_args = progress_reporter is not None or error_policy is not None
        eager = has_ctx_args

        if not progress_reporter:
            progress_reporter = NullProgressReporter()
        pbars = progress_reporter.split(len(detected_sources))

        try:
            extractors = []
            for src_conf, pbar in zip(detected_sources, pbars):
                if not isinstance(src_conf, Source):
                    src_conf = Source(src_conf)

                extractor_kwargs = dict(src_conf.options)

                assert "ctx" not in extractor_kwargs
                extractor_kwargs["ctx"] = ImportContext(
                    progress_reporter=pbar, error_policy=error_policy
                )

                merge_policy = extractor_kwargs.get("merge_policy", DEFAULT_MERGE_POLICY)

                try:
                    extractors.append(
                        env.make_extractor(src_conf.format, src_conf.url, **extractor_kwargs)
                    )
                except TypeError as e:
                    # TODO: for backward compatibility. To be removed after 0.3
                    if "unexpected keyword argument 'ctx'" not in str(e):
                        raise

                    if has_ctx_args:
                        warnings.warn(
                            f"It seems that '{src_conf.format}' extractor "
                            "does not support progress and error reporting. "
                            "It will be disabled in datumaro==1.5.0.",
                            DeprecationWarning,
                        )
                    extractor_kwargs.pop("ctx")

                    extractors.append(
                        env.make_extractor(src_conf.format, src_conf.url, **extractor_kwargs)
                    )
            dataset = (
                cls(source=extractor_merger(extractors), env=env)
                if extractor_merger is not None
                else cls.from_extractors(*extractors, env=env, merge_policy=merge_policy)
            )
            if eager:
                dataset.init_cache()
        except _ImportFail as e:
            cause = e.__cause__ if getattr(e, "__cause__", None) is not None else e
            cause.__traceback__ = e.__traceback__
            raise DatasetImportError(f"Failed to import dataset '{format}' at '{path}'.") from cause
        except DatumaroError as e:
            cause = e.__cause__ if getattr(e, "__cause__", None) is not None else e
            cause.__traceback__ = e.__traceback__
            raise DatasetImportError(f"Failed to import dataset '{format}' at '{path}'.") from cause
        except Exception as e:
            raise DatasetImportError(f"Failed to import dataset '{format}' at '{path}'.") from e

        dataset._source_path = path
        dataset._format = format

        return dataset

    @staticmethod
    def detect(path: str, *, env: Optional[Environment] = None, depth: int = 2) -> str:
        """
        Attempts to detect dataset format of a given directory.

        This function tries to detect a single format and fails if it's not
        possible. Check Environment.detect_dataset() for a function that
        reports status for each format checked.

        Args:
            path: The directory to check
            depth: The maximum depth for recursive search
            env: A plugin collection. If not set, the built-in plugins are used
        """

        if env is None:
            env = Environment()

        if depth < 0:
            raise ValueError("Depth cannot be less than zero")

        matches = env.detect_dataset(path, depth=depth)
        if not matches:
            raise NoMatchingFormatsError()
        elif 1 < len(matches):
            raise MultipleFormatsMatchError(matches)
        else:
            return matches[0]

    @property
    def is_stream(self) -> bool:
        return self._data.is_stream

    def clone(self) -> "Dataset":
        """Create a deep copy of this dataset.

        Returns:
            A cloned instance of the `Dataset`.
        """
        return deepcopy(self)

    def __getitem__(self, idx: int) -> DatasetItem:
        if not self._data.is_stream:
            return self._data[idx]
        raise StreamedItemError()


class StreamDataset(Dataset):
    _stream = True

    def __init__(
        self,
        source: Optional[IDataset] = None,
        *,
        infos: Optional[DatasetInfo] = None,
        categories: Optional[CategoriesInfo] = None,
        media_type: Optional[Type[MediaElement]] = None,
        ann_types: Optional[Set[AnnotationType]] = None,
        env: Optional[Environment] = None,
    ) -> None:
        assert env is None or isinstance(env, Environment), env
        self._env = env

        self._data = StreamDatasetStorage(
            source,
            infos=infos,
            categories=categories,
            media_type=media_type,
            ann_types=ann_types,
        )

        self._format = DEFAULT_FORMAT
        self._source_path = None
        self._options = {}

    @property
    def is_eager(self) -> bool:
        return False

    @classmethod
    def from_extractors(
        cls,
        *sources: IDataset,
        env: Optional[Environment] = None,
        merge_policy: str = DEFAULT_MERGE_POLICY,
    ) -> Dataset:
        """
        Creates a new dataset from one or several `Extractor`s.

        In case of a single input, creates a lazy wrapper around the input.
        In case of several inputs, unifies them and caches the resulting
        dataset. We cannot apply regular dataset merge, since items list cannot be accessed.

        Parameters:
            sources: one or many input extractors
            env: A context for plugins, which will be used for this dataset.
                If not specified, the builtin plugins will be used.
            merge_policy: Policy on how to merge multiple datasets.
                Possible options are "exact", "intersect", and "union".

        Returns:
            dataset: A new dataset with contents produced by input extractors
        """

        if len(sources) == 1:
            source = sources[0]
            dataset = cls(source=source, env=env)
        else:

            class _MergedStreamDataset(cls):
                def __init__(self, *sources: IDataset):
                    from datumaro.components.hl_ops import HLOps

                    self._merged = HLOps.merge(*sources, merge_policy=merge_policy)
                    self._data = self._merged._data
                    self._env = env
                    self._format = DEFAULT_FORMAT
                    self._source_path = None
                    self._options = {}

                def __iter__(self):
                    yield from self._merged

                @property
                def is_stream(self):
                    return True

                def subsets(self) -> Dict[str, DatasetSubset]:
                    return self._merged.subsets()

            return _MergedStreamDataset(*sources)

        return dataset


@contextmanager
def eager_mode(new_mode: bool = True, dataset: Optional[Dataset] = None) -> None:
    if dataset is not None:
        old_mode = dataset.eager

        try:
            dataset.eager = new_mode
            yield
        finally:
            dataset.eager = old_mode
    else:
        old_mode = Dataset._global_eager

        try:
            Dataset._global_eager = new_mode
            yield
        finally:
            Dataset._global_eager = old_mode
