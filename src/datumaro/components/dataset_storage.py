# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Type, Union

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset_base import (
    DEFAULT_SUBSET_NAME,
    CategoriesInfo,
    DatasetBase,
    DatasetInfo,
    DatasetItem,
    IDataset,
)
from datumaro.components.dataset_item_storage import (
    DatasetItemStorage,
    DatasetItemStorageDatasetView,
    ItemStatus,
)
from datumaro.components.errors import (
    CategoriesRedefinedError,
    ConflictingCategoriesError,
    DatasetInfosRedefinedError,
    MediaTypeError,
    NotAvailableError,
    RepeatedItemError,
)
from datumaro.components.importer import _ImportFail
from datumaro.components.media import MediaElement
from datumaro.components.transformer import ItemTransform, Transform
from datumaro.util import is_method_redefined

__all__ = ["DatasetPatch", "DatasetStorage"]


class DatasetPatch:
    class DatasetPatchWrapper(DatasetItemStorageDatasetView):
        # The purpose of this class is to indicate that the input dataset is
        # a patch and autofill patch info in Exporter
        def __init__(self, patch: "DatasetPatch", parent: IDataset):
            super().__init__(
                patch.data,
                infos=parent.infos(),
                categories=parent.categories(),
                media_type=parent.media_type(),
                ann_types=parent.ann_types(),
            )
            self.patch = patch

        def subsets(self):
            return {s: self.get_subset(s) for s in self.patch.updated_subsets}

    def __init__(
        self,
        data: DatasetItemStorage,
        infos: DatasetInfo,
        categories: CategoriesInfo,
        updated_items: Dict[Tuple[str, str], ItemStatus],
        updated_subsets: Dict[str, ItemStatus] = None,
    ):
        self.data = data
        self.infos = infos
        self.categories = categories
        self.updated_items = updated_items
        self._updated_subsets = updated_subsets

    @property
    def updated_subsets(self) -> Dict[str, ItemStatus]:
        if self._updated_subsets is None:
            self._updated_subsets = {s: ItemStatus.modified for s in self.data.subsets()}
        return self._updated_subsets

    def __contains__(self, x: Union[DatasetItem, Tuple[str, str]]) -> bool:
        return x in self.data

    def as_dataset(self, parent: IDataset) -> IDataset:
        return __class__.DatasetPatchWrapper(self, parent)


class _StackedTransform(Transform):
    def __init__(self, source: IDataset, transforms: List[Transform]):
        super().__init__(source)

        self.is_local = True
        self.transforms: List[Transform] = []
        self.malformed_transform_indices: Dict[int, Exception] = {}
        for idx, transform in enumerate(transforms):
            try:
                source = transform[0](source, *transform[1], **transform[2])
            except Exception as e:
                self.malformed_transform_indices[idx] = e

            self.transforms.append(source)

            if self.is_local and not isinstance(source, ItemTransform):
                self.is_local = False

    def transform_item(self, item: DatasetItem) -> DatasetItem:
        for t in self.transforms:
            if item is None:
                break
            item = t.transform_item(item)
        return item

    def __iter__(self) -> Iterator[DatasetItem]:
        yield from self.transforms[-1]

    def infos(self) -> DatasetInfo:
        return self.transforms[-1].infos()

    def categories(self) -> CategoriesInfo:
        return self.transforms[-1].categories()

    def media_type(self) -> Type[MediaElement]:
        return self.transforms[-1].media_type()

    def ann_types(self) -> Set[AnnotationType]:
        return self.transforms[-1].ann_types()


class DatasetStorage(IDataset):
    def __init__(
        self,
        source: Union[IDataset, DatasetItemStorage],
        infos: Optional[DatasetInfo] = None,
        categories: Optional[CategoriesInfo] = None,
        media_type: Optional[Type[MediaElement]] = None,
        ann_types: Optional[Set[AnnotationType]] = None,
    ):
        if source is None and categories is None:
            categories = {}
        elif isinstance(source, IDataset) and categories is not None:
            raise ValueError("Can't use both source and categories")
        self._categories = categories

        if source is None and infos is None:
            infos = {}
        elif isinstance(source, IDataset) and infos is not None:
            raise ValueError("Can't use both source and infos")
        self._infos = infos

        if media_type:
            pass
        elif isinstance(source, IDataset) and source.media_type():
            media_type = source.media_type()
        else:
            raise ValueError("Media type must be provided for a dataset")
        assert issubclass(media_type, MediaElement)

        self._media_type = media_type

        if ann_types:
            pass
        elif isinstance(source, IDataset) and source.ann_types():
            ann_types = source.ann_types()
        else:
            ann_types = set()

        self._ann_types = ann_types

        # Possible combinations:
        # 1. source + storage
        #      - Storage contains a patch to the Source data.
        # 2. no source + storage
        #      - a dataset created from scratch
        #      - a dataset from a source or transform, which was cached
        if isinstance(source, DatasetItemStorage):
            self._source = None
            self._storage = source
        else:
            self._source = source
            self._storage = DatasetItemStorage()  # patch or cache
        self._transforms = []  # A stack of postponed transforms

        # Describes changes in the dataset since initialization
        self._updated_items = {}  # (id, subset) -> ItemStatus

        self._flush_changes = False  # Deferred flush indicator

        self._length = len(self._storage) if self._source is None else None

    def is_cache_initialized(self) -> bool:
        return self._source is None and not self._transforms

    @property
    def _is_unchanged_wrapper(self) -> bool:
        return self._source is not None and self._storage.is_empty() and not self._transforms

    def init_cache(self) -> None:
        if not self.is_cache_initialized():
            for _ in self._iter_init_cache():
                pass

    def _iter_init_cache(self) -> Iterable[DatasetItem]:
        try:
            # Can't just return from the method, because it won't add exception handling
            # It covers cases when we save the null error handler in the source
            for item in self._iter_init_cache_unchecked():
                yield item
        except _ImportFail as e:
            raise e.__cause__

    def _iter_init_cache_unchecked(self) -> Iterable[DatasetItem]:
        # Merges the source, source transforms and patch, caches the result
        # and provides an iterator for the resulting item sequence.
        #
        # If iterated in parallel, the result is undefined.
        # If storage is changed during iteration, the result is undefined.
        #
        # TODO: can potentially be optimized by sharing
        # the cache between parallel consumers and introducing some kind of lock
        #
        # Cases:
        # 1. Has source and patch
        # 2. Has source, transforms and patch
        #   a. Transforms affect only an item (i.e. they are local)
        #   b. Transforms affect whole dataset
        #
        # The patch is always applied on top of the source / transforms stack.

        def _update_status(item_id, new_status: ItemStatus):
            current_status = self._updated_items.get(item_id)

            if current_status is None:
                self._updated_items[item_id] = new_status
            elif new_status == ItemStatus.removed:
                if current_status == ItemStatus.added:
                    self._updated_items.pop(item_id)
                else:
                    self._updated_items[item_id] = ItemStatus.removed
            elif new_status == ItemStatus.modified:
                if current_status != ItemStatus.added:
                    self._updated_items[item_id] = ItemStatus.modified
            elif new_status == ItemStatus.added:
                if current_status != ItemStatus.added:
                    self._updated_items[item_id] = ItemStatus.modified
            else:
                assert False, "Unknown status %s" % new_status

        def _add_ann_types(item: DatasetItem):
            for ann in item.annotations:
                self._ann_types.add(ann.type)

        media_type = self._media_type
        patch = self._storage  # must be empty after transforming
        cache = DatasetItemStorage()
        source = self._source or DatasetItemStorageDatasetView(
            self._storage,
            infos=self._infos,
            categories=self._categories,
            media_type=media_type,
            ann_types=self._ann_types,
        )
        transform = None

        if self._transforms:
            transform = _StackedTransform(source, self._transforms)
            if transform.is_local:
                # An optimized way to find modified items:
                # Transform items inplace and analyze transform outputs
                pass
            else:
                # A generic way to find modified items:
                # Collect all the dataset original ids and compare
                # with transform outputs.
                # TODO: introduce DatasetBase.items() / .ids() to avoid extra
                # dataset traversals?
                old_ids = set((item.id, item.subset) for item in source)
                source = transform

            if not issubclass(transform.media_type(), media_type):
                # TODO: make it statically available
                raise MediaTypeError(
                    "Transforms are not allowed to change media " "type of dataset items"
                )

            self._drop_malformed_transforms(transform.malformed_transform_indices)

        i = -1
        for i, item in enumerate(source):
            if item.media and not isinstance(item.media, media_type):
                raise MediaTypeError(
                    "Unexpected media type of a dataset item '%s'. "
                    "Expected '%s', actual '%s' " % (item.id, media_type, type(item.media))
                )

            if transform and transform.is_local:
                old_id = (item.id, item.subset)
                item = transform.transform_item(item)

            item_id = (item.id, item.subset) if item else None

            if item_id in cache:
                raise RepeatedItemError(item_id)

            if item in patch:
                # Apply changes from the patch
                item = patch.get(*item_id)
            elif transform and not self._flush_changes:
                # Find changes made by transforms, if not overridden by patch
                if transform.is_local:
                    if not item:
                        _update_status(old_id, ItemStatus.removed)
                    elif old_id != item_id:
                        _update_status(old_id, ItemStatus.removed)
                        _update_status(item_id, ItemStatus.added)
                    else:
                        # Consider all items modified without comparison,
                        # because such comparison would be very expensive
                        _update_status(old_id, ItemStatus.modified)
                else:
                    if item:
                        if item_id not in old_ids:
                            _update_status(item_id, ItemStatus.added)
                        else:
                            _update_status(item_id, ItemStatus.modified)

            if not item:
                continue

            cache.put(item)
            yield item
            _add_ann_types(item)

        if i == -1:
            cache = patch
            for item in patch:
                if not self._flush_changes:
                    _update_status((item.id, item.subset), ItemStatus.added)
                yield item
                _add_ann_types(item)
        else:
            for item in patch:
                if item in cache:  # already processed
                    continue
                if not self._flush_changes:
                    _update_status((item.id, item.subset), ItemStatus.added)
                cache.put(item)
                yield item
                _add_ann_types(item)

        if not self._flush_changes and transform and not transform.is_local:
            # Mark removed items that were not produced by transforms
            for old_id in old_ids:
                if old_id not in self._updated_items:
                    self._updated_items[old_id] = ItemStatus.removed

        self._storage = cache
        self._length = len(cache)

        if transform:
            source_cat = transform.categories()
        else:
            source_cat = source.categories()
        if source_cat is not None:
            # Don't need to override categories if already defined
            self._categories = source_cat

        if transform:
            source_infos = transform.infos()
        else:
            source_infos = source.infos()
        if source_infos is not None:
            self._infos = source_infos

        self._source = None
        self._transforms = []

        if self._flush_changes:
            self._flush_changes = False
            self._updated_items = {}

    def __iter__(self) -> Iterator[DatasetItem]:
        if self._is_unchanged_wrapper:
            yield from self._iter_init_cache()
        else:
            yield from self._merged()

    def _merged(self) -> IDataset:
        if self._is_unchanged_wrapper:
            return self._source
        elif self._source is not None:
            self.init_cache()
        return DatasetItemStorageDatasetView(
            self._storage,
            infos=self._infos,
            categories=self._categories,
            media_type=self._media_type,
            ann_types=self._ann_types,
        )

    def __len__(self) -> int:
        if self._length is None:
            self.init_cache()
        return self._length

    def infos(self) -> DatasetInfo:
        if self.is_cache_initialized():
            return self._infos
        elif self._infos is not None:
            return self._infos
        elif any(is_method_redefined("infos", Transform, t[0]) for t in self._transforms):
            self.init_cache()
            return self._infos
        else:
            return self._source.infos()

    def define_infos(self, infos: DatasetInfo):
        if self._infos or self._source is not None:
            raise DatasetInfosRedefinedError()
        self._infos = infos

    def categories(self) -> CategoriesInfo:
        if self.is_cache_initialized():
            return self._categories
        elif self._categories is not None:
            return self._categories
        elif any(is_method_redefined("categories", Transform, t[0]) for t in self._transforms):
            self.init_cache()
            return self._categories
        else:
            return self._source.categories()

    def define_categories(self, categories: CategoriesInfo):
        if self._categories or self._source is not None:
            raise CategoriesRedefinedError()
        self._categories = categories

    def media_type(self) -> Type[MediaElement]:
        return self._media_type

    def ann_types(self) -> Set[AnnotationType]:
        self.init_cache()
        return self._ann_types

    def put(self, item: DatasetItem) -> None:
        if item.media and not isinstance(item.media, self._media_type):
            raise MediaTypeError(
                "Mismatching item media type '%s', "
                "the dataset contains '%s' items." % (type(item.media), self._media_type)
            )

        is_new = self._storage.put(item)

        if not self.is_cache_initialized() or is_new:
            self._updated_items[(item.id, item.subset)] = ItemStatus.added
        else:
            self._updated_items[(item.id, item.subset)] = ItemStatus.modified

        if is_new and not self.is_cache_initialized():
            self._length = None
            self._ann_types = set()
        if self._length is not None:
            self._length += is_new

    def get(self, id: str, subset: Optional[str] = None) -> Optional[DatasetItem]:
        id = str(id)
        subset = subset or DEFAULT_SUBSET_NAME

        item = self._storage.get(id, subset)
        if item is None and not self.is_cache_initialized():
            if self._source.get.__func__ == DatasetBase.get:
                # can be improved if IDataset is ABC
                self.init_cache()
                item = self._storage.get(id, subset)
            else:
                item = self._source.get(id, subset)
                if item:
                    self._storage.put(item)
        return item

    def remove(self, id: str, subset: Optional[str] = None) -> None:
        id = str(id)
        subset = subset or DEFAULT_SUBSET_NAME

        self._storage.remove(id, subset)
        is_removed = self._updated_items.get((id, subset)) != ItemStatus.removed
        if is_removed:
            self._updated_items[(id, subset)] = ItemStatus.removed
        if is_removed and not self.is_cache_initialized():
            self._length = None
            self._ann_types = set()
        if self._length is not None:
            self._length -= is_removed

    def get_subset(self, name: str) -> IDataset:
        return self._merged().get_subset(name)

    def subsets(self) -> Dict[str, IDataset]:
        # TODO: check if this can be optimized in case of transforms
        # and other cases
        return self._merged().subsets()

    def get_annotated_items(self) -> int:
        return self._storage.get_annotated_items()

    def get_annotations(self) -> int:
        return self._storage.get_annotations()

    def get_datasetitem_by_path(self, path: str) -> Optional[DatasetItem]:
        return self._storage.get_datasetitem_by_path(path)

    def transform(self, method: Type[Transform], *args, **kwargs) -> None:
        # Flush accumulated changes
        if not self._storage.is_empty():
            source = self._merged()
            self._storage = DatasetItemStorage()
        else:
            source = self._source

        if not self._transforms:
            # The stack of transforms only needs a single source
            self._source = source
        self._transforms.append((method, args, kwargs))

        if is_method_redefined("infos", Transform, method):
            self._infos = None

        if is_method_redefined("categories", Transform, method):
            self._categories = None
        self._length = None
        self._ann_types = set()

    def has_updated_items(self):
        return bool(self._transforms) or bool(self._updated_items)

    def get_patch(self) -> DatasetPatch:
        # Patch includes only added or modified items.
        # To find removed items, one needs to consult updated_items list.
        if self._transforms:
            self.init_cache()

        # The current patch (storage)
        # - can miss some removals done so we add them manually
        # - can include items than not in the patch
        #     (e.g. an item could get there after source was cached)
        # So we reconstruct the patch instead of copying storage.
        patch = DatasetItemStorage()
        for (item_id, subset), status in self._updated_items.items():
            if status is ItemStatus.removed:
                patch.remove(item_id, subset)
            else:
                patch.put(self._storage.get(item_id, subset))

        return DatasetPatch(
            patch, infos=self._infos, categories=self._categories, updated_items=self._updated_items
        )

    def flush_changes(self):
        self._updated_items = {}
        if not (self.is_cache_initialized() or self._is_unchanged_wrapper):
            self._flush_changes = True

    def update(self, source: Union[DatasetPatch, IDataset, Iterable[DatasetItem]]):
        # TODO: provide a more efficient implementation with patch reuse

        if isinstance(source, DatasetPatch):
            if source.categories() != self.categories():
                raise ConflictingCategoriesError()

            for item_id, status in source.updated_items.items():
                if status == ItemStatus.removed:
                    self.remove(*item_id)
                else:
                    self.put(source.data.get(*item_id))
        elif isinstance(source, IDataset):
            from datumaro.plugins.transforms import ProjectLabels

            for item in ProjectLabels(
                source, self.categories().get(AnnotationType.label, LabelCategories())
            ):
                self.put(item)
        else:
            for item in source:
                self.put(item)

    def _drop_malformed_transforms(self, malformed_transform_indices: Dict[int, Exception]) -> None:
        safe_transforms = []
        for idx, transform in enumerate(self._transforms):
            if idx in malformed_transform_indices:
                log.error(
                    f"Automatically drop {transform} from the transform stack because an error is raised. "
                    "Therefore, the dataset will not be transformed by this transformation since it is droped.",
                    exc_info=malformed_transform_indices[idx],
                )
                continue

            safe_transforms += [transform]

        self._transforms = safe_transforms

    def __getitem__(self, idx: int) -> DatasetItem:
        try:
            return self._storage[idx]
        except IndexError:  # Data storage should be initialized
            self.init_cache()
            return self._storage[idx]


class StreamSubset(IDataset):
    def __init__(self, source: IDataset, subset: str) -> None:
        if not source.is_stream:
            raise ValueError("source should be a stream.")
        self._source = source
        self._subset = subset
        self._length = None

    def __iter__(self) -> Iterator[DatasetItem]:
        for item in self._source:
            if item.subset == self._subset:
                yield item

    def __len__(self) -> int:
        if self._length is None:
            self._length = sum(1 for _ in self)
        return self._length

    def subsets(self) -> Dict[str, IDataset]:
        raise NotAvailableError("Cannot get subsets of the subset.")

    def get_subset(self, name) -> IDataset:
        raise NotAvailableError("Cannot get a subset of the subset.")

    def infos(self) -> DatasetInfo:
        return self._source.infos()

    def categories(self) -> CategoriesInfo:
        return self._source.categories()

    def get(self, id: str, subset: Optional[str] = None) -> Optional[DatasetItem]:
        raise NotAvailableError(
            "Random access to the dataset item is not allowed in streaming. "
            "You can access to the dataset item only by using its iterator."
        )

    def media_type(self) -> Type[MediaElement]:
        return self._source.media_type()

    def ann_types(self) -> Set[AnnotationType]:
        return self._source.ann_types()

    @property
    def is_stream(self) -> bool:
        return True


class StreamDatasetStorage(DatasetStorage):
    def __init__(
        self,
        source: IDataset,
        infos: Optional[DatasetInfo] = None,
        categories: Optional[CategoriesInfo] = None,
        media_type: Optional[Type[MediaElement]] = None,
        ann_types: Optional[Set[AnnotationType]] = None,
    ):
        if not source.is_stream:
            raise ValueError("source should be a stream.")
        self._subset_names = list(source.subsets().keys())
        self._transform_ids_for_latest_subset_names = []
        super().__init__(source, infos, categories, media_type, ann_types)

    def is_cache_initialized(self) -> bool:
        log.debug("This function has no effect on streaming.")
        return True

    def init_cache(self) -> None:
        log.debug("This function has no effect on streaming.")
        pass

    @property
    def stacked_transform(self) -> IDataset:
        if self._transforms:
            transform = _StackedTransform(self._source, self._transforms)
            self._drop_malformed_transforms(transform.malformed_transform_indices)
        else:
            transform = self._source

        self._flush_changes = True
        return transform

    def __iter__(self) -> Iterator[DatasetItem]:
        for item in self.stacked_transform:
            yield item

            for ann in item.annotations:
                self._ann_types.add(ann.type)

    def __len__(self) -> int:
        if self._length is None:
            self._length = len(self._source)
        return self._length

    def put(self, item: DatasetItem) -> None:
        raise NotAvailableError("Drop-in replacement is not allowed in streaming.")

    def get(self, id: str, subset: Optional[str] = None) -> Optional[DatasetItem]:
        raise NotAvailableError(
            "Random access to the dataset item is not allowed in streaming. "
            "You can access to the dataset item only by using its iterator."
        )

    def remove(self, id: str, subset: Optional[str] = None) -> None:
        raise NotAvailableError("Drop-in removal is not allowed in streaming.")

    def get_subset(self, name: str) -> IDataset:
        return self.subsets()[name]

    @property
    def subset_names(self):
        if self._transform_ids_for_latest_subset_names != [id(t) for t in self._transforms]:
            self._subset_names = {item.subset for item in self}
            self._transform_ids_for_latest_subset_names = [id(t) for t in self._transforms]

        return self._subset_names

    def subsets(self) -> Dict[str, IDataset]:
        return {subset: StreamSubset(self, subset) for subset in self.subset_names}

    def transform(self, method: Type[Transform], *args, **kwargs) -> None:
        super().transform(method, *args, **kwargs)

    def get_annotated_items(self) -> int:
        return super().get_annotated_items()

    def get_annotations(self) -> int:
        return super().get_annotations()

    def get_datasetitem_by_path(self, path: str) -> Optional[DatasetItem]:
        raise NotAvailableError("Get dataset item by path is not allowed in streaming.")

    def get_patch(self):
        raise NotAvailableError("Get patch is not allowed in streaming.")

    def flush_changes(self):
        raise NotAvailableError("Flush changes is not allowed in streaming.")

    def update(self, source: Union[DatasetPatch, IDataset, Iterable[DatasetItem]]):
        raise NotAvailableError("Update is not allowed in streaming.")

    def infos(self) -> DatasetInfo:
        return self.stacked_transform.infos()

    def categories(self) -> CategoriesInfo:
        return self.stacked_transform.categories()

    @property
    def is_stream(self) -> bool:
        return True
