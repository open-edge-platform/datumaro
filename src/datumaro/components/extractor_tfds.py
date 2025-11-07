# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import itertools
import logging as log
import os.path as osp
from importlib.util import find_spec
from types import SimpleNamespace as namespace
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Set, Tuple, Type, Union

import attrs
import numpy as np
from attrs import field, frozen

from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories
from datumaro.components.dataset_base import CategoriesInfo, DatasetInfo, DatasetItem, IDataset
from datumaro.components.media import Image, MediaElement
from datumaro.util.import_util import lazy_import
from datumaro.util.tf_util import import_tf

TFDS_EXTRACTOR_AVAILABLE = True if find_spec("tensorflow_datasets") is not None else False

if TFDS_EXTRACTOR_AVAILABLE:
    try:
        if TYPE_CHECKING:
            tf = import_tf()
            import tensorflow_datasets as tfds
        else:
            tfds = lazy_import("tensorflow_datasets")
    except ImportError:
        log.debug("Unable to import TensorFlow or TensorFlow Datasets. Dataset downloading via TFDS is disabled.")


@frozen(kw_only=True)
class TfdsDatasetMetadata:
    # If you add attributes to this class, make sure to update the reporting logic
    # in the `describe-downloads` command to include them.

    human_name: str
    default_output_format: str
    media_type: Type[MediaElement]

    # Every TFDS dataset has a home page (the TFDS documentation page), but
    # this field is still optional for a couple of reasons:
    # * We might want to reuse this class later if we implement downloading
    #   without the use of TFDS, and at that point it might happen that a dataset
    #   has no home page (e.g. it's been taken down).
    # * It's convenient to initially leave this field blank and mass-update it later,
    #   and in order for us to not break typing, it should be optional.
    home_url: Optional[str] = None


@frozen
class _TfdsAdapter:
    category_transformers: Sequence[Callable[[tfds.core.DatasetBuilder, CategoriesInfo, namespace], None]]
    data_transformers: Sequence[Callable[[Any, DatasetItem, namespace], None]]
    id_generator: Callable[[Any], str] = field(default=None, kw_only=True)

    metadata: TfdsDatasetMetadata

    def transform_categories(
        self,
        tfds_builder: tfds.core.DatasetBuilder,
        categories: CategoriesInfo,
        state: namespace,
    ) -> None:
        for t in self.category_transformers:
            t(tfds_builder, categories, state)

    def transform_data(
        self,
        tfds_example: Any,
        item: DatasetItem,
        state: namespace,
    ) -> None:
        for t in self.data_transformers:
            t(tfds_example, item, state)


_FeaturePath = Union[str, Tuple[str, ...]]


def _resolve_feature_path(
    feature_path: _FeaturePath,
    root: tfds.features.FeaturesDict,
) -> tfds.features.FeatureConnector:
    if isinstance(feature_path, str):
        return root[feature_path]

    feature_connector = root

    for segment in feature_path:
        assert isinstance(
            feature_connector,
            (
                tfds.features.FeaturesDict,
                tfds.features.Sequence,
            ),
        )

        if isinstance(feature_connector, tfds.features.Sequence):
            assert segment == "feature"
            feature_connector = feature_connector.feature
        else:
            feature_connector = feature_connector[segment]

    return feature_connector


@frozen
class _SetLabelCategoriesFromClassLabelFeature:
    feature_path: _FeaturePath

    def __call__(
        self,
        tfds_builder: tfds.core.DatasetBuilder,
        categories: CategoriesInfo,
        state: namespace,
    ) -> None:
        assert AnnotationType.label not in categories

        feature_connector = _resolve_feature_path(self.feature_path, tfds_builder.info.features)

        assert isinstance(feature_connector, tfds.features.ClassLabel)
        categories[AnnotationType.label] = LabelCategories.from_iterable(feature_connector.names)


@frozen
class _SetImageFromImageFeature:
    feature_name: str
    filename_feature_name: Optional[str] = field(default=None)

    def __call__(
        self,
        tfds_example: Any,
        item: DatasetItem,
        state: namespace,
    ) -> None:
        if self.filename_feature_name:
            filename = tfds_example[self.filename_feature_name].numpy().decode("UTF-8")
            if osp.exists(filename):
                item.media = Image.from_file(path=filename)
                return
        data = tfds_example[self.feature_name].numpy()
        if isinstance(data, np.ndarray):
            item.media = Image.from_numpy(data=data)
        else:
            item.media = Image.from_bytes(data=data)


@frozen
class _AddLabelFromClassLabelFeature:
    feature_name: str

    def __call__(
        self,
        tfds_example: Any,
        item: DatasetItem,
        state: namespace,
    ) -> None:
        item.annotations.append(
            Label(tfds_example[self.feature_name].numpy()),
        )


@frozen
class _AttributeMemberMapping:
    member_name: str
    attribute_name: str = field()
    value_converter: Optional[Callable[[Any, namespace], Any]] = None

    @attribute_name.default
    def _attribute_name_default(self):
        return self.member_name


@frozen
class _AddObjectsFromFeature:
    feature_name: str
    bbox_member: str
    label_member: Optional[str] = field(default=None, kw_only=True)
    attribute_members: Tuple[_AttributeMemberMapping, ...] = field(
        default=(),
        kw_only=True,
        converter=lambda values: tuple(
            value if isinstance(value, _AttributeMemberMapping) else _AttributeMemberMapping(value) for value in values
        ),
    )

    def __call__(
        self,
        tfds_example: Any,
        item: DatasetItem,
        state: namespace,
    ) -> None:
        tfds_objects = tfds_example[self.feature_name]
        tfds_bboxes = tfds_objects[self.bbox_member]
        num_objects = tfds_bboxes.shape[0]

        tfds_labels = None
        if self.label_member is not None:
            tfds_labels = tfds_objects[self.label_member]
            assert tfds_labels.shape[0] == num_objects

        for am_mapping in self.attribute_members:
            assert tfds_objects[am_mapping.member_name].shape[0] == num_objects

        for i in range(num_objects):
            norm_ymin, norm_xmin, norm_ymax, norm_xmax = tfds_bboxes[i].numpy()

            new_bbox = Bbox(
                x=norm_xmin * item.media.size[1],
                y=norm_ymin * item.media.size[0],
                w=(norm_xmax - norm_xmin) * item.media.size[1],
                h=(norm_ymax - norm_ymin) * item.media.size[0],
            )

            if tfds_labels is not None:
                new_bbox.label = tfds_labels[i].numpy()

            for am_mapping in self.attribute_members:
                attr_value = tfds_objects[am_mapping.member_name][i].numpy()

                if am_mapping.value_converter:
                    attr_value = am_mapping.value_converter(attr_value, state)

                new_bbox.attributes[am_mapping.attribute_name] = attr_value

            item.annotations.append(new_bbox)


@frozen
class _SetAttributeFromFeature:
    feature_name: str
    attribute_name: str

    def __call__(
        self,
        tfds_example: Any,
        item: DatasetItem,
        state: namespace,
    ) -> None:
        item.attributes[self.attribute_name] = tfds_example[self.feature_name].numpy()


@frozen
class _GenerateIdFromTextFeature:
    feature_name: str

    def __call__(self, tfds_example: Any) -> str:
        return tfds_example[self.feature_name].numpy().decode("UTF-8")


@frozen
class _GenerateIdFromFilenameFeature:
    feature_name: str

    def __call__(self, tfds_example: Any) -> str:
        file_name = tfds_example[self.feature_name].numpy().decode("UTF-8")
        return osp.splitext(file_name)[0]


_MNIST_ADAPTER = _TfdsAdapter(
    category_transformers=[_SetLabelCategoriesFromClassLabelFeature("label")],
    data_transformers=[
        _SetImageFromImageFeature("image"),
        _AddLabelFromClassLabelFeature("label"),
    ],
    metadata=TfdsDatasetMetadata(
        human_name="MNIST",
        default_output_format="mnist",
        media_type=Image,
    ),
)

_CIFAR_ADAPTER = _TfdsAdapter(
    category_transformers=[_SetLabelCategoriesFromClassLabelFeature("label")],
    data_transformers=[
        _SetImageFromImageFeature("image"),
        _AddLabelFromClassLabelFeature("label"),
    ],
    id_generator=_GenerateIdFromTextFeature("id"),
    metadata=TfdsDatasetMetadata(human_name="CIFAR", default_output_format="cifar", media_type=Image),
)

_COCO_ADAPTER = _TfdsAdapter(
    category_transformers=[_SetLabelCategoriesFromClassLabelFeature(("objects", "feature", "label"))],
    data_transformers=[
        _SetImageFromImageFeature("image", filename_feature_name="image/filename"),
        _AddObjectsFromFeature("objects", "bbox", label_member="label", attribute_members=("is_crowd",)),
        _SetAttributeFromFeature("image/id", "id"),
    ],
    id_generator=_GenerateIdFromFilenameFeature("image/filename"),
    metadata=TfdsDatasetMetadata(human_name="COCO", default_output_format="coco_instances", media_type=Image),
)

_IMAGENET_ADAPTER = _TfdsAdapter(
    category_transformers=[_SetLabelCategoriesFromClassLabelFeature("label")],
    data_transformers=[
        _SetImageFromImageFeature("image", filename_feature_name="file_name"),
        _AddLabelFromClassLabelFeature("label"),
    ],
    id_generator=_GenerateIdFromFilenameFeature("file_name"),
    metadata=TfdsDatasetMetadata(human_name="ImageNet", default_output_format="imagenet_txt", media_type=Image),
)

_EUROSAT_ADAPTER = _TfdsAdapter(
    category_transformers=[_SetLabelCategoriesFromClassLabelFeature("label")],
    data_transformers=[
        _SetImageFromImageFeature("image"),
        _AddLabelFromClassLabelFeature("label"),
    ],
    id_generator=_GenerateIdFromFilenameFeature("filename"),
    metadata=TfdsDatasetMetadata(human_name="EuroSAT", default_output_format="imagenet_txt", media_type=Image),
)


_UC_MERCED_ADAPTER = _TfdsAdapter(
    category_transformers=[_SetLabelCategoriesFromClassLabelFeature("label")],
    data_transformers=[
        _SetImageFromImageFeature("image"),
        _AddLabelFromClassLabelFeature("label"),
    ],
    id_generator=_GenerateIdFromFilenameFeature("filename"),
    metadata=TfdsDatasetMetadata(human_name="UCMerced", default_output_format="imagenet_txt", media_type=Image),
)


def _voc_save_pose_names(
    tfds_builder: tfds.core.DatasetBuilder,
    categories: CategoriesInfo,
    state: namespace,
) -> None:
    # TFDS represents poses as indexes, but Datumaro represents them as strings.
    # To convert between representations, save the pose names at the start and
    # use them when we're converting boxes.
    # TFDS also provides the pose names in lower case, even though they're title
    # case in the original dataset. Fix them back to title case so that the
    # output dataset better resembles the original dataset.

    state.pose_names = [name.title() for name in tfds_builder.info.features["objects"].feature["pose"].names]


_VOC_ADAPTER = _TfdsAdapter(
    category_transformers=[
        _SetLabelCategoriesFromClassLabelFeature(("objects", "feature", "label")),
        _voc_save_pose_names,
    ],
    data_transformers=[
        _SetImageFromImageFeature("image", filename_feature_name="image/filename"),
        _AddObjectsFromFeature(
            "objects",
            "bbox",
            label_member="label",
            attribute_members=(
                _AttributeMemberMapping("is_difficult", "difficult"),
                _AttributeMemberMapping("is_truncated", "truncated"),
                _AttributeMemberMapping("pose", value_converter=lambda idx, state: state.pose_names[idx]),
            ),
        ),
    ],
    id_generator=_GenerateIdFromFilenameFeature("image/filename"),
    metadata=TfdsDatasetMetadata(human_name="PASCAL VOC", default_output_format="voc", media_type=Image),
)


def _evolve_adapter_meta(adapter: _TfdsAdapter, **kwargs):
    return attrs.evolve(adapter, metadata=attrs.evolve(adapter.metadata, **kwargs))


_TFDS_ADAPTERS = {
    "cifar10": _evolve_adapter_meta(_CIFAR_ADAPTER, human_name="CIFAR-10"),
    "cifar100": _evolve_adapter_meta(_CIFAR_ADAPTER, human_name="CIFAR-100"),
    "coco/2014": _evolve_adapter_meta(_COCO_ADAPTER, human_name="COCO (2014-2015)"),
    "imagenet_v2": _evolve_adapter_meta(_IMAGENET_ADAPTER, human_name="ImageNetV2"),
    "mnist": _MNIST_ADAPTER,
    "voc/2012": _evolve_adapter_meta(_VOC_ADAPTER, human_name="PASCAL VOC 2012"),
    "eurosat": _evolve_adapter_meta(_EUROSAT_ADAPTER, human_name="EuroSAT"),
    "uc_merced": _evolve_adapter_meta(_UC_MERCED_ADAPTER, human_name="UCMerced"),
}

# Assign the TFDS catalog page as the documentation URL for all datasets.
_TFDS_ADAPTERS = {
    name: _evolve_adapter_meta(
        adapter,
        home_url="https://www.tensorflow.org/datasets/catalog/" + name.split("/", maxsplit=1)[0],
    )
    for name, adapter in _TFDS_ADAPTERS.items()
}


class _TfdsSplitExtractor(IDataset):
    def __init__(
        self,
        parent: _TfdsExtractor,
        tfds_split: tf.data.Dataset,
        tfds_split_info: tfds.core.SplitInfo,
    ):
        self._parent = parent
        self._tfds_split = tfds_split
        self._tfds_split_info = tfds_split_info

    def __len__(self) -> int:
        return self._tfds_split_info.num_examples

    def __iter__(self) -> Iterator[DatasetItem]:
        for example_index, tfds_example in enumerate(self._tfds_split):
            if self._parent._adapter.id_generator:
                item_id = self._parent._adapter.id_generator(tfds_example)
            else:
                item_id = str(example_index)

            dm_item = DatasetItem(id=item_id, subset=self._tfds_split_info.name)
            self._parent._adapter.transform_data(tfds_example, dm_item, self._parent._state)

            yield dm_item

    def infos(self) -> DatasetInfo:
        return self._parent.infos()

    def categories(self) -> CategoriesInfo:
        return self._parent.categories()

    def subsets(self) -> Dict[str, IDataset]:
        return {self._tfds_split_info.name: self}

    def get_subset(self, name) -> IDataset:
        assert name == self._tfds_split_info.name
        return self

    def get(self, id, subset=None) -> Optional[DatasetItem]:
        if subset is not None and subset != self._tfds_split_info.name:
            return None

        for item in self:
            if item.id == id:
                return item

        return None

    def media_type(self) -> Type[MediaElement]:
        return self._parent._media_type

    def ann_types(self) -> Set[AnnotationType]:
        return self._parent.ann_types()


class _TfdsExtractor(IDataset):
    _categories: CategoriesInfo
    _infos: DatasetInfo

    def __init__(self, tfds_ds_name: str) -> None:
        self._adapter = _TFDS_ADAPTERS[tfds_ds_name]
        tfds_builder = tfds.builder(tfds_ds_name)
        tfds_ds_info = tfds_builder.info

        self._infos = {}
        self._categories = {}
        self._state = namespace()
        self._adapter.transform_categories(tfds_builder, self._categories, self._state)
        self._media_type = self._adapter.metadata.media_type

        tfds_decoders = {}
        for tfds_feature_name, tfds_fc in tfds_ds_info.features.items():
            if isinstance(tfds_fc, tfds.features.Image):
                tfds_decoders[tfds_feature_name] = tfds.decode.SkipDecoding()

        tfds_builder.download_and_prepare()
        self._tfds_ds = tfds_builder.as_dataset(decoders=tfds_decoders)

        self._split_extractors = {
            split_name: _TfdsSplitExtractor(self, split, tfds_ds_info.splits[split_name])
            # Since dicts in Python 3.7+ (and de facto in 3.6+) are
            # order-preserving, sort the splits by name so that we always
            # iterate over them in alphabetical order.
            for split_name, split in sorted(self._tfds_ds.items())
        }

    def __len__(self) -> int:
        return sum(len(ex) for ex in self._split_extractors.values())

    def __iter__(self) -> Iterator[DatasetItem]:
        return itertools.chain.from_iterable(self._split_extractors.values())

    def infos(self) -> DatasetInfo:
        return self._infos

    def categories(self) -> CategoriesInfo:
        return self._categories

    def subsets(self) -> Dict[str, IDataset]:
        return self._split_extractors

    def get_subset(self, name) -> IDataset:
        return self._split_extractors[name]

    def get(self, id, subset=None) -> Optional[DatasetItem]:
        if subset is None:
            for ex in self._split_extractors.values():
                item = ex.get(id)
                if item is not None:
                    return item
            return None

        if subset not in self._split_extractors:
            return None
        return self._split_extractors[subset].get(id)

    def media_type(self) -> Type[MediaElement]:
        return self._media_type

    def ann_types(self) -> Set[AnnotationType]:
        ann_types = set()
        for items in self._split_extractors.values():
            for item in items:
                for ann in item.annotations:
                    ann_types.add(ann.type)
        return ann_types


# Some dataset metadata elements are either inconvenient to hardcode, or may change
# depending on the version of TFDS. We fetch them from the attributes of the `tfds.Builder`
# object. However, creating the builder may be time-consuming, because if the dataset
# is not already downloaded, TFDS fetches some data from its Google Cloud bucket.
# We therefore only fetch this metadata (which we call _remote_ metadata) when
# we actually need it.

# If you add attributes to either of these metadata classes, make sure to update
# the reporting logic in the `describe-downloads` command to include them.


@frozen(kw_only=True)
class TfdsSubsetRemoteMetadata:
    num_items: int


@frozen(kw_only=True)
class TfdsDatasetRemoteMetadata(TfdsDatasetMetadata):
    # For convenience, the remote metadata also includes the local metadata.
    description: str
    download_size: int
    subsets: Mapping[str, TfdsSubsetRemoteMetadata]

    # the dataset might not have the notion of classes, so `num_classes` may
    # be None.
    num_classes: Optional[int]
    version: str


class TfdsDataset:
    def __init__(self, tfds_ds_name: str):
        self._tfds_ds_name = tfds_ds_name
        self._adapter = _TFDS_ADAPTERS[tfds_ds_name]

    @property
    def metadata(self) -> TfdsDatasetMetadata:
        return self._adapter.metadata

    def make_extractor(self) -> IDataset:
        return _TfdsExtractor(self._tfds_ds_name)

    def query_remote_metadata(self) -> TfdsDatasetRemoteMetadata:
        tfds_builder = tfds.builder(self._tfds_ds_name)

        categories = {}
        state = namespace()
        self._adapter.transform_categories(tfds_builder, categories, state)

        num_classes = None
        if AnnotationType.label in categories:
            num_classes = len(categories[AnnotationType.label])

        return TfdsDatasetRemoteMetadata(
            **attrs.asdict(self._adapter.metadata),
            description=tfds_builder.info.description,
            download_size=int(tfds_builder.info.download_size),
            num_classes=num_classes,
            subsets={
                name: TfdsSubsetRemoteMetadata(num_items=split_info.num_examples)
                for name, split_info in tfds_builder.info.splits.items()
            },
            version=str(tfds_builder.info.version),
        )


AVAILABLE_TFDS_DATASETS: Mapping[str, TfdsDataset] = {name: TfdsDataset(name) for name in _TFDS_ADAPTERS}
