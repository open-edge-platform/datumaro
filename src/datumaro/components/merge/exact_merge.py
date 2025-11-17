# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

from datumaro.components.annotation import Annotation
from datumaro.components.dataset_base import DatasetItem, IDataset
from datumaro.components.dataset_item_storage import DatasetItemStorage
from datumaro.components.errors import (
    DatasetMergeError,
    MismatchingAttributesError,
    MismatchingImageInfoError,
    MismatchingMediaError,
    MismatchingMediaPathError,
    VideoMergeError,
)
from datumaro.components.media import Image, MediaElement, MultiframeImage, PointCloud, Video
from datumaro.components.merge import Merger

__all__ = ["ExactMerge"]


class ExactMerge(Merger):
    """
    Merges several datasets using the "simple" algorithm:
        - All datasets should have the same categories
        - items are matched by (id, subset) pairs
        - matching items share the media info available:
            - nothing + nothing = nothing
            - nothing + something = something
            - something A + something B = conflict
        - annotations are matched by value and shared
        - in case of conflicts, throws an error
    """

    def __init__(self, **options):
        super().__init__(**options)

    @classmethod
    def merge(cls, sources: Sequence[IDataset]) -> DatasetItemStorage:
        items = DatasetItemStorage()
        for source_idx, source in enumerate(sources):
            for item in source:
                existing_item = items.get(item.id, item.subset)
                if existing_item is not None:
                    try:
                        item = cls.merge_items(existing_item, item)
                    except DatasetMergeError as e:
                        e.sources = set(range(source_idx))
                        raise e

                items.put(item)
        return items

    @classmethod
    def _match_annotations_equal(cls, a, b):
        matches = []
        a_unmatched = a[:]
        b_unmatched = b[:]
        for a_ann in a:
            for b_ann in b_unmatched:
                if a_ann != b_ann:
                    continue

                matches.append((a_ann, b_ann))
                a_unmatched.remove(a_ann)
                b_unmatched.remove(b_ann)
                break

        return matches, a_unmatched, b_unmatched

    @classmethod
    def _merge_annotations_equal(cls, a, b):
        matches, a_unmatched, b_unmatched = cls._match_annotations_equal(a, b)
        return [ann_a for (ann_a, _) in matches] + a_unmatched + b_unmatched

    @classmethod
    def merge_items(cls, existing_item: DatasetItem, current_item: DatasetItem) -> DatasetItem:
        return existing_item.wrap(
            media=cls._merge_media(existing_item, current_item),
            attributes=cls._merge_attrs(
                existing_item.attributes,
                current_item.attributes,
                item_id=(existing_item.id, existing_item.subset),
            ),
            annotations=cls._merge_anno(existing_item.annotations, current_item.annotations),
        )

    @classmethod
    def _merge_attrs(cls, a: Dict[str, Any], b: Dict[str, Any], item_id: Tuple[str, str]) -> Dict:
        merged = {}

        for name in a.keys() | b.keys():
            a_val = a.get(name)
            b_val = b.get(name)

            if name not in a:
                m_val = b_val
            elif name not in b:
                m_val = a_val
            elif a_val != b_val:
                raise MismatchingAttributesError(item_id, name, a_val, b_val)
            else:
                m_val = a_val

            merged[name] = m_val

        return merged

    @classmethod
    def _merge_media(cls, item_a: DatasetItem, item_b: DatasetItem) -> Union[Image, PointCloud, Video]:
        if (not item_a.media or isinstance(item_a.media, Image)) and (
            not item_b.media or isinstance(item_b.media, Image)
        ):
            media = cls._merge_images(item_a, item_b)
        elif (not item_a.media or isinstance(item_a.media, PointCloud)) and (
            not item_b.media or isinstance(item_b.media, PointCloud)
        ):
            media = cls._merge_point_clouds(item_a, item_b)
        elif (not item_a.media or isinstance(item_a.media, Video)) and (
            not item_b.media or isinstance(item_b.media, Video)
        ):
            media = cls._merge_videos(item_a, item_b)
        elif (not item_a.media or isinstance(item_a.media, MultiframeImage)) and (
            not item_b.media or isinstance(item_b.media, MultiframeImage)
        ):
            media = cls._merge_multiframe_images(item_a, item_b)
        elif (not item_a.media or isinstance(item_a.media, MediaElement)) and (
            not item_b.media or isinstance(item_b.media, MediaElement)
        ):
            if isinstance(item_a.media, MediaElement) and isinstance(item_b.media, MediaElement):
                item_a_path = getattr(item_a.media, "path", None)
                item_b_path = getattr(item_b.media, "path", None)

                if item_a_path and item_b_path and item_a_path != item_b_path:
                    raise MismatchingMediaPathError((item_a.id, item_a.subset), item_a_path, item_b_path)
                if item_a_path is None and item_b_path is None:
                    raise MismatchingMediaError((item_a.id, item_a.subset), item_a.media, item_b.media)

                media = item_a.media if item_a_path else item_b.media

            elif isinstance(item_a.media, MediaElement):
                media = item_a.media
            else:
                media = item_b.media
        else:
            raise MismatchingMediaError((item_a.id, item_a.subset), item_a.media, item_b.media)
        return media

    @classmethod
    def _merge_images(cls, item_a: DatasetItem, item_b: DatasetItem) -> Image:
        media = None

        if isinstance(item_a.media, Image) and isinstance(item_b.media, Image):
            item_a_path = getattr(item_a.media, "path", None)
            item_b_path = getattr(item_b.media, "path", None)

            if (
                item_a_path
                and item_b_path
                and item_a_path != item_b_path
                and item_a.media.has_data is item_b.media.has_data
            ):
                # We use has_data as a replacement for path existence check
                # - If only one image has data, we'll use it. The other
                #   one is just a path metainfo, which is not significant
                #   in this case.
                # - If both images have data or both don't, we need
                #   to compare paths.
                #
                # Different paths can aclually point to the same file,
                # but it's not the case we'd like to allow here to be
                # a "simple" merging strategy used for extractor joining
                raise MismatchingMediaPathError((item_a.id, item_a.subset), item_a_path, item_b_path)

            if item_a.media.has_size and item_b.media.has_size and item_a.media.size != item_b.media.size:
                raise MismatchingImageInfoError((item_a.id, item_a.subset), item_a.media.size, item_b.media.size)

            # Avoid direct comparison here for better performance
            # If there are 2 "data-only" images, they won't be compared and
            # we just use the first one
            if item_a.media.has_data:
                media = item_a.media
            elif item_b.media.has_data:
                media = item_b.media
            elif item_a_path:
                media = item_a.media
            elif item_b_path:
                media = item_b.media
            elif item_a.media.has_size:
                media = item_a.media
            elif item_b.media.has_size:
                media = item_b.media
            else:
                raise MismatchingMediaError((item_a.id, item_a.subset), item_a.media, item_b.media)

            if not media.has_data or not media.has_size:
                if item_a.media._size:
                    media._size = item_a.media._size
                elif item_b.media._size:
                    media._size = item_b.media._size
        elif isinstance(item_a.media, Image):
            media = item_a.media
        else:
            media = item_b.media

        return media

    @classmethod
    def _merge_point_clouds(cls, item_a: DatasetItem, item_b: DatasetItem) -> PointCloud:
        media = None

        if isinstance(item_a.media, PointCloud) and isinstance(item_b.media, PointCloud):
            item_a_path = getattr(item_a.media, "path", None)
            item_b_path = getattr(item_b.media, "path", None)

            if item_a_path and item_b_path and item_a_path != item_b_path:
                raise MismatchingMediaPathError((item_a.id, item_a.subset), item_a_path, item_b_path)

            # Avoid direct comparison here for better performance
            # If there are 2 "data-only" pointclouds, they won't be compared and
            # we just use the first one
            if item_a.media.has_data or item_a.media.extra_images:
                media = item_a.media

                if item_b.media.extra_images:
                    for image in item_b.media.extra_images:
                        if image not in media.extra_images:
                            media.extra_images.append(image)
            elif item_b.media.has_data or item_b.media.extra_images:
                media = item_b.media

                if item_a.media.extra_images:
                    for image in item_a.media.extra_images:
                        if image not in media.extra_images:
                            media.extra_images.append(image)
            else:
                raise MismatchingMediaError((item_a.id, item_a.subset), item_a.media, item_b.media)

        elif isinstance(item_a.media, PointCloud):
            media = item_a.media
        else:
            media = item_b.media

        return media

    @classmethod
    def _merge_videos(cls, item_a: DatasetItem, item_b: DatasetItem) -> Video:
        media = None

        if isinstance(item_a.media, Video) and isinstance(item_b.media, Video):
            if (
                item_a.media.path is not item_b.media.path
                or item_a.media._start_frame is not item_b.media._start_frame
                or item_a.media._end_frame is not item_b.media._end_frame
                or item_a.media._step is not item_b.media._step
            ):
                raise VideoMergeError(item_a.id)

            media = item_a.media
        elif isinstance(item_a.media, Video):
            media = item_a.media
        else:
            media = item_b.media

        return media

    @classmethod
    def _merge_multiframe_images(cls, item_a: DatasetItem, item_b: DatasetItem) -> MultiframeImage:
        media = None

        if isinstance(item_a.media, MultiframeImage) and isinstance(item_b.media, MultiframeImage):
            if item_a.media.path and item_b.media.path and item_a.media.path != item_b.media.path:
                raise MismatchingMediaPathError((item_a.id, item_a.subset), item_a.media.path, item_b.media.path)

            if item_a.media.path or item_a.media.data:
                media = item_a.media

                if item_b.media.data:
                    for image in item_b.media.data:
                        if image not in media.data:
                            media.data.append(image)
            else:
                media = item_b.media

                if item_a.media.data:
                    for image in item_a.media.data:
                        if image not in media.data:
                            media.data.append(image)

        elif isinstance(item_a.media, MultiframeImage):
            media = item_a.media
        else:
            media = item_b.media

        return media

    @classmethod
    def _merge_anno(cls, a: Iterable[Annotation], b: Iterable[Annotation]) -> List[Annotation]:
        return cls._merge_annotations_equal(a, b)
