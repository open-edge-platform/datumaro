# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from collections import OrderedDict
from typing import Dict, Sequence

import attr
from attr import attrib, attrs

from datumaro.components.annotation import (
    AnnotationType,
    LabelCategories,
    MaskCategories,
    PointsCategories,
)
from datumaro.components.annotations.merger import (
    AnnotationMerger,
    BboxMerger,
    CaptionsMerger,
    Cuboid2DMerger,
    Cuboid3dMerger,
    EllipseMerger,
    ImageAnnotationMerger,
    LabelMerger,
    LineMerger,
    MaskMerger,
    PointsMerger,
    PolygonMerger,
    RotatedBboxMerger,
    TabularMerger,
)
from datumaro.components.dataset_base import DatasetItem, IDataset
from datumaro.components.dataset_item_storage import (
    DatasetItemStorage,
    DatasetItemStorageDatasetView,
)
from datumaro.components.errors import (
    AnnotationsTooCloseError,
    ConflictingCategoriesError,
    FailedAttrVotingError,
    NoMatchingAnnError,
    NoMatchingItemError,
    WrongGroupError,
)
from datumaro.components.merge import Merger
from datumaro.util import find
from datumaro.util.annotation_util import find_instances, max_bbox
from datumaro.util.attrs_util import ensure_cls

__all__ = ["IntersectMerge"]


@attrs
class IntersectMerge(Merger):
    """
    Merge several datasets with "intersect" policy:

    - If there are two or more dataset items whose (id, subset) pairs match each other,
    we can consider this as having an intersection in our dataset. This method merges
    the annotations of the corresponding :class:`DatasetItem` into one :class:`DatasetItem`
    to handle this intersection. The rule to handle merging annotations is provided by
    :class:`AnnotationMerger` according to their annotation types. For example,
    DatasetItem(id="item_1", subset="train", annotations=[Bbox(0, 0, 1, 1)]) from Dataset-A and
    DatasetItem(id="item_1", subset="train", annotations=[Bbox(.5, .5, 1, 1)]) from Dataset-B can be
    merged into DatasetItem(id="item_1", subset="train", annotations=[Bbox(0, 0, 1, 1)]).

    - Label categories are merged according to the union of their label names
    (Same as `UnionMerge`). For example, if Dataset-A has {"car", "cat", "dog"}
    and Dataset-B has {"car", "bus", "truck"} labels, the merged dataset will have
    {"bust", "car", "cat", "dog", "truck"} labels.

    - This merge has configuration parameters (`conf`) to control the annotation merge behaviors.

    For example,

    ```python
    merge = IntersectMerge(
        conf=IntersectMerge.Conf(
            pairwise_dist=0.25,
            groups=[],
            output_conf_thresh=0.0,
            quorum=0,
        )
    )
    ```

    For more details for the parameters, please refer to :class:`IntersectMerge.Conf`.
    """

    def __init__(self, **options):
        super().__init__(**options)

    @attrs(kw_only=True)
    class Conf:
        """
        Parameters
        ----------
        pairwise_dist
            IoU match threshold for segments
        sigma
            Parameter for Object Keypoint Similarity metric
            (https://cocodataset.org/#keypoints-eval)
        output_conf_thresh
            Confidence threshold for output annotations
        quorum
            Minimum count for a label and attribute voting results to be counted
        ignored_attributes
            Attributes to be ignored in the merged :class:`DatasetItem`
        groups
            A comma-separated list of labels in annotation groups to check.
            '?' postfix can be added to a label to make it optional in the group (repeatable)
        close_distance
            Distance threshold between annotations to decide their closeness. If they are decided
            to be close, it will be enrolled to the error tracker.
        """

        pairwise_dist = attrib(converter=float, default=0.5)
        sigma = attrib(converter=list, factory=list)

        output_conf_thresh = attrib(converter=float, default=0)
        quorum = attrib(converter=int, default=0)
        ignored_attributes = attrib(converter=set, factory=set)

        def _groups_converter(value):
            result = []
            for group in value:
                rg = set()
                for label in group:
                    optional = label.endswith("?")
                    name = label if not optional else label[:-1]
                    rg.add((name, optional))
                result.append(rg)
            return result

        groups = attrib(converter=_groups_converter, factory=list)
        close_distance = attrib(converter=float, default=0.75)

    conf = attrib(converter=ensure_cls(Conf), factory=Conf)

    # Error trackers:
    errors = attrib(factory=list, init=False)

    def add_item_error(self, error, *args, **kwargs):
        self.errors.append(error(self._item_id, *args, **kwargs))

    # Indexes:
    _dataset_map = attrib(init=False)  # id(dataset) -> (dataset, index)
    _item_map = attrib(init=False)  # id(item) -> (item, id(dataset))
    _ann_map = attrib(init=False)  # id(ann) -> (ann, id(item))
    _item_id = attrib(init=False)
    _item = attrib(init=False)

    # Misc.
    _infos = attrib(init=False)  # merged infos
    _categories = attrib(init=False)  # merged categories

    def merge(self, sources: Sequence[IDataset]) -> DatasetItemStorage:
        self._infos = self.merge_infos([d.infos() for d in sources])
        self._categories = self.merge_categories([d.categories() for d in sources])
        merged = DatasetItemStorage()
        self._check_groups_definition()

        item_matches, item_map = self.match_items(sources)
        self._item_map = item_map
        self._dataset_map = {id(d): (d, i) for i, d in enumerate(sources)}

        for item_id, items in item_matches.items():
            self._item_id = item_id

            if len(items) < len(sources):
                missing_sources = set(id(s) for s in sources) - set(items)
                missing_sources = [self._dataset_map[s][1] for s in missing_sources]
                self.add_item_error(NoMatchingItemError, sources=missing_sources)
            merged.put(self.merge_items(items))

        return merged

    def get_ann_source(self, ann_id):
        return self._item_map[self._ann_map[ann_id][1]][1]

    def __call__(self, *datasets: IDataset) -> DatasetItemStorageDatasetView:
        # TODO: self.merge() should be the first since this order matters for
        # IntersectMerge.
        merged = self.merge(datasets)
        infos = self.merge_infos(d.infos() for d in datasets)
        categories = self.merge_categories(d.categories() for d in datasets)
        media_type = self.merge_media_types(datasets)
        ann_types = self.merge_ann_types(datasets)
        return DatasetItemStorageDatasetView(
            parent=merged,
            infos=infos,
            categories=categories,
            media_type=media_type,
            ann_types=ann_types,
        )

    def merge_categories(self, sources: Sequence[IDataset]) -> Dict:
        # TODO: This is a temporary workaround to minimize code changes.
        # We have to revisit it to make this class stateless.
        if hasattr(self, "_categories"):
            return self._categories

        dst_categories = {}

        label_cat = self._merge_label_categories(sources)
        if label_cat is None:
            label_cat = LabelCategories()
        dst_categories[AnnotationType.label] = label_cat

        points_cat = self._merge_point_categories(sources, label_cat)
        if points_cat is not None:
            dst_categories[AnnotationType.points] = points_cat

        mask_cat = self._merge_mask_categories(sources, label_cat)
        if mask_cat is not None:
            dst_categories[AnnotationType.mask] = mask_cat

        return dst_categories

    def merge_items(self, items: Dict[int, DatasetItem]) -> DatasetItem:
        self._item = next(iter(items.values()))

        self._ann_map = {}
        sources = []
        for item in items.values():
            self._ann_map.update({id(a): (a, id(item)) for a in item.annotations})
            sources.append(item.annotations)
        log.debug(
            "Merging item %s: source annotations %s" % (self._item_id, list(map(len, sources)))
        )

        annotations = self.merge_annotations(sources)

        annotations = [
            a for a in annotations if self.conf.output_conf_thresh <= a.attributes.get("score", 1)
        ]

        return self._item.wrap(annotations=annotations)

    def merge_annotations(self, sources):
        self._make_mergers(sources)

        clusters = self._match_annotations(sources)

        joined_clusters = sum(clusters.values(), [])
        group_map = self._find_cluster_groups(joined_clusters)

        annotations = []
        for t, clusters in clusters.items():
            for cluster in clusters:
                self._check_cluster_sources(cluster)

            merged_clusters = self._merge_clusters(t, clusters)

            for merged_ann, cluster in zip(merged_clusters, clusters):
                attributes = self._find_cluster_attrs(cluster, merged_ann)
                attributes = {
                    k: v for k, v in attributes.items() if k not in self.conf.ignored_attributes
                }
                attributes.update(merged_ann.attributes)
                merged_ann.attributes = attributes

                new_group_id = find(enumerate(group_map), lambda e: id(cluster) in e[1][0])
                if new_group_id is None:
                    new_group_id = 0
                else:
                    new_group_id = new_group_id[0] + 1
                merged_ann.group = new_group_id

            if self.conf.close_distance:
                self._check_annotation_distance(t, merged_clusters)

            annotations += merged_clusters

        if self.conf.groups:
            self._check_groups(annotations)

        return annotations

    def match_items(self, datasets):
        item_ids = set((item.id, item.subset) for d in datasets for item in d)

        item_map = {}  # id(item) -> (item, id(dataset))

        matches = OrderedDict()
        for item_id, item_subset in sorted(item_ids, key=lambda e: e[0]):
            items = {}
            for d in datasets:
                item = d.get(item_id, subset=item_subset)
                if item:
                    items[id(d)] = item
                    item_map[id(item)] = (item, id(d))
            matches[(item_id, item_subset)] = items

        return matches, item_map

    def _merge_label_categories(self, sources):
        same = True
        common = None
        for src_categories in sources:
            src_cat = src_categories.get(AnnotationType.label)
            if common is None:
                common = src_cat
            elif common != src_cat:
                same = False
                break

        if same:
            return common

        dst_cat = LabelCategories()
        for src_id, src_categories in enumerate(sources):
            src_cat = src_categories.get(AnnotationType.label)
            if src_cat is None:
                continue

            for src_label in src_cat.items:
                dst_label = dst_cat.find(src_label.name)[1]
                if dst_label is not None:
                    if dst_label != src_label:
                        if (
                            src_label.parent
                            and dst_label.parent
                            and src_label.parent != dst_label.parent
                        ):
                            raise ConflictingCategoriesError(
                                "Can't merge label category %s (from #%s): "
                                "parent label conflict: %s vs. %s"
                                % (src_label.name, src_id, src_label.parent, dst_label.parent),
                                sources=list(range(src_id)),
                            )
                        dst_label.parent = dst_label.parent or src_label.parent
                        dst_label.attributes |= src_label.attributes
                    else:
                        pass
                else:
                    dst_cat.add(src_label.name, src_label.parent, src_label.attributes)

        return dst_cat

    def _merge_point_categories(self, sources, label_cat):
        dst_point_cat = PointsCategories()

        for src_id, src_categories in enumerate(sources):
            src_label_cat = src_categories.get(AnnotationType.label)
            src_point_cat = src_categories.get(AnnotationType.points)
            if src_label_cat is None or src_point_cat is None:
                continue

            for src_label_id, src_cat in src_point_cat.items.items():
                src_label = src_label_cat.items[src_label_id].name
                dst_label_id = label_cat.find(src_label)[0]
                dst_cat = dst_point_cat.items.get(dst_label_id)
                if dst_cat is not None:
                    if dst_cat != src_cat:
                        raise ConflictingCategoriesError(
                            "Can't merge point category for label "
                            "%s (from #%s): %s vs. %s" % (src_label, src_id, src_cat, dst_cat),
                            sources=list(range(src_id)),
                        )
                    else:
                        pass
                else:
                    dst_point_cat.add(dst_label_id, src_cat.labels, src_cat.joints)

        if len(dst_point_cat.items) == 0:
            return None

        return dst_point_cat

    def _merge_mask_categories(self, sources, label_cat):
        dst_mask_cat = MaskCategories()

        for src_id, src_categories in enumerate(sources):
            src_label_cat = src_categories.get(AnnotationType.label)
            src_mask_cat = src_categories.get(AnnotationType.mask)
            if src_label_cat is None or src_mask_cat is None:
                continue

            for src_label_id, src_cat in src_mask_cat.colormap.items():
                src_label = src_label_cat.items[src_label_id].name
                dst_label_id = label_cat.find(src_label)[0]
                dst_cat = dst_mask_cat.colormap.get(dst_label_id)
                if dst_cat is not None:
                    if dst_cat != src_cat:
                        raise ConflictingCategoriesError(
                            "Can't merge mask category for label "
                            "%s (from #%s): %s vs. %s" % (src_label, src_id, src_cat, dst_cat),
                            sources=list(range(src_id)),
                        )
                    else:
                        pass
                else:
                    dst_mask_cat.colormap[dst_label_id] = src_cat

        if len(dst_mask_cat.colormap) == 0:
            return None

        return dst_mask_cat

    def _match_annotations(self, sources):
        all_by_type = {}
        for s in sources:
            src_by_type = {}
            for a in s:
                src_by_type.setdefault(a.type, []).append(a)
            for k, v in src_by_type.items():
                all_by_type.setdefault(k, []).append(v)

        clusters = {}
        for k, v in all_by_type.items():
            clusters.setdefault(k, []).extend(self._match_ann_type(k, v))

        return clusters

    def _make_mergers(self, sources):
        def _make(c, **kwargs):
            kwargs.update(attr.asdict(self.conf))
            fields = attr.fields_dict(c)
            return c(**{k: v for k, v in kwargs.items() if k in fields}, context=self)

        def _for_type(t, **kwargs):
            if t is AnnotationType.unknown:
                return _make(AnnotationMerger, **kwargs)
            elif t is AnnotationType.label:
                return _make(LabelMerger, **kwargs)
            elif t is AnnotationType.bbox:
                return _make(BboxMerger, **kwargs)
            elif t is AnnotationType.mask:
                return _make(MaskMerger, **kwargs)
            elif t is AnnotationType.polygon:
                return _make(PolygonMerger, **kwargs)
            elif t is AnnotationType.polyline:
                return _make(LineMerger, **kwargs)
            elif t is AnnotationType.points:
                return _make(PointsMerger, **kwargs)
            elif t is AnnotationType.caption:
                return _make(CaptionsMerger, **kwargs)
            elif t is AnnotationType.cuboid_3d:
                return _make(Cuboid3dMerger, **kwargs)
            elif t is AnnotationType.super_resolution_annotation:
                return _make(ImageAnnotationMerger, **kwargs)
            elif t is AnnotationType.depth_annotation:
                return _make(ImageAnnotationMerger, **kwargs)
            elif t is AnnotationType.ellipse:
                return _make(EllipseMerger, **kwargs)
            elif t is AnnotationType.tabular:
                return _make(TabularMerger, **kwargs)
            elif t is AnnotationType.rotated_bbox:
                return _make(RotatedBboxMerger, **kwargs)
            elif t is AnnotationType.cuboid_2d:
                return _make(Cuboid2DMerger, **kwargs)
            else:
                raise NotImplementedError("Type %s is not supported" % t)

        instance_map = {}
        for s in sources:
            s_instances = find_instances(s)
            for inst in s_instances:
                inst_bbox = max_bbox(
                    [
                        a
                        for a in inst
                        if a.type
                        in {AnnotationType.polygon, AnnotationType.mask, AnnotationType.bbox}
                    ]
                )
                for ann in inst:
                    instance_map[id(ann)] = [inst, inst_bbox]

        self._mergers = {t: _for_type(t, instance_map=instance_map) for t in AnnotationType}

    def _match_ann_type(self, t, sources):
        return self._mergers[t].match_annotations(sources)

    def _merge_clusters(self, t, clusters):
        return self._mergers[t].merge_clusters(clusters)

    def _find_cluster_groups(self, clusters):
        cluster_groups = []
        visited = set()
        for a_idx, cluster_a in enumerate(clusters):
            if a_idx in visited:
                continue
            visited.add(a_idx)

            cluster_group = {id(cluster_a)}

            # find segment groups in the cluster group
            a_groups = set(ann.group for ann in cluster_a)
            for cluster_b in clusters[a_idx + 1 :]:
                b_groups = set(ann.group for ann in cluster_b)
                if a_groups & b_groups:
                    a_groups |= b_groups

            # now we know all the segment groups in this cluster group
            # so we can find adjacent clusters
            for b_idx, cluster_b in enumerate(clusters[a_idx + 1 :]):
                b_idx = a_idx + 1 + b_idx
                b_groups = set(ann.group for ann in cluster_b)
                if a_groups & b_groups:
                    cluster_group.add(id(cluster_b))
                    visited.add(b_idx)

            if a_groups == {0}:
                continue  # skip annotations without a group
            cluster_groups.append((cluster_group, a_groups))
        return cluster_groups

    def _find_cluster_attrs(self, cluster, ann):
        quorum = self.conf.quorum or 0

        # TODO: when attribute types are implemented, add linear
        # interpolation for contiguous values

        attr_votes = {}  # name -> { value: score , ... }
        for s in cluster:
            for name, value in s.attributes.items():
                votes = attr_votes.get(name, {})
                votes[value] = 1 + votes.get(value, 0)
                attr_votes[name] = votes

        attributes = {}
        for name, votes in attr_votes.items():
            winner, count = max(votes.items(), key=lambda e: e[1])
            if count < quorum:
                if sum(votes.values()) < quorum:
                    # blame provokers
                    missing_sources = set(
                        self.get_ann_source(id(a))
                        for a in cluster
                        if s.attributes.get(name) == winner
                    )
                else:
                    # blame outliers
                    missing_sources = set(
                        self.get_ann_source(id(a))
                        for a in cluster
                        if s.attributes.get(name) != winner
                    )
                missing_sources = [self._dataset_map[s][1] for s in missing_sources]
                self.add_item_error(
                    FailedAttrVotingError, name, votes, ann, sources=missing_sources
                )
                continue
            attributes[name] = winner

        return attributes

    def _check_cluster_sources(self, cluster):
        if len(cluster) == len(self._dataset_map):
            return

        def _has_item(s):
            item = self._dataset_map[s][0].get(*self._item_id)
            if not item:
                return False
            if len(item.annotations) == 0:
                return False
            return True

        missing_sources = set(self._dataset_map) - set(self.get_ann_source(id(a)) for a in cluster)
        missing_sources = [self._dataset_map[s][1] for s in missing_sources if _has_item(s)]
        if missing_sources:
            self.add_item_error(NoMatchingAnnError, cluster[0], sources=missing_sources)

    def _check_annotation_distance(self, t, annotations):
        for a_idx, a_ann in enumerate(annotations):
            for b_ann in annotations[a_idx + 1 :]:
                d = self._mergers[t].distance(a_ann, b_ann)
                if self.conf.close_distance < d:
                    self.add_item_error(AnnotationsTooCloseError, a_ann, b_ann, d)

    def _check_groups(self, annotations):
        check_groups = []
        for check_group_raw in self.conf.groups:
            check_group = set(l[0] for l in check_group_raw)
            optional = set(l[0] for l in check_group_raw if l[1])
            check_groups.append((check_group, optional))

        def _check_group(group_labels, group):
            for check_group, optional in check_groups:
                common = check_group & group_labels
                real_miss = check_group - common - optional
                extra = group_labels - check_group
                if common and (extra or real_miss):
                    self.add_item_error(WrongGroupError, group_labels, check_group, group)
                    break

        groups = find_instances(annotations)
        for group in groups:
            group_labels = set()
            for ann in group:
                if not hasattr(ann, "label"):
                    continue
                label = self._get_label_name(ann.label)

                if ann.group:
                    group_labels.add(label)
                else:
                    _check_group({label}, [ann])

            if not group_labels:
                continue
            _check_group(group_labels, group)

    def _get_label_name(self, label_id):
        if label_id is None:
            return None
        return self._categories[AnnotationType.label].items[label_id].name

    def _get_label_id(self, label):
        return self._categories[AnnotationType.label].find(label)[0]

    def _get_src_label_name(self, ann, label_id):
        if label_id is None:
            return None
        item_id = self._ann_map[id(ann)][1]
        dataset_id = self._item_map[item_id][1]
        return (
            self._dataset_map[dataset_id][0].categories()[AnnotationType.label].items[label_id].name
        )

    def get_any_label_name(self, ann, label_id):
        if label_id is None:
            return None
        try:
            return self._get_src_label_name(ann, label_id)
        except KeyError:
            return self._get_label_name(label_id)

    def _check_groups_definition(self):
        for group in self.conf.groups:
            for label, _ in group:
                _, entry = self._categories[AnnotationType.label].find(label)
                if entry is None:
                    raise ValueError(
                        "Datasets do not contain "
                        "label '%s', available labels %s"
                        % (label, [i.name for i in self._categories[AnnotationType.label].items])
                    )
