# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from collections import defaultdict
from random import Random
from typing import List, Mapping, Optional, Tuple

from datumaro.components.annotation import AnnotationType
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetItem, IDataset
from datumaro.components.transformer import Transform
from datumaro.util import cast


class RandomSampler(Transform, CliPlugin):
    r"""
    Sampler that keeps no more than required number of items in the dataset.|n
    |n
    Notes:|n
    |s|s- Items are selected uniformly|n
    |s|s- Requesting a sample larger than the number of all images will|n
    |s|s|s|sreturn all images|n
    |n
    Example: select subset of 20 images randomly|n

    .. code-block::

    |s|s%(prog)s -k 20 |n
    |n
    Example: select subset of 20 images, modify only 'train' subset|n

    .. code-block::

    |s|s%(prog)s -k 20 -s train
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("-k", "--count", type=int, required=True, help="Maximum number of items to sample")
        parser.add_argument(
            "-s",
            "--subset",
            default=None,
            help="Limit changes to this subset (default: affect all dataset)",
        )
        parser.add_argument("--seed", type=int, help="Initial value for random number generator")
        return parser

    def __init__(
        self,
        extractor: IDataset,
        count: int,
        *,
        subset: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(extractor)

        self._seed = seed
        self._count = count
        self._indices = None
        self._subset = subset

    def __iter__(self):
        if self._indices is None:
            rng = Random(self._seed)  # nosec B311

            if self._subset:
                n = len(self._extractor.get_subset(self._subset))
            else:
                n = len(self._extractor)
            self._indices = rng.sample(range(n), min(self._count, n))
            self._indices.sort()

        idx_iter = iter(self._indices)

        try:
            next_pick = next(idx_iter)
        except StopIteration:
            if not self._subset:
                return
            else:
                next_pick = -1

        i = 0
        for item in self._extractor:
            if self._subset and self._subset != item.subset:
                yield item
            else:
                if i == next_pick:
                    yield item

                    try:
                        next_pick = next(idx_iter)
                    except StopIteration:
                        if self._subset:
                            next_pick = -1
                            continue
                        else:
                            return

                i += 1


class LabelRandomSampler(Transform, CliPlugin):
    """
    Sampler that keeps at least the required number of annotations of
    each class in the dataset for each subset separately.|n
    |n
    Consider using the "stats" command to get class distribution in
    the dataset.|n
    |n
    Notes:|n
    |s|s- Items can contain annotations of several selected classes|n
    |s|s|s|s(e.g. 3 bounding boxes per image). The number of annotations in the|n
    |s|s|s|sresulting dataset varies between max(class counts) and sum(class counts)|n
    |s|s- If the input dataset does not has enough class annotations, the result|n
    |s|s|s|swill contain only what is available|n
    |s|s- Items are selected uniformly|n
    |s|s- For reasons above, the resulting class distribution in the dataset may|n
    |s|s|s|snot be the same as requested|n
    |s|s- The resulting dataset will only keep annotations for|n
    |s|s|s|sclasses with specified count > 0|n |n
    Example: select at least 5 annotations of each class randomly|n

    .. code-block::

    |s|s%(prog)s -k 5 |n |n
    Example: select at least 5 images with "cat" annotations and 3 "person"|n

    .. code-block::

    |s|s%(prog)s -l "cat:5" -l "person:3"
    """

    @staticmethod
    def _parse_label_count(s: str) -> Tuple[str, int]:
        label, count = s.split(":", maxsplit=1)
        count = cast(count, int, default=None)

        if not label:
            raise argparse.ArgumentError(None, "Class name cannot be empty")
        if count is None or count < 0:
            raise argparse.ArgumentError(None, f"Class '{label}' count is invalid")

        return label, count

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-k",
            "--count",
            type=int,
            required=True,
            help="Minimum number of annotations of each class",
        )
        parser.add_argument(
            "-l",
            "--label",
            dest="label_counts",
            action="append",
            type=cls._parse_label_count,
            help="Minimum number of annotations of a specific class. "
            "Overrides the `-k/--count` setting for the class. "
            "The format is 'label_name:count' (repeatable)",
        )
        parser.add_argument("--seed", type=int, help="Initial value for random number generator")
        return parser

    def __init__(
        self,
        extractor: IDataset,
        *,
        count: Optional[int] = None,
        label_counts: Optional[Mapping[str, int]] = None,
        seed: Optional[int] = None,
    ):
        from datumaro.plugins.transforms import ProjectLabels

        count = count or 0
        label_counts = dict(label_counts or {})
        assert count or any(label_counts.values())

        new_labels = {}
        for label in extractor.categories()[AnnotationType.label]:
            label_count = label_counts.get(label.name, count)
            if label_count:
                new_labels[label.name] = label_count
        self._label_counts = {idx: count for idx, count in enumerate(new_labels.values())}
        super().__init__(ProjectLabels(extractor, new_labels.keys()))

        self._seed = seed

        # for repeated calls
        self._selected_items: List[DatasetItem] = None

    def __iter__(self):
        if self._selected_items is not None:
            yield from self._selected_items
            return

        # Uses the reservoir sampling algorithm for each class
        # https://en.wikipedia.org/wiki/Reservoir_sampling

        def _make_bucket():
            # label -> bucket
            return {label: [] for label in self._label_counts}

        buckets = defaultdict(_make_bucket)  # subset -> subset_buckets

        rng = Random(self._seed)  # nosec B311

        for i, item in enumerate(self._extractor):
            labels = set(getattr(ann, "label", None) for ann in item.annotations)
            labels.discard(None)
            for label in labels:
                if len(buckets[item.subset][label]) < self._label_counts[label]:
                    buckets[item.subset][label].append(item)
                else:
                    j = rng.randint(1, i)
                    if j <= self._label_counts[label]:
                        buckets[item.subset][label][j - 1] = item

        selected_items = {}
        for subset_buckets in buckets.values():
            for label_bucket in subset_buckets.values():
                for item in label_bucket:
                    if item:
                        selected_items.setdefault((item.id, item.subset), item)

        self._selected_items = selected_items.values()
        yield from self._selected_items
