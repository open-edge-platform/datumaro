# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from .base import Merger
from .exact_merge import ExactMerge
from .intersect_merge import IntersectMerge
from .union_merge import UnionMerge

DEFAULT_MERGE_POLICY = "exact"


def get_merger(merge_policy: str = DEFAULT_MERGE_POLICY, *args, **kwargs) -> Merger:
    """
    Get :class:`Merger` according to `merge_policy`. You have to choose an appropriate `Merger`
    for your purpose. The available merge policies are "union", "intersect", and "exact".

    1. :class:`UnionMerge`

    Merge several datasets with "union" policy:

    - Label categories are merged according to the union of their label names.
    For example, if Dataset-A has {"car", "cat", "dog"} and Dataset-B has
    {"car", "bus", "truck"} labels, the merged dataset will have
    {"bust", "car", "cat", "dog", "truck"} labels.

    - If there are two or more dataset items whose (id, subset) pairs match each other,
    both are included in the merged dataset. At this time, since the same (id, subset)
    pair cannot be duplicated in the dataset, we add a suffix to the id of each source item.
    For example, if Dataset-A has DatasetItem(id="magic", subset="train") and Dataset-B has
    also DatasetItem(id="magic", subset="train"), the merged dataset will have
    DatasetItem(id="magic-0", subset="train") and DatasetItem(id="magic-1", subset="train").

    2. :class:`IntersectMerge`

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

    3. :class:`ExactMerge`

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
    if merge_policy == "union":
        return UnionMerge(*args, **kwargs)
    if merge_policy == "intersect":
        return IntersectMerge(*args, **kwargs)
    if merge_policy == "exact":
        return ExactMerge(*args, **kwargs)

    raise ValueError(f"{merge_policy} is invalid Merger name.")
