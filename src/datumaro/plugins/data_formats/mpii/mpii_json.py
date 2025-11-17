# Copyright (C) 2022-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import errno
import os.path as osp
from typing import List, Optional

import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories, Points, PointsCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.util import parse_json_file

from .format import MPII_POINTS_JOINTS, MPII_POINTS_LABELS


class MpiiJsonPath:
    ANNOTATION_FILE = "mpii_annotations.json"
    HEADBOXES_FILE = "mpii_headboxes.npy"
    VISIBILITY_FILE = "jnt_visible.npy"
    POS_GT_FILE = "mpii_pos_gt.npy"


class MpiiJsonBase(SubsetBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not osp.isfile(path):
            raise FileNotFoundError(errno.ENOENT, "Can't find annotations file", path)

        super().__init__(subset=subset, ctx=ctx)

        self._categories = {
            AnnotationType.label: LabelCategories.from_iterable(["human"]),
            AnnotationType.points: PointsCategories.from_iterable([(0, MPII_POINTS_LABELS, MPII_POINTS_JOINTS)]),
        }

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        root_dir = osp.dirname(path)

        hb_path = osp.join(root_dir, MpiiJsonPath.HEADBOXES_FILE)
        if osp.isfile(hb_path):
            headboxes = np.load(hb_path)
        else:
            headboxes = np.array([[[]]])

        vis_path = osp.join(root_dir, MpiiJsonPath.VISIBILITY_FILE)
        if osp.isfile(vis_path):
            visibility = np.load(vis_path).T
        else:
            visibility = np.array([])

        pos_gt_path = osp.join(root_dir, MpiiJsonPath.POS_GT_FILE)
        if osp.isfile(pos_gt_path):
            gt_pose = np.transpose(np.load(pos_gt_path), (2, 0, 1))
        else:
            gt_pose = np.array([])

        for i, ann in enumerate(parse_json_file(path)):
            item_id = osp.splitext(ann.get("img_paths", ""))[0]

            center = ann.get("objpos", [])
            scale = float(ann.get("scale_provided", 0))

            if i < gt_pose.shape[0]:
                points = gt_pose[i].ravel()

                if i < visibility.shape[0]:
                    vis = visibility[i]
                else:
                    vis = np.ones(len(points) // 2, dtype=np.int8)
            else:
                keypoints = np.array(ann.get("joint_self", []))
                points = keypoints[:, 0:2].ravel()

                vis = keypoints[:, 2]
                if i < visibility.shape[0]:
                    vis = visibility[i]

            vis = [int(val) for val in vis]

            group_num = 1

            annotations = [
                Points(
                    points,
                    vis,
                    label=0,
                    group=group_num,
                    attributes={"center": center, "scale": scale},
                )
            ]

            if i < headboxes.shape[2]:
                bbox = headboxes[:, :, i]
                annotations.append(
                    Bbox(
                        bbox[0][0],
                        bbox[0][1],
                        bbox[1][0] - bbox[0][0],
                        bbox[1][1] - bbox[0][1],
                        label=0,
                        group=group_num,
                    )
                )

            group_num += 1

            joint_others = ann.get("joint_others")
            if joint_others:
                num_others = int(ann.get("numOtherPeople", 1))
                center = ann.get("objpos_other", [])
                scale = ann.get("scale_provided_other", 0)

                if num_others == 1:
                    center = [center]
                    scale = [scale]
                    joint_others = [joint_others]

                for i in range(num_others):
                    keypoints = np.array(joint_others[i])
                    points = keypoints[:, 0:2].ravel()
                    vis = keypoints[:, 2]
                    vis = [int(val) for val in vis]

                    attributes = {}
                    if i < len(center):
                        attributes["center"] = center[i]
                    if i < len(scale):
                        attributes["scale"] = scale[i]

                    annotations.append(Points(points, vis, label=0, group=group_num, attributes=attributes))

                    group_num += 1

            items[item_id] = DatasetItem(
                id=item_id,
                subset=self._subset,
                media=Image.from_file(path=osp.join(root_dir, ann.get("img_paths", ""))),
                annotations=annotations,
            )

            for ann in annotations:
                self._ann_types.add(ann.type)

        return items


class MpiiJsonImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, ".json", "mpii_json")

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(MpiiJsonPath.ANNOTATION_FILE)

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [osp.splitext(MpiiJsonPath.ANNOTATION_FILE)[1]]
