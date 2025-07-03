# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from typing import List, Optional, Tuple

from datumaro.components.dataset import Dataset
from datumaro.util.deprecation import deprecated

log.basicConfig(level=log.INFO)


try:
    import torch

    class BboxOverlaps2D:
        """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

        def __init__(self, scale=1.0, dtype=None):
            self.scale = scale
            self.dtype = dtype

        def _bbox_overlaps(self, bboxes1, bboxes2, eps=1e-6):
            """Calculate overlap between two set of bboxes."""
            # Either the boxes are empty or the length of boxes' last dimension is 4
            assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
            assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

            # Batch dim must be the same
            # Batch dim: (B1, B2, ... Bn)
            assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
            batch_shape = bboxes1.shape[:-2]

            rows = bboxes1.size(-2)
            cols = bboxes2.size(-2)

            if rows * cols == 0:
                return bboxes1.new(batch_shape + (rows, cols))

            area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

            lt = torch.max(
                bboxes1[..., :, None, :2], bboxes2[..., None, :, :2]
            )  # [B, rows, cols, 2]
            rb = torch.min(
                bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:]
            )  # [B, rows, cols, 2]

            wh = (rb - lt).clamp(min=0)
            overlap = wh[..., 0] * wh[..., 1]

            union = area1[..., None] + area2[..., None, :] - overlap

            eps = union.new_tensor([eps])
            union = torch.max(union, eps)
            ious = overlap / union

            return ious

        def __call__(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor):
            """Calculate IoU between 2D bboxes.

            Args:
                bboxes1 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                    in <x1, y1, x2, y2> format, or shape (m, 5) in <x1, y1, x2,
                    y2, score> format.
                bboxes2 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                    in <x1, y1, x2, y2> format, shape (m, 5) in <x1, y1, x2, y2,
                    score> format, or be empty. If ``is_aligned `` is ``True``,
                    then m and n must be equal.
                mode (str): "iou" (intersection over union), "iof" (intersection
                    over foreground), or "giou" (generalized intersection over
                    union).
                is_aligned (bool, optional): If True, then m and n must be equal.
                    Default False.

            Returns:
                Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
            """
            assert bboxes1.size(-1) in [0, 4, 5]
            assert bboxes2.size(-1) in [0, 4, 5]
            if bboxes2.size(-1) == 5:
                bboxes2 = bboxes2[..., :4]
            if bboxes1.size(-1) == 5:
                bboxes1 = bboxes1[..., :4]

            return self._bbox_overlaps(bboxes1, bboxes2)

    @deprecated(deprecated_version="1.11", removed_version="1.12")
    class DataAwareAnchorGenerator:
        def __init__(
            self,
            img_size: Tuple[int, int],
            strides: List[int],
            scales: List[List[float]],
            ratios: List[List[float]],
            pos_thr: float,
            neg_thr: float,
            device: Optional[str] = "cpu",
        ):
            """Data-aware anchor generator for optimizing appropriate anchor scales and ratios.
            In general, anchor generator gets img_size and strides, and its assigner gets positive
            and negative thresholds for solving matching problem in object detection tasks.

            Args:
                img_size (Tuple[int, int]): Image size of height and width.
                strides (List[int]): Strides of feature map from feature pyramid network.
                This implicitly indicates receptive field size and base size of anchor generator.
                scales (List[float]): Initial scales for data-aware optimization.
                ratios (List[float]): Initial ratios for data-aware optimization.
                pos_thr (float): Positive threshold for matching in the following assigner.
                neg_thr (float): Negative threshold for matching in the following assigner.
                device (str): Device for computing gradient. Please refer to `torch.device`
            """
            assert len(strides) == len(scales) or len(strides) == len(ratios)
            assert pos_thr >= neg_thr

            self.img_size = img_size
            self.strides = strides
            self.scales = scales
            self.ratios = ratios
            self.pos_thr = pos_thr
            self.neg_thr = neg_thr

            self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
            self.shifts = [self.get_shifts(stride) for stride in self.strides]

            self.iou_calculator = BboxOverlaps2D()

        def get_shifts(self, stride: int) -> torch.Tensor:
            """Bounding box proposals from anchor generator is composed of shifts and base anchors,
            where shifts is generated in mesh-grid manner and base anchors is combinations of ratios and
            scales. This function is to create mesh-grid shifts in the original image space.

            Args:
                stride (int): Strides of feature map from feature pyramid network.

            Returns:
                Tensor: Shift point coordinates.
            """

            def _meshgrid(x: torch.Tensor, y: torch.Tensor):
                # use shape instead of len to keep tracing while exporting to onnx
                xx = x.repeat(y.shape[0])
                yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
                return xx, yy

            feat_h, feat_w = self.img_size[0] // stride, self.img_size[1] // stride
            shift_x = torch.arange(0, feat_w) * stride
            shift_y = torch.arange(0, feat_h) * stride

            shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
            shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1).to(self.device)

            return shifts

        def get_anchors(
            self, base_size: int, shifts: torch.Tensor, scales: torch.Tensor, ratios: torch.Tensor
        ) -> torch.Tensor:
            """This function is to create base anchors, which combinates ratios and scales.

            Args:
                base_size (int): Strides of feature map from feature pyramid network.
                shifts (Tensor): Shift point coordinates in the original image space.
                scales (Tensor): Scales for creating base anchors.
                ratios (Tensor): Ratios for creating base anchors.

            Returns:
                Tensor: Set of anchor bounding box coordinates.
            """
            w, h = base_size, base_size
            x_center, y_center = 0.5 * w, 0.5 * h

            h_ratios = torch.sqrt(ratios)
            w_ratios = 1 / h_ratios

            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)

            base_anchors = [
                x_center - 0.5 * ws,
                y_center - 0.5 * hs,
                x_center + 0.5 * ws,
                y_center + 0.5 * hs,
            ]
            base_anchors = torch.stack(base_anchors, dim=-1).to(self.device)

            anchors = base_anchors[None, :, :] + shifts[:, None, :]
            anchors = anchors.view(-1, 4)

            return anchors

        def initialize(self, targets, scales, ratios):
            init_ratios = []
            init_scales = []
            for level, base_size in enumerate(self.strides):
                representatives_dict = {
                    1: [0.5],
                    2: [0.4, 0.6],
                    3: [0.25, 0.5, 0.75],
                    4: [0.25, 0.4, 0.6, 0.75],
                    5: [0.1, 0.25, 0.5, 0.75, 0.9],
                }
                representatives = representatives_dict.get(len(scales[level]), None)
                if not representatives:
                    log.error("less than 5 number of strides are available.")

                anchors = self.get_anchors(
                    base_size, self.shifts[level], scales[level], ratios[level]
                )

                overlaps = self.iou_calculator(targets, anchors)
                gt_max_overlaps, _ = overlaps.max(dim=1)

                scale_samples, ratio_samples = [], []
                for target in targets[gt_max_overlaps > 0.1]:
                    h = target[3] - target[1]
                    w = target[2] - target[0]
                    r = h / w
                    affine_h = base_size * torch.sqrt(r)
                    affine_w = base_size / torch.sqrt(r)
                    s = max(h / affine_h, w / affine_w)

                    scale_samples.append(s)
                    ratio_samples.append(r)

                if len(scale_samples) < len(representatives):
                    init_scales.append(scales[level])
                    init_ratios.append(ratios[level])
                else:
                    init_scales.append(
                        [
                            torch.kthvalue(
                                torch.Tensor(scale_samples), int(p * len(scale_samples))
                            ).values.item()
                            for p in representatives
                        ]
                    )
                    init_ratios.append(
                        [
                            torch.kthvalue(
                                torch.Tensor(ratio_samples), int(p * len(ratio_samples))
                            ).values.item()
                            for p in representatives
                        ]
                    )

            return torch.Tensor(init_scales), torch.Tensor(init_ratios)

        def get_loss(self, targets: torch.Tensor, scales: torch.Tensor, ratios: torch.Tensor):
            """This function is to create base anchors, which combinates ratios and scales.

            Args:
                targets (Tensor): Set of target bounding box coordinates.
                scales (Tensor): Scales for creating base anchors.
                ratios (Tensor): Ratios for creating base anchors.

            Returns:
                float: Cost.
                float: Coverage rate.
            """
            all_anchors = []
            for level, base_size in enumerate(self.strides):
                anchors = self.get_anchors(
                    base_size, self.shifts[level], scales[level], ratios[level]
                )
                all_anchors.append(anchors)
            all_anchors = torch.cat(all_anchors, dim=0)

            overlaps = self.iou_calculator(targets, all_anchors)

            max_overlaps, _ = overlaps.max(dim=0)
            log.info(
                f"[ANCHOR] total: {overlaps.shape[1]}, "
                f"pos: {sum(max_overlaps >= self.pos_thr)}, "
                f"neg: {sum(max_overlaps < self.neg_thr)}"
            )

            gt_max_overlaps, _ = overlaps.max(dim=1)
            log.info(
                f"[GT] total: {overlaps.shape[0]}, "
                f"pos: {sum(gt_max_overlaps >= self.pos_thr)}, "
                f"neg: {sum(gt_max_overlaps < self.neg_thr)}"
            )

            cost = (
                500 * gt_max_overlaps[(gt_max_overlaps >= self.pos_thr)].sum()
                + max_overlaps[(max_overlaps >= self.pos_thr)].sum()
                - max_overlaps[(max_overlaps < self.neg_thr)].sum()
            )
            log.info(f"Cost: {cost}")

            num_gts = overlaps.size(0)
            pos_gts = 0
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.pos_thr:
                    max_iou_inds = (overlaps[i, :] == gt_max_overlaps[i]).nonzero()
                    if max_iou_inds.numel() != 0:
                        overlaps[:, max_iou_inds[0]] = 0
                        pos_gts += 1
            coverage_rate = pos_gts / num_gts
            log.info(f"Coverage rate: {coverage_rate}")

            return cost, coverage_rate

        def optimize(
            self,
            dataset: Dataset,
            subset: Optional[str] = None,
            batch_size: Optional[int] = 1024,
            learning_rate: Optional[float] = 0.1,
            num_iters: Optional[int] = 100,
        ):
            """This function is to create base anchors, which combinates ratios and scales.

            Args:
                dataset (Dataset): Desired dataset to optimize anchor scales and ratios.
                batch_size (int): Minibatch size.
                learning_rate (float): Learning rate.
                num_iters (int): Number of iterations.

            Returns:
                List[List[float]]: Optimized scales.
                List[List[float]]: Optimized ratios.
            """
            targets = []
            for item in dataset:
                if subset and item.subset != subset:
                    continue
                height, width = item.media.data.shape[:2]
                scale_h, scale_w = height // self.img_size[0], width // self.img_size[1]

                for ann in item.annotations:
                    x, y, w, h = ann.get_bbox()
                    x /= scale_w
                    y /= scale_h
                    w /= scale_w
                    h /= scale_h
                    targets.append([x, y, x + w, y + h])
            targets = torch.Tensor(targets).to(self.device)

            scales, ratios = torch.Tensor(self.scales).to(self.device), torch.Tensor(
                self.ratios
            ).to(self.device)
            scales, ratios = self.initialize(targets, scales, ratios)

            scales = scales.detach().requires_grad_(True)
            ratios = ratios.detach().requires_grad_(True)
            optimizer = torch.optim.Adam([scales, ratios], lr=learning_rate)

            opt_iter = 0
            opt_cov_rate = 0
            opt_scales = scales
            opt_ratios = ratios
            for iter in range(num_iters):
                random_indices = torch.randperm(targets.size(0))[:batch_size]
                cost, cov_rate = self.get_loss(
                    targets=targets[random_indices],
                    scales=scales,
                    ratios=ratios,
                )
                log.info(f"iter: {iter - 1} / cost: {cost} / scales: {scales} / ratios: {ratios}")

                if cov_rate >= opt_cov_rate:
                    opt_iter = iter
                    opt_scales = scales
                    opt_ratios = ratios

                optimizer.zero_grad()
                (-cost).backward()
                optimizer.step()

                ratios.data = torch.clamp(ratios.data, min=0.125, max=8)
                scales.data = torch.clamp(scales.data, min=0.0625, max=16)

            log.info(f"optimized scale/ratio: {opt_scales}/{opt_ratios} @ iter {opt_iter}")
            return opt_scales.tolist(), opt_ratios.tolist()

except ImportError:

    @deprecated(deprecated_version="1.11", removed_version="1.12")
    class DataAwareAnchorGenerator:
        def __init__(self):
            raise ImportError("Torch package not found. Cannot optimize anchor generator.")
