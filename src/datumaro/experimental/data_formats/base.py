# Copyright (C) 2022-2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from enum import Enum


class DataFormat(Enum):
    """Supported data formats for load/save."""

    DATUMARO = "DATUMARO"
    DATUMARO_LEGACY = "DATUMARO_LEGACY"
    COCO = "COCO"
    VOC = "VOC"
    YOLO = "YOLO"
    YOLO_ULTRALYTICS = "YOLO_ULTRALYTICS"
    UNKNOWN = "UNKNOWN"
