# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: MIT

"""
This module defines semantic tags for fields used in Datumaro's supported data formats. The semantics can be used to
ensure generic fields such as NumericField, BoolField, and StringField are kept when converting to specific data format
Sample schemas.
"""

# Dataset organization semantics
IMAGE_ID = "image_id"  # NumericField | stores the image id

# COCO semantics
AREAS = "areas"  # NumericField | stores the area of an annotation
ISCROWD = "iscrowd"  # BoolField | stores whether an annotation is a crowd
CAPTION_GROUP_IDS = "caption_group_ids"  # NumericField | stores group ids for captions to link multiple captions
