# Copyright (C) 2022-2026 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""
Unit tests for YOLO sample classes.
"""

import numpy as np

from datumaro.experimental.data_formats.yolo.sample import YoloSample
from datumaro.experimental.fields import ImageInfo, Subset

# ================
# YoloSample Tests
# ================


def test_yolo_sample_create_with_all_fields():
    """Test creating a sample with all fields populated."""
    bboxes = np.array([[100.0, 100.0, 50.0, 50.0]], dtype=np.float32)
    labels = np.array([0], dtype=np.int32)

    sample = YoloSample(
        image="/path/to/image.jpg",
        image_info=ImageInfo(height=480, width=640),
        bboxes=bboxes,
        labels=labels,
        subset=Subset.TRAINING,
    )

    assert sample.image.path == "/path/to/image.jpg"
    assert sample.image_info.height == 480
    assert sample.image_info.width == 640
    assert sample.subset == Subset.TRAINING
    np.testing.assert_array_equal(sample.bboxes, bboxes)
    np.testing.assert_array_equal(sample.labels, labels)


def test_yolo_sample_create_with_none_annotations():
    """Test creating a sample without annotations."""
    sample = YoloSample(
        image="/path/to/image.jpg",
        image_info=ImageInfo(height=480, width=640),
        bboxes=None,
        labels=None,
        subset=Subset.VALIDATION,
    )

    assert sample.image.path == "/path/to/image.jpg"
    assert sample.bboxes is None
    assert sample.labels is None
    assert sample.subset == Subset.VALIDATION


def test_yolo_sample_create_with_multiple_bboxes():
    """Test creating a sample with multiple bounding boxes."""
    bboxes = np.array(
        [
            [100.0, 100.0, 50.0, 50.0],
            [200.0, 200.0, 60.0, 40.0],
            [300.0, 150.0, 30.0, 80.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 1, 0], dtype=np.int32)

    sample = YoloSample(
        image="/path/to/image.jpg",
        image_info=ImageInfo(height=480, width=640),
        bboxes=bboxes,
        labels=labels,
        subset=Subset.TESTING,
    )

    assert sample.bboxes.shape == (3, 4)
    assert sample.labels.shape == (3,)


def test_yolo_sample_subset_variations():
    """Test creating samples with different subsets."""
    for subset in [Subset.TRAINING, Subset.VALIDATION, Subset.TESTING, Subset.UNASSIGNED]:
        sample = YoloSample(
            image="/path/to/image.jpg",
            image_info=ImageInfo(height=480, width=640),
            bboxes=None,
            labels=None,
            subset=subset,
        )
        assert sample.subset == subset
