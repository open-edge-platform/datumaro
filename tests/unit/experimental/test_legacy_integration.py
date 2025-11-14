# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import tempfile
from typing import cast

import numpy as np
import pytest

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset import Dataset as LegacyDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, ImageFromFile
from datumaro.experimental.dataset import Dataset as ExperimentalDataset
from datumaro.experimental.legacy import convert_from_legacy, convert_to_legacy


@pytest.fixture()
def fxt_dataset() -> LegacyDataset:
    h = w = 8
    n_labels = 5
    n_items = 5

    return LegacyDataset.from_iterable(  # pyright: ignore[reportUnknownMemberType]
        [
            DatasetItem(
                id=f"img_{item_id}",
                subset=subset,
                media=Image.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                    data=np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8), ext=".png"
                ),
                annotations=[
                    Bbox(
                        *np.random.randint(0, h, size=(4,)).tolist(),
                        id=item_id,
                        label=np.random.randint(0, n_labels),
                        group=item_id,
                        z_order=0,
                        attributes={},
                    )
                ],
            )
            for subset in ["Test", "Train", "Validation"]
            for item_id in range(n_items)
        ],
        categories={AnnotationType.label: LabelCategories.from_iterable([f"label_{idx}" for idx in range(n_labels)])},
    )


@pytest.mark.parametrize("input_format", ["coco", "yolo", "datumaro"], ids=lambda x: f"[if:{x}]")
def test_object_detection(
    fxt_dataset: LegacyDataset,
    input_format: str,
):
    """Test converting legacy datasets from various formats to v2 format."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Export the fixture dataset to the specified format
        src_dir = osp.join(temp_dir, "src")
        fxt_dataset.export(src_dir, format=input_format, save_media=True)  # pyright: ignore[reportUnknownMemberType]

        # Step 2: Import it back as a legacy dataset
        legacy_dataset = LegacyDataset.import_from(src_dir, format=input_format)  # pyright: ignore[reportUnknownMemberType]

        # Step 3: Convert to v2 format
        experimental_dataset = convert_from_legacy(legacy_dataset)

        # Step 4: Verify the conversion
        assert isinstance(experimental_dataset, ExperimentalDataset)
        assert len(experimental_dataset) == len(legacy_dataset)

        # Verify schema attributes exist
        schema = experimental_dataset.schema
        assert "bboxes" in schema.attributes
        assert "image_path" in schema.attributes
        assert "bboxes" in schema.attributes
        assert "labels" in schema.attributes

        # Verify data conversion for first few samples
        for legacy_item, experimental_sample in zip(legacy_dataset, experimental_dataset):
            # Check image path is preserved
            assert getattr(experimental_sample, "image_path") == legacy_item.media.path  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportOptionalMemberAccess]

            # Check bbox data
            assert hasattr(experimental_sample, "bboxes")
            assert hasattr(experimental_sample, "labels")

            # Get bbox annotations from legacy item
            bbox_anns = [ann for ann in legacy_item.annotations if isinstance(ann, Bbox)]
            assert bbox_anns

            # Verify bbox array shape and values
            bboxes = getattr(experimental_sample, "bboxes")
            labels = getattr(experimental_sample, "labels")

            assert bboxes.shape[0] == len(bbox_anns)
            assert bboxes.shape[1] == 4  # x1, y1, x2, y2
            assert labels.shape[0] == len(bbox_anns)

            # Verify first bbox conversion (x,y,w,h -> x1,y1,x2,y2)
            first_bbox = bbox_anns[0]
            expected_x1y1x2y2 = [
                first_bbox.x,
                first_bbox.y,
                first_bbox.x + first_bbox.w,
                first_bbox.y + first_bbox.h,
            ]
            np.testing.assert_array_almost_equal(bboxes[0], expected_x1y1x2y2)
            assert labels[0] == first_bbox.label

        # Step 5: Test conversion back to legacy format
        reconstructed_legacy_dataset = convert_to_legacy(experimental_dataset)  # type: ignore

        # Step 6: Verify the round-trip conversion
        assert isinstance(reconstructed_legacy_dataset, LegacyDataset)
        assert len(reconstructed_legacy_dataset) == len(experimental_dataset)

        # Verify data consistency in round-trip conversion
        for original_item, reconstructed_item in zip(legacy_dataset, reconstructed_legacy_dataset):
            # Check media paths are preserved
            original_media = cast("ImageFromFile", original_item.media)  # pyright: ignore[reportUnknownMemberType]
            reconstructed_media = cast("ImageFromFile", reconstructed_item.media)  # pyright: ignore[reportUnknownMemberType]
            assert original_media.path == reconstructed_media.path

            # Get bbox annotations from both datasets
            original_bboxes = [ann for ann in original_item.annotations if isinstance(ann, Bbox)]
            reconstructed_bboxes = [ann for ann in reconstructed_item.annotations if isinstance(ann, Bbox)]

            assert len(original_bboxes) == len(reconstructed_bboxes)

            # Sort both lists by bbox coordinates for consistent comparison
            original_bboxes_sorted = sorted(original_bboxes, key=lambda b: (b.x, b.y, b.w, b.h))
            reconstructed_bboxes_sorted = sorted(reconstructed_bboxes, key=lambda b: (b.x, b.y, b.w, b.h))

            # Verify each bbox is preserved through round-trip conversion
            for orig_bbox, recon_bbox in zip(original_bboxes_sorted, reconstructed_bboxes_sorted):
                assert orig_bbox.x == recon_bbox.x
                assert orig_bbox.y == recon_bbox.y
                assert orig_bbox.w == recon_bbox.w
                assert orig_bbox.h == recon_bbox.h
                assert orig_bbox.label == recon_bbox.label
