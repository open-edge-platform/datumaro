# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest

from datumaro.experimental.categories import (
    Colormap,
    GroupType,
    LabelCategories,
    MaskCategories,
    RgbColor,
)


def test_group_type_to_str():
    assert GroupType.EXCLUSIVE.to_str() == "exclusive"
    assert GroupType.INCLUSIVE.to_str() == "inclusive"


def test_group_type_from_str():
    assert GroupType.from_str("exclusive") == GroupType.EXCLUSIVE
    assert GroupType.from_str("INCLUSIVE") == GroupType.INCLUSIVE


def test_group_type_from_str_invalid():
    with pytest.raises(ValueError, match="Invalid GroupType: invalid"):
        GroupType.from_str("invalid")


def test_label_categories_empty_creation():
    categories = LabelCategories()
    assert len(categories) == 0
    assert list(categories) == []


def test_label_categories_add_label():
    categories = LabelCategories(labels=("person",))
    assert len(categories) == 1
    assert "person" in categories
    assert categories[0] == "person"


def test_label_categories_add_duplicate_name_raises_error():
    with pytest.raises(ValueError):
        LabelCategories(labels=("person", "person"))


def test_label_categories_find_existing_category():
    categories = LabelCategories(labels=("person",))

    index, category = categories.find("person")
    assert index == 0
    assert category == "person"


def test_label_categories_find_nonexistent_category():
    categories = LabelCategories(labels=())

    index, category = categories.find("nonexistent")
    assert index is None
    assert category is None


def test_label_categories_contains_by_name():
    categories = LabelCategories(labels=("person",))

    assert "person" in categories
    assert "nonexistent" not in categories


def test_label_categories_contains_by_index():
    categories = LabelCategories(labels=("person",))

    assert 0 in categories
    assert 1 not in categories
    assert -1 not in categories


def test_label_categories():
    categories = LabelCategories(labels=("person", "car", "bicycle"))

    assert len(categories) == 3
    assert "person" in categories
    assert "car" in categories
    assert "bicycle" in categories


def test_label_categories_iteration():
    categories = LabelCategories(labels=("person", "car"))

    names = [cat for cat in categories]
    assert names == ["person", "car"]


def test_mask_categories_empty_creation():
    categories = MaskCategories()
    assert len(categories.colormap) == 0


def test_mask_categories_manual_colormap():
    colormap = Colormap(data={0: RgbColor(0, 0, 0), 1: RgbColor(255, 0, 0), 2: RgbColor(0, 255, 0)})

    categories = MaskCategories(colormap=colormap)

    assert len(categories.colormap) == 3
    assert categories.colormap[0] == RgbColor(0, 0, 0)
    assert categories.colormap[1] == RgbColor(255, 0, 0)
    assert categories.colormap[2] == RgbColor(0, 255, 0)


def test_mask_categories_immutability():
    categories = MaskCategories(colormap=Colormap(data={0: RgbColor(255, 255, 255)}))

    assert categories.colormap[0] == RgbColor(255, 255, 255)
    assert 0 in categories.colormap

    # Test immutability
    with pytest.raises(Exception):
        categories.colormap[1] = RgbColor(0, 0, 0)


def test_mask_categories_inverse_colormap():
    colormap = Colormap(data={0: RgbColor(0, 0, 0), 1: RgbColor(255, 0, 0)})

    categories = MaskCategories(colormap=colormap)

    inverse = categories.colormap.inverse_colormap
    assert inverse[RgbColor(0, 0, 0)] == 0
    assert inverse[RgbColor(255, 0, 0)] == 1


def test_mask_categories_inverse_colormap_caching():
    categories = MaskCategories(
        colormap=Colormap(data={0: RgbColor(255, 255, 255), 1: RgbColor(0, 0, 0)})
    )

    # Access inverse colormap multiple times - should use cache
    inverse1 = categories.colormap.inverse_colormap
    inverse2 = categories.colormap.inverse_colormap

    assert len(inverse1) == 2
    assert inverse1[RgbColor(255, 255, 255)] == 0
    assert inverse1[RgbColor(0, 0, 0)] == 1
    assert inverse1 is inverse2  # Should return same cached object


def test_mask_categories_iteration():
    colormap = Colormap(data={0: RgbColor(0, 0, 0), 1: RgbColor(255, 0, 0)})

    categories = MaskCategories(colormap=colormap)

    items = list(categories.colormap)
    assert items == [(0, RgbColor(0, 0, 0)), (1, RgbColor(255, 0, 0))]
