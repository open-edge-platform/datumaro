# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest

from datumaro.experimental.categories import (
    Colormap,
    GroupType,
    LabelCategories,
    LabelCategory,
    LabelGroup,
    MaskCategories,
    RgbColor,
)


def test_label_category_creation():
    category = LabelCategory(name="child", parent="person")
    assert category.name == "child"
    assert category.parent == "person"


def test_label_category_empty_name_raises_error():
    with pytest.raises(ValueError, match="Category name cannot be empty"):
        LabelCategory(name="")


def test_label_group_basic_creation():
    group = LabelGroup(name="vehicles")
    assert group.name == "vehicles"
    assert group.labels == []
    assert group.group_type == GroupType.EXCLUSIVE


def test_label_group_creation_with_labels_and_type():
    group = LabelGroup(name="vehicles", labels=["car", "truck"], group_type=GroupType.INCLUSIVE)
    assert group.name == "vehicles"
    assert group.labels == ["car", "truck"]
    assert group.group_type == GroupType.INCLUSIVE


def test_label_group_empty_name_raises_error():
    with pytest.raises(ValueError, match="Label group name cannot be empty"):
        LabelGroup(name="")


def test_group_type_to_str():
    assert GroupType.EXCLUSIVE.to_str() == "exclusive"
    assert GroupType.INCLUSIVE.to_str() == "inclusive"
    assert GroupType.RESTRICTED.to_str() == "restricted"


def test_group_type_from_str():
    assert GroupType.from_str("exclusive") == GroupType.EXCLUSIVE
    assert GroupType.from_str("INCLUSIVE") == GroupType.INCLUSIVE
    assert GroupType.from_str("Restricted") == GroupType.RESTRICTED


def test_group_type_from_str_invalid():
    with pytest.raises(ValueError, match="Invalid GroupType: invalid"):
        GroupType.from_str("invalid")


def test_label_categories_empty_creation():
    categories = LabelCategories()
    assert len(categories) == 0
    assert list(categories) == []


def test_label_categories_add_category():
    categories = LabelCategories()
    index = categories.add("person")

    assert index == 0
    assert len(categories) == 1
    assert "person" in categories
    assert categories[0].name == "person"


def test_label_categories_add_category_with_parent():
    categories = LabelCategories()
    categories.add("person")
    index = categories.add("child", parent="person")

    assert index == 1
    assert categories[1].name == "child"
    assert categories[1].parent == "person"


def test_label_categories_add_duplicate_name_raises_error():
    categories = LabelCategories()
    categories.add("person")

    with pytest.raises(ValueError):
        categories.add("person")


def test_label_categories_find_existing_category():
    categories = LabelCategories()
    categories.add("person")

    index, category = categories.find("person")
    assert index == 0
    assert category.name == "person"


def test_label_categories_find_nonexistent_category():
    categories = LabelCategories()

    index, category = categories.find("nonexistent")
    assert index is None
    assert category is None


def test_label_categories_contains_by_name():
    categories = LabelCategories()
    categories.add("person")

    assert "person" in categories
    assert "nonexistent" not in categories


def test_label_categories_contains_by_index():
    categories = LabelCategories()
    categories.add("person")

    assert 0 in categories
    assert 1 not in categories
    assert -1 not in categories


def test_label_categories_add_label_group():
    categories = LabelCategories()
    index = categories.add_label_group("vehicles", ["car", "truck"], GroupType.INCLUSIVE)

    assert index == 0
    assert len(categories.label_groups) == 1
    group = categories.label_groups[0]
    assert group.name == "vehicles"
    assert group.labels == ["car", "truck"]
    assert group.group_type == GroupType.INCLUSIVE


def test_label_categories_from_iterable_strings():
    categories = LabelCategories.from_iterable(["person", "car", "bicycle"])

    assert len(categories) == 3
    assert "person" in categories
    assert "car" in categories
    assert "bicycle" in categories


def test_label_categories_from_iterable_tuples():
    categories = LabelCategories.from_iterable(
        [("person",), ("car", "vehicle"), ("bicycle", "vehicle")]
    )

    assert len(categories) == 3
    assert categories[1].parent == "vehicle"


def test_label_categories_iteration():
    categories = LabelCategories()
    categories.add("person")
    categories.add("car")

    names = [cat.name for cat in categories]
    assert names == ["person", "car"]


def test_mask_categories_empty_creation():
    categories = MaskCategories()
    assert len(categories.colormap) == 0


def test_mask_categories_manual_colormap():
    colormap = Colormap()
    colormap[0] = RgbColor(0, 0, 0)
    colormap[1] = RgbColor(255, 0, 0)
    colormap[2] = RgbColor(0, 255, 0)

    categories = MaskCategories(colormap=colormap)

    assert len(categories.colormap) == 3
    assert categories.colormap[0] == RgbColor(0, 0, 0)
    assert categories.colormap[1] == RgbColor(255, 0, 0)
    assert categories.colormap[2] == RgbColor(0, 255, 0)


def test_mask_categories_setitem_getitem():
    categories = MaskCategories()
    categories.colormap[0] = RgbColor(255, 255, 255)

    assert categories.colormap[0] == RgbColor(255, 255, 255)
    assert 0 in categories.colormap


def test_mask_categories_inverse_colormap():
    colormap = Colormap()
    colormap[0] = RgbColor(0, 0, 0)
    colormap[1] = RgbColor(255, 0, 0)

    categories = MaskCategories(colormap=colormap)

    inverse = categories.colormap.inverse_colormap
    assert inverse[RgbColor(0, 0, 0)] == 0
    assert inverse[RgbColor(255, 0, 0)] == 1


def test_mask_categories_inverse_colormap_invalidation():
    categories = MaskCategories()
    categories.colormap[0] = RgbColor(255, 255, 255)

    # Access inverse colormap to cache it
    inverse1 = categories.colormap.inverse_colormap
    assert inverse1[RgbColor(255, 255, 255)] == 0

    # Modify colormap - should invalidate cache
    categories.colormap[1] = RgbColor(0, 0, 0)
    inverse2 = categories.colormap.inverse_colormap

    # Should be a new dictionary with both colors
    assert len(inverse2) == 2
    assert inverse2[RgbColor(255, 255, 255)] == 0
    assert inverse2[RgbColor(0, 0, 0)] == 1


def test_mask_categories_iteration():
    colormap = Colormap()
    colormap[0] = RgbColor(0, 0, 0)
    colormap[1] = RgbColor(255, 0, 0)

    categories = MaskCategories(colormap=colormap)

    items = list(categories.colormap)
    assert items == [(0, RgbColor(0, 0, 0)), (1, RgbColor(255, 0, 0))]
