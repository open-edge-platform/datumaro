# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest

from datumaro.experimental.categories import (
    Colormap,
    GroupType,
    HierarchicalLabelCategories,
    HierarchicalLabelCategory,
    LabelCategories,
    LabelGroup,
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


# HierarchicalLabelCategory Tests
def test_hierarchical_label_category_empty_name_raises_error():
    """Test that empty name raises ValueError."""
    with pytest.raises(ValueError, match="Label name cannot be empty"):
        HierarchicalLabelCategory(name="")


def test_hierarchical_label_category_hash():
    """Test that categories with same attributes have same hash."""
    cat1 = HierarchicalLabelCategory(
        name="person", parent="human", label_semantics={"type": "living"}
    )
    cat2 = HierarchicalLabelCategory(
        name="person", parent="human", label_semantics={"type": "living"}
    )
    cat3 = HierarchicalLabelCategory(name="person", parent="", label_semantics={"type": "living"})

    assert hash(cat1) == hash(cat2)
    assert hash(cat1) != hash(cat3)


# LabelGroup Tests
def test_label_group_converts_list_to_tuple():
    """Test that list labels are converted to tuple."""
    group = LabelGroup(name="test", labels=["a", "b", "c"])
    assert group.labels == ("a", "b", "c")
    assert isinstance(group.labels, tuple)


def test_label_group_empty_name_raises_error():
    """Test that empty name raises ValueError."""
    with pytest.raises(ValueError, match="Label group name cannot be empty"):
        LabelGroup(name="", labels=("a", "b"))


def test_label_group_invalid_labels_type_raises_error():
    """Test that invalid labels type raises TypeError."""
    with pytest.raises(TypeError, match="labels must be a tuple of strings"):
        LabelGroup(name="test", labels="invalid")


# HierarchicalLabelCategories Tests
def test_hierarchical_label_categories_empty_creation():
    """Test creation of empty hierarchical categories."""
    categories = HierarchicalLabelCategories()
    assert len(categories) == 0
    assert list(categories) == []
    assert categories.labels == ()


def test_hierarchical_label_categories_basic_creation():
    """Test creation with basic items."""
    items = [
        HierarchicalLabelCategory("animal"),
        HierarchicalLabelCategory("dog", parent="animal"),
        HierarchicalLabelCategory("cat", parent="animal"),
    ]
    categories = HierarchicalLabelCategories(items=items)

    assert len(categories) == 3
    assert categories.labels == ("animal", "dog", "cat")
    assert "dog" in categories
    assert "cat" in categories


def test_hierarchical_label_categories_duplicate_names_raises_error():
    """Test that duplicate names raise ValueError."""
    items = [
        HierarchicalLabelCategory("duplicate"),
        HierarchicalLabelCategory("duplicate"),
    ]
    with pytest.raises(ValueError, match="Duplicate label name: duplicate"):
        HierarchicalLabelCategories(items=items)


def test_hierarchical_label_categories_invalid_parent_raises_error():
    """Test that invalid parent raises ValueError."""
    items = [
        HierarchicalLabelCategory("child", parent="nonexistent"),
    ]
    with pytest.raises(ValueError, match="Parent 'nonexistent' not found for label 'child'"):
        HierarchicalLabelCategories(items=items)


def test_hierarchical_label_categories_invalid_group_label_raises_error():
    """Test that invalid group label raises ValueError."""
    items = [HierarchicalLabelCategory("existing")]
    groups = [LabelGroup("test_group", labels=("nonexistent",))]

    with pytest.raises(
        ValueError, match="Label 'nonexistent' in group 'test_group' not found in items"
    ):
        HierarchicalLabelCategories(items=items, label_groups=groups)


def test_hierarchical_label_categories_get_children():
    """Test getting children of a parent label."""
    items = [
        HierarchicalLabelCategory("animal"),
        HierarchicalLabelCategory("dog", parent="animal"),
        HierarchicalLabelCategory("cat", parent="animal"),
        HierarchicalLabelCategory("bird"),
    ]
    categories = HierarchicalLabelCategories(items=items)

    children = categories.get_children("animal")
    assert set(children) == {"dog", "cat"}

    # Test label with no children
    children = categories.get_children("bird")
    assert children == ()

    # Test non-existing label
    children = categories.get_children("nonexistent")
    assert children == ()


def test_hierarchical_label_categories_get_parent():
    """Test getting parent of a label."""
    items = [
        HierarchicalLabelCategory("animal"),
        HierarchicalLabelCategory("dog", parent="animal"),
    ]
    categories = HierarchicalLabelCategories(items=items)

    # Test label with parent
    parent = categories.get_parent("dog")
    assert parent == "animal"

    # Test root label
    parent = categories.get_parent("animal")
    assert parent == ""

    # Test non-existing label
    parent = categories.get_parent("nonexistent")
    assert parent is None


def test_hierarchical_label_categories_get_hierarchy_level():
    """Test getting hierarchy level of labels."""
    items = [
        HierarchicalLabelCategory("root"),
        HierarchicalLabelCategory("level1", parent="root"),
        HierarchicalLabelCategory("level2", parent="level1"),
    ]
    categories = HierarchicalLabelCategories(items=items)

    assert categories.get_hierarchy_level("root") == 0
    assert categories.get_hierarchy_level("level1") == 1
    assert categories.get_hierarchy_level("level2") == 2


def test_hierarchical_label_categories_indexing():
    """Test indexing and contains operations."""
    items = [
        HierarchicalLabelCategory("first"),
        HierarchicalLabelCategory("second"),
    ]
    categories = HierarchicalLabelCategories(items=items)

    # Test indexing
    assert categories[0].name == "first"
    assert categories[1].name == "second"

    # Test contains by name
    assert "first" in categories
    assert "second" in categories
    assert "nonexistent" not in categories

    # Test contains by index
    assert 0 in categories
    assert 1 in categories
    assert 2 not in categories
    assert -1 not in categories


def test_hierarchical_label_categories_iteration():
    """Test iteration over categories."""
    items = [
        HierarchicalLabelCategory("first"),
        HierarchicalLabelCategory("second"),
    ]
    categories = HierarchicalLabelCategories(items=items)

    names = [cat.name for cat in categories]
    assert names == ["first", "second"]


def test_hierarchical_label_categories_hash():
    """Test hashing of hierarchical categories."""
    items = [HierarchicalLabelCategory("test")]
    groups = [LabelGroup("group1", labels=("test",))]

    cat1 = HierarchicalLabelCategories(items=items, label_groups=groups)
    cat2 = HierarchicalLabelCategories(items=items, label_groups=groups)
    cat3 = HierarchicalLabelCategories(items=items)

    assert hash(cat1) == hash(cat2)
    assert hash(cat1) != hash(cat3)
