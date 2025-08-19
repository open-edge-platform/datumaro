# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
Categories definitions for the experimental dataset system.

This module provides category management functionality using standard dataclasses
instead of attrs, taking inspiration from the original Categories implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, Union


class GroupType(IntEnum):
    """Types of label groups for organizing labels."""

    EXCLUSIVE = 0
    INCLUSIVE = 1
    RESTRICTED = 2

    def to_str(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, text: str) -> GroupType:
        try:
            return cls[text.upper()]
        except KeyError:
            raise ValueError(f"Invalid GroupType: {text}")


@dataclass
class Categories:
    """
    A base class for annotation metainfo. It is supposed to include
    dataset-wide metainfo like available labels, label colors,
    label attributes etc.
    """

    pass


@dataclass
class LabelCategory:
    """Represents a single label category with optional parent and attributes."""

    name: str
    parent: str = ""

    def __post_init__(self):
        if not self.name:
            raise ValueError("Category name cannot be empty")


@dataclass
class LabelGroup:
    """Represents a group of labels with a specific group type."""

    name: str
    labels: List[str] = field(default_factory=list)
    group_type: GroupType = GroupType.EXCLUSIVE

    def __post_init__(self):
        if not self.name:
            raise ValueError("Label group name cannot be empty")


@dataclass
class LabelCategories(Categories):
    """
    Label categories management with support for hierarchical labels and groups.
    """

    items: List[LabelCategory] = field(default_factory=list)
    label_groups: List[LabelGroup] = field(default_factory=list)
    _indices: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._reindex()

    @classmethod
    def from_iterable(
        cls,
        iterable: Union[
            List[str],
            List[Tuple[str]],
            List[Tuple[str, str]],
            List[Tuple[str, str, List[str]]],
        ],
    ) -> LabelCategories:
        """
        Creates a LabelCategories from iterable.

        Args:
            iterable: This iterable object can be:
                - a list of str - will be interpreted as list of Category names
                - a list of positional arguments - will generate Categories
                  with these arguments

        Returns: a LabelCategories object
        """
        temp_categories = cls()

        for category in iterable:
            if isinstance(category, str):
                category = [category]
            temp_categories.add(*category)

        return temp_categories

    def _reindex(self):
        """Rebuild the internal index mapping names to positions."""
        indices = {}
        for index, item in enumerate(self.items):
            if item.name in indices:
                raise ValueError(f"Duplicate category name: {item.name}")
            indices[item.name] = index
        self._indices = indices

    def add(
        self,
        name: str,
        parent: Optional[str] = None,
    ) -> int:
        """
        Add a new label category.

        Args:
            name: The category name
            parent: Optional parent category name

        Returns:
            The index of the newly added category

        Raises:
            AssertionError: If name is empty or already exists
        """
        if name in self._indices:
            raise ValueError(f"Category '{name}' already exists")

        index = len(self.items)
        category = LabelCategory(
            name=name,
            parent=parent or "",
        )
        self.items.append(category)
        self._indices[name] = index
        return index

    def add_label_group(
        self,
        name: str,
        labels: List[str],
        group_type: GroupType = GroupType.EXCLUSIVE,
    ) -> int:
        """
        Add a new label group.

        Args:
            name: The group name
            labels: List of label names in this group
            group_type: The type of group (exclusive, inclusive, restricted)

        Returns:
            The index of the newly added label group
        """

        index = len(self.label_groups)
        label_group = LabelGroup(name=name, labels=labels, group_type=group_type)
        self.label_groups.append(label_group)
        return index

    def find(self, name: str) -> Tuple[Optional[int], Optional[LabelCategory]]:
        """
        Find a category by name.

        Args:
            name: The category name to find

        Returns:
            A tuple of (index, category) or (None, None) if not found
        """
        index = self._indices.get(name)
        if index is not None:
            return index, self.items[index]
        return None, None

    def __getitem__(self, idx: int) -> LabelCategory:
        """Get category by index."""
        return self.items[idx]

    def __contains__(self, value: Union[int, str]) -> bool:
        """Check if a category exists by name or index."""
        if isinstance(value, str):
            return self.find(value)[1] is not None
        else:
            return 0 <= value < len(self.items)

    def __len__(self) -> int:
        """Get the number of categories."""
        return len(self.items)

    def __iter__(self):
        """Iterate over categories."""
        return iter(self.items)


class RgbColor(NamedTuple):
    """RGB color representation with named fields."""

    r: int
    g: int
    b: int


@dataclass
class Colormap:
    """
    A colormap that stores index-to-color mappings and provides efficient
    reverse lookup via an inverse colormap property.
    """

    _data: Dict[int, RgbColor] = field(default_factory=dict)
    _inverse_colormap: Optional[Dict[RgbColor, int]] = field(default=None, init=False, repr=False)

    @property
    def inverse_colormap(self) -> Dict[RgbColor, int]:
        """Get the inverse colormap (color -> index mapping)."""
        if self._inverse_colormap is None:
            self._inverse_colormap = {v: k for k, v in self._data.items()}
        return self._inverse_colormap

    def __setitem__(self, index: int, color: RgbColor):
        """Set a color for an index."""
        self._data[index] = color
        # Invalidate cached inverse colormap
        self._inverse_colormap = None

    def __getitem__(self, index: int) -> RgbColor:
        """Get color by index."""
        return self._data[index]

    def __contains__(self, index: int) -> bool:
        """Check if an index exists in the colormap."""
        return index in self._data

    def __len__(self) -> int:
        """Get the number of colors in the colormap."""
        return len(self._data)

    def __iter__(self) -> Iterator[Tuple[int, RgbColor]]:
        """Iterate over colormap items."""
        return iter(self._data.items())

    def get(self, index: int, default=None):
        """Get color by index with default."""
        return self._data.get(index, default)

    def __eq__(self, other):
        """Compare with another Colormap or dictionary."""
        if isinstance(other, Colormap):
            return self._data == other._data
        elif isinstance(other, dict):
            return self._data == other
        return False


@dataclass
class MaskCategories(Categories):
    """
    Describes a color map for segmentation masks.
    """

    colormap: Colormap = field(default_factory=Colormap)

    @classmethod
    def generate(cls, size: int = 255, include_background: bool = True) -> MaskCategories:
        """
        Generates MaskCategories with the specified size.

        If include_background is True, the result will include the item
            "0: (0, 0, 0)", which is typically used as a background color.
        """
        # Import here to avoid circular dependencies
        from datumaro.util.mask_tools import generate_colormap

        # TODO: Refactor generate_colormap to return a Colormap.
        colormap_dict = generate_colormap(size, include_background=include_background)
        colormap = Colormap()
        for index, color in colormap_dict.items():
            colormap[index] = RgbColor(*color) if isinstance(color, tuple) else color

        return cls(colormap=colormap)
