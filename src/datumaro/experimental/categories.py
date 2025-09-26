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

    EXCLUSIVE = 0  # Only one label from the group can be assigned
    INCLUSIVE = 1  # Multiple labels from the group can be assigned

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
class LabelCategories(Categories):
    """Represents a group of labels with a specific group type."""

    labels: List[str] = field(default_factory=list)
    group_type: GroupType = GroupType.EXCLUSIVE
    _indices: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._reindex()

    def _reindex(self):
        """Rebuild the internal index mapping names to positions."""
        indices = {}
        for index, label in enumerate(self.labels):
            if label in indices:
                raise ValueError(f"Duplicate label: {label}")
            indices[label] = index
        self._indices = indices

    def add(self, label: str) -> int:
        """
        Add a new label.

        Args:
            label: The label name

        Returns:
            The index of the newly added category

        Raises:
            ValueError: If label already exists
        """
        if label in self.labels:
            raise ValueError(f"Duplicate label: {label}")
        index = len(self.labels)
        self.labels.append(label)
        self._indices[label] = index
        return index

    def find(self, name: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Find a label by name.

        Args:
            name: The label name to find

        Returns:
            A tuple of (index, category) or (None, None) if not found
        """
        index = self._indices.get(name)
        if index is not None:
            return index, self.labels[index]
        return None, None

    def __getitem__(self, idx: int) -> str:
        """Get category by index."""
        return self.labels[idx]

    def __contains__(self, value: Union[int, str]) -> bool:
        """Check if a label exists by name or index."""
        if isinstance(value, str):
            return value in self.labels
        else:
            return 0 <= value < len(self.labels)

    def __len__(self) -> int:
        """Get the number of labels."""
        return len(self.labels)

    def __iter__(self):
        """Iterate over label."""
        return iter(self.labels)


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

    labels: List[str] = field(default_factory=list)
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
