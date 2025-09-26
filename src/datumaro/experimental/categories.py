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
from functools import cache
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, Union


class LabelSemantic(IntEnum):
    """
    Semantic meaning of a label for classification tasks.
    NORMAL: Indicates a label representing normal/expected data.
    ANOMALOUS: Indicates a label representing anomalous/outlier data.
    """

    NORMAL = 1
    ANOMALOUS = 2


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


@dataclass(frozen=True)
class Categories:
    """
    A base class for annotation metainfo. It is supposed to include
    dataset-wide metainfo like available labels, label colors,
    label attributes etc.
    """

    pass


@dataclass(frozen=True)
class LabelCategories(Categories):
    """Represents a group of labels with a specific group type and semantics."""

    labels: Tuple[str, ...] = field(default_factory=tuple)
    group_type: GroupType = GroupType.EXCLUSIVE
    label_semantics: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate that there are no duplicate labels."""
        if isinstance(self.labels, list):
            object.__setattr__(self, "labels", tuple(self.labels))
        elif not isinstance(self.labels, tuple):
            raise TypeError("labels must be a tuple of strings")

        seen = set()
        for label in self.labels:
            if label in seen:
                raise ValueError(f"Duplicate label: {label}")
            seen.add(label)

    @property
    @cache
    def _index_map(self) -> Dict[str, int]:
        """Cached mapping from label names to indices."""
        return {label: idx for idx, label in enumerate(self.labels)}

    def find(self, name_or_semantic: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Find a label by name or LabelSemantic.

        Args:
            name_or_semantic: The label name or LabelSemantic to find

        Returns:
            A tuple of (index, category) or (None, None) if not found
        """
        if isinstance(name_or_semantic, LabelSemantic):
            label = self.label_semantics.get(name_or_semantic)
            if label is None:
                return None, None
        else:
            label = name_or_semantic

        index = self._index_map.get(name_or_semantic)
        if index is not None:
            return index, self.labels[index]
        return None, None

    def __getitem__(self, idx: int) -> str:
        """Get category by index."""
        return self.labels[idx]

    def __contains__(self, value: Union[int, str, LabelSemantic]) -> bool:
        """Check if a label exists by name, index, or semantic."""
        if isinstance(value, LabelSemantic):
            return value in self.label_semantics
        elif isinstance(value, str):
            return value in self.labels
        else:
            return 0 <= value < len(self.labels)

    def __len__(self) -> int:
        """Get the number of labels."""
        return len(self.labels)

    def __iter__(self):
        """Iterate over label."""
        return iter(self.labels)

    def __hash__(self):
        # Include label_semantics in the hash
        return hash((self.labels, self.group_type, frozenset(self.label_semantics.items())))


class RgbColor(NamedTuple):
    """RGB color representation with named fields."""

    r: int
    g: int
    b: int


@dataclass(frozen=True)
class Colormap:
    """
    A colormap that stores index-to-color mappings and provides efficient
    reverse lookup via an inverse colormap property.
    """

    data: Dict[int, RgbColor] = field(default_factory=dict)

    def __post_init__(self):
        """Validate that there are no duplicate colors."""
        object.__setattr__(self, "_inverse_colormap", {v: k for k, v in self.data.items()})

    @property
    def inverse_colormap(self) -> Dict[RgbColor, int]:
        """Get the inverse colormap (color -> index mapping)."""
        return getattr(self, "_inverse_colormap")

    def __getitem__(self, index: int) -> RgbColor:
        """Get color by index."""
        return self.data[index]

    def __contains__(self, index: int) -> bool:
        """Check if an index exists in the colormap."""
        return index in self.data

    def __len__(self) -> int:
        """Get the number of colors in the colormap."""
        return len(self.data)

    def __iter__(self) -> Iterator[Tuple[int, RgbColor]]:
        """Iterate over colormap items."""
        return iter(self.data.items())

    def get(self, index: int, default=None):
        """Get color by index with default."""
        return self.data.get(index, default)

    def __eq__(self, other):
        """Compare with another Colormap or dictionary."""
        if isinstance(other, Colormap):
            return self.data == other.data
        elif isinstance(other, dict):
            return self.data == other
        return False


@dataclass(frozen=True)
class MaskCategories(Categories):
    """
    Describes a color map for segmentation masks.
    """

    labels: List[str] = field(default_factory=list)
    colormap: Colormap = field(default_factory=Colormap)

    def __hash__(self):
        return hash((tuple(self.labels), frozenset(self.colormap.items())))

    @classmethod
    def generate(cls, size: int = 255, include_background: bool = True) -> "MaskCategories":
        """
        Generates MaskCategories with the specified size.

        If include_background is True, the result will include the item
            "0: (0, 0, 0)", which is typically used as a background color.
        """
        # Import here to avoid circular dependencies
        from datumaro.util.mask_tools import generate_colormap

        # TODO: Refactor generate_colormap to return a Colormap.
        colormap_dict = generate_colormap(size, include_background=include_background)
        colormap_data = {}
        for index, color in colormap_dict.items():
            colormap_data[index] = RgbColor(*color) if isinstance(color, tuple) else color

        colormap = Colormap(data=colormap_data)
        return cls(colormap=colormap)
