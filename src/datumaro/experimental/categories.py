# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
Categories definitions for the experimental dataset system.

This module provides category management functionality using standard dataclasses
instead of attrs, taking inspiration from the original Categories implementation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cache
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, Type, TypeVar, Union


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
    RESTRICTED = 2  # For empty labels

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

    Provides a unified JSON (de)serialization API used by Dataset I/O.
    """

    def to_json_obj(self) -> dict:
        """Serialize this Categories instance to a JSON-serializable dict.

        Subclasses must implement this method and include a 'type' field
        so the factory can reconstruct the instance.
        """
        raise NotImplementedError

    @staticmethod
    def from_json_obj(obj: dict) -> "Categories":
        """Deserialize a Categories instance from its JSON-serializable dict.

        Dispatches based on the 'type' field added by to_json_obj().
        """
        typ = obj.get("type") if isinstance(obj, dict) else None
        try:
            cls = _CATEGORIES_REGISTRY[typ]
        except Exception:
            raise ValueError(f"Unknown Categories type: {typ}")
        return cls.from_json_obj(obj)

    @staticmethod
    def serialize_map(categories_map: Dict[str, "Categories"]) -> dict:
        """Serialize a mapping of attribute name -> Categories to JSON object.

        Output format:
        { "attributes": { name: <category json>, ... } }
        """
        return {"attributes": {name: cat.to_json_obj() for name, cat in categories_map.items()}}

    @staticmethod
    def deserialize_map(data: dict) -> Dict[str, "Categories"]:
        """Deserialize mapping produced by serialize_map()."""
        out: Dict[str, Categories] = {}
        attrs = data.get("attributes", {}) if isinstance(data, dict) else {}
        for name, payload in attrs.items():
            try:
                out[name] = Categories.from_json_obj(payload)
            except Exception:
                # Skip unknown/invalid categories payloads to keep I/O robust
                continue
        return out


# Categories registry for factory-based deserialization
_CATEGORIES_REGISTRY: Dict[str, Type["Categories"]] = {}
T_Cat = TypeVar("T_Cat", bound="Categories")


def register_category(cls: Type[T_Cat]) -> Type[T_Cat]:
    _CATEGORIES_REGISTRY[cls.__name__] = cls
    return cls


@register_category
@dataclass(frozen=True)
class LabelCategories(Categories):
    """
    Represents a group of labels with a specific group type and semantics.
    Use this for simple, non-hierarchical tasks.
    """

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

        index = self._index_map.get(label)
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

    def to_json_obj(self) -> dict:
        ls = {}
        for k, v in self.label_semantics.items():
            if isinstance(k, LabelSemantic):
                ls[k.name] = v
            else:
                ls[str(k)] = v
        return {
            "type": "LabelCategories",
            "labels": list(self.labels),
            "group_type": self.group_type.to_str()
            if isinstance(self.group_type, GroupType)
            else GroupType.EXCLUSIVE.to_str(),
            "label_semantics": ls,
        }

    @classmethod
    def from_json_obj(cls, obj: dict) -> "LabelCategories":
        ls_in = obj.get("label_semantics", {})
        ls: dict = {}
        for k, v in ls_in.items():
            try:
                ls[LabelSemantic[k]] = v
            except Exception:
                ls[k] = v
        labels = tuple(obj.get("labels", ()))
        gt = GroupType.from_str(obj.get("group_type", GroupType.EXCLUSIVE.to_str()))
        return cls(labels=labels, group_type=gt, label_semantics=ls)


@dataclass(frozen=True)
class HierarchicalLabelCategory:
    """Represents a single label category with hierarchical support."""

    name: str
    parent: str = field(default="")
    label_semantics: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate that name is not empty."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Label name cannot be empty and must be a string")

    def __hash__(self):
        return hash((self.name, self.parent, frozenset(self.label_semantics.items())))


@dataclass(frozen=True)
class LabelGroup:
    """Represents a group of labels with a specific group type."""

    name: str
    labels: Tuple[str, ...] = field(default_factory=tuple)
    group_type: GroupType = GroupType.EXCLUSIVE

    def __post_init__(self):
        """Validate that name is not empty and labels is a tuple."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Label group name cannot be empty and must be a string")


@register_category
@dataclass(frozen=True)
class HierarchicalLabelCategories(Categories):
    """
    Represents hierarchical label categories with groups and parent-child relationships.
    Use this for complex hierarchical classification tasks.
    """

    items: Tuple[HierarchicalLabelCategory, ...] = field(default_factory=tuple)
    label_groups: Tuple[LabelGroup, ...] = field(default_factory=tuple)
    label_semantics: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.items, tuple):
            raise TypeError("items must be a tuple of HierarchicalLabelCategory")
        if not isinstance(self.label_groups, tuple):
            raise TypeError("label_groups must be a tuple of LabelGroup")

        # Validate no duplicate names
        seen_names = set()
        for item in self.items:
            if item.name in seen_names:
                raise ValueError(f"Duplicate label name: {item.name}")
            seen_names.add(item.name)

            # Also check for name label_semantics
            # TODO: Remove this check after migrating completely to new system
            if "name" in item.label_semantics:
                name = item.label_semantics["name"]
                if name not in seen_names:
                    seen_names.add(name)

        # Validate that all parents exist
        for item in self.items:
            if item.parent and item.parent not in seen_names:
                raise ValueError(f"Parent '{item.parent}' not found for label '{item.name}'")

        # Validate that all labels in groups exist
        for group in self.label_groups:
            for label_name in group.labels:
                if label_name not in seen_names:
                    raise ValueError(
                        f"Label '{label_name}' in group '{group.name}' not found in items"
                    )

    @property
    @cache
    def _index_map(self) -> Dict[str, int]:
        """Cached mapping from label names to indices."""
        return {item.name: idx for idx, item in enumerate(self.items)}

    @property
    @cache
    def _children_map(self) -> Dict[str, Tuple[str, ...]]:
        """Cached mapping from parent names to child names."""
        children_map: defaultdict[str, List[str]] = defaultdict(list)
        for item in self.items:
            if item.parent:
                children_map[item.parent].append(item.name)
        return {parent: tuple(children) for parent, children in children_map.items()}

    @property
    def labels(self) -> Tuple[str, ...]:
        """Get all label names for compatibility."""
        return tuple(item.name for item in self.items)

    def find(self, name: str) -> Tuple[Optional[int], Optional[HierarchicalLabelCategory]]:
        """
        Find a label by name.

        Args:
            name: The label name to find

        Returns:
            A tuple of (index, category) or (None, None) if not found
        """
        index = self._index_map.get(name)
        if index is not None:
            return index, self.items[index]
        return None, None

    def get_children(self, parent_name: str) -> Tuple[str, ...]:
        """
        Get all children of a parent label.

        Args:
            parent_name: The name of the parent label

        Returns:
            Tuple of child label names
        """
        return self._children_map.get(parent_name, ())

    def get_parent(self, label_name: str) -> Optional[str]:
        """
        Get the parent of a label.

        Args:
            label_name: The name of the label

        Returns:
            Parent name or None if no parent
        """
        index = self._index_map.get(label_name)
        if index is not None:
            return self.items[index].parent
        return None

    def get_hierarchy_level(self, label_name: str) -> int:
        """
        Get the hierarchy level of a label (0 for root, 1 for first level children, etc.)

        Args:
            label_name: The name of the label

        Returns:
            Hierarchy level
        """
        level = 0
        current = label_name
        while True:
            parent = self.get_parent(current)
            if not parent:
                break
            level += 1
            current = parent
        return level

    def __getitem__(self, idx: int) -> HierarchicalLabelCategory:
        """Get category by index."""
        return self.items[idx]

    def __contains__(self, value: Union[int, str]) -> bool:
        """Check if a label exists by name or index."""
        if isinstance(value, str):
            return value in self._index_map
        else:
            return 0 <= value < len(self.items)

    def __len__(self) -> int:
        """Get the number of labels."""
        return len(self.items)

    def __iter__(self) -> Iterator[HierarchicalLabelCategory]:
        """Iterate over label categories."""
        return iter(self.items)

    def __hash__(self):
        # Hash label_groups via value-based representation to avoid relying on their own hash implementation.
        lg_repr = tuple((lg.name, tuple(lg.labels), lg.group_type) for lg in self.label_groups)
        return hash((self.items, lg_repr, frozenset(self.label_semantics.items())))

    def to_json_obj(self) -> dict:
        def _ser_item(it: HierarchicalLabelCategory) -> dict:
            ls = {}
            for k, v in it.label_semantics.items():
                if isinstance(k, LabelSemantic):
                    ls[k.name] = v
                else:
                    ls[str(k)] = v
            return {"name": it.name, "parent": it.parent, "label_semantics": ls}

        def _ser_group(g: LabelGroup) -> dict:
            return {
                "name": g.name,
                "labels": list(g.labels),
                "group_type": g.group_type.to_str()
                if isinstance(g.group_type, GroupType)
                else GroupType.EXCLUSIVE.to_str(),
            }

        top_ls = {}
        for k, v in self.label_semantics.items():
            if isinstance(k, LabelSemantic):
                top_ls[k.name] = v
            else:
                top_ls[str(k)] = v

        return {
            "type": "HierarchicalLabelCategories",
            "items": [_ser_item(it) for it in self.items],
            "label_groups": [_ser_group(g) for g in self.label_groups],
            "label_semantics": top_ls,
        }

    @classmethod
    def from_json_obj(cls, obj: dict) -> "HierarchicalLabelCategories":
        def _deser_label_semantics(d: dict) -> dict:
            out = {}
            for k, v in d.items():
                try:
                    out[LabelSemantic[k]] = v
                except Exception:
                    out[k] = v
            return out

        items = []
        for it in obj.get("items", []):
            ls = _deser_label_semantics(it.get("label_semantics", {}))
            items.append(
                HierarchicalLabelCategory(
                    name=it.get("name", ""), parent=it.get("parent", ""), label_semantics=ls
                )
            )
        groups = []
        for g in obj.get("label_groups", []):
            gt = GroupType.from_str(g.get("group_type", GroupType.EXCLUSIVE.to_str()))
            groups.append(
                LabelGroup(name=g.get("name", ""), labels=tuple(g.get("labels", [])), group_type=gt)
            )
        top_ls = _deser_label_semantics(obj.get("label_semantics", {}))
        return cls(items=tuple(items), label_groups=tuple(groups), label_semantics=top_ls)


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


@register_category
@dataclass(frozen=True)
class MaskCategories(Categories):
    """
    Describes a color map for segmentation masks.
    """

    labels: List[str] = field(default_factory=list)
    colormap: Colormap = field(default_factory=Colormap)

    def __hash__(self):
        return hash((tuple(self.labels), frozenset(self.colormap.items())))

    def to_json_obj(self) -> dict:
        cmap: dict[str, list[int]] = {}
        for idx, color in self.colormap:
            cmap[str(idx)] = [int(color.r), int(color.g), int(color.b)]
        return {
            "type": "MaskCategories",
            "labels": list(self.labels),
            "colormap": cmap,
        }

    @classmethod
    def from_json_obj(cls, obj: dict) -> "MaskCategories":
        labels = list(obj.get("labels", []))
        cmap_in: dict[str, list[int]] = obj.get("colormap", {})
        cmap: dict[int, RgbColor] = {}
        for k, rgb in cmap_in.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
                cmap[idx] = RgbColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        return cls(labels=labels, colormap=Colormap(data=cmap))

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
