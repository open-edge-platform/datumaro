"""
Sidecar metadata utilities for Datumaro Experimental.

This module is responsible for serializing/deserializing the sidecar JSON file
that accompanies Parquet datasets. The sidecar contains:
- a compact representation of the schema (attribute names, field types, semantics),
- categories metadata for annotated attributes, and
- optional `media` information used to rebase/resolve image paths during load.

Keeping this logic separate from the Dataset class helps maintain a clear
separation between core data structures and IO concerns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..categories import Categories
from ..schema import Schema


def serialize_schema_dict(schema: Schema) -> dict[str, Any]:
    """
    Convert a `Schema` into a compact JSON-serializable dictionary.

    The resulting dict lists attributes with their name, underlying Python type
    name, field class, and optional semantic tag. This is sufficient to
    reconstruct type information for loaders and for debugging/inspection.

    Args:
        schema: The dataset schema to serialize.

    Returns:
        A dict of the form `{ "attributes": [ {name, type, field, semantic}, ... ] }`.
    """
    attrs: list[dict[str, Any]] = []
    for attr_name, attr in schema.attributes.items():
        try:
            type_name = getattr(attr.type, "__name__", repr(attr.type))
        except Exception:
            type_name = repr(attr.type)
        semantic = getattr(attr.annotation, "semantic", None)
        semantic_name = getattr(semantic, "name", None) if semantic is not None else None
        attrs.append(
            {
                "name": attr_name,
                "type": type_name,
                "field": attr.annotation.__class__.__name__,
                "semantic": semantic_name,
            }
        )
    return {"attributes": attrs}


def collect_categories_map(schema: Schema) -> Dict[str, Categories]:
    categories_map: Dict[str, Categories] = {}
    for name, info in schema.attributes.items():
        if info.categories is not None:
            categories_map[name] = info.categories
    return categories_map


def build_sidecar_dict(schema: Schema, media_meta: dict[str, Any] | None) -> dict[str, Any]:
    sidecar: dict[str, Any] = {}
    sidecar["schema"] = serialize_schema_dict(schema)
    categories_map = collect_categories_map(schema)
    sidecar["categories"] = Categories.serialize_map(categories_map) if categories_map else {}
    if media_meta is not None:
        sidecar["media"] = media_meta
    return sidecar


def write_sidecar_file(parquet_path: Path, sidecar: dict[str, Any]) -> None:
    try:
        json_path = Path(parquet_path).with_suffix(".json")
        json_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2))
    except Exception as e:
        # Best-effort: if sidecar write fails, we still have the parquet file
        import warnings

        warnings.warn(f"Failed to write sidecar JSON next to {parquet_path}: {e}")


def read_sidecar_file(parquet_path: Path) -> dict[str, Any] | None:
    try:
        json_path = Path(parquet_path).with_suffix(".json")
        if not json_path.exists():
            return None
        return json.loads(json_path.read_text())
    except Exception:
        return None


def deserialize_categories_map(data: dict[str, Any]) -> Dict[str, Categories]:
    return Categories.deserialize_map(data)
