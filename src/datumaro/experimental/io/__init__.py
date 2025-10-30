"""
Experimental IO utilities for Datumaro.

This package groups helpers used by the experimental Parquet backend, including:
- media packaging and path rewriting (copy/symlink/move) at save time,
- sidecar JSON serialization/deserialization for schema and categories,
- zip archive creation and extraction for portable dataset bundles.

These modules are intentionally lightweight and independent from `Dataset` so
that the core remains focused on data modeling and transformations.
"""

from .archive import create_zip_archive, extract_zip_to_dir
from .media_packager import (
    MediaCopyMode,
    PathStyle,
    detect_file_path_fields,
    package_and_rewrite_paths,
    rebase_media_paths,
)
from .sidecar import (
    build_sidecar_dict,
    collect_categories_map,
    deserialize_categories_map,
    read_sidecar_file,
    serialize_schema_dict,
    write_sidecar_file,
)

__all__ = [
    "MediaCopyMode",
    "PathStyle",
    "detect_file_path_fields",
    "package_and_rewrite_paths",
    "rebase_media_paths",
    "build_sidecar_dict",
    "write_sidecar_file",
    "read_sidecar_file",
    "serialize_schema_dict",
    "collect_categories_map",
    "deserialize_categories_map",
    "create_zip_archive",
    "extract_zip_to_dir",
]
