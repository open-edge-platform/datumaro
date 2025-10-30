"""
Media packaging utilities for Datumaro Experimental.

This module provides small, focused helpers used by the Parquet IO backend to:
- detect which Dataset columns store image file paths (ImagePathField),
- copy/symlink/move referenced media into a co-located directory while saving,
- rewrite table values to portable paths (relative or absolute), and
- rebase/resolve those paths when loading datasets back.

All functions here are pure helpers and do not depend on the Dataset class
itself, only on primitive inputs such as DataFrames and Schema. This keeps the
core Dataset free of IO concerns and makes testing easier.
"""
from __future__ import annotations

import shutil
from enum import Enum
from pathlib import Path
from typing import Any

import polars as pl

from ..fields import ImagePathField
from ..schema import Schema


class MediaCopyMode(Enum):
    """How media files are materialized into the media directory.

    Note: values are stringly-typed to be stored verbatim in sidecar JSON.
    """

    COPY = "copy"
    SYMLINK = "symlink"
    MOVE = "move"


class PathStyle(Enum):
    """How paths are stored in the parquet table."""

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


def detect_file_path_fields(schema: Schema) -> list[str]:
    """
    Return the list of column names that store file paths.

    The function inspects the provided `schema` and collects attributes whose
    field type is `ImagePathField`.

    Args:
        schema: The dataset schema to inspect.

    Returns:
        A list of column names to treat as image path columns.
    """
    detected: list[str] = []
    for name, info in schema.attributes.items():
        if isinstance(info.annotation, ImagePathField):
            detected.append(name)
    return detected


def _prepare_media_root(out_parquet_path: Path, media_dir: str) -> Path:
    out_dir = Path(out_parquet_path).parent
    media_root = out_dir / media_dir
    media_root.mkdir(parents=True, exist_ok=True)
    return media_root


def _resolve_media_destination(media_root: Path, src: Path) -> Path:
    dst = media_root / src.name
    if dst.exists():
        try:
            same = src.exists() and dst.exists() and src.samefile(dst)
        except Exception:
            same = False
        if not same:
            stem, ext = src.stem, src.suffix
            i = 1
            candidate = media_root / f"{stem}_{i}{ext}"
            while candidate.exists():
                i += 1
                candidate = media_root / f"{stem}_{i}{ext}"
            dst = candidate
    return dst


def _transfer_media(src: Path, dst: Path, copy_mode: MediaCopyMode) -> None:
    if copy_mode == MediaCopyMode.COPY:
        shutil.copy2(src, dst)
    elif copy_mode == MediaCopyMode.SYMLINK:
        try:
            if not dst.exists():
                dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
    elif copy_mode == MediaCopyMode.MOVE:
        if not dst.exists():
            shutil.move(str(src), str(dst))
    else:
        raise ValueError("Unsupported copy_mode: %r" % copy_mode)


def _format_stored_path(dst: Path, media_dir: str, path_style: PathStyle) -> str:
    if path_style == PathStyle.RELATIVE:
        rel = Path(media_dir) / dst.name
        return rel.as_posix()
    if path_style == PathStyle.ABSOLUTE:
        return str(dst.resolve())
    raise ValueError("Unsupported path_style: %r" % path_style)


def package_and_rewrite_paths(
    *,
    df: pl.DataFrame,
    out_parquet_path: Path,
    path_fields: list[str],
    media_dir: str,
    copy_mode: MediaCopyMode,
    path_style: PathStyle,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Copy/symlink/move media files next to the Parquet and rewrite path columns.

    For each column listed in `path_fields`, the function:
    - copies (or symlinks/moves) the referenced files into `<parquet_dir>/<media_dir>/`,
    - rewrites the column values to either relative (`media/<name>`) or absolute
      paths depending on `path_style`.

    A sidecar `media_meta` dict describing the chosen layout is returned so the
    loader can later rebase/resolve the paths.

    Args:
        df: The input DataFrame containing path columns.
        out_parquet_path: Destination Parquet path (used to locate the media dir).
        path_fields: Names of the columns that hold image paths.
        media_dir: The directory name under which media will be stored.
        copy_mode: One of "copy", "symlink", or "move".
        path_style: Either "relative" or "absolute" for stored values.

    Returns:
        A tuple of `(new_df, media_meta)` where `new_df` is the updated DataFrame
        and `media_meta` is a JSON-serializable dict with keys: `root`,
        `path_fields`, `style`, and `copy_mode`.
    """
    media_root = _prepare_media_root(out_parquet_path, media_dir)
    updates: list[pl.Series] = []
    for col in path_fields:
        if col not in df.columns:
            continue
        values = df[col].to_list()
        new_vals: list[str | None] = []
        for v in values:
            if v is None:
                new_vals.append(None)
                continue
            src = Path(str(v))
            dst = _resolve_media_destination(media_root, src)
            _transfer_media(src, dst, copy_mode)
            new_vals.append(_format_stored_path(dst, media_dir, path_style))
        updates.append(pl.Series(col, new_vals, dtype=df.schema[col]))
    if updates:
        df = df.with_columns(updates)
    media_meta: dict[str, Any] = {
        "root": media_dir,
        "path_fields": path_fields,
        "style": path_style.value,
        "copy_mode": copy_mode.value,
    }
    return df, media_meta


def rebase_media_paths(
    *,
    df: pl.DataFrame,
    path_fields: list[str],
    sidecar_media_meta: dict[str, Any],
    parquet_path: Path,
    rebase_media_root: str | None,
    make_paths_absolute: bool,
) -> pl.DataFrame:
    """
    Rebase and/or make absolute the image paths stored in the DataFrame.

    This uses the `media` metadata block from the sidecar JSON to determine how
    paths were stored (relative vs absolute) and what the media root was. It can
    optionally:
    - rebase relative paths under a new root (`rebase_media_root`), and/or
    - convert any paths to absolute OS paths (`make_paths_absolute`).

    Args:
        df: Input DataFrame whose columns include the path fields.
        path_fields: Column names that store image paths.
        sidecar_media_meta: The `media` section from the sidecar JSON.
        parquet_path: Path to the Parquet file adjacent to the media directory.
        rebase_media_root: If provided, a new root directory under which the
            stored relative paths will be resolved.
        make_paths_absolute: If True, resulting paths will be resolved to absolute.

    Returns:
        A new DataFrame with updated path values where applicable.
    """
    media_root_name = sidecar_media_meta.get("root")
    style = sidecar_media_meta.get("style", "relative")
    base_dir = Path(parquet_path).parent
    if rebase_media_root is not None:
        root_path = Path(rebase_media_root)
    else:
        root_path = base_dir / (media_root_name or "media")

    updates: list[pl.Series] = []
    for col in path_fields:
        if col not in df.columns:
            continue
        vals = df[col].to_list()
        new_vals: list[str | None] = []
        for v in vals:
            if v is None:
                new_vals.append(None)
                continue
            p = Path(str(v))
            if style == "relative" and not p.is_absolute():
                p = root_path / p
            if make_paths_absolute:
                p = p.resolve()
            new_vals.append(str(p))
        updates.append(pl.Series(col, new_vals, dtype=df.schema[col]))
    if updates:
        df = df.with_columns(updates)
    return df
