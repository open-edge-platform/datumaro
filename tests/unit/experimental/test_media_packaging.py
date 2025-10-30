# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

import io
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image as PILImage

from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import image_path_field, label_field
from datumaro.experimental.io import MediaCopyMode, PathStyle


def _make_temp_image(path: Path, size=(8, 6), color=(10, 20, 30)) -> bytes:
    img = PILImage.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


class PathSample(Sample):
    image_path: str | None = image_path_field()
    label: int = label_field()


def test_save_with_media_relative(tmp_path: Path):
    # Prepare two distinct images
    img1 = tmp_path / "inputs" / "a" / "img.png"
    img2 = tmp_path / "inputs" / "b" / "img2.png"
    _make_temp_image(img1)
    _make_temp_image(img2)

    ds = Dataset(PathSample)
    ds.append(PathSample(image_path=str(img1), label=0))
    ds.append(PathSample(image_path=str(img2), label=1))

    out_parquet = tmp_path / "out" / "data.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    ds.save_parquet(
        str(out_parquet), save_media=True, media_dir="media", path_style=PathStyle.RELATIVE
    )

    # Media directory exists with two files
    media_dir = out_parquet.parent / "media"
    assert media_dir.exists() and media_dir.is_dir()
    media_files = sorted(p.name for p in media_dir.iterdir())
    assert len(media_files) == 2

    # Sidecar has media block
    sidecar = out_parquet.with_suffix(".json").read_text()
    import json

    meta = json.loads(sidecar)
    assert "media" in meta
    assert meta["media"]["root"] == "media"
    assert meta["media"]["style"] == "relative"
    assert meta["media"]["copy_mode"] in {"copy", "symlink", "move"}
    assert "image_path" in meta["media"]["path_fields"]

    # Parquet contains relative paths
    df = pl.read_parquet(out_parquet)
    vals = df["image_path"].to_list()
    assert all(isinstance(v, str) and not Path(v).is_absolute() for v in vals)
    assert all(v.startswith("media/") for v in vals)


def test_from_parquet_rebase_and_absolute(tmp_path: Path):
    # Create an image and save with media packed relative
    img = tmp_path / "inputs" / "img.png"
    _make_temp_image(img)

    ds = Dataset(PathSample)
    ds.append(PathSample(image_path=str(img), label=1))

    out_dir = tmp_path / "pkg"
    out_dir.mkdir()
    out_parquet = out_dir / "data.parquet"

    ds.save_parquet(
        str(out_parquet), save_media=True, media_dir="media", path_style=PathStyle.RELATIVE
    )

    # Move/copy media to a new root and rebase during load
    new_root = tmp_path / "relocated_media"
    (new_root / "media").mkdir(parents=True, exist_ok=True)
    # Copy files from original media under the same relative subdir name used in paths
    for p in (out_dir / "media").iterdir():
        (new_root / "media" / p.name).write_bytes(p.read_bytes())

    loaded = Dataset.from_parquet(
        str(out_parquet),
        PathSample,
        rebase_media_root=str(new_root),
        make_paths_absolute=True,
    )

    # Check that the loaded path is absolute and points to a file in the new_root
    v = loaded.df["image_path"][0]
    assert Path(v).is_absolute()
    # Because saved paths are relative to the declared media root (e.g., 'media/..'),
    # rebasing will keep that subdir under the provided root.
    assert Path(v).parent == new_root / "media"
    assert Path(v).exists()


def test_filename_collision_handling(tmp_path: Path):
    # Two different source files with the same basename
    src1 = tmp_path / "src1" / "same.png"
    src2 = tmp_path / "src2" / "same.png"
    _make_temp_image(src1, color=(255, 0, 0))
    _make_temp_image(src2, color=(0, 255, 0))

    ds = Dataset(PathSample)
    ds.append(PathSample(image_path=str(src1), label=0))
    ds.append(PathSample(image_path=str(src2), label=1))

    out_parquet = tmp_path / "out" / "data.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    ds.save_parquet(
        str(out_parquet), save_media=True, media_dir="media", path_style=PathStyle.RELATIVE
    )

    media_dir = out_parquet.parent / "media"
    files = sorted(p.name for p in media_dir.iterdir())
    # Should have two files with distinct names (one likely suffixed)
    assert len(files) == 2
    assert len(set(files)) == 2

    df = pl.read_parquet(out_parquet)
    vals = df["image_path"].to_list()
    assert len(set(vals)) == 2  # different relative targets


def test_none_path_preserved(tmp_path: Path):
    img = tmp_path / "i.png"
    _make_temp_image(img)

    ds = Dataset(PathSample)
    ds.append(PathSample(image_path=str(img), label=0))
    ds.append(PathSample(image_path=None, label=1))

    out_parquet = tmp_path / "o.parquet"
    ds.save_parquet(str(out_parquet), save_media=True)

    df = pl.read_parquet(out_parquet)
    assert df["image_path"][1] is None

    loaded = Dataset.from_parquet(str(out_parquet), PathSample, make_paths_absolute=True)
    assert loaded.df["image_path"][1] is None


def test_copy_mode_move(tmp_path: Path):
    src = tmp_path / "src" / "m.png"
    _make_temp_image(src)

    ds = Dataset(PathSample)
    ds.append(PathSample(image_path=str(src), label=0))

    out_parquet = tmp_path / "out" / "d.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    ds.save_parquet(str(out_parquet), save_media=True, copy_mode=MediaCopyMode.MOVE)

    # Source should have been moved (no longer exists)
    assert not src.exists()

    # Media destination should contain the file
    media_dir = out_parquet.parent / "media"
    files = list(media_dir.iterdir())
    assert len(files) == 1 and files[0].exists()
