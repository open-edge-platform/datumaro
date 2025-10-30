# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

import io
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image as PILImage

from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import image_path_field, label_field
from datumaro.experimental.io import PathStyle


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


def test_save_with_media_zip(tmp_path: Path):
    # Prepare sample with two images
    img1 = tmp_path / "inputs" / "img1.png"
    img2 = tmp_path / "inputs" / "img2.png"
    _make_temp_image(img1)
    _make_temp_image(img2)

    ds = Dataset(PathSample)
    ds.append(PathSample(image_path=str(img1), label=0))
    ds.append(PathSample(image_path=str(img2), label=1))

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "dataset.parquet"

    # Save with media and archive
    ds.save_parquet(
        str(parquet_path),
        save_media=True,
        media_dir="media",
        path_style=PathStyle.RELATIVE,
        archive=True,
    )

    # Zip should exist next to parquet
    zip_path = parquet_path.with_suffix(".zip")
    assert zip_path.exists()

    # Check contents of the zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "dataset.parquet" in names
        assert "dataset.json" in names
        # Media files present
        media_names = [n for n in names if n.startswith("media/") and not n.endswith("/")]
        assert len(media_names) == 2

    # Parquet should contain relative paths starting with media/
    df = pl.read_parquet(parquet_path)
    vals = df["image_path"].to_list()
    assert all(isinstance(v, str) and v.startswith("media/") for v in vals)


def test_load_from_zip_rebase_absolute(tmp_path: Path):
    # Create a dataset and save to zip with media
    img = tmp_path / "inputs" / "im.png"
    _make_temp_image(img)

    ds = Dataset(PathSample)
    ds.append(PathSample(image_path=str(img), label=3))

    parquet_path = tmp_path / "pkg" / "data.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    ds.save_parquet(str(parquet_path), save_media=True, path_style=PathStyle.RELATIVE, archive=True)

    zip_path = parquet_path.with_suffix(".zip")
    assert zip_path.exists()

    # Load directly from zip, result should have absolute paths to extracted media
    loaded = Dataset.from_parquet(
        str(zip_path),
        PathSample,
        make_paths_absolute=True,
    )

    p = Path(loaded.df["image_path"][0])
    assert p.is_absolute()
    assert p.exists()


def test_zip_without_media(tmp_path: Path):
    # Save without media but archive enabled
    ds = Dataset(PathSample)
    ds.append(PathSample(image_path="/tmp/nonexistent.png", label=1))

    parquet_path = tmp_path / "plain" / "tab.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    ds.save_parquet(str(parquet_path), save_media=False, archive=True)

    zip_path = parquet_path.with_suffix(".zip")
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "tab.parquet" in names
        assert "tab.json" in names
        # No media directory expected because we didn't save_media
        assert not any(n.startswith("media/") for n in names)

    # Loading from this zip should still work
    loaded = Dataset.from_parquet(str(zip_path), PathSample)
    assert loaded.df.height == 1
