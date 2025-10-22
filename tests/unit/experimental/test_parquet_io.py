# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: MIT

import io
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image as PILImage

from datumaro.experimental.categories import GroupType, LabelCategories
from datumaro.experimental.dataset import Dataset, Sample
from datumaro.experimental.fields import (
    ImageBytesField,
    ImagePathField,
    image_field,
    image_path_field,
    label_field,
)


def _make_temp_image(path: Path, size=(8, 6), color=(10, 20, 30)) -> bytes:
    img = PILImage.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    path.write_bytes(data)
    return data


class PathSample(Sample):
    image_path: str = image_path_field()
    label: int = label_field()


class ImageSample(Sample):
    image: np.ndarray = image_field(dtype=pl.UInt8)
    label: int = label_field()


def test_save_and_load_parquet(tmp_path: Path):
    img_path1 = tmp_path / "img1.png"
    _make_temp_image(path=img_path1)
    ds = Dataset(PathSample)
    ds.append(PathSample(image_path=str(img_path1), label=1))

    # Save to parquet
    out_file = tmp_path / "data.parquet"
    ds.save_parquet(str(out_file))

    # Load from parquet
    ds2 = Dataset.from_parquet(str(out_file), PathSample)
    assert ds[0].image_path == ds2[0].image_path
    assert ds[0].label == ds2[0].label

    out_file2 = tmp_path / "bytes.parquet"
    bytes_dataset = ds2.convert_to_schema(ImageSample)
    bytes_dataset.save_parquet(str(out_file2))


def test_save_and_load_np_image(tmp_path: Path):
    img_bytes1 = _make_temp_image(path=tmp_path / "img1.png")

    ds = Dataset(ImageSample)
    ds.append(ImageSample(image=np.frombuffer(img_bytes1, dtype=np.uint8), label=0))

    # Save to parquet
    out_file = tmp_path / "data.parquet"
    ds.save_parquet(str(out_file))

    # Load from parquet
    ds2 = Dataset.from_parquet(str(out_file), ImageSample)
    s0 = ds[0]
    s1 = ds2[0]

    assert s0.label == s1.label
    assert isinstance(s0.image, np.ndarray)
    assert isinstance(s1.image, np.ndarray)
    assert s1.image.tobytes() == img_bytes1


def test_categories_embedded_in_parquet(tmp_path: Path):
    # Define label categories for the label field
    categories = LabelCategories(labels=("cat", "dog"), group_type=GroupType.EXCLUSIVE)

    ds = Dataset(PathSample, categories={"label": categories})
    # minimal data to save
    img_path = tmp_path / "img.png"
    _make_temp_image(img_path)
    ds.append(PathSample(image_path=str(img_path), label=0))

    out_file = tmp_path / "categories.parquet"
    ds.save_parquet(str(out_file))

    # Load without passing categories; they should be restored from embedded metadata
    ds2 = Dataset.from_parquet(str(out_file), PathSample)
    restored = ds2.schema.attributes["label"].categories
    assert restored == categories
