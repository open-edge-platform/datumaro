# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from typing import Generator

import numpy as np
import pytest

from datumaro.components.crypter import Crypter
from datumaro.components.dataset import Dataset
from datumaro.components.media import Image
from datumaro.errors import DatasetImportError
from datumaro.plugins.data_formats.datumaro_binary.format import DatumaroBinaryPath

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestCaseHelper, TestDir, compare_datasets
from tests.utils.test_utils import run_datum as run

yolo_dir = get_test_asset_path("yolo_dataset")


def get_image(export_dir: str) -> Generator[Image, None, None]:
    for root, _, files in os.walk(export_dir):
        for file in files:
            fpath = osp.join(root, file)
            _, ext = osp.splitext(fpath)
            if ext == ".jpg":
                yield Image.from_file(path=fpath)


@pytest.fixture
def export_dir():
    with TestDir() as export_dir:
        yield export_dir


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("no_media_encryption", [True, False])
@pytest.mark.skip(
    reason="Round-trip conversion YOLO -> datumaro_binary -> YOLO loses image metadata needed for YOLO parsing"
)
def test_yolo_to_dm_binary_encryption(
    test_dir: str,
    export_dir: str,
    helper_tc: TestCaseHelper,
    no_media_encryption: bool,
    num_workers: int,
):
    """
    1. Import yolo format dataset
    2. Export it to DatumaroBinary format with encryption
    3. Check the encryption
    4. Succeed to import the encrypted dataset with the true key
    5. Re-export it to the yolo format
    6. Test whether it is the same as the original
    """
    yolo_dir = get_test_asset_path("yolo_dataset")

    # 1. Import and export directly without project
    # Export yolo dataset to DatumaroBinary format with encryption
    cmd = [
        "convert",
        "-if",
        "yolo",
        "-i",
        yolo_dir,
        "-f",
        "datumaro_binary",
        "-o",
        osp.join(export_dir, "dm_binary"),
        "--",
        "--save-media",
        "--encryption",
        "--num-workers",
        str(num_workers),
    ]
    if no_media_encryption:
        cmd += ["--no-media-encryption"]

    run(helper_tc, *cmd)

    # Check whether the key exists
    key_path = osp.join(export_dir, "dm_binary", DatumaroBinaryPath.SECRET_KEY_FILE)
    assert osp.exists(key_path)

    # 4-0. Get secret key
    with open(key_path, "r") as fp:
        true_key = fp.read().encode()
        wrong_key = Crypter.gen_key(true_key)

    # 4-1. Wrong key cannot import the encrypted dataset.
    with pytest.raises(DatasetImportError):
        Dataset.import_from(
            osp.join(export_dir, "dm_binary"), format="datumaro_binary", encryption_key=wrong_key
        )

    # 4-2-1. You cannot open the encrypted image.
    if not no_media_encryption:
        for img in get_image(export_dir):
            with pytest.raises(Exception):
                assert img.data is None
    # 4-2-2. You can open the encrypted image (--no-media-encryption).
    else:
        for img in get_image(export_dir):
            assert isinstance(img.data, np.ndarray)

    # 5. Convert the encrypted dataset back to yolo format with the true key
    run(
        helper_tc,
        "convert",
        "-if",
        "datumaro_binary",
        "-i",
        osp.join(export_dir, "dm_binary"),
        "-f",
        "yolo",
        "-o",
        osp.join(export_dir, "yolo"),
        "--encryption-key",
        true_key.decode(),
    )

    # 6. Test whether it is the same as the original
    expect = Dataset.import_from(yolo_dir, format="yolo")
    actual = Dataset.import_from(osp.join(export_dir, "yolo"), format="yolo")

    compare_datasets(helper_tc, expect, actual)
