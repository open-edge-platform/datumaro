# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from typing import Tuple

DEFAULT_SUBSET_NAME = "default"
BboxIntCoords = Tuple[int, int, int, int]  # (x, y, w, h)
SUBSET_NAME_BLACKLIST = {"labels", "images", "annotations", "instances"}
SUBSET_NAME_WHITELIST = {"train", "test", "val"}


def get_datumaro_cache_dir(
    _CACHE_DIR: str = osp.expanduser(os.getenv("XDG_CACHE_HOME", osp.join("~", ".cache"))),
) -> str:
    """Get DATUMARO_CACHE_DIR. If it does not exists, create it."""
    DATUMARO_CACHE_DIR = osp.join(_CACHE_DIR, "datumaro")

    try:
        if not osp.exists(DATUMARO_CACHE_DIR):
            os.makedirs(DATUMARO_CACHE_DIR)
    except Exception as e:
        log.error(f"Cannot create DATUMARO_CACHE_DIR={DATUMARO_CACHE_DIR} since {e}.")

    return DATUMARO_CACHE_DIR
