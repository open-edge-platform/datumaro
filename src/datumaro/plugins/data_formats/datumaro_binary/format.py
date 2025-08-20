# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.errors import DatasetImportError

_SIGNATURE = "signature:datumaro_binary"


class DatumaroBinaryPath:
    IMAGES_DIR = "images"
    ANNOTATIONS_DIR = "annotations"
    PCD_DIR = "point_clouds"
    VIDEO_DIR = "videos"
    MASKS_DIR = "masks"

    ANNOTATION_EXT = ".datum"
    IMAGE_EXT = ".jpg"
    MASK_EXT = ".png"
    SIGNATURE = _SIGNATURE
    SIGNATURE_LEN = len(_SIGNATURE)

    MAX_BLOB_SIZE = 2**20  # 1 Mega bytes
    MP_TIMEOUT = 300.0  # 5 minutes

    @classmethod
    def check_signature(cls, signature: str):
        if signature != cls.SIGNATURE:
            raise DatasetImportError(
                f"Input signature={signature} is not aligned with the ground truth signature={cls.SIGNATURE}"
            )
