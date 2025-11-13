# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import re
from typing import List, Optional

from datumaro.components.importer import ImportContext, Importer
from datumaro.components.lazy_plugin import extra_deps
from datumaro.plugins.data_formats.tf_detection_api.base import TfDetectionApiBase
from datumaro.plugins.data_formats.tf_detection_api.format import TfrecordImporterType
from datumaro.util.tf_util import has_feature
from datumaro.util.tf_util import import_tf as _import_tf

tf = _import_tf()


@extra_deps("tensorflow")
class RoboflowTfrecordImporter(Importer):
    _ANNO_EXT = ".tfrecord"

    @classmethod
    def find_sources(cls, path):
        sources = cls._find_sources_recursive(
            path=path,
            ext=cls._ANNO_EXT,
            extractor_name="roboflow_tfrecord",
        )
        if len(sources) == 0:
            return []

        undesired_feature = {
            "image/source_id": tf.io.FixedLenFeature([], tf.string),
        }

        subsets = {}
        for source in sources:
            if has_feature(path=source["url"], feature=undesired_feature):
                continue
            subset_name = os.path.dirname(source["url"]).split(os.sep)[-1]
            subsets[subset_name] = source["url"]

        return [
            {
                "url": url,
                "format": "roboflow_tfrecord",
                "options": {
                    "subset": subset,
                },
            }
            for subset, url in subsets.items()
        ]

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]


class RoboflowTfrecordBase(TfDetectionApiBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(
            path=path,
            subset=subset,
            tfrecord_importer_type=TfrecordImporterType.roboflow,
            ctx=ctx,
        )

    @staticmethod
    def _parse_labelmap(text):
        entry_pattern = r'name:\s*"([^"]+)"\s*,\s*id:\s*(\d+)'
        entry_pattern = re.compile(entry_pattern)

        matches = re.findall(entry_pattern, text)

        return {name: int(id) for name, id in matches}
