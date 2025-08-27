# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import contextlib
import logging as log
import os
import os.path as osp
import sys
from typing import Dict, Tuple

from datumaro.components.dataset_base import IDataset
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.extractor_tfds import (
    AVAILABLE_TFDS_DATASETS,
    TFDS_EXTRACTOR_AVAILABLE,
    TfdsDatasetRemoteMetadata,
)
from datumaro.util import dump_json
from datumaro.util.os_util import make_file_name

from ...util.dataset_utils import generate_next_file_name
from ...util.errors import CliException
from .downloader import IDatasetDownloader


class TfdsDatasetDownloader(IDatasetDownloader):
    @classmethod
    def get_extractor(cls, dataset_id: str) -> Tuple[str, IDataset]:
        if dataset_id.startswith("tfds:"):
            if TFDS_EXTRACTOR_AVAILABLE:
                tfds_ds_name = dataset_id[5:]
                tfds_ds = AVAILABLE_TFDS_DATASETS.get(tfds_ds_name)
                if tfds_ds:
                    default_output_format = tfds_ds.metadata.default_output_format
                    extractor_factory = tfds_ds.make_extractor
                    return default_output_format, extractor_factory
                else:
                    raise CliException(f"Unsupported TFDS dataset '{tfds_ds_name}'")
            else:
                raise CliException(
                    "TFDS datasets are not available, because TFDS and/or"
                    "TensorFlow are not installed.\n"
                    "You can install them with: pip install datumaro[tf,tfds]"
                )
        else:
            raise CliException(
                f"Unknown dataset ID TFDS dataset '{dataset_id}': TFDS datasets must start with the prefix 'tfds:'"
            )

    @staticmethod
    def _describe_txt(dataset_metas: Dict[str, TfdsDatasetRemoteMetadata], report_file=None):
        with open(report_file, "w") if report_file else contextlib.nullcontext() as report_file:
            if dataset_metas:
                print("Available datasets:", file=report_file)
                for name, meta in sorted(dataset_metas.items()):
                    print(
                        f"""
{name} ({meta.human_name}):
  default output format: {meta.default_output_format}
  description:""",
                        file=report_file,
                    )
                    for line in meta.description.rstrip("\n").split("\n"):
                        print(f"    {line}", file=report_file)
                    print(
                        f"""  download size: {meta.download_size} bytes
  home URL: {meta.home_url or 'N/A'}
  number of classes: {meta.num_classes}
  subsets:""",
                        file=report_file,
                    )
                    for subset_name, subset_meta in sorted(meta.subsets.items()):
                        print(f"    {subset_name}: {subset_meta.num_items} items", file=report_file)
                    print(f"  version: {meta.version}" "", file=report_file)
            else:
                print(
                    """No datasets available.

"You can enable TFDS datasets by installing TensorFlow and TensorFlow Datasets:
    pip install datumaro[tf,tfds]""",
                    file=report_file,
                )

    @staticmethod
    def _describe_json(dataset_metas, report_file):
        def meta_to_raw(meta: TfdsDatasetRemoteMetadata):
            raw = {}

            # We omit the media type from the output, because there is currently no mechanism
            # for mapping media types to strings. The media type could be useful information
            # for users, though, so we might want to implement such a mechanism eventually.

            for attribute in (
                "default_output_format",
                "description",
                "download_size",
                "home_url",
                "human_name",
                "num_classes",
                "version",
            ):
                raw[attribute] = getattr(meta, attribute)

            raw["subsets"] = {
                name: {"num_items": subset.num_items} for name, subset in meta.subsets.items()
            }

            return raw

        with (
            open(report_file, "w") if report_file else contextlib.nullcontext(sys.stdout)
        ) as report_file:
            report_file.write(
                dump_json(
                    {name: meta_to_raw(meta) for name, meta in dataset_metas.items()},
                    indent=True,
                    append_newline=True,
                ).decode()
            )

    @classmethod
    def describe(cls, report_format, report_file=None):
        dataset_metas: Dict[str, TfdsDatasetRemoteMetadata] = {}

        if TFDS_EXTRACTOR_AVAILABLE:
            for dataset_name, dataset in AVAILABLE_TFDS_DATASETS.items():
                dataset_metas[f"tfds:{dataset_name}"] = dataset.query_remote_metadata()

        if report_format == "text":
            cls._describe_txt(dataset_metas, report_file)

        elif report_format == "json":
            cls._describe_json(dataset_metas, report_file)

    @classmethod
    def describe_command_description(_):
        return """More detailed
        information can be found in the TFDS Catalog:
        <https://www.tensorflow.org/datasets/catalog/overview>."""

    @classmethod
    def get_command_description(cls):
        builtin_writers = sorted(DEFAULT_ENVIRONMENT.exporters)
        if TFDS_EXTRACTOR_AVAILABLE:
            available_datasets = ", ".join(f"tfds:{name}" for name in AVAILABLE_TFDS_DATASETS)
        else:
            available_datasets = "N/A (TensorFlow and/or TensorFlow Datasets are not installed)"
            return f"""
Supported datasets: {available_datasets}|n
|n
Supported output formats: {", ".join(builtin_writers)}|n
|n
Examples:|n
- Download the MNIST dataset:|n
|s|s%(prog)s -i tfds:mnist -- --save-media|n
|n
- Download the VOC 2012 dataset, saving only the annotations in the COCO
format into a specific directory:|n
|s|s%(prog)s -i tfds:voc/2012 -f coco -o path/I/like/
"""

    @classmethod
    def download(cls, dataset_id, dst_dir, overwrite, output_format, subset, extra_args):
        env = DEFAULT_ENVIRONMENT
        default_output_format, extractor_factory = cls.get_extractor(dataset_id)
        output_format = output_format or default_output_format

        try:
            exporter = env.exporters[output_format]
        except KeyError:
            raise CliException(f"Exporter for format '{output_format}' is not found")
        extra_args = exporter.parse_cmdline(extra_args)

        if dst_dir:
            if not overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
                raise CliException(
                    f"Directory '{dst_dir}' already exists (pass --overwrite to overwrite)"
                )
        else:
            dst_dir = generate_next_file_name(
                f"{make_file_name(dataset_id)}-{make_file_name(output_format)}"
            )
        dst_dir = osp.abspath(dst_dir)

        log.info("Downloading the dataset")
        extractor = extractor_factory()

        if subset:
            try:
                extractor = extractor.subsets()[subset]
            except KeyError:
                raise CliException(f"Subset '{subset}' is not present in the dataset")

        log.info("Exporting the dataset")
        exporter.convert(extractor, dst_dir, default_image_ext=".png", **extra_args)

        log.info(f"Dataset exported to '{dst_dir}' as '{output_format}'")
