# Copyright (C) 2021-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from typing import Dict

from ..util import MultilineFormatter
from .downloaders import IDatasetDownloader, KaggleDatasetDownloader, TfdsDatasetDownloader


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Download a publicly available dataset",
        description=f"""
        Downloads a publicly available dataset and saves it in a given format.|n
        |n
        To download the dataset, run "datum download <dataset_type> get". On the other hand,
        for information about the datasets, run "datum download <dataset_type> describe".
        Supported dataset types are: {list(DOWNLOADERS.keys())}
        """,
        formatter_class=MultilineFormatter,
    )
    subparsers = parser.add_subparsers(title="Dataset types")
    for name, downloader in DOWNLOADERS.items():
        dataset_type_parser = subparsers.add_parser(
            name=name,
            help=f"Download {name} dataset",
            formatter_class=MultilineFormatter,
        )
        _subparsers = dataset_type_parser.add_subparsers(title="Commands")
        build_get_subparser(_subparsers, name, downloader)
        build_describe_subparser(_subparsers, name, downloader)


def build_get_subparser(
    subparsers: argparse._SubParsersAction, name: str, downloader: IDatasetDownloader
):
    parser = subparsers.add_parser(
        name="get",
        help="Download a publicly available dataset",
        description=downloader.get_command_description(),
        formatter_class=MultilineFormatter,
    )
    parser.add_argument("-i", "--dataset-id", required=True, help="Which dataset to download")
    parser.add_argument(
        "-f", "--output-format", help="Output format (default: original format of the dataset)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        help="Directory to save output (default: a subdir in the current one)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.add_argument("-s", "--subset", help="Save only the specified subset")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for output format (pass '-- -h' for help). "
        "Must be specified after the main command arguments",
    )

    parser.set_defaults(command=download_command, downloader=downloader)

    return parser


def build_describe_subparser(
    subparsers: argparse._SubParsersAction, name: str, downloader: IDatasetDownloader
):
    parser = subparsers.add_parser(
        name="describe",
        help="Print information about downloadable datasets",
        description=f"""
        Reports information about datasets that can be downloaded with the
        "datum download {name}" command. The information is reported either as
        human-readable text (the default) or as a JSON object."""
        + downloader.describe_command_description(),
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "--report-format",
        choices=("text", "json"),
        default="text",
        help="Format in which to report the information (default: %(default)s)",
    )
    parser.add_argument(
        "--report-file", help="File to which to write the report (default: standard output)"
    )
    parser.set_defaults(command=describe_downloads_command, downloader=downloader)

    return parser


DOWNLOADERS: Dict[str, IDatasetDownloader] = {
    "tfds": TfdsDatasetDownloader,
    "kaggle": KaggleDatasetDownloader,
}


def download_command(args):
    args.downloader.download(
        args.dataset_id,
        args.dst_dir,
        args.overwrite,
        args.output_format,
        args.subset,
        args.extra_args,
    )


def describe_downloads_command(args):
    return args.downloader.describe(args.report_format, args.report_file)
