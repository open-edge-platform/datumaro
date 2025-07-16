# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp
from shutil import rmtree

from datumaro.cli.util.errors import CliException
from datumaro.util.definitions import get_datumaro_cache_dir

from ..util import MultilineFormatter

deprecated = "[DEPRECATED, will be removed in 1.12]"


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help=f"{deprecated} Generate synthetic dataset",
        description=f"""
        {deprecated} Creates a synthetic dataset with elements of the specified type and shape,
        and saves it in the provided directory.|n
        |n
        Currently, can only generate fractal images, useful for network compression.|n
        To create 3-channel images, you should provide the number of images, height and width.|n
        The images are colorized with a model, which will be downloaded automatically.|n
        Uses the algorithm from the article: https://arxiv.org/abs/2103.13023 |n
        |n
        Examples:|n
        - Generate 300 3-channel images with H=224, W=256 and store to data_dir:|n
        |s|s%(prog)s -o data_dir -k 300 --shape 224 256
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "-o", "--output-dir", required=True, help="Output directory to store generated dataset"
    )
    parser.add_argument(
        "-k", "--count", type=int, required=True, help="Number of images to be generated"
    )
    parser.add_argument(
        "--shape",
        nargs=2,
        metavar="DIM",
        type=int,
        required=True,
        help="Dimensions of data to be generated (height, width)",
    )
    parser.add_argument(
        "-t",
        "--type",
        default="image",
        choices=["image"],
        help="Specify type of data to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=get_datumaro_cache_dir(),
        help="Path to load the colorization model from. "
        "If no model is found, the model will be downloaded (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )

    parser.set_defaults(command=generate_command)

    return parser


def get_sensitive_args():
    return {generate_command: ["output_dir", "model_dir"]}


def generate_command(args):
    log.warning("This command is deprecated and will be removed in Datumaro 1.12")
    from datumaro.plugins.synthetic_data import FractalImageGenerator

    log.info("Generating dataset...")
    output_dir = args.output_dir

    if osp.isdir(output_dir) and os.listdir(output_dir):
        if args.overwrite:
            rmtree(output_dir)
            os.mkdir(output_dir)
        else:
            raise CliException(
                f"Directory '{output_dir}' already exists (pass --overwrite to overwrite)"
            )

    if args.type == "image":
        FractalImageGenerator(
            count=args.count, output_dir=output_dir, shape=args.shape, model_path=args.model_dir
        ).generate_dataset()
    else:
        raise NotImplementedError(f"Data type: {args.type} is not supported")

    log.info(f"Results have been saved to '{output_dir}'")

    return 0
