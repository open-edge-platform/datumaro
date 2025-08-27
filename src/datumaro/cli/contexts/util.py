# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.cli.util import MultilineFormatter, add_subparser
from datumaro.cli.util.errors import CliException
from datumaro.components.dataset import Dataset
from datumaro.util.os_util import generate_next_name


def build_split_video_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Split video into frames",
        description="""
        Splits a video into separate frames and saves them in a directory.
        |n
        This command is useful for making a dataset from a video file.
        Unlike direct video reading during model training, which can produce
        different results if the system environment changes, this command
        allows to split the video into frames and use them instead, making
        the dataset reproducible and stable.|n
        |n
        This command provides different options like setting the frame step,
        file name pattern, starting and finishing frame etc.|n
        |n
        Examples:|n
        - Split a video into frames, use each 30-rd frame:|n
        |s|s%(prog)s -i video.mp4 -o video.mp4-frames --step 30|n
        - Split a video into frames, save as 'frame_xxxxxx.png' files:|n
        |s|s%(prog)s -i video.mp4 --image-ext=.png --name-pattern='frame_%%06d'
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "-i", "--input-path", dest="src_path", required=True, help="Path to the video file"
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
    parser.add_argument(
        "-n",
        "--name-pattern",
        default="%06d",
        help="Name pattern for the produced images (default: %(default)s)",
    )
    parser.add_argument(
        "-s", "--step", type=int, default=1, help="Frame step (default: %(default)s)"
    )
    parser.add_argument(
        "-b", "--start-frame", type=int, default=0, help="Starting frame (default: %(default)s)"
    )
    parser.add_argument(
        "-e", "--end-frame", type=int, default=None, help="Finishing frame (default: %(default)s)"
    )
    parser.add_argument(
        "-x", "--image-ext", default=".jpg", help="Output image extension (default: %(default)s)"
    )
    parser.set_defaults(command=split_video_command)

    return parser


def get_split_video_sensitive_args():
    return {
        split_video_command: ["src_path", "dst_dir", "name_pattern"],
    }


def split_video_command(args):
    src_path = osp.abspath(args.src_path)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException(
                "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
            )
    else:
        basename = "%s-frames" % osp.basename(src_path)
        dst_dir = generate_next_name(os.listdir("."), basename)
    dst_dir = osp.abspath(dst_dir)

    log.info("Exporting frames...")

    dataset = Dataset.import_from(
        src_path,
        "video_frames",
        name_pattern=args.name_pattern,
        step=args.step,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    dataset.export(format="image_dir", save_dir=dst_dir, image_ext=args.image_ext)

    log.info("Frames are exported into '%s'" % dst_dir)

    return 0


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    subparsers = parser.add_subparsers()
    add_subparser(subparsers, "split_video", build_split_video_parser)

    return parser


def get_sensitive_args():
    return {
        **get_split_video_sensitive_args(),
    }
