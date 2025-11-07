# Copyright (C) 2021-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from typing import List, Optional, Tuple

import cv2
import numpy as np

from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetBase, DatasetItem
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Video, VideoFrame
from datumaro.util.os_util import find_files

# Taken from https://en.wikipedia.org/wiki/Comparison_of_video_container_formats
# An extension does not define file contents, but it can be a good file filter
VIDEO_EXTENSIONS = [
    "3gp",
    "3g2",
    "asf",
    "wmv",
    "avi",
    "divx",
    "evo",
    "f4v",
    "flv",
    "mkv",
    "mk3d",
    "mp4",
    "mpg",
    "mpeg",
    "m2p",
    "ps",
    "ts",
    "m2ts",
    "mxf",
    "ogg",
    "ogv",
    "ogx",
    "mov",
    "qt",
    "rmvb",
    "vob",
    "webm",
]


class VideoFramesImporter(Importer):
    """
    Reads video frames as a dataset.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--subset",
            help="The name of the subset for the produced dataset items (default: none)",
        )
        parser.add_argument(
            "-p",
            "--name-pattern",
            default="%06d",
            help="The name pattern for the produced dataset items (default: %(default)s).",
        )
        parser.add_argument("-s", "--step", type=int, default=1, help="Frame step (default: %(default)s)")
        parser.add_argument("-b", "--start-frame", type=int, default=0, help="Starting frame (default: %(default)s)")
        parser.add_argument(
            "-e",
            "--end-frame",
            type=int,
            default=None,
            help="Finishing frame (default: %(default)s)",
        )
        return parser

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        try:
            next(find_files(context.root_path, VIDEO_EXTENSIONS, recursive=True))
            return FormatDetectionConfidence.LOW
        except StopIteration:
            context.fail(
                "No video files found in '%s'. "
                "Checked extensions: %s" % (context.root_path, ", ".join(VIDEO_EXTENSIONS))
            )

    @classmethod
    def find_sources(cls, path):
        if not osp.isfile(path):
            return []
        return [{"url": path, "format": VideoFramesBase.NAME}]

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return list({f".{ext}" for ext in VIDEO_EXTENSIONS})


class VideoFramesBase(DatasetBase):
    def __init__(
        self,
        url: str,
        *,
        name_pattern: str = "%06d",
        step: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ) -> None:
        self._subset = subset or DEFAULT_SUBSET_NAME
        super().__init__(subsets=[self._subset], media_type=VideoFrame, ctx=ctx)

        assert osp.isfile(url), url

        self._name_pattern = name_pattern
        self._reader = Video(url, step=step, start_frame=start_frame, end_frame=end_frame)
        self._length = self._reader.length  # NOTE: the value is often incorrect
        self._ann_types = set()

    def __iter__(self):
        for frame in self._reader:
            yield DatasetItem(id=self._name_pattern % (frame.index,), subset=self._subset, media=frame)

    def get(self, id, subset=None):
        assert subset == self._subset, "%s != %s" % (subset, self._subset)
        return super().get(id, subset or self._subset)


class VideoKeyframesImporter(VideoFramesImporter):
    """
    Reads video frames as a dataset.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-t",
            "--threshold",
            type=float,
            default=0.3,
            help="Similarity threshold (default: %(default)s)",
        )
        parser.add_argument(
            "-r",
            "--resize",
            type=float,
            default=(64, 64),
            help="Image size for comuting ZNCC score (default: %(default)s)",
        )
        return parser

    @classmethod
    def find_sources(cls, path):
        if not osp.isfile(path):
            return []
        return [{"url": path, "format": VideoKeyframesBase.NAME}]


class VideoKeyframesBase(VideoFramesBase):
    def __init__(
        self,
        url: str,
        *,
        subset: Optional[str] = None,
        name_pattern: str = "%06d",
        step: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        resize: Tuple[int, int] = (64, 64),
        threshold: float = 0.3,
    ) -> None:
        super().__init__(
            url=url,
            subset=subset,
            step=step,
            start_frame=start_frame,
            end_frame=end_frame,
            name_pattern=name_pattern,
        )

        self._resize = resize
        self._threshold = threshold
        self._keyframe = None

    def _is_keyframe(self, frame):
        if self._keyframe is None:
            self._keyframe = cv2.resize(cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2GRAY), self._resize).astype(
                "float64"
            )
            self._keyframe -= np.mean(self._keyframe)
            return True

        _curr_frame = cv2.resize(cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2GRAY), self._resize).astype("float64")
        _curr_frame -= np.mean(_curr_frame)

        zncc_score = np.sum(self._keyframe * _curr_frame)

        # added the epsilon 1e-6 for numerical stability during division operation
        zncc_norm = np.sqrt(np.sum(self._keyframe**2)) * np.sqrt(np.sum(_curr_frame**2)) + 1e-6
        zncc_score /= zncc_norm

        if zncc_score < self._threshold:
            self._keyframe = _curr_frame
            return True

        return False

    def __iter__(self):
        for frame in self._reader:
            frame_data = frame.video.get_frame_data(frame.index)
            if self._is_keyframe(frame_data):
                yield DatasetItem(id=self._name_pattern % (frame.index,), subset=self._subset, media=frame)
