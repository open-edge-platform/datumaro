# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from unittest.case import TestCase
from unittest.mock import PropertyMock, patch

import numpy as np

from datumaro.util.scope import scope_add, scoped
from tests.utils.test_utils import TestDir
from tests.utils.test_utils import run_datum as run
from tests.utils.video import make_sample_video


class VideoSplittingTest:
    @scoped
    @patch("datumaro.components.media.VideoFrame.data", new_callable=PropertyMock)
    def test_can_split_video(self, mock_video_frame_data):
        mock_video_frame_data.return_value = np.full((32, 32, 3), fill_value=0, dtype=np.uint8)

        test_dir = scope_add(TestDir())
        video_path = osp.join(test_dir, "video.avi")
        make_sample_video(video_path, frames=10)

        output_dir = osp.join(test_dir, "result")

        run(
            TestCase(),
            "util",
            "split_video",
            "-i",
            video_path,
            "-o",
            output_dir,
            "--image-ext",
            ".jpg",
            "--start-frame",
            "2",
            "--end-frame",
            "8",
            "--step",
            "2",
        )

        assert set(os.listdir(output_dir)) == {"%06d.jpg" % n for n in range(2, 9, 2)}
