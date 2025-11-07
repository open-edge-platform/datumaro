# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp

import pytest

from datumaro.components.dataset_base import DatasetItem
from datumaro.components.exporter import ExportContextComponent
from datumaro.components.media import MediaElement, Video, VideoFrame
from tests.utils.test_utils import TestDir
from tests.utils.video import make_sample_video


@pytest.fixture()
def fxt_sample_video(test_dir):
    video_path = osp.join(test_dir, "video.avi")
    make_sample_video(video_path, frame_size=(4, 6), frames=4)
    yield video_path


@pytest.fixture
def fxt_export_context_component(test_dir):
    return ExportContextComponent(
        save_dir=test_dir,
        save_media=True,
        images_dir="images",
        pcd_dir="point_clouds",
        video_dir="videos",
    )


class ExportContextComponentTest:
    def test_make_video_filename(self, fxt_export_context_component, fxt_sample_video):
        video = Video(fxt_sample_video)
        frame = VideoFrame(video, index=1)

        ecc: ExportContextComponent = fxt_export_context_component

        for media in [video, frame]:
            assert ecc.make_video_filename(DatasetItem(0, media=media)) == "video.avi"

        # error cases
        for item in [None, DatasetItem(0, media=MediaElement())]:
            with pytest.raises(AssertionError):
                ecc.make_video_filename(item)

    def test_save_video(self, fxt_export_context_component, fxt_sample_video):
        video = Video(fxt_sample_video)
        frame = VideoFrame(video, index=1)
        ecc: ExportContextComponent = fxt_export_context_component

        with TestDir() as test_dir:
            ecc.save_video(DatasetItem(0, media=video), basedir=test_dir)
            expected_path = osp.join(test_dir, "video.avi")
            assert osp.exists(expected_path)

        with TestDir() as test_dir:
            ecc.save_video(DatasetItem(0, media=frame), basedir=test_dir)
            expected_path = osp.join(test_dir, "video.avi")
            assert osp.exists(expected_path)

        # cannot save items with no media
        with TestDir() as test_dir:
            ecc.save_video(DatasetItem(0), basedir=test_dir)
            files = os.listdir(test_dir)
            assert not files
