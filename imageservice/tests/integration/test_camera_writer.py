"""
tests/integration/test_camera_writer.py
Integration tests: camera → writer pipeline
"""

import time
import json
import os
import pytest
import numpy as np

from workers.camera import XimeaCamera
from workers.writer import FrameWriter

pytestmark = pytest.mark.integration


class TestCameraWriterPipeline:

    def test_frames_from_camera_written_to_disk(
        self, single_roi_cfg, fake_xiapi, star_roi, tmp_data_dir
    ):
        n = 10
        with FrameWriter(
            tmp_data_dir, "integ_001", "roi",
            star_roi.shape, buffer_n_frames=32,
        ) as writer:
            with XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi) as cam:
                cam._img.set_data(star_roi)
                for _ in range(n):
                    frame = cam.acquire_frame()
                    writer.push(next(iter(frame.rois.values())), frame)
            time.sleep(0.3)

        assert writer.stats.frames_written == n
        assert writer.stats.frames_dropped == 0

    def test_metadata_frame_ids_sequential(
        self, single_roi_cfg, fake_xiapi, star_roi, tmp_data_dir
    ):
        n = 5
        with FrameWriter(
            tmp_data_dir, "integ_002", "roi",
            star_roi.shape, buffer_n_frames=32,
        ) as writer:
            with XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi) as cam:
                cam._img.set_data(star_roi)
                for _ in range(n):
                    frame = cam.acquire_frame()
                    writer.push(next(iter(frame.rois.values())), frame)
            time.sleep(0.3)

        meta_path = os.path.join(
            tmp_data_dir, "integ_002_meta.jsonl"
        )
        lines = open(meta_path).readlines()
        frame_ids = [json.loads(l)["frame_id"] for l in lines]
        assert frame_ids == sorted(frame_ids)
        assert len(set(frame_ids)) == n   # all unique

    def test_centroid_from_acquired_frame(
        self, single_roi_cfg, fake_xiapi, star_roi
    ):
        from workers.centroid import compute_pointing_error
        from tests.conftest import SENSOR_WIDTH as SENSOR_W, SENSOR_HEIGHT as SENSOR_H, ROI_OFFSET_X, ROI_OFFSET_Y

        with XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi) as cam:
            cam._img.set_data(star_roi)
            frame = cam.acquire_frame()

        roi = next(iter(frame.rois.values()))
        pe = compute_pointing_error(
            roi, ROI_OFFSET_X, ROI_OFFSET_Y,
            SENSOR_W, SENSOR_H,
            timestamp_ns=frame.timestamp_ns,
            frame_id=frame.frame_id,
        )
        assert pe is not None
        # Star is near centre of ROI → pointing error near zero
        assert abs(pe.dx) < 5.0
        assert abs(pe.dy) < 5.0


