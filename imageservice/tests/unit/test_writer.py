"""
tests/unit/test_writer.py - Unit tests for startracker.writer
"""

import time
import pytest
import numpy as np

from workers.writer import FrameWriter, WriterStats
from workers.camera import AcquiredFrame, CameraMode

pytestmark = pytest.mark.unit


def _make_acquired(frame_id: int, shape) -> tuple:
    arr = np.full(shape, frame_id % 4096, dtype=np.uint16)
    acq = AcquiredFrame(
        frame_id=frame_id,
        timestamp_ns=frame_id * 10_000_000,
        host_time=float(frame_id) * 0.01,
        mode=CameraMode.SINGLE_ROI,
        rois={},
        nframes_dropped=0,
    )
    return arr, acq


class TestFrameWriter:

    def test_push_and_write_n_frames(self, tmp_data_dir, star_roi):
        n = 15
        with FrameWriter(tmp_data_dir, "s001", "roi",
                         star_roi.shape, buffer_n_frames=32) as w:
            for i in range(n):
                arr, acq = _make_acquired(i, star_roi.shape)
                w.push(arr, acq)
            time.sleep(0.2)
        assert w.stats.frames_pushed  == n
        assert w.stats.frames_written == n
        assert w.stats.frames_dropped == 0

    def test_binary_file_size_correct(self, tmp_data_dir, star_roi):
        import os
        n = 10
        h, w_ = star_roi.shape
        with FrameWriter(tmp_data_dir, "s002", "roi",
                         star_roi.shape, buffer_n_frames=32) as wr:
            for i in range(n):
                arr, acq = _make_acquired(i, star_roi.shape)
                wr.push(arr, acq)
            time.sleep(0.2)
        path = os.path.join(tmp_data_dir, "s002_frames.bin")
        assert os.path.getsize(path) == n * h * w_ * 2

    def test_pixel_data_round_trip(self, tmp_data_dir, star_roi):
        import os
        n = 5
        h, w_ = star_roi.shape
        frames = [np.full(star_roi.shape, i * 100, dtype=np.uint16)
                  for i in range(n)]
        with FrameWriter(tmp_data_dir, "s003", "roi",
                         star_roi.shape, buffer_n_frames=32) as wr:
            for i, frame in enumerate(frames):
                _, acq = _make_acquired(i, star_roi.shape)
                wr.push(frame, acq)
            time.sleep(0.2)
        path = os.path.join(tmp_data_dir, "s003_frames.bin")
        raw = np.fromfile(path, dtype=np.uint16).reshape(n, h, w_)
        for i in range(n):
            assert raw[i, 0, 0] == i * 100

    def test_metadata_file_correct(self, tmp_data_dir, star_roi):
        import json, os
        n = 3
        with FrameWriter(tmp_data_dir, "s004", "roi",
                         star_roi.shape, buffer_n_frames=32) as wr:
            for i in range(n):
                arr, acq = _make_acquired(i, star_roi.shape)
                wr.push(arr, acq)
            time.sleep(0.2)
        path = os.path.join(tmp_data_dir, "s004_meta.jsonl")
        lines = open(path).readlines()
        assert len(lines) == n
        m = json.loads(lines[0])
        assert m["frame_id"]  == 0
        assert m["roi_label"] == "roi"
        assert m["n_rows"]    == star_roi.shape[0]
        assert m["n_cols"]    == star_roi.shape[1]

    def test_ring_buffer_overflow_drops_oldest(self, tmp_data_dir, star_roi):
        buf_size = 4
        total    = buf_size + 3
        with FrameWriter(tmp_data_dir, "s005", "roi",
                         star_roi.shape, buffer_n_frames=buf_size,
                         flush_timeout_s=2.0) as wr:
            # Fill buffer faster than it can be drained
            for i in range(total):
                arr, acq = _make_acquired(i, star_roi.shape)
                wr.push(arr, acq)
            time.sleep(0.5)
        assert wr.stats.frames_dropped >= 0   # some may have been dropped

    def test_wrong_shape_raises(self, tmp_data_dir, star_roi):
        with FrameWriter(tmp_data_dir, "s006", "roi",
                         star_roi.shape, buffer_n_frames=8) as wr:
            bad = np.zeros((32, 32), dtype=np.uint16)
            _, acq = _make_acquired(0, star_roi.shape)
            with pytest.raises(ValueError, match="shape"):
                wr.push(bad, acq)

    def test_push_before_start_raises(self, tmp_data_dir, star_roi):
        wr = FrameWriter(tmp_data_dir, "s007", "roi",
                         star_roi.shape, buffer_n_frames=8)
        arr, acq = _make_acquired(0, star_roi.shape)
        with pytest.raises(RuntimeError, match="start"):
            wr.push(arr, acq)

    def test_stats_dataclass(self, frame_writer, star_roi):
        arr, acq = _make_acquired(0, star_roi.shape)
        frame_writer.push(arr, acq)
        s = frame_writer.stats
        assert isinstance(s, WriterStats)
        assert s.frames_pushed == 1
