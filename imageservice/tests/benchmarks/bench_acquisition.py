"""
tests/benchmarks/bench_acquisition.py
Benchmarks for the acquisition loop hot path components.

These measure the per-frame overhead of parse_roi_strip, frame copy,
and writer.push - the operations that must complete within the frame budget.
"""

import numpy as np
import pytest

from workers.camera import RoiDefinition, _parse_roi_strip

pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="module")
def synthetic_strip_9roi():
    """
    Synthetic multi-ROI strip: 9 regions of alternating heights 64/4 pixels,
    width 4504. Mirrors the expected multi-ROI strip geometry.
    """
    heights = [64, 4, 64, 4, 64, 4, 64, 4, 64]
    total_h = sum(heights)
    strip   = np.random.randint(0, 4095, (total_h, 4504), dtype=np.uint16)

    rois = []
    y = 0
    labels = ["top_left", "tm", "top_right", "ml", "centre",
              "mr", "bot_left", "bm", "bot_right"]
    is_data = [True, False, True, False, True, False, True, False, True]
    for i, (h, lbl, data) in enumerate(zip(heights, labels, is_data)):
        rois.append(RoiDefinition(h, y, 4504, 0, lbl, data))
        y += h
    return strip, rois


class TestAcquisitionBenchmarks:

    def test_bench_parse_roi_strip(self, benchmark, synthetic_strip_9roi):
        """
        Benchmark: _parse_roi_strip on a full multi-ROI strip.
        Target: < 200 µs (strip copy dominates; parse itself is O(n_regions)).
        """
        strip, rois = synthetic_strip_9roi
        result = benchmark(_parse_roi_strip, strip, rois)
        assert len(result) == 5   # five data regions

    def test_bench_strip_copy(self, benchmark, synthetic_strip_9roi):
        """
        Benchmark: np.ndarray.copy() on the full strip.
        This is the defensive copy in acquire_frame() - its cost is a floor
        for acquisition latency regardless of what else happens.
        """
        strip, _ = synthetic_strip_9roi
        benchmark(strip.copy)

    def test_bench_writer_push(self, benchmark, tmp_path, star_roi):
        """
        Benchmark: FrameWriter.push() - the call made from the hot path.
        Measures lock contention + np.copyto() cost.
        """
        import time
        from workers.camera import AcquiredFrame, CameraMode
        from workers.writer import FrameWriter

        writer = FrameWriter(
            data_dir=str(tmp_path),
            session_id="bench",
            roi_label="roi",
            frame_shape=star_roi.shape,
            buffer_n_frames=256,
        )
        writer.start()

        acq = AcquiredFrame(
            frame_id=0, timestamp_ns=0, host_time=0.0,
            mode=CameraMode.SINGLE_ROI, rois={}, nframes_dropped=0,
        )

        benchmark(writer.push, star_roi, acq)
        writer.stop()
