"""
tests/integration/test_compress_pipeline.py
Integration tests: writer output → compressor
"""

import json, os, tempfile
import numpy as np
import pytest

pytestmark = pytest.mark.integration


class TestCompressionPipeline:

    def test_make_diffs_then_reconstruct(self, frame_stack):
        from workers.compression import _make_diffs
        diffs, dtype_name, scale = _make_diffs(frame_stack)
        # Reconstruct frame 1 from reference + diff[0]
        reference   = frame_stack[0].astype(np.int32)
        diff_0      = diffs[0].astype(np.int32) * int(scale)
        reconstructed = (reference + diff_0).astype(np.uint16)
        np.testing.assert_array_equal(reconstructed, frame_stack[1])

    def test_diffs_written_as_correct_dtype(self, frame_stack):
        from workers.compression import _make_diffs
        _, dtype_name, _ = _make_diffs(frame_stack)
        assert dtype_name in ("int8", "int16")

    def test_writer_output_loadable_as_frame_stack(
        self, tmp_data_dir, star_roi, frame_stack
    ):
        """Frames written by FrameWriter can be reloaded as a numpy stack."""
        import time
        from workers.writer import FrameWriter
        from workers.camera import AcquiredFrame, CameraMode

        n, h, w = frame_stack.shape
        with FrameWriter(
            tmp_data_dir, "integ_comp", "roi",
            (h, w), buffer_n_frames=32,
        ) as writer:
            for i in range(n):
                acq = AcquiredFrame(
                    frame_id=i, timestamp_ns=i * 1_000_000,
                    host_time=float(i), mode=CameraMode.SINGLE_ROI,
                    rois={}, nframes_dropped=0,
                )
                writer.push(frame_stack[i], acq)
            time.sleep(0.3)

        path = os.path.join(tmp_data_dir, "integ_comp_frames.bin")
        raw = np.fromfile(path, dtype=np.uint16).reshape(n, h, w)
        np.testing.assert_array_equal(raw, frame_stack)
