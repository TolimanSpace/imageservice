"""
tests/e2e/test_simulated_session.py
End-to-end test: full simulated acquisition session using fake hardware.

Exercises the complete pipeline:
  SystemConfig → SessionConfig → CameraConfig →
  XimeaCamera → centroid → FrameWriter → written files
"""

import json
import os
import time

import numpy as np
import pytest

from workers.acquisition import _run, _should_continue, AcquisitionResult
from workers.camera import (
    AcquiredFrame,
    CameraMode,
    RoiDefinition,
    XimeaCamera,
)
from workers.centroid import CENTRE_ROI_LABEL, compute_pointing_error
from workers.writer import FrameWriter
from tests.conftest import (
    ROI_HEIGHT,
    ROI_OFFSET_X,
    ROI_OFFSET_Y,
    ROI_WIDTH,
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
)
from tests.fake_xiapi import FakeXiapi

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_star_roi(seed: int = 42) -> np.ndarray:
    """Synthetic 128×128 ROI with a Gaussian PSF."""
    rng = np.random.default_rng(seed)
    img = rng.integers(5, 15, (ROI_HEIGHT, ROI_WIDTH), dtype=np.uint16)
    cy, cx = ROI_HEIGHT // 2, ROI_WIDTH // 2
    y, x = np.ogrid[:ROI_HEIGHT, :ROI_WIDTH]
    psf = 3000 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 3.0**2))
    return np.clip(img.astype(np.float32) + psf, 0, 4095).astype(np.uint16)


# ---------------------------------------------------------------------------
# E2E: manual pipeline (no multiprocessing - easier to debug)
# ---------------------------------------------------------------------------

class TestSimulatedSession:

    def test_full_session_writes_correct_frame_count(
        self, system_config, session_config_single,
        fake_xiapi, tmp_data_dir
    ):
        """
        A complete 20-frame SINGLE_ROI session:
        - Camera delivers 20 frames via fake hardware
        - FrameWriter receives all 20
        - Binary file has correct size
        - Metadata file has 20 lines
        """
        star_roi = _make_star_roi()
        n = session_config_single.n_frames

        camera_config = system_config.to_camera_config(session_config_single)
        roi = session_config_single.rois[0]

        writers = {
            roi.label: FrameWriter(
                data_dir=tmp_data_dir,
                session_id=session_config_single.session_id,
                roi_label=roi.label,
                frame_shape=(roi.height, roi.width),
                buffer_n_frames=32,
                flush_timeout_s=2.0,
            )
        }

        for w in writers.values():
            w.start()

        with XimeaCamera(camera_config, _xiapi=fake_xiapi) as cam:
            cam._img.set_data(star_roi)
            for _ in range(n):
                frame = cam.acquire_frame()
                writers[roi.label].push(frame.rois[roi.label], frame)

        time.sleep(0.3)
        for w in writers.values():
            w.stop()

        s = writers[roi.label].stats
        assert s.frames_written == n
        assert s.frames_dropped == 0

        # Check file size
        bin_path = os.path.join(
            tmp_data_dir,
            f"{session_config_single.session_id}_{roi.label}_frames.bin"
        )
        expected_bytes = n * roi.height * roi.width * 2
        assert os.path.getsize(bin_path) == expected_bytes

        # Check metadata
        meta_path = bin_path.replace("_frames.bin", "_meta.jsonl")
        lines = open(meta_path).readlines()
        assert len(lines) == n

    def test_centroids_computed_for_all_frames(
        self, system_config, session_config_single, fake_xiapi
    ):
        """All 20 frames should yield a valid PointingError."""
        star_roi = _make_star_roi()
        n = session_config_single.n_frames
        camera_config = system_config.to_camera_config(session_config_single)
        roi = session_config_single.rois[0]

        pointing_errors = []
        with XimeaCamera(camera_config, _xiapi=fake_xiapi) as cam:
            cam._img.set_data(star_roi)
            for _ in range(n):
                frame = cam.acquire_frame()
                centre = frame.rois.get(roi.label)
                if centre is not None:
                    pe = compute_pointing_error(
                        centre,
                        roi_offset_x=ROI_OFFSET_X,
                        roi_offset_y=ROI_OFFSET_Y,
                        sensor_width=SENSOR_WIDTH,
                        sensor_height=SENSOR_HEIGHT,
                        timestamp_ns=frame.timestamp_ns,
                        frame_id=frame.frame_id,
                    )
                    pointing_errors.append(pe)

        assert len(pointing_errors) == n
        assert all(pe is not None for pe in pointing_errors)
        # Star is near sensor centre → |dx|, |dy| < 5 pixels
        assert all(abs(pe.dx) < 5.0 for pe in pointing_errors)
        assert all(abs(pe.dy) < 5.0 for pe in pointing_errors)

    def test_camera_error_recovery(
        self, system_config, session_config_single, fake_xiapi, tmp_data_dir
    ):
        """
        Inject 3 consecutive errors - within MAX_CONSECUTIVE_ERRORS (5).
        Session should continue and recover automatically.
        """
        star_roi = _make_star_roi()
        camera_config = system_config.to_camera_config(session_config_single)
        roi = session_config_single.rois[0]

        with XimeaCamera(camera_config, _xiapi=fake_xiapi) as cam:
            cam._img.set_data(star_roi)

            # Inject 3 errors after the 5th frame
            error_count = 0
            frames_acquired = 0
            n = session_config_single.n_frames

            for attempt in range(n + 10):
                if frames_acquired == 5 and error_count < 3:
                    cam._cam.inject_error()
                    error_count += 1

                try:
                    frame = cam.acquire_frame()
                    frames_acquired += 1
                    if frames_acquired >= n:
                        break
                except Exception:
                    pass

        # Should have acquired all n frames despite the injected errors
        assert frames_acquired == n

    def test_session_stops_at_n_frames(
        self, system_config, session_config_single, fake_xiapi
    ):
        from workers.acquisition import _should_continue
        import time

        n = session_config_single.n_frames
        session_start = time.monotonic()

        # Simulate counting frames
        for i in range(n + 5):
            if not _should_continue(session_config_single, i, session_start):
                stopped_at = i
                break
        else:
            stopped_at = n + 5

        assert stopped_at == n

    def test_session_stops_at_duration(self):
        from workers.camera import CameraMode, RoiDefinition
        from workers.config import SessionConfig
        from workers.acquisition import _should_continue
        import time

        session = SessionConfig(
            mode=CameraMode.FULL_FRAME, rois=None,
            frame_rate_hz=1.0, exposure_us=1_000,
            n_frames=None, duration_s=0.1,
            session_id="duration_test",
        )
        session_start = time.monotonic() - 0.2   # already 0.2s ago
        assert not _should_continue(session, 0, session_start)

    def test_full_frame_session_writes_correct_size(
        self, system_config, session_config_full_frame,
        fake_xiapi, tmp_data_dir
    ):
        """FULL_FRAME session writes full 4504×4504 frames."""
        full_frame = np.zeros((SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.uint16)
        full_frame[SENSOR_HEIGHT // 2, SENSOR_WIDTH // 2] = 3000
        n = session_config_full_frame.n_frames

        camera_config = system_config.to_camera_config(
            session_config_full_frame
        )

        writer = FrameWriter(
            data_dir=tmp_data_dir,
            session_id=session_config_full_frame.session_id,
            roi_label="full_frame",
            frame_shape=(SENSOR_HEIGHT, SENSOR_WIDTH),
            buffer_n_frames=4,
            flush_timeout_s=5.0,
        )
        writer.start()

        with XimeaCamera(camera_config, _xiapi=fake_xiapi) as cam:
            cam._img.set_data(full_frame)
            for _ in range(n):
                frame = cam.acquire_frame()
                writer.push(frame.rois["full_frame"], frame)

        time.sleep(0.5)
        writer.stop()

        assert writer.stats.frames_written == n
        expected = n * SENSOR_HEIGHT * SENSOR_WIDTH * 2
        bin_path = os.path.join(
            tmp_data_dir,
            f"{session_config_full_frame.session_id}_full_frame_frames.bin"
        )
        assert os.path.getsize(bin_path) == expected
