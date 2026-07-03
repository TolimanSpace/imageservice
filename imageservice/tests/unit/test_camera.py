"""
tests/unit/test_camera.py — Unit tests for startracker.camera

Tests are grouped by class/function under TestCameraConfig,
TestCameraLifecycle, TestAcquisition, and TestFactories.
All tests use FakeXiapi — no real hardware required.
"""

import pytest
import numpy as np

from workers.camera import (
    AcquiredFrame,
    CameraConfig,
    CameraMode,
    RoiDefinition,
    XimeaCamera,
    _check_alignment,
    _parse_roi_strip,
    _select_timing_mode,
    full_frame_config,
    single_roi_config,
)
from tests.fake_xiapi import FakeXiapi, Xi_error

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# CameraConfig validation
# ---------------------------------------------------------------------------

class TestCameraConfig:

    def test_single_roi_valid(self, single_roi_cfg):
        assert single_roi_cfg.mode is CameraMode.SINGLE_ROI
        # output camera.py uses single_roi field (SingleRoiDefinition)
        assert single_roi_cfg.single_roi is not None
        assert single_roi_cfg.rois is None

    def test_full_frame_valid(self, full_frame_cfg):
        assert full_frame_cfg.mode is CameraMode.FULL_FRAME
        assert full_frame_cfg.rois is None

    def test_invalid_bit_depth_raises(self):
        with pytest.raises(ValueError, match="output_bit_depth"):
            CameraConfig(
                serial_number=None,
                mode=CameraMode.FULL_FRAME,
                exposure_us=5_000,
                frame_rate_hz=1.0,
                output_bit_depth=8,
            )

    def test_zero_frame_rate_raises(self):
        with pytest.raises(ValueError, match="frame_rate_hz"):
            full_frame_config(exposure_us=5_000, frame_rate_hz=0.0)

    def test_single_roi_requires_single_roi(self):
        with pytest.raises(ValueError, match="single_roi"):
            CameraConfig(
                serial_number=None,
                mode=CameraMode.SINGLE_ROI,
                exposure_us=5_000,
                frame_rate_hz=10.0,
                rois=None,
            )

    def test_single_roi_rejects_rois_list(self):
        """output camera.py: SINGLE_ROI uses single_roi field, rois must be None."""
        roi = RoiDefinition(128, 0, 128, 0, "roi", True)
        with pytest.raises(ValueError, match="single_roi must be provided"):
            CameraConfig(
                serial_number=None,
                mode=CameraMode.SINGLE_ROI,
                exposure_us=5_000,
                frame_rate_hz=10.0,
                rois=[roi],
            )

    def test_full_frame_rejects_rois(self):
        roi = RoiDefinition(128, 0, 128, 0, "roi", True)
        with pytest.raises(ValueError, match="None"):
            CameraConfig(
                serial_number=None,
                mode=CameraMode.FULL_FRAME,
                exposure_us=5_000,
                frame_rate_hz=1.0,
                rois=[roi],
            )


# ---------------------------------------------------------------------------
# Camera lifecycle
# ---------------------------------------------------------------------------

class TestCameraLifecycle:

    def test_open_by_serial_number(self, single_roi_cfg, fake_xiapi):
        cfg = single_roi_config(
            width=128, height=128, offset_x=0, offset_y=0,
            exposure_us=5_000, frame_rate_hz=10.0,
            serial_number="SN123456",
        )
        cam = XimeaCamera(cfg, _xiapi=fake_xiapi)
        cam.open()
        cam.close()
        cam_instance = fake_xiapi.Camera.instances[-1] \
            if hasattr(fake_xiapi.Camera, 'instances') else None
        # Verify open_device_by_SN was called (not open_device)
        all_calls = [c for obj in [fake_xiapi._last_camera]
                     for c in obj.calls] \
            if hasattr(fake_xiapi, '_last_camera') else []
        # Simpler: check via the camera object directly
        cam2 = XimeaCamera(cfg, _xiapi=fake_xiapi)
        cam2.open()
        # The FakeCamera created will have the call
        assert any(
            c[0] == "open_device_by_SN" and c[1] == "SN123456"
            for c in cam2._cam.calls
        )
        cam2.close()

    def test_open_without_serial_number(self, single_roi_cfg, fake_xiapi):
        cam = XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi)
        cam.open()
        assert any(c[0] == "open_device" for c in cam._cam.calls)
        cam.close()

    def test_configure_sets_imgdataformat(self, single_roi_cfg, fake_xiapi):
        cam = XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi)
        cam.open()
        cam.configure()
        assert cam._cam.was_called("set_imgdataformat", "XI_MONO16")
        cam.close()

    def test_configure_sets_bit_depth(self, single_roi_cfg, fake_xiapi):
        cam = XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi)
        cam.open()
        cam.configure()
        assert cam._cam.was_called("set_output_bit_depth", "XI_BPP_12")
        cam.close()

    def test_configure_sets_roi_geometry(self, single_roi_cfg, fake_xiapi):
        cam = XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi)
        cam.open()
        cam.configure()
        assert cam._cam.was_called("set_width",  128)
        assert cam._cam.was_called("set_height", 128)
        cam.close()

    def test_configure_sets_exposure(self, single_roi_cfg, fake_xiapi):
        cam = XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi)
        cam.open()
        cam.configure()
        assert cam._cam.was_called("set_exposure", 5_000)
        cam.close()

    def test_configure_during_acquisition_raises(self, single_roi_cfg,
                                                  fake_xiapi, star_roi):
        cam = XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi)
        cam.open()
        cam.configure()
        cam.start()
        with pytest.raises(RuntimeError, match="reconfigure"):
            cam.configure()
        cam.stop()
        cam.close()

    def test_context_manager_calls_stop_and_close(self, single_roi_cfg,
                                                    fake_xiapi, star_roi):
        cam = XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi)
        cam_instance = None
        with cam:
            cam_instance = cam._cam
            cam._img.set_data(star_roi)
        assert any(c[0] == "stop_acquisition" for c in cam_instance.calls)
        assert any(c[0] == "close_device"     for c in cam_instance.calls)

    def test_context_manager_closes_on_exception(self, single_roi_cfg,
                                                   fake_xiapi):
        cam = XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi)
        cam_instance = None
        with pytest.raises(ValueError):
            with cam:
                cam_instance = cam._cam
                raise ValueError("deliberate test error")
        assert any(c[0] == "close_device" for c in cam_instance.calls)

    def test_full_frame_does_not_call_region_selector(self, full_frame_cfg,
                                                        fake_xiapi):
        cam = XimeaCamera(full_frame_cfg, _xiapi=fake_xiapi)
        cam.open()
        cam.configure()
        region_calls = [c for c in cam._cam.calls
                        if c == ("set_param", "region_selector", 0)
                        or (len(c) > 1 and c[1] == "region_selector")]
        assert len(region_calls) == 0, \
            "FULL_FRAME must not call region_selector"
        cam.close()

    def test_full_frame_uses_maximum_dimensions(self, full_frame_cfg,
                                                 fake_xiapi):
        cam = XimeaCamera(full_frame_cfg, _xiapi=fake_xiapi)
        cam.open()
        cam.configure()
        assert cam._cam.was_called("set_width",  4504)
        assert cam._cam.was_called("set_height", 4504)
        assert cam._cam.was_called("set_offsetX", 0)
        assert cam._cam.was_called("set_offsetY", 0)
        cam.close()

    def test_buffer_size_clamped_to_camera_range(self, fake_xiapi):
        # Request a buffer size outside the camera's [2, 64] range
        cfg = full_frame_config(
            exposure_us=50_000, frame_rate_hz=1.0,
            transport_buffer_size=1,   # below minimum of 2
        )
        cam = XimeaCamera(cfg, _xiapi=fake_xiapi)
        cam.open()
        cam.configure()
        # Should have been clamped to 2
        set_buf_calls = cam._cam.call_args_for("set_buffers_queue_size")
        assert len(set_buf_calls) == 1
        assert set_buf_calls[0][0] == 2
        cam.close()


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

class TestAcquisition:

    def test_acquire_frame_returns_acquired_frame(self, single_roi_cfg,
                                                    fake_xiapi, star_roi):
        with XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi) as cam:
            cam._img.set_data(star_roi)
            frame = cam.acquire_frame()
        assert isinstance(frame, AcquiredFrame)

    def test_acquire_frame_has_correct_shape(self, single_roi_cfg,
                                               fake_xiapi, star_roi):
        with XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi) as cam:
            cam._img.set_data(star_roi)
            frame = cam.acquire_frame()
        roi_arr = frame.rois["roi"]
        assert roi_arr.shape == star_roi.shape
        assert roi_arr.dtype == np.uint16

    def test_acquire_frame_returns_copy(self, single_roi_cfg,
                                         fake_xiapi, star_roi):
        """Mutating the source buffer must not affect the acquired frame."""
        with XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi) as cam:
            cam._img.set_data(star_roi)
            frame = cam.acquire_frame()
            # Mutate the fake image buffer
            cam._img._data[:] = 0
        assert frame.rois["roi"][star_roi > 100].any(), \
            "Frame data was affected by post-acquire mutation"

    def test_acquire_frame_increments_frame_counter(self, single_roi_cfg,
                                                      fake_xiapi, star_roi):
        with XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi) as cam:
            cam._img.set_data(star_roi)
            for _ in range(5):
                cam.acquire_frame()
        assert cam.frames_acquired == 5

    def test_acquire_frame_outside_acquisition_raises(self, single_roi_cfg,
                                                        fake_xiapi):
        cam = XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi)
        cam.open()
        cam.configure()
        with pytest.raises(RuntimeError, match="outside"):
            cam.acquire_frame()
        cam.close()

    def test_dropped_frame_detected(self, single_roi_cfg,
                                     fake_xiapi, star_roi):
        """Simulate a dropped frame by advancing acq_nframe by 2."""
        with XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi) as cam:
            cam._img.set_data(star_roi)
            frame1 = cam.acquire_frame()
            # Manually skip an acq_nframe to simulate a dropped frame
            cam._cam._frame_counter += 1
            frame2 = cam.acquire_frame()
        assert frame2.nframes_dropped == 1

    def test_timestamp_populated(self, single_roi_cfg, fake_xiapi, star_roi):
        with XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi) as cam:
            cam._img.set_data(star_roi)
            frame = cam.acquire_frame()
        assert frame.timestamp_ns > 0

    def test_full_frame_mode_rois_key(self, full_frame_cfg,
                                       fake_xiapi):
        full_frame = np.zeros((4504, 4504), dtype=np.uint16)
        with XimeaCamera(full_frame_cfg, _xiapi=fake_xiapi) as cam:
            cam._img.set_data(full_frame)
            frame = cam.acquire_frame()
        assert "full_frame" in frame.rois
        assert frame.rois["full_frame"].shape == (4504, 4504)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_check_alignment_valid(self):
        _check_alignment("width", 128, 4)   # no exception

    def test_check_alignment_invalid(self):
        with pytest.raises(ValueError, match="increment"):
            _check_alignment("width", 130, 4)

    def test_check_alignment_zero_increment(self):
        _check_alignment("width", 130, 0)   # zero increment -> skip check

    def test_select_timing_mode_mc_series(self):
        assert _select_timing_mode(b"MC203MG-SY-UB") == \
            "XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT"

    def test_select_timing_mode_mq_series(self):
        assert _select_timing_mode(b"MQ013MG-E2") == \
            "XI_ACQ_TIMING_MODE_FRAME_RATE"

    def test_select_timing_mode_md_series(self):
        assert _select_timing_mode(b"MD120MU-SY") == \
            "XI_ACQ_TIMING_MODE_FRAME_RATE"

    def test_parse_roi_strip_returns_data_regions_only(self):
        # Build a synthetic strip: 3 regions of height 10, 4, 10
        rois = [
            RoiDefinition(10, 0,  64, 0, "top",   True),
            RoiDefinition(4,  10, 64, 0, "mid",   False),
            RoiDefinition(10, 14, 64, 0, "bot",   True),
        ]
        strip = np.arange(24 * 64, dtype=np.uint16).reshape(24, 64)
        result = _parse_roi_strip(strip, rois)
        assert "top" in result and "bot" in result
        assert "mid" not in result
        assert result["top"].shape == (10, 64)
        assert result["bot"].shape == (10, 64)

    def test_parse_roi_strip_correct_row_offsets(self):
        rois = [
            RoiDefinition(10, 0,  8, 0, "top", True),
            RoiDefinition(4,  10, 8, 0, "mid", False),
            RoiDefinition(10, 14, 8, 0, "bot", True),
        ]
        strip = np.zeros((24, 8), dtype=np.uint16)
        strip[0,  0] = 111   # first row of top
        strip[14, 0] = 222   # first row of bot
        result = _parse_roi_strip(strip, rois)
        assert result["top"][0, 0] == 111
        assert result["bot"][0, 0] == 222


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

class TestFactories:

    def test_single_roi_config_factory(self):
        cfg = single_roi_config(
            width=64, height=64, offset_x=100, offset_y=100,
            exposure_us=5_000, frame_rate_hz=10.0,
        )
        assert cfg.mode is CameraMode.SINGLE_ROI
        # output camera.py uses single_roi: SingleRoiDefinition
        assert cfg.single_roi is not None
        assert cfg.single_roi.width == 64
        assert cfg.single_roi.label == "roi"  # default label

    def test_single_roi_config_custom_label(self):
        cfg = single_roi_config(
            width=64, height=64, offset_x=0, offset_y=0,
            exposure_us=5_000, frame_rate_hz=10.0,
            label="my_roi",
        )
        assert cfg.single_roi.label == "my_roi"

    def test_full_frame_config_factory(self):
        cfg = full_frame_config(exposure_us=50_000, frame_rate_hz=2.0)
        assert cfg.mode is CameraMode.FULL_FRAME
        assert cfg.rois is None
        assert cfg.exposure_us == 50_000
