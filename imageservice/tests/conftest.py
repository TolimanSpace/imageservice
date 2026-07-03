"""
conftest.py - Shared pytest fixtures.

Available to all tests via automatic fixture injection. Organised into:

  Infrastructure   - FakeXiapi, tmp directories
  Camera configs   - pre-built CameraConfig objects for each mode
  System/Session   - SystemConfig and SessionConfig instances
  Image data       - synthetic ROI arrays, frame stacks
  Writers          - pre-started FrameWriter instances
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from typing import Dict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the fake ximea module is injected before any worker imports
# ---------------------------------------------------------------------------

def _inject_fake_ximea() -> None:
    """
    Inject a fake ximea module so worker.camera can be imported
    without the Ximea C extension installed.
    """
    if "ximea" not in sys.modules:
        fake_ximea  = types.ModuleType("ximea")
        fake_xiapi  = types.ModuleType("ximea.xiapi")

        class _FakeCam:  pass
        class _FakeImg:  pass
        class _FakeErr(Exception): pass

        fake_xiapi.Camera   = _FakeCam
        fake_xiapi.Image    = _FakeImg
        fake_xiapi.Xi_error = _FakeErr
        fake_ximea.xiapi    = fake_xiapi
        sys.modules["ximea"]        = fake_ximea
        sys.modules["ximea.xiapi"]  = fake_xiapi

_inject_fake_ximea()

# Now safe to import worker modules
from workers.camera import (
    CameraConfig,
    CameraMode,
    RoiDefinition,
    XimeaCamera,
    full_frame_config,
    single_roi_config,
)
from workers.centroid import compute_pointing_error
from workers.config import SessionConfig, SystemConfig
from workers.writer import FrameWriter
from tests.fake_xiapi import FakeXiapi, FakeCamera, FakeImage


# ---------------------------------------------------------------------------
# Sensor / geometry constants shared across tests
# ---------------------------------------------------------------------------

SENSOR_WIDTH  = 4504
SENSOR_HEIGHT = 4504
ROI_WIDTH     = 128
ROI_HEIGHT    = 128
ROI_OFFSET_X  = (SENSOR_WIDTH  - ROI_WIDTH)  // 2   # 2188
ROI_OFFSET_Y  = (SENSOR_HEIGHT - ROI_HEIGHT) // 2   # 2188


# ---------------------------------------------------------------------------
# Infrastructure fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_xiapi() -> FakeXiapi:
    """A fresh FakeXiapi instance for each test."""
    return FakeXiapi()


@pytest.fixture
def fake_camera(fake_xiapi) -> FakeCamera:
    """
    A FakeCamera instance with a pre-loaded blank frame.
    Use fake_camera.inject_error() to simulate hardware faults.
    """
    cam = FakeCamera(sensor_width=SENSOR_WIDTH, sensor_height=SENSOR_HEIGHT)
    return cam


@pytest.fixture
def tmp_data_dir(tmp_path) -> str:
    """Temporary directory for writer output files."""
    d = tmp_path / "data"
    d.mkdir()
    return str(d)


@pytest.fixture
def tmp_log_dir(tmp_path) -> str:
    """Temporary directory for log files."""
    d = tmp_path / "logs"
    d.mkdir()
    return str(d)


# ---------------------------------------------------------------------------
# Camera config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_roi_cfg() -> CameraConfig:
    """CameraConfig for SINGLE_ROI mode centred on the sensor."""
    return single_roi_config(
        width=ROI_WIDTH,
        height=ROI_HEIGHT,
        offset_x=ROI_OFFSET_X,
        offset_y=ROI_OFFSET_Y,
        exposure_us=5_000,
        frame_rate_hz=10.0,
        label="roi",
        serial_number=None,
    )


@pytest.fixture
def full_frame_cfg() -> CameraConfig:
    """CameraConfig for FULL_FRAME diagnostic mode."""
    return full_frame_config(
        exposure_us=50_000,
        frame_rate_hz=1.0,
        serial_number=None,
    )


# ---------------------------------------------------------------------------
# System / session config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def system_config_dict(tmp_data_dir, tmp_log_dir) -> dict:
    """Raw dict matching the SystemConfig JSON schema."""
    return {
        "camera": {
            "serial_number": "TEST000001",
            "transport_buffer_size": 8,
            "output_bit_depth": 12,
        },
        "sensor": {
            "width": SENSOR_WIDTH,
            "height": SENSOR_HEIGHT,
        },
        "centroid": {"min_total_intensity": 1.0},
        "writer":   {"ring_buffer_n_frames": 32, "flush_timeout_s": 2.0},
        "compression": {"batch_size_n": 10, "deflate_level": 4},
        "serial":   {"baud_rate": 115200},
        "paths": {
            "data_dir": tmp_data_dir,
            "log_dir":  tmp_log_dir,
        },
    }


@pytest.fixture
def system_config(system_config_dict, tmp_path) -> SystemConfig:
    """Loaded SystemConfig from a temporary JSON file."""
    cfg_path = tmp_path / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(system_config_dict, f)
    return SystemConfig.from_file(str(cfg_path))


@pytest.fixture
def session_config_single() -> SessionConfig:
    """SessionConfig for a short SINGLE_ROI session."""
    return SessionConfig(
        mode=CameraMode.SINGLE_ROI,
        rois=[RoiDefinition(
            height=ROI_HEIGHT,
            offset_y=ROI_OFFSET_Y,
            width=ROI_WIDTH,
            offset_x=ROI_OFFSET_X,
            label="roi",
            is_data=True,
        )],
        frame_rate_hz=10.0,
        exposure_us=5_000,
        n_frames=20,
        duration_s=None,
        session_id="test_session_001",
    )


@pytest.fixture
def session_config_full_frame() -> SessionConfig:
    """SessionConfig for a short FULL_FRAME diagnostic session."""
    return SessionConfig(
        mode=CameraMode.FULL_FRAME,
        rois=None,
        frame_rate_hz=1.0,
        exposure_us=50_000,
        n_frames=3,
        duration_s=None,
        session_id="test_diag_001",
    )


# ---------------------------------------------------------------------------
# Image data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_roi() -> np.ndarray:
    """All-zero uint16 ROI array."""
    return np.zeros((ROI_HEIGHT, ROI_WIDTH), dtype=np.uint16)


@pytest.fixture
def star_roi() -> np.ndarray:
    """
    Synthetic uint16 ROI with a Gaussian star PSF near the centre.
    Peak value ~3000 ADU, background ~10 ADU.
    """
    rng = np.random.default_rng(42)
    img = rng.integers(5, 15, (ROI_HEIGHT, ROI_WIDTH), dtype=np.uint16)
    cy, cx = ROI_HEIGHT // 2, ROI_WIDTH // 2
    y, x = np.ogrid[:ROI_HEIGHT, :ROI_WIDTH]
    sigma = 3.0
    psf = 3000 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    img = np.clip(img.astype(np.float32) + psf, 0, 4095).astype(np.uint16)
    return img


@pytest.fixture
def two_star_roi() -> np.ndarray:
    """
    Synthetic uint16 ROI with two Gaussian PSFs (binary star system).
    Stars at offsets (-10, -10) and (+10, +10) from ROI centre.
    """
    rng = np.random.default_rng(99)
    img = rng.integers(5, 15, (ROI_HEIGHT, ROI_WIDTH), dtype=np.uint16)
    cy, cx = ROI_HEIGHT // 2, ROI_WIDTH // 2
    y, x = np.ogrid[:ROI_HEIGHT, :ROI_WIDTH]
    sigma = 3.0
    for dy, dx in [(-10, -10), (10, 10)]:
        psf = 2500 * np.exp(
            -((x - (cx + dx))**2 + (y - (cy + dy))**2) / (2 * sigma**2)
        )
        img = np.clip(img.astype(np.float32) + psf, 0, 4095).astype(np.uint16)
    return img


@pytest.fixture
def frame_stack(star_roi) -> np.ndarray:
    """
    Stack of 10 slightly varying uint16 frames for compression tests.
    Each frame shifts the star by 1 pixel to create non-trivial diffs.
    """
    n = 10
    h, w = star_roi.shape
    stack = np.zeros((n, h, w), dtype=np.uint16)
    stack[0] = star_roi
    rng = np.random.default_rng(7)
    for i in range(1, n):
        noise = rng.integers(-5, 6, (h, w))
        stack[i] = np.clip(
            star_roi.astype(np.int32) + noise, 0, 4095
        ).astype(np.uint16)
    return stack


# ---------------------------------------------------------------------------
# XimeaCamera fixture (fake hardware)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_ximea_camera(single_roi_cfg, star_roi, fake_xiapi):
    """
    XimeaCamera configured for SINGLE_ROI, backed by FakeXiapi.
    The fake image is pre-loaded with a star_roi frame.
    Returns the camera instance (not yet entered as context manager).
    """
    cam = XimeaCamera(single_roi_cfg, _xiapi=fake_xiapi)
    cam.open()
    cam.configure()
    # Pre-load image data
    cam._img.set_data(star_roi)
    cam.start()
    yield cam
    cam.stop()
    cam.close()


# ---------------------------------------------------------------------------
# FrameWriter fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def frame_writer(tmp_data_dir, star_roi):
    """
    A started FrameWriter for a single ROI, using tmp_data_dir.
    Yields the writer; stops it after the test.
    """
    writer = FrameWriter(
        data_dir=tmp_data_dir,
        session_id="fixture_session",
        roi_label="roi",
        frame_shape=star_roi.shape,
        buffer_n_frames=32,
        flush_timeout_s=2.0,
    )
    writer.start()
    yield writer
    writer.stop()
