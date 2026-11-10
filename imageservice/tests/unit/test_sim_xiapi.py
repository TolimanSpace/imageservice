"""
tests/unit/test_sim_xiapi.py - The simulated camera (workers/sim_xiapi.py)
behind XimeaCamera, with a fixed star and with a fake pointing plant.
"""

import os
import socket
import struct
import threading
import time

import numpy as np
import pytest

from workers import sim_xiapi
from workers.camera import XimeaCamera, single_roi_config
from workers.centroid import compute_pointing_error

pytestmark = pytest.mark.unit

SENSOR = 4512
ROI = 128
OFF = (SENSOR - ROI) // 2


@pytest.fixture
def sim_env(monkeypatch):
    def set_source(source, **extra):
        monkeypatch.setenv("TOLIMAN_CAMERA_SIM", source)
        monkeypatch.setenv("TOLIMAN_CAMERA_SIM_SEED", "1")
        for k, v in extra.items():
            monkeypatch.setenv(f"TOLIMAN_CAMERA_SIM_{k}", str(v))
    return set_source


def _config(rate=100.0, exposure=4000):
    return single_roi_config(width=ROI, height=ROI, offset_x=OFF, offset_y=OFF,
                             exposure_us=exposure, frame_rate_hz=rate)


def _pointing(frame):
    return compute_pointing_error(frame.rois["single_roi"], OFF, OFF, SENSOR, SENSOR,
                                  frame.timestamp_ns, frame.frame_id, subtract_background=True)


def test_selected_by_the_environment(sim_env):
    sim_env("static:0,0")
    cam = XimeaCamera(_config())
    assert cam._xiapi is sim_xiapi


def test_not_selected_without_the_environment(monkeypatch):
    monkeypatch.delenv("TOLIMAN_CAMERA_SIM", raising=False)
    cam = XimeaCamera(_config())
    assert cam._xiapi is not sim_xiapi


def test_renders_the_star_where_it_is(sim_env):
    sim_env("static:2.5,-1.25")
    with XimeaCamera(_config()) as cam:
        frames = [cam.acquire_frame() for _ in range(5)]
    for f in frames:
        img = f.rois["single_roi"]
        assert img.dtype == np.uint16 and img.shape == (ROI, ROI)
        assert img.max() <= 4095
        pe = _pointing(f)
        assert abs(pe.dx - 2.5) < 0.1 and abs(pe.dy + 1.25) < 0.1
    # Bias around 100 ADU (MC203MG-SY-UB), peak well above it
    assert 90 < np.median(frames[0].rois["single_roi"]) < 110
    assert frames[0].rois["single_roi"].max() > 1000


def test_exposure_scales_the_signal(sim_env):
    sim_env("static:0,0", READ_NOISE=0)
    peaks = []
    for exposure in (1000, 4000):
        with XimeaCamera(_config(exposure=exposure)) as cam:
            peaks.append(int(cam.acquire_frame().rois["single_roi"].max()) - 100)
    assert 3.0 < peaks[1] / peaks[0] < 5.0


def test_frames_are_paced_and_counted(sim_env):
    sim_env("static:0,0")
    with XimeaCamera(_config(rate=50.0, exposure=1000)) as cam:
        start = time.monotonic()
        frames = [cam.acquire_frame() for _ in range(10)]
        elapsed = time.monotonic() - start
    assert 0.17 < elapsed < 0.4          # 10 frames at 50 Hz
    assert [f.frame_id for f in frames] == list(range(1, 11))
    assert all(f.nframes_dropped == 0 for f in frames)
    ts = [f.timestamp_ns for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert abs((ts[-1] - ts[0]) / 9 - 20e6) < 5e6


def test_a_slow_reader_drops_frames(sim_env):
    sim_env("static:0,0")
    with XimeaCamera(_config(rate=100.0, exposure=1000)) as cam:
        cam.acquire_frame()
        time.sleep(0.1)
        f = cam.acquire_frame()
    assert f.nframes_dropped >= 5


def test_sensor_size_bounds_the_roi(sim_env):
    sim_env("static:0,0", SENSOR="1024x1024")
    with pytest.raises(sim_xiapi.Xi_error):
        with XimeaCamera(_config()) as cam:   # ROI at 2192 is off a 1024 sensor
            cam.acquire_frame()


class FakePlant:
    """Answers plant queries with a star position that the test sets."""

    def __init__(self, path):
        self.path = path
        self.x, self.y = 0.0, 0.0
        self.queries = 0
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(path)
        self.sock.settimeout(0.1)
        self.running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(64)
            except socket.timeout:
                continue
            if data == bytes([0x10, 1, 0, 0]) and addr:
                self.queries += 1
                self.sock.sendto(struct.pack("<BBBBQ5f", 0x11, 1, 3, 0, time.time_ns(),
                                             self.x, self.y, 2047, 2047, 2047), addr)

    def close(self):
        self.running = False
        self.thread.join()
        self.sock.close()


@pytest.fixture
def plant(tmp_path):
    p = FakePlant(str(tmp_path / "plant.sock"))
    yield p
    p.close()


def test_follows_the_plant(sim_env, plant):
    sim_env(plant.path)
    with XimeaCamera(_config()) as cam:
        plant.x, plant.y = -3.0, 4.0
        a = _pointing(cam.acquire_frame())
        plant.x, plant.y = 1.0, 0.5
        b = _pointing(cam.acquire_frame())
    assert abs(a.dx + 3.0) < 0.1 and abs(a.dy - 4.0) < 0.1
    assert abs(b.dx - 1.0) < 0.1 and abs(b.dy - 0.5) < 0.1
    assert plant.queries == 2


def test_no_plant_is_a_camera_error(sim_env, tmp_path):
    sim_env(str(tmp_path / "missing.sock"))
    with XimeaCamera(_config()) as cam:
        with pytest.raises(sim_xiapi.Xi_error):
            cam.acquire_frame()
