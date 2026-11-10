"""
tests/integration/test_sim_pointing.py - The acquisition loop on the
simulated camera sends each frame's centroid to the pointing service socket.
"""

import dataclasses
import socket

import pytest

from workers.acquisition import _run
from workers.pointing_link import CENTROID
from tests.conftest import SENSOR_WIDTH, SENSOR_HEIGHT

pytestmark = pytest.mark.integration


def test_session_sends_centroids(monkeypatch, tmp_path, system_config, session_config_single):
    monkeypatch.setenv("TOLIMAN_CAMERA_SIM", "static:1.5,-2.0")
    monkeypatch.setenv("TOLIMAN_CAMERA_SIM_SENSOR", f"{SENSOR_WIDTH}x{SENSOR_HEIGHT}")
    monkeypatch.setenv("TOLIMAN_CAMERA_SIM_SEED", "2")
    path = str(tmp_path / "pointing.sock")
    monkeypatch.setenv("TOLIMAN_POINTING_SOCKET", path)
    rx = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    rx.bind(path)
    rx.settimeout(1.0)

    system = dataclasses.replace(system_config, centroid_subtract_background=True)
    session = dataclasses.replace(session_config_single, frame_rate_hz=50.0, exposure_us=4000, n_frames=10)
    result = _run(system, session, enable_diagnostics=False)

    assert not result.fault and result.frames_acquired == 10
    got = [CENTROID.unpack(rx.recv(64)) for _ in range(10)]
    rx.close()
    for i, (kind, ver, flags, _, fid, t, dx, dy, total, peak) in enumerate(got):
        assert (kind, ver, flags, fid) == (1, 1, 0, i + 1)
        assert t > 0
        assert abs(dx - 1.5) < 0.1 and abs(dy + 2.0) < 0.1
        assert total > 0 and 1000 < peak < 4095
