"""
tests/unit/test_pointing_link.py - Centroid datagrams to the pointing service.
"""

import os
import socket
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from workers.pointing_link import CENTROID, CentroidSender, encode_centroid

pytestmark = pytest.mark.unit


def test_layout_matches_the_sec_pointing_link():
    # The same bytes as the SEC repository's pointing_link.Centroid.encode()
    data = encode_centroid(7, 1_700_000_000_123_456_789, 1.5, -0.25, 1234.0, 2000.0,
                           no_signal=False, saturated=True)
    assert len(data) == 32
    assert data[:4] == bytes([0x01, 1, 0x02, 0])
    assert struct.unpack("<I", data[4:8]) == (7,)
    assert struct.unpack("<Q", data[8:16]) == (1_700_000_000_123_456_789,)
    assert struct.unpack("<ffff", data[16:]) == (1.5, -0.25, 1234.0, 2000.0)


def test_frame_id_wraps():
    assert struct.unpack_from("<I", encode_centroid(2**32 + 5, 0, 0, 0, 0, 0), 4) == (5,)


@pytest.fixture
def receiver(tmp_path):
    path = str(tmp_path / "pointing.sock")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(path)
    sock.settimeout(1.0)
    yield path, sock
    sock.close()


def _frame(frame_id=3, t=123):
    return SimpleNamespace(frame_id=frame_id, unix_time_ns=t)


def _pointing(dx, dy, total, peak):
    return SimpleNamespace(dx=dx, dy=dy, centroid=SimpleNamespace(total_intensity=total, peak_value=peak))


def test_sends_a_pointing_error(receiver):
    path, sock = receiver
    sender = CentroidSender(path)
    sender.send_pointing(_frame(), _pointing(0.5, -1.0, 5000.0, 4095))
    kind, ver, flags, _, fid, t, dx, dy, total, peak = CENTROID.unpack(sock.recv(64))
    assert (kind, ver, flags, fid, t) == (1, 1, 0x02, 3, 123)  # 4095: saturated
    assert (dx, dy, total, peak) == (0.5, -1.0, 5000.0, 4095.0)
    sender.close()


def test_sends_no_signal(receiver):
    path, sock = receiver
    sender = CentroidSender(path)
    sender.send_no_signal(_frame(9), np.full((4, 4), 101, dtype=np.uint16))
    _, _, flags, _, fid, _, dx, dy, total, peak = CENTROID.unpack(sock.recv(64))
    assert (flags, fid, dx, dy, total, peak) == (0x01, 9, 0.0, 0.0, 0.0, 101.0)


def test_socket_path_from_environment(monkeypatch, receiver):
    path, sock = receiver
    monkeypatch.setenv("TOLIMAN_POINTING_SOCKET", path)
    CentroidSender().send_no_signal(_frame(), None)
    assert len(sock.recv(64)) == 32


def test_drops_without_a_pointing_service(tmp_path):
    sender = CentroidSender(str(tmp_path / "nobody.sock"))
    for _ in range(3):
        sender.send_no_signal(_frame(), None)
    assert (sender.sent, sender.dropped) == (0, 3)


def test_never_blocks_on_a_full_queue(receiver):
    path, _sock = receiver  # never read
    sender = CentroidSender(path)
    for _ in range(10_000):
        sender.send_no_signal(_frame(), None)
    assert sender.dropped > 0
    assert sender.sent + sender.dropped == 10_000
