"""
The supervisor as the SEC Camera service: a fake executive on a Unix socket
sends START over the toliman-ipc wire format, the supervisor runs a session
on the fake camera and reports it as an event.
"""

import json
import os
import signal
import socket
import struct
import sys
import tempfile
import threading
import time

import numpy as np
import pytest

import toliman_service as ts
from supervisor import run_supervisor
from tests.fake_xiapi import FakeCamera, FakeImage

pytestmark = pytest.mark.integration


class FakeExecutive:
    """Accepts one service connection; records its messages."""

    def __init__(self, path):
        self.path = path
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(path)
        self.server.listen(1)
        self.server.settimeout(10)
        self.sid = None
        self.replies = {}
        self.events = []
        self.connected = threading.Event()
        self.reply_received = threading.Condition()
        self.conn = None
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        self.conn, _ = self.server.accept()
        self.connected.set()
        buf = b""
        while True:
            chunk = self.conn.recv(4096)
            if not chunk:
                return
            buf += chunk
            while len(buf) >= ts.HEADER.size:
                src, _dst, msg_type, length = ts.HEADER.unpack_from(buf)
                if len(buf) < ts.HEADER.size + length:
                    break
                payload = buf[ts.HEADER.size:ts.HEADER.size + length]
                buf = buf[ts.HEADER.size + length:]
                self.sid = src
                if msg_type == ts.EVT_LOG:
                    self.events.append(payload.decode())
                elif msg_type == ts.CMD_REP:
                    with self.reply_received:
                        self.replies[payload[:2]] = payload[2]
                        self.reply_received.notify_all()

    def command(self, req_id, command, params):
        rid = struct.pack(">H", req_id)
        payload = rid + bytes([command]) + params
        self.conn.sendall(ts.HEADER.pack(0, 2, ts.CMD_REQ, len(payload)) + payload)
        with self.reply_received:
            self.reply_received.wait_for(lambda: rid in self.replies, timeout=5)
        return self.replies.get(rid)


class PacedCamera(FakeCamera):
    """Delivers frames at about 100 Hz, like a real camera."""

    def get_image(self, img, timeout=1000):
        time.sleep(0.01)
        super().get_image(img, timeout)


@pytest.fixture
def star_camera(monkeypatch, star_roi):
    """Makes XimeaCamera's default xiapi deliver star_roi frames."""
    class StarImage(FakeImage):
        def __init__(self):
            super().__init__()
            self.set_data(star_roi)

    xiapi = sys.modules["ximea.xiapi"]
    monkeypatch.setattr(xiapi, "Camera", PacedCamera, raising=False)
    monkeypatch.setattr(xiapi, "Image", StarImage, raising=False)
    monkeypatch.setattr(xiapi, "Xi_error", Exception, raising=False)


@pytest.fixture
def executive():
    # AF_UNIX paths are limited to 108 bytes, which pytest's tmp_path can exceed
    with tempfile.TemporaryDirectory() as d:
        yield FakeExecutive(os.path.join(d, "executive.sock"))


@pytest.fixture
def restore_signals():
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield
    for s, h in saved.items():
        signal.signal(s, h)


@pytest.fixture
def cfg_path(system_config_dict, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(system_config_dict))
    return path


def _start(session_id, n_frames=-1, duration=0.0, diag=0):
    return struct.pack("<BfIifB32sHH", 0, 100.0, 2_500, n_frames, duration, diag, session_id, 128, 128)


def test_start_runs_a_session(star_camera, executive, cfg_path, tmp_data_dir, restore_signals):
    params = _start(b"e2e001", n_frames=5, diag=1)

    def obc():
        assert executive.connected.wait(10)
        assert executive.command(1, 0x02, b"") == 0x05   # STOP: nothing running
        assert executive.command(2, 0x01, b"\0") == 0x02  # START: wrong length
        assert executive.command(3, 0x01, params) == 0x00

    t = threading.Thread(target=obc)
    t.start()
    code = run_supervisor(str(cfg_path), socket_path=executive.path, once=True)
    t.join(10)
    executive.thread.join(5)

    assert code == 0
    assert executive.sid == 2
    assert any(e.startswith("START e2e001") for e in executive.events)
    status = [e for e in executive.events if e.startswith("session e2e001")]
    assert status and status[0].startswith("session e2e001 ok: 5 frames"), executive.events

    with open(os.path.join(tmp_data_dir, "e2e001_session.json")) as f:
        record = json.load(f)
    assert record["success"] and record["frames_acquired"] == 5
    first = record["first_frame"]
    assert first["clock_offset_ns"] == first["unix_time_ns"] - first["timestamp_ns"]
    assert record["start_unix_ns"] <= first["unix_time_ns"] <= record["last_frame"]["unix_time_ns"]

    meta_path = os.path.join(tmp_data_dir, "e2e001_meta.jsonl")
    with open(meta_path) as f:
        meta = [json.loads(line) for line in f]
    assert len(meta) == 5
    assert all(m["unix_time_ns"] >= record["start_unix_ns"] for m in meta)

    diag = np.load(os.path.join(tmp_data_dir, "e2e001_diagnostics.npy"))
    assert list(diag["unix_time_ns"]) == [m["unix_time_ns"] for m in meta]


def test_stop_ends_the_session(star_camera, executive, cfg_path, restore_signals):
    def obc():
        assert executive.connected.wait(10)
        assert executive.command(1, 0x01, _start(b"stop001", duration=30.0)) == 0x00
        time.sleep(0.5)
        assert executive.command(2, 0x01, _start(b"stop002", duration=30.0)) == 0x04  # busy
        assert executive.command(3, 0x02, b"") == 0x00

    t = threading.Thread(target=obc)
    t0 = time.monotonic()
    t.start()
    code = run_supervisor(str(cfg_path), socket_path=executive.path, once=True)
    t.join(10)
    executive.thread.join(5)

    assert code == 0
    assert time.monotonic() - t0 < 10
    assert "STOP: ending the session" in executive.events
    assert any(e.startswith("session stop001 ok:") for e in executive.events), executive.events


def test_sigterm_stops_the_session_and_reports_it(star_camera, executive, cfg_path, restore_signals):
    def obc():
        assert executive.connected.wait(10)
        assert executive.command(1, 0x01, _start(b"term001", duration=30.0)) == 0x00
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    t = threading.Thread(target=obc)
    t.start()
    code = run_supervisor(str(cfg_path), socket_path=executive.path)
    t.join(10)
    executive.thread.join(5)

    assert code == 0
    assert any(e.startswith("session term001 ok:") for e in executive.events), executive.events
