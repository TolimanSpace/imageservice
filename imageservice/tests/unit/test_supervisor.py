"""Unit tests for the supervisor's SID 2 command handling (no socket, no camera)."""

import struct

import pytest

import toliman_service as ts
from supervisor import (
    BUSY,
    CMD_START,
    CMD_STOP,
    INVALID_PARAMS,
    NOT_RUNNING,
    WRONG_LENGTH,
    CommandError,
    CommandTimeout,
    ExecutiveInterface,
    parse_start_command,
)
from workers.camera import CameraMode

pytestmark = pytest.mark.unit


def start_params(mode=0, rate=100.0, exposure=2_500, n_frames=10, duration=0.0,
                 diag=0, session_id=b"s001", roi=(128, 128)):
    params = struct.pack("<BfIifB32s", mode, rate, exposure, n_frames, duration, diag, session_id)
    if roi is not None:
        params += struct.pack("<HH", *roi)
    return params


class TestParseStart:

    def test_full_packet(self):
        cmd = parse_start_command(start_params(diag=1))
        assert cmd.session_id == "s001"
        assert cmd.mode is CameraMode.SINGLE_ROI
        assert cmd.frame_rate_hz == 100.0
        assert cmd.exposure_us == 2_500
        assert cmd.n_frames == 10
        assert cmd.duration_s is None
        assert (cmd.roi_width, cmd.roi_height) == (128, 128)
        assert cmd.enable_diagnostics

    def test_without_roi_size(self):
        cmd = parse_start_command(start_params(mode=1, n_frames=-1, duration=5.0, roi=None))
        assert cmd.mode is CameraMode.MULTI_ROI
        assert cmd.n_frames is None
        assert cmd.duration_s == 5.0
        assert cmd.roi_width is None and cmd.roi_height is None

    def test_zero_roi_size_means_default(self):
        cmd = parse_start_command(start_params(roi=(0, 0)))
        assert cmd.roi_width is None and cmd.roi_height is None

    @pytest.mark.parametrize("params", [b"", start_params()[:-1], start_params() + b"\0"])
    def test_wrong_length(self, params):
        with pytest.raises(CommandError) as e:
            parse_start_command(params)
        assert e.value.code == WRONG_LENGTH

    @pytest.mark.parametrize("kwargs", [
        {"mode": 3},
        {"n_frames": 0},
        {"n_frames": -2},
        {"duration": -1.0},
        {"duration": float("nan")},
        {"rate": float("inf")},
        {"diag": 2},
        {"session_id": b""},
        {"session_id": b"../etc"},
        {"session_id": b".hidden"},
        {"session_id": b"a/b"},
        {"session_id": b"caf\xc3\xa9"},
    ])
    def test_invalid(self, kwargs):
        with pytest.raises(CommandError) as e:
            parse_start_command(start_params(**kwargs))
        assert e.value.code == INVALID_PARAMS


class TestExecutiveInterface:

    @pytest.fixture
    def iface(self, system_config):
        return ExecutiveInterface(system_config, socket_path="/nonexistent")

    def test_start_queues_session(self, iface):
        ctx = ts.Context()
        assert iface.command(ctx, CMD_START, start_params()) == ts.OK
        command, session = iface.receive_command(timeout_s=0)
        assert command.session_id == "s001"
        assert session.n_frames == 10
        assert session.rois[0].width == 128

    def test_second_start_is_busy_until_finished(self, iface):
        ctx = ts.Context()
        assert iface.command(ctx, CMD_START, start_params()) == ts.OK
        assert iface.command(ctx, CMD_START, start_params(session_id=b"s002")) == BUSY
        iface.session_finished()
        assert iface.command(ctx, CMD_START, start_params(session_id=b"s002")) == ts.OK

    def test_start_rejected_by_session_config(self, iface):
        # exposure longer than the frame period
        ctx = ts.Context()
        assert iface.command(ctx, CMD_START, start_params(exposure=20_000)) == INVALID_PARAMS
        with pytest.raises(CommandTimeout):
            iface.receive_command(timeout_s=0)

    def test_rejected_start_does_not_block_next(self, iface):
        ctx = ts.Context()
        assert iface.command(ctx, CMD_START, start_params(mode=9)) == INVALID_PARAMS
        assert iface.command(ctx, CMD_START, start_params()) == ts.OK

    def test_stop(self, iface):
        ctx = ts.Context()
        assert iface.command(ctx, CMD_STOP, b"") == NOT_RUNNING
        assert iface.command(ctx, CMD_START, start_params()) == ts.OK
        assert not iface.stop_session.is_set()
        assert iface.command(ctx, CMD_STOP, b"\0") == WRONG_LENGTH
        assert iface.command(ctx, CMD_STOP, b"") == ts.OK
        assert iface.stop_session.is_set()

    def test_start_clears_previous_stop(self, iface):
        ctx = ts.Context()
        iface.command(ctx, CMD_START, start_params())
        iface.command(ctx, CMD_STOP, b"")
        iface.session_finished()
        iface.command(ctx, CMD_START, start_params(session_id=b"s002"))
        assert not iface.stop_session.is_set()

    def test_unknown_command(self, iface):
        assert iface.command(ts.Context(), 0x7F, b"") == ts.ERROR
