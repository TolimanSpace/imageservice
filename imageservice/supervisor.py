"""
supervisor.py — Supervisor process for the Toliman image acquisition system.

The supervisor is the top-level entry point for the image acquisition system.
It is responsible for:

  1. Loading SystemConfig from the JSON config file at startup
  2. Connecting to the SEC executive's Unix socket as the Camera service
     (SID 2) and receiving session commands the OBC sends on CSP port 23
  3. Validating the command and constructing a SessionConfig
  4. Launching the acquisition process (acquisition.py) as a subprocess
  5. Monitoring the subprocess and handling its AcquisitionResult
  6. Reporting session status and faults back to the OBC as events
  7. Logging all events to a session log file

Lifecycle
---------
The supervisor stays resident and runs one imaging session per START
command. On the SEC it runs as the systemd unit toliman-camera.service,
which the OBC starts, stops and restarts with the SID 0 Start/Stop/Reset
Payload commands.

    OBC sends START → supervisor launches acquisition subprocess
                    → acquisition runs until n_frames / duration_s / STOP
                    → supervisor receives AcquisitionResult
                    → supervisor sends the status as an event to the OBC
                    → supervisor waits for the next START

A camera fault (camera SEL, unrecoverable error) ends the session with a
fault status; the supervisor then waits for the next START, which reopens
the camera (after the OBC has power cycled it). With --once the supervisor
exits after one session, with a non-zero exit code on failure.

Executive interface (SID 2)
---------------------------
Only the executive speaks CSP on the SEC. Payload services connect to its
Unix socket (TOLIMAN_IPC_SOCKET, default /tmp/toliman-executive.sock) with
toliman_service.py, a copy of the SEC repository's Python service template.
The executive forwards port 23 commands with SID 2 and stores the events in
its event log, which the OBC downloads.

Port 23 commands (SID 2):

  0x01 START  54 bytes, little endian (the trailing ROI size may be omitted):
      uint8    mode           0=SINGLE_ROI, 1=MULTI_ROI, 2=FULL_FRAME
      float32  frame_rate_hz
      uint32   exposure_us
      int32    n_frames       -1 = no frame limit
      float32  duration_s     0.0 = no time limit
      uint8    diagnostics    0 or 1
      char[32] session_id     NUL-padded; letters, digits, '_', '-', '.'
      uint16   roi_width      SINGLE_ROI only; 0 = default
      uint16   roi_height     SINGLE_ROI only; 0 = default
  0x02 STOP   no parameters: ends the running session after the current
              frame; the session reports its status as usual.

Reply codes: 0x00 accepted, 0x01 unknown command, 0x02 wrong length,
0x03 invalid parameters, 0x04 a session is already running, 0x05 no session
is running (STOP). START replies as soon as the session is accepted; the
outcome follows as an event.

Usage
-----
    python3 supervisor.py --config /etc/imageservice/config.json

    # Run one session, then exit
    python3 supervisor.py --config /etc/imageservice/config.json --once

    # With timeout: exit if no START arrives within 60 s
    python3 supervisor.py --config /etc/imageservice/config.json --timeout 60

    # Dry run: validate config without connecting or launching acquisition
    python3 supervisor.py --config /etc/imageservice/config.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple


import toliman_service as ts
from workers.acquisition import AcquisitionResult, run_acquisition
from workers.camera import CameraMode, RoiDefinition
from workers.config import ConfigError, SessionConfig, SystemConfig

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Command and status dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SessionCommand:
    """
    A parsed START command received from the OBC via CSP.

    Attributes
    ----------
    session_id : str
        Unique identifier for this imaging session, assigned by the OBC.
    mode : CameraMode
        Acquisition mode (SINGLE_ROI, MULTI_ROI, FULL_FRAME).
    frame_rate_hz : float
        Target frame rate in Hz.
    exposure_us : int
        Exposure time in microseconds.
    n_frames : int or None
        Number of frames to acquire, or None for time-limited session.
    duration_s : float or None
        Session duration in seconds, or None for frame-limited session.
    enable_diagnostics : bool
        Whether to collect per-frame timing diagnostics.
    """
    session_id: str
    mode: CameraMode
    frame_rate_hz: float
    exposure_us: int
    n_frames: Optional[int]
    duration_s: Optional[float]
    roi_width: Optional[int] = None
    roi_height: Optional[int] = None
    enable_diagnostics: bool = False


@dataclass
class SessionStatus:
    """
    Session outcome reported to the OBC via CSP.

    Attributes
    ----------
    session_id : str
    success : bool
        True if the session completed normally.
    fault : bool
        True if the session ended due to a camera fault (e.g. SEL).
    fault_message : str or None
        Human-readable fault description if fault=True.
    frames_acquired : int
    frames_dropped : int
    elapsed_s : float
    actual_rate_hz : float
        Approximate actual frame rate (frames_acquired / elapsed_s).
    """
    session_id: str
    success: bool
    fault: bool
    fault_message: Optional[str]
    frames_acquired: int
    frames_dropped: int
    elapsed_s: float
    actual_rate_hz: float

    @classmethod
    def from_result(cls, result: AcquisitionResult) -> "SessionStatus":
        buffer_dropped = sum(result.frames_dropped_buffer.values())
        actual_rate = (
            result.frames_acquired / result.elapsed_s
            if result.elapsed_s > 0 else 0.0
        )
        return cls(
            session_id=result.session_id,
            success=not result.fault,
            fault=result.fault,
            fault_message=result.fault_message,
            frames_acquired=result.frames_acquired,
            frames_dropped=buffer_dropped,
            elapsed_s=result.elapsed_s,
            actual_rate_hz=actual_rate,
        )



# ---------------------------------------------------------------------------
# Executive interface (SID 2 on the executive's Unix socket)
# ---------------------------------------------------------------------------

SID_CAMERA = 2

CMD_START = 0x01
CMD_STOP  = 0x02

# Service-defined reply codes (0x02-0xEF, docs/icd.md "Camera")
WRONG_LENGTH   = 0x02
INVALID_PARAMS = 0x03
BUSY           = 0x04
NOT_RUNNING    = 0x05

# START parameters; the ROI size at the end is optional
_START = struct.Struct("<BfIifB32s")
_START_ROI = struct.Struct("<HH")

_MODES = {
    0: CameraMode.SINGLE_ROI,
    1: CameraMode.MULTI_ROI,
    2: CameraMode.FULL_FRAME,
}

# session_id ends up in file names in data_dir and log_dir
_SESSION_ID = re.compile(r"[A-Za-z0-9_\-][A-Za-z0-9_.\-]*")

# Keeps an event line short in the executive's event log
_EVENT_MAX = 200


class CommandTimeout(Exception):
    """Raised by ExecutiveInterface.receive_command() if no command arrives in time."""


class CommandError(Exception):
    """A rejected command; `code` is the port 23 reply."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def parse_start_command(params: bytes) -> SessionCommand:
    """
    Deserialise the START parameters (see the module docstring) into a
    SessionCommand. Raises CommandError with the reply code if they are
    malformed; range checks against the system config happen in
    command_to_session_config().
    """
    if len(params) not in (_START.size, _START.size + _START_ROI.size):
        raise CommandError(
            WRONG_LENGTH,
            f"START has {len(params)} bytes, expected {_START.size} or "
            f"{_START.size + _START_ROI.size}",
        )
    mode, rate, exposure, n_frames, duration, diag, raw_id = _START.unpack_from(params)
    roi_w, roi_h = (
        _START_ROI.unpack_from(params, _START.size)
        if len(params) > _START.size else (0, 0)
    )

    if mode not in _MODES:
        raise CommandError(INVALID_PARAMS, f"unknown mode {mode}")
    if not math.isfinite(rate) or not math.isfinite(duration) or duration < 0:
        raise CommandError(INVALID_PARAMS, "frame rate or duration out of range")
    if n_frames == 0 or n_frames < -1:
        raise CommandError(INVALID_PARAMS, f"n_frames {n_frames}")
    if diag not in (0, 1):
        raise CommandError(INVALID_PARAMS, f"diagnostics flag {diag}")
    session_id = raw_id.split(b"\0", 1)[0].decode("ascii", "replace")
    if not _SESSION_ID.fullmatch(session_id):
        raise CommandError(INVALID_PARAMS, f"session id {session_id!r}")

    return SessionCommand(
        session_id=session_id,
        mode=_MODES[mode],
        frame_rate_hz=rate,
        exposure_us=exposure,
        n_frames=None if n_frames == -1 else n_frames,
        duration_s=None if duration == 0.0 else duration,
        roi_width=roi_w or None,
        roi_height=roi_h or None,
        enable_diagnostics=bool(diag),
    )


class ExecutiveInterface(ts.Service):
    """
    The Camera service (SID 2) on the executive's Unix socket.

    toliman_service.run() keeps the connection (and reconnects after an
    executive restart) on a background thread and calls command() for each
    port 23 command. START is validated there and handed to the supervisor's
    main loop through receive_command(); the reply goes back at once, since
    the OBC gives up on a reply after 5 s.

    Parameters
    ----------
    system : SystemConfig
        Used to validate START against the sensor geometry.
    socket_path : str, optional
        The executive's socket; default TOLIMAN_IPC_SOCKET or
        /tmp/toliman-executive.sock.
    """

    def __init__(self, system: SystemConfig, socket_path: Optional[str] = None) -> None:
        self._system = system
        self._socket_path = socket_path
        self._ctx = ts.Context()
        self._commands: "queue.Queue[Tuple[SessionCommand, SessionConfig]]" = queue.Queue()
        self._lock = threading.Lock()
        self._busy = False
        self._disconnect = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Shared with the acquisition process; set by STOP and on shutdown
        self.stop_session = multiprocessing.Event()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=ts.run,
            args=(SID_CAMERA, self),
            kwargs={
                "socket_path": self._socket_path,
                "stop": self._disconnect,
                "ctx": self._ctx,
            },
            name="executive-ipc",
            daemon=True,
        )
        self._thread.start()

    def close(self, flush_s: float = 1.0) -> None:
        """Gives queued events (the last status) flush_s to go out, then disconnects."""
        deadline = time.monotonic() + flush_s
        while not self._ctx._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.1)   # the last event may be dequeued but not yet sent
        self._disconnect.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # --- toliman_service.Service ---

    def command(self, ctx: ts.Context, command: int, params: bytes) -> int:
        if command == CMD_START:
            return self._start(ctx, params)
        if command == CMD_STOP:
            if params:
                return WRONG_LENGTH
            with self._lock:
                if not self._busy:
                    return NOT_RUNNING
                self.stop_session.set()
            ctx.event("STOP: ending the session")
            return ts.OK
        logger.warning("Unknown command 0x%02X", command)
        return ts.ERROR

    def connected(self, ctx: ts.Context) -> None:
        logger.info("Connected to the executive as SID %d.", SID_CAMERA)

    def _start(self, ctx: ts.Context, params: bytes) -> int:
        try:
            command = parse_start_command(params)
            session = command_to_session_config(command, self._system)
        except CommandError as e:
            logger.warning("START rejected: %s", e)
            ctx.event(f"START rejected: {e}"[:_EVENT_MAX])
            return e.code
        except (ConfigError, ValueError) as e:
            logger.warning("START rejected: %s", e)
            ctx.event(f"START rejected: {e}"[:_EVENT_MAX])
            return INVALID_PARAMS
        with self._lock:
            if self._busy:
                return BUSY
            self._busy = True
            self.stop_session.clear()
        self._commands.put((command, session))
        ctx.event(
            f"START {command.session_id}: {command.mode.name} "
            f"{command.frame_rate_hz:g} Hz {command.exposure_us} us "
            f"n_frames={command.n_frames} duration={command.duration_s}"[:_EVENT_MAX]
        )
        return ts.OK

    # --- used by the supervisor main loop ---

    def receive_command(self, timeout_s: Optional[float]) -> Tuple[SessionCommand, SessionConfig]:
        """
        Block until a START has been accepted, for at most timeout_s seconds
        (None = indefinitely). Raises CommandTimeout.
        """
        try:
            return self._commands.get(timeout=timeout_s)
        except queue.Empty:
            raise CommandTimeout() from None

    def session_finished(self) -> None:
        """Allows the next START."""
        with self._lock:
            self._busy = False

    def send_status(self, status: SessionStatus) -> None:
        """Report a session outcome to the OBC as an event."""
        logger.info(
            "SESSION STATUS: session=%s success=%s fault=%s "
            "frames=%d dropped=%d elapsed=%.1fs rate=%.2fHz",
            status.session_id, status.success, status.fault,
            status.frames_acquired, status.frames_dropped,
            status.elapsed_s, status.actual_rate_hz,
        )
        if status.fault:
            logger.critical("CAMERA FAULT: %s", status.fault_message)
        outcome = "ok" if status.success else ("FAULT" if status.fault else "failed")
        line = (
            f"session {status.session_id} {outcome}: "
            f"{status.frames_acquired} frames, {status.frames_dropped} dropped, "
            f"{status.elapsed_s:.1f} s, {status.actual_rate_hz:.2f} Hz"
        )
        if status.fault_message:
            line += f"; {status.fault_message}"
        self._ctx.event(line[:_EVENT_MAX])


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def command_to_session_config(
    command: SessionCommand,
    system: SystemConfig,
) -> SessionConfig:
    """
    Convert a SessionCommand from the OBC into a SessionConfig.

    For SINGLE_ROI and FULL_FRAME, ROI geometry comes from SystemConfig.
    For MULTI_ROI, ROI geometry is built from SystemConfig.multi_roi via
    make_rois().

    Parameters
    ----------
    command : SessionCommand
    system : SystemConfig

    Returns
    -------
    SessionConfig

    Raises
    ------
    ConfigError
        If the command parameters are invalid or incompatible with
        the system configuration.
    """
    from workers.camera import make_rois

    if command.mode is CameraMode.SINGLE_ROI:
        # Use ROI dimensions from the command if provided otherwise
        # default to a centred 128x128 region on the sensor
        roi_w = command.roi_width if command.roi_width else 864
        roi_h = command.roi_height if command.roi_height else 864
        offset_x = (system.sensor_width - roi_w) // 2
        offset_y = (system.sensor_height - roi_h) // 2
        rois = [RoiDefinition(
            height = roi_h,
            offset_y = offset_y,
            width = roi_w,
            offset_x = offset_x,
            label = "single_roi",
            is_data  = True,
        )]

    elif command.mode is CameraMode.MULTI_ROI:
        # Build the default geometry spanning the full sensor
        # TODO: make configuratble
        rois = make_rois(
            sensor_width = system.sensor_width,
            sensor_height = system.sensor_height,
            offset_x = 0,
            offset_y = 0,
            sidelobe_width = 360,
            sidelobe_height = 360,
            core_width = 128,
            core_height = 128,
            full_width = system.sensor_width,
            full_height = system.sensor_height,
        )

    else:  # FULL_FRAME
        rois = None

    return SessionConfig(
        mode=command.mode,
        rois=rois,
        frame_rate_hz=command.frame_rate_hz,
        exposure_us=command.exposure_us,
        n_frames=command.n_frames,
        duration_s=command.duration_s,
        session_id=command.session_id,
    )



def _session_log_handler(system: SystemConfig, session_id: str) -> logging.Handler:
    """Add a DEBUG file log for one session; remove it with _end_session_log()."""
    os.makedirs(system.log_dir, exist_ok=True)
    log_path = os.path.join(
        system.log_dir, f"{session_id}_supervisor.log"
    )
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    ))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    return handler


def _end_session_log(handler: logging.Handler) -> None:
    root = logging.getLogger()
    root.removeHandler(handler)
    handler.close()
    root.setLevel(logging.INFO)


def _acquisition_entry(*args, **kwargs) -> None:
    """run_acquisition() with default signal handling (fork copies the supervisor's)."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    run_acquisition(*args, **kwargs)


# ---------------------------------------------------------------------------
# Main supervisor logic
# ---------------------------------------------------------------------------

# Time the acquisition process gets to finish after a stop request
STOP_GRACE_S = 10.0


def run_supervisor(
    config_path: str,
    socket_path: Optional[str] = None,
    command_timeout_s: float = 0.0,
    once: bool = False,
    dry_run: bool = False,
) -> int:
    """
    Main supervisor entry point. Returns an exit code (0 = success).

    Parameters
    ----------
    config_path : str
        Path to the JSON system config file.
    socket_path : str, optional
        The executive's Unix socket (default TOLIMAN_IPC_SOCKET or
        /tmp/toliman-executive.sock).
    command_timeout_s : float
        Exit if no START arrives for this many seconds (0 = wait indefinitely).
    once : bool
        Exit after one session: 0 = success, 1 = invalid or crashed session,
        2 = camera fault.
    dry_run : bool
        If True, validate the config only — do not connect or launch acquisition.
    """

    # --- Basic logging before we have a session ID ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    # Session file logs record DEBUG; the console stays at INFO
    for h in logging.getLogger().handlers:
        h.setLevel(logging.INFO)

    # --- Load system config ---
    try:
        system = SystemConfig.from_file(config_path)
    except ConfigError as e:
        logger.critical("Failed to load system config: %s", e)
        return 1
    logger.info("Supervisor starting. Config: %s", config_path)

    if dry_run:
        logger.info("Dry run complete — config OK.")
        return 0

    # --- Connect to the executive ---
    executive = ExecutiveInterface(system, socket_path)
    shutdown = threading.Event()

    def _on_signal(signum, _frame):
        logger.info("Signal %d — stopping.", signum)
        shutdown.set()
        executive.stop_session.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    executive.start()

    try:
        idle_since = time.monotonic()
        while not shutdown.is_set():
            # Poll so a signal or the idle timeout ends the wait
            try:
                command, session = executive.receive_command(timeout_s=0.5)
            except CommandTimeout:
                if (command_timeout_s > 0
                        and time.monotonic() - idle_since >= command_timeout_s):
                    logger.warning(
                        "No command received within %.0fs — exiting.",
                        command_timeout_s,
                    )
                    return 0
                continue

            try:
                code = _run_session(system, executive, command, session, shutdown)
            finally:
                executive.session_finished()
            if once:
                return code
            idle_since = time.monotonic()
        return 0
    finally:
        executive.close()


def _run_session(
    system: SystemConfig,
    executive: ExecutiveInterface,
    command: SessionCommand,
    session: SessionConfig,
    shutdown: threading.Event,
) -> int:
    """Run one accepted START to completion; returns the --once exit code."""
    logger.info(
        "Received command: session=%s mode=%s rate=%.1fHz "
        "exposure=%dus n_frames=%s duration=%s",
        command.session_id, command.mode.name,
        command.frame_rate_hz, command.exposure_us,
        command.n_frames, command.duration_s,
    )
    log_handler = _session_log_handler(system, command.session_id)
    try:
        return _supervise_acquisition(system, executive, command, session, shutdown)
    finally:
        _end_session_log(log_handler)


def _supervise_acquisition(
    system: SystemConfig,
    executive: ExecutiveInterface,
    command: SessionCommand,
    session: SessionConfig,
    shutdown: threading.Event,
) -> int:
    # --- Launch acquisition subprocess ---
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_acquisition_entry,
        args=(system, session, result_queue),
        kwargs={
            "enable_diagnostics": command.enable_diagnostics,
            "stop_event": executive.stop_session,
        },
        name=f"acquisition-{command.session_id}",
    )

    logger.info(
        "Launching acquisition process for session %s...",
        command.session_id,
    )
    t_launch = time.monotonic()
    proc.start()
    logger.info("Acquisition process started (pid=%d).", proc.pid)

    # --- Monitor subprocess ---
    result: Optional[AcquisitionResult] = None
    stop_deadline: Optional[float] = None
    try:
        # Block until the acquisition process posts its result.
        # Use a poll loop so we can log heartbeat messages, detect
        # if the process dies without posting a result, and give up on
        # it if it ignores a stop request.
        while proc.is_alive():
            try:
                result = result_queue.get(timeout=1.0)
                break
            except queue.Empty:
                now = time.monotonic()
                if shutdown.is_set() and stop_deadline is None:
                    stop_deadline = now + STOP_GRACE_S
                if stop_deadline is not None and now >= stop_deadline:
                    logger.error("Acquisition did not stop within %.0fs.", STOP_GRACE_S)
                    break
                logger.debug(
                    "Acquisition running... elapsed=%.0fs pid=%d",
                    now - t_launch, proc.pid,
                )

        # If process exited without posting a result, something went wrong
        if result is None:
            try:
                result = result_queue.get(timeout=0.5)
            except queue.Empty:
                logger.error(
                    "Acquisition process (pid=%d) exited without posting "
                    "a result (exit code=%s).",
                    proc.pid, proc.exitcode,
                )
                executive.send_status(SessionStatus(
                    session_id=command.session_id,
                    success=False, fault=True,
                    fault_message=(
                        f"Acquisition process exited unexpectedly "
                        f"(exit code={proc.exitcode})"
                    ),
                    frames_acquired=0, frames_dropped=0,
                    elapsed_s=time.monotonic() - t_launch,
                    actual_rate_hz=0.0,
                ))
                return 1

    finally:
        # Ensure process is cleaned up
        if proc.is_alive():
            proc.join(timeout=5.0)
        if proc.is_alive():
            logger.warning("Acquisition process still alive — terminating.")
            proc.terminate()
            proc.join(timeout=5.0)
            if proc.is_alive():
                logger.error("Process did not terminate — killing.")
                proc.kill()
        proc.join()

    # --- Process result ---
    logger.info(
        "Acquisition complete: frames=%d dropped=%d elapsed=%.1fs fault=%s",
        result.frames_acquired,
        sum(result.frames_dropped_buffer.values()),
        result.elapsed_s,
        result.fault,
    )

    if result.fault:
        logger.critical(
            "CAMERA FAULT: %s — OBC should command a power cycle.",
            result.fault_message,
        )

    # Save diagnostics if collected
    if result.diagnostics is not None:
        _save_diagnostics(result, system)

    # --- Report to OBC ---
    status = SessionStatus.from_result(result)
    _save_session_record(result, status, system)
    executive.send_status(status)

    return 0 if not result.fault else 2


def _save_diagnostics(result: AcquisitionResult, system: SystemConfig) -> None:
    """Save per-frame timing diagnostics to a .npy file in the data directory."""
    import numpy as np
    path = os.path.join(
        system.data_dir,
        f"{result.session_id}_diagnostics.npy",
    )
    try:
        np.save(path, result.diagnostics)
        logger.info("Diagnostics saved to: %s", path)
    except OSError as e:
        logger.error("Failed to save diagnostics: %s", e)


def _save_session_record(
    result: AcquisitionResult,
    status: SessionStatus,
    system: SystemConfig,
) -> None:
    """
    Save <session_id>_session.json in the data directory: the outcome and the
    camera-clock to Unix time reference, so frames, diagnostics and centroids
    stamped with the camera's timestamp_ns can be placed in UTC on the ground.

    clock_offset_ns = unix_time_ns - timestamp_ns, at the first and last
    frame; the two differ by the camera clock's drift over the session.
    """
    def _clock(pair):
        if pair is None:
            return None
        camera_ns, unix_ns = pair
        return {
            "timestamp_ns": camera_ns,
            "unix_time_ns": unix_ns,
            "clock_offset_ns": unix_ns - camera_ns,
        }

    record = {
        "session_id": result.session_id,
        "start_unix_ns": result.start_unix_ns,
        "elapsed_s": result.elapsed_s,
        "first_frame": _clock(result.clock_first),
        "last_frame": _clock(result.clock_last),
        "frames_acquired": result.frames_acquired,
        "frames_written": result.frames_written,
        "frames_dropped_buffer": result.frames_dropped_buffer,
        "n_centroid_none": result.n_centroid_none,
        "n_camera_errors": result.n_camera_errors,
        "n_reconnects": result.n_reconnects,
        "success": status.success,
        "fault": status.fault,
        "fault_message": status.fault_message,
    }
    path = os.path.join(system.data_dir, f"{result.session_id}_session.json")
    try:
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        logger.info("Session record saved to: %s", path)
    except OSError as e:
        logger.error("Failed to save session record: %s", e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Toliman image service supervisor (SEC Camera service, SID 2)."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to system config JSON file.",
    )
    parser.add_argument(
        "--socket", default=None,
        help="The executive's Unix socket (default: $TOLIMAN_IPC_SOCKET or "
             f"{ts.DEFAULT_SOCKET}).",
    )
    parser.add_argument(
        "--timeout", type=float, default=0.0,
        help="Exit if no START arrives for this many seconds (0 = indefinite).",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Exit after one session, with a non-zero code on failure.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config only — do not connect or launch acquisition.",
    )

    args = parser.parse_args()

    exit_code = run_supervisor(
        config_path=args.config,
        socket_path=args.socket,
        command_timeout_s=args.timeout,
        once=args.once,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
