"""
supervisor.py — Supervisor process for the Toliman image acquisition system.

The supervisor is the top-level entry point for the image acquisition system.
It is responsible for:

  1. Loading SystemConfig from the JSON config file at startup
  2. Listening for session commands from the OBC via CSP
  3. Validating the command and constructing a SessionConfig
  4. Launching the acquisition process (acquisition.py) as a subprocess
  5. Monitoring the subprocess and handling its AcquisitionResult
  6. Reporting session status and faults back to the OBC via CSP
  7. Logging all events to a session log file

Lifecycle
---------
The supervisor runs a single imaging session per invocation and then exits.
It is expected to be launched by the OBC (or a higher-level process manager)
for each imaging session.

    OBC sends START command → supervisor launches acquisition subprocess
                            → acquisition runs until n_frames / duration_s
                            → supervisor receives AcquisitionResult
                            → supervisor sends STATUS to OBC
                            → supervisor exits

If the acquisition subprocess faults (camera SEL, unrecoverable error), the
supervisor reports the fault to the OBC and exits with a non-zero exit code.

CSP interface (stubbed)
-----------------------
The CspInterface class below is a stub. It defines the API that the real
implementation must satisfy:

  receive_command() → SessionCommand
      Block until a START command arrives from the OBC, parse it into a
      SessionCommand, and return it.  Raise CommandTimeout if no command
      arrives within the configured timeout.

  send_status(status: SessionStatus) → None
      Send a status message to the OBC. Called after session completion
      (success or fault) and optionally at intermediate checkpoints.

Todo:
  1. Fill in CspInterface.__init__() to initialise the CSP node with
     node address and port assignment
  2. Implement receive_command() to deserialise the command packet
     format into a SessionCommand
  3. Implement send_status() to serialise SessionStatus into the
     telemetry packet format

Usage
-----
    python3 supervisor.py --config /etc/imageservice/config.json

    # With timeout: wait up to 60s for a START command, then exit
    python3 supervisor.py --config /etc/imageservice/config.json --timeout 60

    # Dry run: validate config and CSP interface without launching acquisition
    python3 supervisor.py --config /etc/imageservice/config.json --dry-run
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Optional


from workers.acquisition import AcquisitionResult, CameraFaultError, run_acquisition
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
# CSP interface — STUB
# ---------------------------------------------------------------------------

class CommandTimeout(Exception):
    """Raised by CspInterface.receive_command() if no command arrives in time."""


class CspInterface:
    """
    Interface to the CubeSat Space Protocol stack.

    THIS IS A STUB. The methods below define the API; the implementation
    must be filled in once mission parameters are agreed (node address, 
    port assignments, packet format).

    Parameters
    ----------
    node_address : int
        This node's CSP address. To be assigned during mission integration.
    command_port : int
        CSP port to listen on for incoming START commands from the OBC.
    telemetry_port : int
        CSP port to send status messages to on the OBC node.
    obc_address : int
        CSP address of the OBC node.
    command_timeout_s : float
        How long to wait for a command before raising CommandTimeout.
    """

    def __init__(
        self,
        node_address: int,
        command_port: int,
        telemetry_port: int,
        obc_address: int,
        command_timeout_s: float = 0.0,
    ) -> None:
        self._node_address      = node_address
        self._command_port      = command_port
        self._telemetry_port    = telemetry_port
        self._obc_address       = obc_address
        self._command_timeout_s = command_timeout_s

        logger.info(
            "CspInterface stub initialised: node=%d command_port=%d "
            "telemetry_port=%d obc=%d",
            node_address, command_port, telemetry_port, obc_address,
        )

        # TODO: initialise libcsp
        # import csp
        # csp.init(node_address, "imageservice", "1.0", 10, 300, 10)
        # csp.zmqhub_init(node_address, "localhost")  # or appropriate interface
        # csp.rtable_load("0/0 ZMQHUB")
        # csp.route_start_task()

    def receive_command(self) -> SessionCommand:
        """
        Block until a START command packet arrives from the OBC.

        Returns a parsed SessionCommand.
        Raises CommandTimeout if no command arrives within command_timeout_s
        (or blocks indefinitely if command_timeout_s == 0).

        STUB — returns a hardcoded test command for development.

        TODO: replace with real CSP receive logic:
            socket = csp.socket()
            csp.bind(socket, self._command_port)
            csp.listen(socket, 10)
            conn = csp.accept(socket, int(self._command_timeout_s * 1000))
            if conn is None:
                raise CommandTimeout(...)
            packet = csp.read(conn, 1000)
            return self._parse_command_packet(packet.data)
        """
        logger.warning(
            "CspInterface.receive_command() is a STUB — "
            "returning hardcoded test command."
        )
        # Stub: return a test command immediately
        return SessionCommand(
            session_id=f"stub_{int(time.time())}",
            mode=CameraMode.SINGLE_ROI,
            frame_rate_hz=100.0,
            exposure_us=5_000,
            n_frames=None,
            duration_s=60,
            enable_diagnostics=True,
        )

    def send_status(self, status: SessionStatus) -> None:
        """
        Send a SessionStatus message to the OBC.

        STUB — logs the status to file only.

        TODO: replace with real CSP send logic:
            packet_bytes = self._serialise_status(status)
            csp.transaction(
                self._obc_address,
                self._telemetry_port,
                1000,  # timeout ms
                packet_bytes, len(packet_bytes),
                None, 0,
            )
        """
        logger.warning(
            "CspInterface.send_status() is a STUB — "
            "status not transmitted to OBC."
        )
        logger.info(
            "SESSION STATUS: session=%s success=%s fault=%s "
            "frames=%d dropped=%d elapsed=%.1fs rate=%.2fHz",
            status.session_id, status.success, status.fault,
            status.frames_acquired, status.frames_dropped,
            status.elapsed_s, status.actual_rate_hz,
        )
        if status.fault:
            logger.critical(
                "CAMERA FAULT: %s", status.fault_message
            )

    def _parse_command_packet(self, data: bytes) -> SessionCommand:
        """
        Deserialise a raw CSP packet into a SessionCommand.

        STUB — packet format to be defined during mission integration.

        Suggested format (struct, little-endian):
            uint8   mode          (0=SINGLE_ROI, 1=MULTI_ROI, 2=FULL_FRAME)
            float32 frame_rate_hz
            uint32  exposure_us
            int32   n_frames      (-1 = None)
            float32 duration_s    (0.0 = None)
            uint8   diagnostics   (0 or 1)
            char[32] session_id   (null-terminated)
        """
        raise NotImplementedError(
            "_parse_command_packet() must be implemented once the "
            "CSP packet format is agreed with the OBC team."
        )

    def _serialise_status(self, status: SessionStatus) -> bytes:
        """
        Serialise a SessionStatus into a raw CSP telemetry packet.

        STUB — packet format to be defined during mission integration.
        """
        raise NotImplementedError(
            "_serialise_status() must be implemented once the "
            "CSP telemetry format is agreed with the OBC team."
        )


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
        roi_w = command.roi_width if command.roi_width else 848
        roi_h = command.roi_height if command.roi_height else 848
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


def _configure_logging(system: SystemConfig, session_id: str) -> None:
    """Set up file logging for this supervisor session."""
    os.makedirs(system.log_dir, exist_ok=True)
    log_path = os.path.join(
        system.log_dir, f"{session_id}_supervisor.log"
    )
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    ))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    # Also print INFO and above to stdout for local monitoring
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s"
    ))
    root.addHandler(console)


# ---------------------------------------------------------------------------
# Main supervisor logic
# ---------------------------------------------------------------------------

def run_supervisor(
    config_path: str,
    csp_node_address: int,
    csp_command_port: int,
    csp_telemetry_port: int,
    csp_obc_address: int,
    command_timeout_s: float = 0.0,
    dry_run: bool = False,
) -> int:
    """
    Main supervisor entry point. Returns an exit code (0 = success).

    Parameters
    ----------
    config_path : str
        Path to the JSON system config file.
    csp_node_address : int
        This node's CSP address.
    csp_command_port : int
        CSP port to listen on for commands.
    csp_telemetry_port : int
        CSP port to send status to on the OBC.
    csp_obc_address : int
        CSP address of the OBC.
    command_timeout_s : float
        Seconds to wait for a command (0 = wait indefinitely).
    dry_run : bool
        If True, validate config and CSP init only — do not launch acquisition.
    """

    # --- Load system config ---
    try:
        system = SystemConfig.from_file(config_path)
    except ConfigError as e:
        logging.critical("Failed to load system config: %s", e)
        return 1

    # --- Basic logging before we have a session ID ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    logger.info("Supervisor starting. Config: %s", config_path)

    # --- Initialise CSP interface ---
    csp = CspInterface(
        node_address=csp_node_address,
        command_port=csp_command_port,
        telemetry_port=csp_telemetry_port,
        obc_address=csp_obc_address,
        command_timeout_s=command_timeout_s,
    )

    if dry_run:
        logger.info("Dry run complete — config and CSP interface initialised OK.")
        return 0

    # --- Wait for START command from OBC ---
    logger.info(
        "Waiting for START command on CSP port %d%s...",
        csp_command_port,
        f" (timeout {command_timeout_s:.0f}s)" if command_timeout_s > 0 else "",
    )
    try:
        command = csp.receive_command()
    except CommandTimeout:
        logger.warning(
            "No command received within %.0fs — exiting.", command_timeout_s
        )
        return 0

    logger.info(
        "Received command: session=%s mode=%s rate=%.1fHz "
        "exposure=%dus n_frames=%s duration=%s",
        command.session_id, command.mode.name,
        command.frame_rate_hz, command.exposure_us,
        command.n_frames, command.duration_s,
    )

    # --- Set up session logging ---
    _configure_logging(system, command.session_id)

    # --- Build SessionConfig ---
    try:
        session = command_to_session_config(command, system)
    except ConfigError as e:
        logger.error("Invalid session command: %s", e)
        csp.send_status(SessionStatus(
            session_id=command.session_id,
            success=False, fault=False,
            fault_message=f"Invalid command: {e}",
            frames_acquired=0, frames_dropped=0,
            elapsed_s=0.0, actual_rate_hz=0.0,
        ))
        return 1

    # --- Launch acquisition subprocess ---
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=run_acquisition,
        args=(system, session, result_queue),
        kwargs={"enable_diagnostics": command.enable_diagnostics},
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
    try:
        # Block until the acquisition process posts its result.
        # Use a poll loop so we can log heartbeat messages and detect
        # if the process dies without posting a result.
        while proc.is_alive():
            try:
                result = result_queue.get(timeout=5.0)
                break
            except Exception:
                # Queue empty after 5s — process still running, log heartbeat
                elapsed = time.monotonic() - t_launch
                logger.debug(
                    "Acquisition running... elapsed=%.0fs pid=%d",
                    elapsed, proc.pid,
                )

        # If process exited without posting a result, something went wrong
        if result is None:
            try:
                result = result_queue.get_nowait()
            except Exception:
                logger.error(
                    "Acquisition process (pid=%d) exited without posting "
                    "a result (exit code=%s).",
                    proc.pid, proc.exitcode,
                )
                status = SessionStatus(
                    session_id=command.session_id,
                    success=False, fault=True,
                    fault_message=(
                        f"Acquisition process exited unexpectedly "
                        f"(exit code={proc.exitcode})"
                    ),
                    frames_acquired=0, frames_dropped=0,
                    elapsed_s=time.monotonic() - t_launch,
                    actual_rate_hz=0.0,
                )
                csp.send_status(status)
                return 1

    finally:
        # Ensure process is cleaned up
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
    csp.send_status(status)

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Toliman image service supervisor."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to system config JSON file.",
    )
    parser.add_argument(
        "--csp-node", type=int, default=0,
        help="This node's CSP address (default: 0 — update for your mission).",
    )
    parser.add_argument(
        "--csp-command-port", type=int, default=10,
        help="CSP port to listen on for commands (default: 10).",
    )
    parser.add_argument(
        "--csp-telemetry-port", type=int, default=11,
        help="CSP port to send status to on the OBC (default: 11).",
    )
    parser.add_argument(
        "--csp-obc", type=int, default=1,
        help="CSP address of the OBC node (default: 1).",
    )
    parser.add_argument(
        "--timeout", type=float, default=0.0,
        help="Seconds to wait for a command before exiting (0 = indefinite).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and CSP init only — do not launch acquisition.",
    )

    args = parser.parse_args()

    exit_code = run_supervisor(
        config_path=args.config,
        csp_node_address=args.csp_node,
        csp_command_port=args.csp_command_port,
        csp_telemetry_port=args.csp_telemetry_port,
        csp_obc_address=args.csp_obc,
        command_timeout_s=args.timeout,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
