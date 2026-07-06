"""
acquisition.py - Main acquisition loop for the Toliman imaging system.

Runs as a dedicated process (via multiprocessing) to isolate the time-critical
acquisition loop from other system processes and allow real-time scheduling
priority to be set independently.

Architecture
------------
The acquisition loop owns a single XimeaCamera and drives it at the
configured frame rate. On each frame:

  1. acquire_frame()        - blocking call to the Ximea driver
  2. compute_pointing_error() - centroid of the centre ROI
  3. publish_pointing_error() - send (dx, dy, timestamp) to serial port (stubbed)
  4. push to FrameWriter(s)  - one writer per data-bearing ROI

The loop runs until one of:
  - n_frames frames have been acquired   (if SessionConfig.n_frames is set)
  - duration_s seconds have elapsed      (if SessionConfig.duration_s is set)
  - a stop event is set externally
  - an unrecoverable camera fault occurs

Error handling
--------------
Camera errors are classified into two tiers:

  Transient  - a single get_image() timeout or transport error. The loop
               skips the frame, increments a consecutive-error counter, and
               continues. Likely causes: SET-induced SEFI, USB glitch.

  Persistent - MAX_CONSECUTIVE_ERRORS consecutive transient errors, or a
               single error that occurs during camera reconfiguration.
               The loop attempts one full camera close/reopen cycle. If the
               camera comes back, acquisition resumes from the next frame.
               If it does not, CameraFaultError is raised and the session
               ends. Likely cause at this point: Single Event Latch-up or 
               hardware damage requiring OBC-commanded power cycle.

Diagnostics
-----------
Per-frame timing diagnostics are controlled by the ENABLE_DIAGNOSTICS flag
passed to run_acquisition(). When enabled, a pre-allocated NumPy structured
array records timing breakdowns for every frame. When disabled, the
diagnostic code paths are skipped entirely.
The diagnostics array is returned in the result dict and can be saved or
analysed after the session.

IPC - serial publisher
----------------------
The serial publisher (pointing error → serial port → pointing controller)
is stubbed here as _publish_pointing_error(). This will be implemented in
serial_publisher.py once the protocol is agreed. The stub logs the pointing
error at DEBUG level so the data flow is visible during development.

Usage
-----
Typically launched via multiprocessing from a supervisor process::

    import multiprocessing as mp
    result_queue = mp.Queue()
    proc = mp.Process(
        target=run_acquisition,
        args=(system_config, session_config, result_queue),
        kwargs={"enable_diagnostics": True},
    )
    proc.start()
    proc.join()
    result = result_queue.get()
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from workers.camera import (
    AcquiredFrame,
    CameraMode,
    XimeaCamera,
)
from workers.centroid import (
    CENTRE_ROI_LABEL,
    PointingError,
    compute_pointing_error,
)
from workers.config import SessionConfig, SystemConfig
from workers.writer import FrameWriter
from workers.process_timing_logger import TimingLogger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of consecutive frame errors before attempting a camera reconnect
MAX_CONSECUTIVE_ERRORS: int = 5

# Number of reconnect attempts before declaring an unrecoverable fault
MAX_RECONNECT_ATTEMPTS: int = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CameraFaultError(Exception):
    """
    Raised when the camera cannot be recovered after persistent errors.

    This indicates a hardware-level fault (likely Latch-Up)
    that requires an OBC-commanded power cycle. The acquisition process
    exits cleanly after raising this.
    """


# ---------------------------------------------------------------------------
# Result and diagnostics types
# ---------------------------------------------------------------------------

@dataclass
class AcquisitionResult:
    """
    Summary returned to the supervisor process via the result queue.

    Attributes
    ----------
    session_id : str
    frames_acquired : int
        Total frames successfully acquired from the camera.
    frames_written : dict
        Mapping of roi_label -> frames written to disk for that ROI.
    frames_dropped_buffer : dict
        Mapping of roi_label -> frames dropped due to ring buffer overflow.
    n_centroid_none : int
        Number of frames where the centre ROI had insufficient signal.
    n_camera_errors : int
        Total transient camera errors encountered.
    n_reconnects : int
        Number of camera reconnect attempts made.
    fault : bool
        True if the session ended due to an unrecoverable camera fault.
    fault_message : str or None
        Description of the fault if fault=True.
    elapsed_s : float
        Total session wall time in seconds.
    diagnostics : np.ndarray or None
        Per-frame timing array if diagnostics were enabled; None otherwise.
        dtype fields: frame_id (i8), t_acquire_us (f4), t_centroid_us (f4),
        t_push_us (f4), t_total_us (f4), interframe_us (f4).
    """
    session_id: str
    frames_acquired: int
    frames_written: Dict[str, int]
    frames_dropped_buffer: Dict[str, int]
    n_centroid_none: int
    n_camera_errors: int
    n_reconnects: int
    fault: bool
    fault_message: Optional[str]
    elapsed_s: float
    diagnostics: Optional[np.ndarray]


# Structured dtype for per-frame diagnostics (pre-allocated)
_DIAG_DTYPE = np.dtype([
    ("frame_id",       np.int64),
    ("t_acquire_us",   np.float32),   # time inside get_image()
    ("t_centroid_us",  np.float32),   # time for centroid computation
    ("t_push_us",      np.float32),   # time to push to all writers
    ("t_total_us",     np.float32),   # total frame processing time
    ("interframe_us",  np.float32),   # wall time since previous frame
])


# ---------------------------------------------------------------------------
# Entry point (run in a subprocess)
# ---------------------------------------------------------------------------

def run_acquisition(
    system: SystemConfig,
    session: SessionConfig,
    result_queue: multiprocessing.Queue,
    enable_diagnostics: bool = False,
) -> None:
    """
    Main acquisition entry point. Intended to run in a dedicated process.

    Sets real-time scheduling priority (SCHED_FIFO) if running as root or
    with the CAP_SYS_NICE capability. Falls back gracefully if not permitted.

    Parameters
    ----------
    system : SystemConfig
        Static hardware configuration.
    session : SessionConfig
        Per-session operational parameters.
    result_queue : multiprocessing.Queue
        Queue to which the AcquisitionResult is posted on exit (success or
        failure). The supervisor process reads from this queue.
    enable_diagnostics : bool
        If True, record per-frame timing diagnostics. Adds two
        perf_counter_ns() calls per frame to the hot path.
    """
    _configure_logging(system.log_dir, session.session_id)
    logger.info(
        "Acquisition process started: session=%s  pid=%d",
        session.session_id, os.getpid(),
    )

    _try_set_realtime_priority()

    result = _run(system, session, enable_diagnostics)
    result_queue.put(result)

    logger.info(
        "Acquisition process exiting: frames=%d  fault=%s",
        result.frames_acquired, result.fault,
    )


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

def _run(
    system: SystemConfig,
    session: SessionConfig,
    enable_diagnostics: bool,
) -> AcquisitionResult:
    """
    Build the camera config, open writers, and run the acquisition loop.
    Returns an AcquisitionResult regardless of how the session ends.
    """
    camera_config = system.to_camera_config(session)
    centre_offset_x, centre_offset_y = system.centre_roi_offset(session)

    # Determine which ROI labels carry data and their shapes
    data_rois = _data_roi_info(session, system)

    # Pre-allocate diagnostics buffer (max n_frames entries, or 100k if unlimited)
    diag_capacity = session.n_frames if session.n_frames else 100_000
    diag_buf: Optional[np.ndarray] = (
        np.zeros(diag_capacity, dtype=_DIAG_DTYPE)
        if enable_diagnostics else None
    )
    diag_idx: int = 0

    # Counters
    frames_acquired:  int = 0
    n_centroid_none:  int = 0
    n_camera_errors:  int = 0
    n_reconnects:     int = 0
    fault:            bool = False
    fault_message:    Optional[str] = None
    session_start:    float = time.monotonic()

    # Open one FrameWriter per data ROI
    writers: Dict[str, FrameWriter] = {
        label: FrameWriter(
            data_dir=system.data_dir,
            session_id=session.session_id,
            roi_label=label,
            frame_shape=shape,
            buffer_n_frames=system.writer_ring_buffer_n_frames,
            flush_timeout_s=system.writer_flush_timeout_s,
        )
        for label, shape in data_rois.items()
    }

    # Acquisition-side timing logger
    acq_tlog = TimingLogger(
        component="acquisition",
        session_id=session.session_id,
        log_dir=system.log_dir,
        enabled=enable_diagnostics,
    )

    try:
        for w in writers.values():
            w.configure_diagnostics(
                session_id=session.session_id,
                log_dir=system.log_dir,
                enable=enable_diagnostics,
            )
            w.start()

        with XimeaCamera(camera_config) as cam:
            logger.info("Camera open and acquiring.")

            consecutive_errors: int = 0
            last_frame_time:    float = time.monotonic()

            while _should_continue(session, frames_acquired, session_start):

                # ---------------------------------------------------------
                # Frame acquisition
                # ---------------------------------------------------------
                t0 = time.perf_counter_ns() if enable_diagnostics else 0

                if enable_diagnostics: acq_tlog.start("acquire")
                try:
                    frame = cam.acquire_frame()
                    if enable_diagnostics: acq_tlog.end("acquire")
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    n_camera_errors += 1
                    logger.warning(
                        "Camera error (consecutive=%d): %s",
                        consecutive_errors, e,
                    )
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.error(
                            "Reached %d consecutive errors - attempting "
                            "camera reconnect.", MAX_CONSECUTIVE_ERRORS,
                        )
                        recovered, n_reconnects = _attempt_reconnect(
                            cam, camera_config, n_reconnects,
                        )
                        if not recovered:
                            fault = True
                            fault_message = (
                                f"Camera unrecoverable after "
                                f"{n_reconnects} reconnect attempt(s). "
                                f"Last error: {e}. "
                                f"Possible SEL - OBC power cycle required."
                            )
                            logger.critical(fault_message)
                            raise CameraFaultError(fault_message)
                        consecutive_errors = 0
                    continue   # skip this frame

                t_acquire = time.perf_counter_ns() if enable_diagnostics else 0
                frames_acquired += 1

                # ---------------------------------------------------------
                # Centroid and pointing error
                # ---------------------------------------------------------
                centre_roi = _get_centre_roi(frame, session)
                pointing: Optional[PointingError] = None

                if centre_roi is not None:
                    if enable_diagnostics: acq_tlog.start("centroid")
                    pointing = compute_pointing_error(
                        centre_roi=centre_roi,
                        roi_offset_x=centre_offset_x,
                        roi_offset_y=centre_offset_y,
                        sensor_width=system.sensor_width,
                        sensor_height=system.sensor_height,
                        timestamp_ns=frame.timestamp_ns,
                        frame_id=frame.frame_id,
                        min_total_intensity=system.centroid_min_total_intensity,
                    )

                if pointing is not None:
                    _publish_pointing_error(pointing)
                    if enable_diagnostics: acq_tlog.end("centroid")
                else:
                    n_centroid_none += 1
                    _publish_no_signal(frame.frame_id, frame.timestamp_ns)
                    if enable_diagnostics: acq_tlog.end("centroid")

                t_centroid = time.perf_counter_ns() if enable_diagnostics else 0

                # ---------------------------------------------------------
                # Push frames to writers
                # ---------------------------------------------------------
                if enable_diagnostics: acq_tlog.start("push")
                for label, writer in writers.items():
                    roi_arr = frame.rois.get(label)
                    if roi_arr is not None:
                        writer.push(roi_arr, frame)
                if enable_diagnostics: acq_tlog.end("push")

                t_push = time.perf_counter_ns() if enable_diagnostics else 0

                # ---------------------------------------------------------
                # Diagnostics
                # ---------------------------------------------------------
                if enable_diagnostics and diag_idx < diag_capacity:
                    now = time.monotonic()
                    interframe_us = (now - last_frame_time) * 1e6
                    last_frame_time = now
                    diag_buf[diag_idx] = (
                        frame.frame_id,
                        (t_acquire  - t0)         / 1e3,
                        (t_centroid - t_acquire)  / 1e3,
                        (t_push     - t_centroid) / 1e3,
                        (t_push     - t0)         / 1e3,
                        interframe_us,
                    )
                    diag_idx += 1
                elif not enable_diagnostics:
                    pass

    except CameraFaultError:
        pass   # fault flags already set; fall through to cleanup
    except Exception as e:
        logger.exception("Unexpected error in acquisition loop: %s", e)
        fault = True
        fault_message = f"Unexpected error: {e}"
    finally:
        acq_tlog.close()
        for w in writers.values():
            w.stop()

    elapsed_s = time.monotonic() - session_start
    actual_rate = frames_acquired / elapsed_s if elapsed_s > 0 else 0.0
    logger.info(
        "Session complete: frames=%d  elapsed=%.1f s  actual_rate=%.2f Hz",
        frames_acquired, elapsed_s, actual_rate,
    )

    return AcquisitionResult(
        session_id=session.session_id,
        frames_acquired=frames_acquired,
        frames_written={
            label: w.stats.frames_written for label, w in writers.items()
        },
        frames_dropped_buffer={
            label: w.stats.frames_dropped for label, w in writers.items()
        },
        n_centroid_none=n_centroid_none,
        n_camera_errors=n_camera_errors,
        n_reconnects=n_reconnects,
        fault=fault,
        fault_message=fault_message,
        elapsed_s=elapsed_s,
        diagnostics=(
            diag_buf[:diag_idx] if enable_diagnostics else None
        ),
    )


# ---------------------------------------------------------------------------
# Session termination predicate
# ---------------------------------------------------------------------------

def _should_continue(
    session: SessionConfig,
    frames_acquired: int,
    session_start: float,
) -> bool:
    """Return True if the acquisition loop should continue."""
    if session.n_frames is not None:
        if frames_acquired >= session.n_frames:
            logger.info(
                "Frame limit reached (%d frames).", session.n_frames
            )
            return False
    if session.duration_s is not None:
        elapsed = time.monotonic() - session_start
        if elapsed >= session.duration_s:
            logger.info(
                "Time limit reached (%.1f s).", session.duration_s
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Camera error recovery
# ---------------------------------------------------------------------------

def _attempt_reconnect(
    cam: XimeaCamera,
    camera_config,
    n_reconnects: int,
) -> tuple:
    """
    Attempt to stop, close, reopen and restart the camera.

    Returns (recovered: bool, updated_n_reconnects: int).
    """
    if n_reconnects >= MAX_RECONNECT_ATTEMPTS:
        logger.error(
            "Max reconnect attempts (%d) exhausted.", MAX_RECONNECT_ATTEMPTS
        )
        return False, n_reconnects

    n_reconnects += 1
    logger.warning(
        "Attempting camera reconnect (attempt %d/%d)...",
        n_reconnects, MAX_RECONNECT_ATTEMPTS,
    )

    try:
        cam.stop()
        cam.close()
        time.sleep(2.0)    # allow USB re-enumeration
        cam.open()
        cam.configure()
        cam.start()
        logger.info("Camera reconnect successful.")
        return True, n_reconnects
    except Exception as e:
        logger.error("Camera reconnect failed: %s", e)
        return False, n_reconnects


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def _data_roi_info(
    session: SessionConfig,
    system: SystemConfig,
) -> Dict[str, tuple]:
    """
    Return a dict of {roi_label: (n_rows, n_cols)} for all data-bearing ROIs.

    For MULTI_ROI: the five quincunx data regions.
    For SINGLE_ROI: the single ROI.
    For FULL_FRAME: the full sensor frame, using sensor dimensions from
    SystemConfig. Full frames are saved to disk for alignment verification
    and background flux characterisation.
    """
    if session.mode is CameraMode.MULTI_ROI:
        return {
            roi.label: (roi.height, roi.width)
            for roi in session.rois
            if roi.is_data
        }
    elif session.mode is CameraMode.SINGLE_ROI:
        roi = session.rois[0]
        return {roi.label: (roi.height, roi.width)}
    else:  # FULL_FRAME
        return {
            "full_frame": (system.sensor_height, system.sensor_width)
        }


def _get_centre_roi(
    frame: AcquiredFrame,
    session: SessionConfig,
) -> Optional[np.ndarray]:
    """
    Extract the centre ROI array from a frame, regardless of camera mode.

    For MULTI_ROI: returns frame.rois[CENTRE_ROI_LABEL].
    For SINGLE_ROI and FULL_FRAME: returns the single available array,
    since in these modes the entire image is the "centre" for centroiding.
    """
    if session.mode is CameraMode.MULTI_ROI:
        return frame.rois.get(CENTRE_ROI_LABEL)
    else:
        # For SINGLE_ROI / FULL_FRAME there is exactly one entry
        if frame.rois:
            return next(iter(frame.rois.values()))
        return None


# ---------------------------------------------------------------------------
# Serial publisher stub
# ---------------------------------------------------------------------------

def _publish_pointing_error(pointing: PointingError) -> None:
    """
    Publish a PointingError to the pointing controller via serial port.

    STUB - to be implemented in serial_publisher.py once the framing
    protocol is agreed with the pointing controller team.

    When implemented this will open/reuse a serial.Serial handle and
    write a framed binary packet containing (dx, dy, timestamp_ns, frame_id).
    """
    logger.debug(
        "POINTING: frame_id=%d  dx=%.4f  dy=%.4f  ts=%d  "
        "intensity=%.1f  peak=%d",
        pointing.frame_id, pointing.dx, pointing.dy,
        pointing.timestamp_ns,
        pointing.centroid.total_intensity,
        pointing.centroid.peak_value,
    )


def _publish_no_signal(frame_id: int, timestamp_ns: int) -> None:
    """
    Publish a sentinel to the pointing controller indicating no signal.

    STUB - to be implemented alongside _publish_pointing_error().

    The sentinel should be a distinguished packet value that the pointing
    controller recognises as "star not detected - hold last good pointing".
    """
    logger.debug(
        "POINTING: frame_id=%d  ts=%d  NO SIGNAL",
        frame_id, timestamp_ns,
    )


# ---------------------------------------------------------------------------
# Process setup helpers
# ---------------------------------------------------------------------------

def _try_set_realtime_priority() -> None:
    """
    Attempt to set SCHED_FIFO real-time scheduling for this process.

    Requires CAP_SYS_NICE or root. Logs a warning if not permitted rather
    than raising - the acquisition loop will still function, but with
    normal scheduler jitter.
    """
    try:
        param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        logger.info("Real-time scheduling (SCHED_FIFO) set successfully.")
    except (AttributeError, PermissionError, OSError) as e:
        logger.warning(
            "Could not set real-time scheduling priority: %s  "
            "(acquisition will run at normal priority).", e,
        )


def _configure_logging(log_dir: str, session_id: str) -> None:
    """
    Configure a file handler for this process's logger.

    Each acquisition session gets its own log file so that logs from
    concurrent diagnostic sessions don't interleave.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{session_id}_acquisition.log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
    ))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)