"""
writer.py - Asynchronous frame writer with in-memory ring buffer.

Decouples the time-critical acquisition loop from disk I/O by buffering
frames in a fixed-size in-memory ring buffer and draining it in a
background writer thread.

Architecture
------------
The ring buffer is a pre-allocated array of NumPy uint16 frames in RAM.
Two integer indices track the state:

  _write_idx  : next slot for the acquisition loop to write into
  _read_idx   : next slot for the writer thread to consume

When the buffer is full (_write_idx has lapped _read_idx), the oldest
unread slot is overwritten and a drop counter is incremented. This keeps
the acquisition loop non-blocking at the cost of losing the oldest
buffered frame. A warning is logged on each drop.

Output files (written per session)
-----------------------------------
  <data_dir>/<session_id>_frames.bin
      Raw uint16 pixel data written sequentially, one frame after another.
      Frame dimensions are recorded in the metadata file; the binary file
      itself has no header. Frames can be re-read as::

          frames = numpy.fromfile(path, dtype=numpy.uint16)
          frames = frames.reshape((-1, n_rows, n_cols))

  <data_dir>/<session_id>_meta.jsonl
      One JSON object per line (JSON Lines format), one line per written
      frame, in the same order as the binary file. Each line contains::

          {
            "frame_id":        int,    # acq_nframe from AcquiredFrame
            "timestamp_ns":    int,    # camera hardware timestamp
            "host_time":       float,  # time.monotonic() at acquisition
            "nframes_dropped": int,    # dropped frames before this one
            "roi_label":       str,    # which ROI this frame came from
            "n_rows":          int,    # frame height in pixels
            "n_cols":          int,    # frame width in pixels
            "session_id":      str,
          }

Thread safety
-------------
The ring buffer uses a threading.Lock to protect the shared indices. The
lock is held only for index arithmetic (microseconds), never during the
actual frame copy or disk write, so it does not introduce meaningful
latency into the acquisition loop.

Usage
-----
::

    writer = FrameWriter(
        data_dir="/data/imageservice",
        session_id="obs_2026_001",
        roi_label="centre",
        frame_shape=(64, 128),
        buffer_n_frames=256,
    )

    with writer:
        for frame in camera:
            writer.push(frame.rois["centre"], frame)

    # On __exit__ the writer flushes remaining frames and closes files.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from workers.camera import AcquiredFrame
from workers.process_timing_logger import TimingLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Writer statistics (returned by writer.stats property)
# ---------------------------------------------------------------------------

@dataclass
class WriterStats:
    """
    Snapshot of writer performance metrics.

    Attributes
    ----------
    frames_pushed : int
        Total frames pushed into the ring buffer by the acquisition loop.
    frames_written : int
        Total frames successfully written to disk.
    frames_dropped : int
        Frames dropped due to ring buffer overflow (oldest overwritten).
    buffer_n_frames : int
        Total ring buffer capacity in frames.
    buffer_occupancy : int
        Number of frames currently waiting in the buffer.
    write_errors : int
        Number of disk write errors encountered (non-fatal; frame skipped).
    """
    frames_pushed: int
    frames_written: int
    frames_dropped: int
    buffer_n_frames: int
    buffer_occupancy: int
    write_errors: int


# ---------------------------------------------------------------------------
# FrameWriter
# ---------------------------------------------------------------------------

class FrameWriter:
    """
    Asynchronous frame writer with in-memory ring buffer.

    Receives frames from the acquisition loop via push(), buffers them in
    RAM, and drains the buffer to disk in a background thread.

    Parameters
    ----------
    data_dir : str
        Directory for output files. Must exist and be writable.
    session_id : str
        Session identifier used as a filename prefix.
    roi_label : str
        Label of the ROI being written (e.g. "centre", "single_roi").
        Recorded in the metadata file per frame.
    frame_shape : tuple of (int, int)
        Expected (n_rows, n_cols) of each frame. Frames with a different
        shape are rejected by push() with a warning.
    buffer_n_frames : int
        Ring buffer capacity in frames. Should be large enough to absorb
        disk I/O bursts. At 100 Hz and 64x4504 uint16 frames:
          - each frame ≈ 578 KB
          - 256 frames ≈ 144 MB RAM
    flush_timeout_s : float
        Maximum time in seconds to wait for the writer thread to drain the
        buffer on close(). Frames remaining after timeout are discarded
        with a warning.
    """

    def __init__(
        self,
        data_dir: str,
        session_id: str,
        roi_label: str,
        frame_shape: Tuple[int, int],
        buffer_n_frames: int = 256,
        flush_timeout_s: float = 5.0,
    ) -> None:
        self._data_dir       = data_dir
        self._session_id     = session_id
        self._roi_label      = roi_label
        self._frame_shape    = frame_shape
        self._buffer_n       = buffer_n_frames
        self._flush_timeout  = flush_timeout_s

        # Pre-allocate ring buffer: shape (buffer_n_frames, n_rows, n_cols)
        n_rows, n_cols = frame_shape
        self._buffer = np.zeros(
            (buffer_n_frames, n_rows, n_cols), dtype=np.uint16
        )

        # Parallel metadata buffer - one dict slot per frame slot
        self._meta_buffer: list = [None] * buffer_n_frames

        # Ring buffer indices (protected by _lock)
        self._write_idx: int = 0   # next slot for push()
        self._read_idx:  int = 0   # next slot for writer thread
        self._occupancy: int = 0   # frames waiting to be written

        # Statistics
        self._frames_pushed:  int = 0
        self._frames_written: int = 0
        self._frames_dropped: int = 0
        self._write_errors:   int = 0

        # Logger
        self._timing_logger: Optional[TimingLogger] = None

        self._lock = threading.Lock()
        self._data_available = threading.Event()
        self._stop_event     = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # File handles (opened in start())
        self._bin_file  = None
        self._meta_file = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "FrameWriter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def configure_diagnostics(
        self,
        session_id: str,
        log_dir: str,
        enable: bool = True,
    ) -> None:

        if not enable:
            return
        self._timing_logger = TimingLogger(
            component=f"writer_{self._roi_label}",
            session_id=session_id,
            log_dir=log_dir,
            enabled=True
        )

    def start(self) -> None:
        """
        Open output files and start the background writer thread.

        Raises
        ------
        RuntimeError
            If the writer is already running.
        OSError
            If the output files cannot be created.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("FrameWriter is already running.")

        os.makedirs(self._data_dir, exist_ok=True)

        bin_path  = self._bin_path()
        meta_path = self._meta_path()

        logger.info("Opening frame file:    %s", bin_path)
        logger.info("Opening metadata file: %s", meta_path)

        self._bin_file  = open(bin_path,  "wb", buffering=0)   # unbuffered
        self._meta_file = open(meta_path, "w",  buffering=1)   # line-buffered

        self._stop_event.clear()
        self._data_available.clear()
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="frame-writer",
            daemon=True,
        )
        self._thread.start()
        logger.info("FrameWriter started (buffer=%d frames, shape=%s).",
                    self._buffer_n, self._frame_shape)

    def stop(self) -> None:
        """
        Signal the writer thread to flush remaining frames and stop.

        Blocks until the thread finishes or flush_timeout_s elapses.
        Frames remaining in the buffer after timeout are discarded.
        """
        if self._thread is None:
            return

        logger.info("FrameWriter stopping - waiting for buffer to drain...")
        self._stop_event.set()
        self._data_available.set()   # wake the thread if it's waiting
        self._thread.join(timeout=self._flush_timeout)

        if self._thread.is_alive():
            remaining = self._occupancy
            logger.warning(
                "FrameWriter flush timeout (%.1f s) - "
                "%d frame(s) in buffer were not written.",
                self._flush_timeout, remaining,
            )
        else:
            logger.info("FrameWriter drained cleanly.")

        self._close_files()

        if self._timing_logger:
            self._timing_logger.close()
            self._timing_logger = None

        self._thread = None

        s = self.stats
        logger.info(
            "FrameWriter session summary: pushed=%d written=%d "
            "dropped=%d errors=%d",
            s.frames_pushed, s.frames_written,
            s.frames_dropped, s.write_errors,
        )

    # ------------------------------------------------------------------
    # Acquisition-loop interface (called from hot path)
    # ------------------------------------------------------------------

    def push(self, frame_data: np.ndarray, acquired: AcquiredFrame) -> None:
        """
        Push a frame into the ring buffer from the acquisition loop.

        This method is designed to be fast: it validates the frame shape,
        copies pixel data into the pre-allocated buffer slot, records
        metadata, advances the write index, and signals the writer thread.
        The lock is held only for index arithmetic.

        If the buffer is full the oldest unread slot is overwritten and the
        drop counter is incremented.

        Parameters
        ----------
        frame_data : np.ndarray
            2-D uint16 array of shape frame_shape. Typically
            acquired.rois[roi_label].
        acquired : AcquiredFrame
            The source AcquiredFrame; used for metadata (timestamps, IDs).

        Raises
        ------
        ValueError
            If frame_data.shape does not match the configured frame_shape.
        RuntimeError
            If the writer has not been started.
        """
        if self._thread is None:
            raise RuntimeError(
                "FrameWriter.push() called before start()."
            )

        if frame_data.shape != self._frame_shape:
            raise ValueError(
                f"Frame shape {frame_data.shape} does not match "
                f"configured shape {self._frame_shape}."
            )

        with self._lock:
            slot = self._write_idx % self._buffer_n

            if self._occupancy == self._buffer_n:
                # Buffer full - overwrite oldest slot, advance read index
                self._frames_dropped += 1
                self._read_idx = (self._read_idx + 1) % self._buffer_n
                logger.warning(
                    "Ring buffer full - oldest frame dropped "
                    "(total dropped: %d).", self._frames_dropped
                )
            else:
                self._occupancy += 1

            # Copy pixel data into pre-allocated slot (no allocation)
            np.copyto(self._buffer[slot], frame_data)

            # Record metadata alongside the frame slot
            self._meta_buffer[slot] = {
                "frame_id":        acquired.frame_id,
                "timestamp_ns":    acquired.timestamp_ns,
                "host_time":       acquired.host_time,
                "nframes_dropped": acquired.nframes_dropped,
                "roi_label":       self._roi_label,
                "n_rows":          self._frame_shape[0],
                "n_cols":          self._frame_shape[1],
                "session_id":      self._session_id,
            }

            self._write_idx = (self._write_idx + 1) % self._buffer_n
            self._frames_pushed += 1

        self._data_available.set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> WriterStats:
        """Return a snapshot of current writer statistics."""
        with self._lock:
            return WriterStats(
                frames_pushed=self._frames_pushed,
                frames_written=self._frames_written,
                frames_dropped=self._frames_dropped,
                buffer_n_frames=self._buffer_n,
                buffer_occupancy=self._occupancy,
                write_errors=self._write_errors,
            )

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Background writer thread
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """
        Background thread: drain the ring buffer to disk.

        Waits on _data_available, then writes all pending frames before
        waiting again. Exits when _stop_event is set and the buffer is empty.
        """
        logger.debug("Writer thread started.")
        while True:
            tlog = self._timing_logger
            if tlog: tlog.start("idle")
            self._data_available.wait()
            self._data_available.clear()
            if tlog: tlog.end("idle")

            # Drain all pending frames
            while True:
                with self._lock:
                    if self._occupancy == 0:
                        break
                    slot = self._read_idx % self._buffer_n
                    # Take a view of the slot (no copy needed - writer thread
                    # owns this slot until read_idx advances)
                    frame_view = self._buffer[slot]
                    meta       = self._meta_buffer[slot]

                # Write pixel data
                try:
                    if tlog: tlog.start("write_bin")
                    self._bin_file.write(frame_view.tobytes())
                    if tlog: tlog.end("write_bin")
                except OSError as e:
                    self._write_errors += 1
                    logger.error(
                        "Frame write error (frame_id=%s): %s",
                        meta.get("frame_id", "?"), e,
                    )
                else:
                    # Write metadata only if pixel write succeeded
                    try:
                        if tlog: tlog.start("write_meta")
                        self._meta_file.write(json.dumps(meta) + "\n")
                        if tlog: tlog.end("write_meta")
                    except OSError as e:
                        self._write_errors += 1
                        logger.error("Metadata write error: %s", e)

                with self._lock:
                    self._read_idx = (self._read_idx + 1) % self._buffer_n
                    self._occupancy -= 1
                    self._frames_written += 1

            # Exit only after buffer is empty
            if self._stop_event.is_set():
                break

        logger.debug("Writer thread exiting.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bin_path(self) -> str:
        return os.path.join(
            self._data_dir, f"{self._session_id}_frames.bin"
        )

    def _meta_path(self) -> str:
        return os.path.join(
            self._data_dir, f"{self._session_id}_meta.jsonl"
        )

    def _close_files(self) -> None:
        for attr, label in [("_bin_file", "frame"), ("_meta_file", "metadata")]:
            f = getattr(self, attr)
            if f is not None:
                try:
                    f.flush()
                    os.fsync(f.fileno())
                    f.close()
                    logger.debug("%s file closed and fsynced.", label)
                except OSError as e:
                    logger.error("Error closing %s file: %s", label, e)
                finally:
                    setattr(self, attr, None)