"""
process_timing_logger.py — Structured per-frame timing logger.

Provides a lightweight logger that writes one line per activity start/end
in a parseable format. Compatible with the existing file logging in
acquisition.py and writer.py — just adds structured DEBUG lines.

Log line format:
    TIMING|<wall_time_s>|<event>|<activity>|<pid>|<cpu>

    wall_time_s : float  — time.monotonic() at the event
    event       : str    — "start" or "end"
    activity    : str    — e.g. "acquire", "centroid", "push", "write"
    pid         : int    — os.getpid()
    cpu         : int    — current CPU core (from os.sched_getaffinity or psutil)

Usage in acquisition.py
-----------------------
    from process_timing_logger import TimingLogger
    tlog = TimingLogger("acquisition", session_id, log_dir)

    # In the hot loop:
    tlog.start("acquire");  frame = cam.acquire_frame();  tlog.end("acquire")
    tlog.start("centroid"); pe = compute_pointing_error(...); tlog.end("centroid")
    tlog.start("push");     writer.push(...);               tlog.end("push")

Usage in writer.py
------------------
    tlog = TimingLogger("writer", session_id, log_dir)
    tlog.start("write"); self._bin_file.write(...); tlog.end("write")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Try to get CPU affinity — graceful fallback if not available
def _get_cpu() -> int:
    try:
        return list(os.sched_getaffinity(0))[0]
    except (AttributeError, OSError):
        pass
    try:
        import psutil
        return psutil.Process().cpu_num()
    except Exception:
        return -1


class TimingLogger:
    """
    Lightweight per-frame timing logger.

    Writes structured TIMING| lines to a dedicated .timing.log file
    alongside the session logs. The file is kept separate from the
    main log so it can be parsed efficiently without filtering.

    Parameters
    ----------
    component : str
        Name of the component, e.g. "acquisition", "writer_roi".
    session_id : str
    log_dir : str
        Directory to write the timing log file.
    enabled : bool
        If False, all calls are no-ops. Allows the logger to be
        disabled without changing call sites.
    """

    def __init__(
        self,
        component: str,
        session_id: str,
        log_dir: str,
        enabled: bool = True,
    ) -> None:
        self._component  = component
        self._session_id = session_id
        self._enabled    = enabled
        self._pid        = os.getpid()
        self._start_times: Dict[str, float] = {}
        self._file       = None

        if not enabled:
            return

        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(
            log_dir, f"{session_id}_{component}.timing.log"
        )
        self._file = open(path, "w", buffering=1)   # line-buffered
        self._file.write(
            f"# session={session_id} component={component} pid={self._pid}\n"
        )
        logger.debug("TimingLogger: writing to %s", path)

    def start(self, activity: str) -> None:
        if not self._enabled or self._file is None:
            return
        t = time.monotonic()
        self._start_times[activity] = t
        cpu = _get_cpu()
        self._file.write(
            f"TIMING|{t:.9f}|start|{activity}|{self._pid}|{cpu}\n"
        )

    def end(self, activity: str) -> None:
        if not self._enabled or self._file is None:
            return
        t = time.monotonic()
        cpu = _get_cpu()
        self._file.write(
            f"TIMING|{t:.9f}|end|{activity}|{self._pid}|{cpu}\n"
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()
