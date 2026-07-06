"""
plot_process_timing.py — Gantt-style process timing plots.

Parses .timing.log files produced by TimingLogger and generates
per-frame activity timeline figures matching the style of
process_timing_start / process_timing_firstminute plots.

Each row is a named activity. Each horizontal bar is one frame's
worth of that activity. Colour encodes the CPU core.

Usage
-----
    # Single log file
    python3 plot_process_timing.py --logs obs_001_acquisition.timing.log

    # Multiple components (acquisition + writer)
    python3 plot_process_timing.py \\
        --logs obs_001_acquisition.timing.log obs_001_writer_roi.timing.log \\
        --output-dir ./figures

    # Zoom to a time window
    python3 plot_process_timing.py --logs *.timing.log --start 0 --end 5
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_project_root = str(pathlib.Path(__file__).parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

FIGURE_DPI   = 150
FONT_SIZE    = 10
BAR_HEIGHT   = 0.6

# Colour palette for CPU cores (up to 8 cores)
CPU_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ActivitySpan:
    __slots__ = ("activity", "t_start", "t_end", "pid", "cpu")

    def __init__(self, activity, t_start, t_end, pid, cpu):
        self.activity = activity
        self.t_start  = t_start
        self.t_end    = t_end
        self.pid      = pid
        self.cpu      = cpu

    @property
    def duration(self):
        return self.t_end - self.t_start


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_timing_log(path: str) -> Tuple[str, str, List[ActivitySpan]]:
    """
    Parse a .timing.log file into a list of ActivitySpan objects.

    Returns (session_id, component, spans).
    Unpaired start/end lines are discarded with a warning.
    """
    session_id = "unknown"
    component  = "unknown"
    pending: Dict[Tuple[str, int], float] = {}   # (activity, pid) -> t_start
    spans: List[ActivitySpan] = []

    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if line.startswith("#"):
                # Header: # session=X component=Y pid=Z
                for part in line.lstrip("# ").split():
                    if part.startswith("session="):
                        session_id = part.split("=", 1)[1]
                    elif part.startswith("component="):
                        component = part.split("=", 1)[1]
                continue

            if not line.startswith("TIMING|"):
                continue

            parts = line.split("|")
            if len(parts) != 6:
                print(f"Warning: malformed line {lineno} in {path}")
                continue

            _, t_str, event, activity, pid_str, cpu_str = parts
            try:
                t   = float(t_str)
                pid = int(pid_str)
                cpu = int(cpu_str)
            except ValueError:
                print(f"Warning: could not parse line {lineno} in {path}")
                continue

            key = (activity, pid)
            if event == "start":
                pending[key] = (t, cpu)
            elif event == "end" and key in pending:
                t_start, cpu_start = pending.pop(key)
                spans.append(ActivitySpan(activity, t_start, t, pid, cpu_start))

    if pending:
        print(f"Warning: {len(pending)} unpaired start events in {path}")

    spans.sort(key=lambda s: s.t_start)
    return session_id, component, spans


def load_logs(paths: List[str]):
    """Load and merge spans from multiple log files. Returns (t0, all_spans)."""
    all_spans: List[ActivitySpan] = []
    for path in paths:
        _, _, spans = parse_timing_log(path)
        all_spans.extend(spans)

    if not all_spans:
        raise ValueError("No timing spans found in the provided log files.")

    t0 = min(s.t_start for s in all_spans)
    for s in all_spans:
        s.t_start -= t0
        s.t_end   -= t0

    return all_spans


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _get_activity_order(spans: List[ActivitySpan]) -> List[str]:
    """Return activity names in a logical display order."""
    preferred = [
        "acquire", "centroid", "push",
        "write_bin", "write_meta", "write",
        "compress", "diff", "netcdf",
    ]
    seen = set()
    ordered = []
    for name in preferred:
        if any(s.activity == name for s in spans):
            ordered.append(name)
            seen.add(name)
    # Append any remaining activities alphabetically
    for name in sorted({s.activity for s in spans} - seen):
        ordered.append(name)
    return ordered


def plot_process_timing(
    spans: List[ActivitySpan],
    t_start_s: float = 0.0,
    t_end_s: Optional[float] = None,
    title: Optional[str] = None,
    show_cpu_legend: bool = True,
) -> plt.Figure:
    """
    Generate a Gantt-style per-frame activity timeline.

    Parameters
    ----------
    spans : list of ActivitySpan
        All spans to plot (already offset so t=0 is session start).
    t_start_s, t_end_s : float
        Time window to display in seconds.
    title : str or None
    show_cpu_legend : bool
    """
    if t_end_s is None:
        t_end_s = max(s.t_end for s in spans)

    window_spans = [
        s for s in spans
        if s.t_end >= t_start_s and s.t_start <= t_end_s
    ]

    activities = _get_activity_order(window_spans)
    n_rows     = len(activities)
    act_index  = {a: i for i, a in enumerate(activities)}

    all_cpus  = sorted({s.cpu for s in window_spans if s.cpu >= 0})
    n_cpus    = len(all_cpus)
    cpu_idx   = {c: i for i, c in enumerate(all_cpus)}

    fig_h = max(3.0, 0.7 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    for span in window_spans:
        row  = act_index[span.activity]
        ypos = row - BAR_HEIGHT / 2
        w    = span.t_end - span.t_start
        cidx = cpu_idx.get(span.cpu, 0)
        color = CPU_COLOURS[cidx % len(CPU_COLOURS)]
        ax.barh(
            row, w,
            left=span.t_start,
            height=BAR_HEIGHT,
            color=color,
            alpha=0.85,
            linewidth=0,
        )

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(activities, fontsize=FONT_SIZE)
    ax.set_xlim(t_start_s, t_end_s)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlabel("Session Time (seconds)", fontsize=FONT_SIZE)
    ax.tick_params(axis="x", labelsize=FONT_SIZE - 1)
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.88", linewidth=0.5, zorder=0)

    if title:
        ax.set_title(title, fontsize=FONT_SIZE + 1)

    if show_cpu_legend and n_cpus > 1:
        patches = [
            mpatches.Patch(
                color=CPU_COLOURS[cpu_idx[c] % len(CPU_COLOURS)],
                label=f"CPU {c}",
                alpha=0.85,
            )
            for c in all_cpus
        ]
        ax.legend(
            handles=patches,
            loc="upper right",
            fontsize=FONT_SIZE - 1,
            framealpha=0.9,
        )

    # Annotate with frame count and duration
    n_frames_acquire = sum(
        1 for s in window_spans if s.activity == "acquire"
    )
    ax.text(
        0.01, 0.02,
        f"{n_frames_acquire} frames   "
        f"{t_end_s - t_start_s:.1f} s window",
        transform=ax.transAxes,
        fontsize=FONT_SIZE - 2,
        va="bottom",
        color="0.4",
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate process timing Gantt plots from .timing.log files."
    )
    parser.add_argument(
        "--logs", nargs="+", required=True,
        help="One or more .timing.log files to plot.",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--format", default="pdf",
        choices=["pdf", "png", "svg"],
    )
    parser.add_argument(
        "--start", type=float, default=0.0,
        help="Start of time window in seconds (default 0).",
    )
    parser.add_argument(
        "--end", type=float, default=None,
        help="End of time window in seconds (default: full session).",
    )
    parser.add_argument(
        "--windows", nargs="+", type=float, default=None,
        metavar="END_S",
        help=(
            "Generate multiple zoom windows. Each value is the end time "
            "in seconds from the session start. "
            "E.g. --windows 5 60 produces a 0-5s and a 0-60s figure."
        ),
    )
    args = parser.parse_args()

    print(f"Loading {len(args.logs)} log file(s)...")
    spans = load_logs(args.logs)
    print(f"Loaded {len(spans):,} activity spans.")

    session_id = os.path.basename(args.logs[0]).split("_acquisition")[0].split("_writer")[0]
    os.makedirs(args.output_dir, exist_ok=True)

    def save(fig, suffix):
        path = os.path.join(
            args.output_dir,
            f"{session_id}_process_timing_{suffix}.{args.format}",
        )
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    windows = args.windows if args.windows else [args.end]

    for end_s in windows:
        suffix = f"{args.start:.0f}to{end_s:.0f}s" if end_s else "full"
        title  = f"Process Timing — {session_id}"
        if end_s:
            title += f"  ({args.start:.0f}–{end_s:.0f} s)"
        fig = plot_process_timing(
            spans,
            t_start_s=args.start,
            t_end_s=end_s,
            title=title,
        )
        save(fig, suffix)

    # Always produce the full-session view too (unless already done)
    if args.windows:
        fig = plot_process_timing(spans, title=f"Process Timing — {session_id} (full)")
        save(fig, "full")


if __name__ == "__main__":
    main()
