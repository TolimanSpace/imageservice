"""
plot_timing.py — Timing diagnostic plots for the Toliman image acquisition system.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_project_root = str(pathlib.Path(__file__).parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

FIGURE_DPI     = 150
SCATTER_COLOR  = "#1f77b4"
REF_LINE_COLOR = "#d62728"
SCATTER_SIZE   = 4
SCATTER_ALPHA  = 0.5
FONT_SIZE      = 11


def load_metadata(meta_path: str) -> List[Dict]:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    records = []
    with open(meta_path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {lineno}: {e}")
    if not records:
        raise ValueError(f"No records found in {meta_path}")
    records.sort(key=lambda r: r["frame_id"])
    return records


def extract_arrays(records: List[Dict]):
    timestamp_s = np.array([r["timestamp_ns"] / 1e9 for r in records], dtype=np.float64)
    host_time   = np.array([r["host_time"]            for r in records], dtype=np.float64)
    frame_ids   = np.array([r["frame_id"]             for r in records], dtype=np.int64)
    return timestamp_s, host_time, frame_ids


def plot_camera_timing(
    records: List[Dict],
    nominal_rate_hz: float,
    session_id: Optional[str] = None,
) -> plt.Figure:
    timestamp_s, _, frame_ids = extract_arrays(records)
    nominal_period_s = 1.0 / nominal_rate_hz

    dt_s         = np.diff(timestamp_s)
    t_mid_s      = timestamp_s[:-1]
    residual_us  = (dt_s - nominal_period_s) * 1e6
    well_behaved = np.abs(dt_s - nominal_period_s) < 1.0

    n_frames   = len(records)
    n_dropped  = int(np.sum(np.diff(frame_ids) - 1))
    duration_s = timestamp_s[-1] - timestamp_s[0]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=False,
        gridspec_kw={"hspace": 0.35},
    )

    title = "Camera Timestamp"
    if session_id:
        title += f"  —  {session_id}"
    fig.suptitle(title, fontsize=FONT_SIZE + 1, y=0.98)

    kw = dict(s=SCATTER_SIZE, color=SCATTER_COLOR, alpha=SCATTER_ALPHA,
              linewidths=0, rasterized=True)

    ax_top.scatter(t_mid_s, dt_s, **kw)
    ax_top.axhline(nominal_period_s, color=REF_LINE_COLOR, linewidth=1.2,
                   zorder=5, label=f"Nominal ({nominal_rate_hz:.1f} Hz)")
    ax_top.set_ylabel(r"$\Delta t$ (seconds)", fontsize=FONT_SIZE)
    ax_top.set_ylim(bottom=0)
    ax_top.legend(fontsize=FONT_SIZE - 1, loc="upper right")
    ax_top.text(
        0.01, 0.97,
        f"n = {n_frames:,}   dropped = {n_dropped}   duration = {duration_s:.0f} s",
        transform=ax_top.transAxes, fontsize=FONT_SIZE - 2, va="top", color="0.4",
    )

    ax_bot.scatter(t_mid_s[well_behaved], residual_us[well_behaved], **kw)
    ax_bot.axhline(0, color=REF_LINE_COLOR, linewidth=1.2, zorder=5)
    ax_bot.set_ylim(-10, 10)
    ax_bot.set_ylabel(
        r"$\Delta t - t_{\rm nom}$ ($\mu$s)", fontsize=FONT_SIZE
    )
    ax_bot.set_xlabel("Camera Time (seconds)", fontsize=FONT_SIZE)

    r_std = float(np.std(residual_us[well_behaved]))
    r_p99 = float(np.percentile(np.abs(residual_us[well_behaved]), 99))
    ax_bot.text(
        0.01, 0.97,
        f"std = {r_std:.2f} µs   |Δ| p99 = {r_p99:.2f} µs",
        transform=ax_bot.transAxes, fontsize=FONT_SIZE - 2, va="top", color="0.4",
    )

    for ax in (ax_top, ax_bot):
        ax.tick_params(labelsize=FONT_SIZE - 1)
        ax.set_xlim(t_mid_s[0], t_mid_s[-1])

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_host_timing(
    records: List[Dict],
    nominal_rate_hz: float,
    session_id: Optional[str] = None,
) -> plt.Figure:
    _, host_time, _ = extract_arrays(records)
    nominal_period_s = 1.0 / nominal_rate_hz

    dt_s   = np.diff(host_time)
    t_rel  = host_time[:-1] - host_time[0]

    fig, ax = plt.subplots(figsize=(12, 4))
    title = "Host Acquisition Timestamp"
    if session_id:
        title += f"  —  {session_id}"
    ax.set_title(title, fontsize=FONT_SIZE + 1)

    ax.scatter(t_rel, dt_s, s=SCATTER_SIZE, color=SCATTER_COLOR,
               alpha=SCATTER_ALPHA, linewidths=0, rasterized=True)
    ax.axhline(nominal_period_s, color=REF_LINE_COLOR, linewidth=1.2,
               zorder=5, label=f"Nominal ({nominal_rate_hz:.1f} Hz)")
    ax.set_xlabel("Session Time (seconds)", fontsize=FONT_SIZE)
    ax.set_ylabel(r"$\Delta t$ (seconds)", fontsize=FONT_SIZE)
    ax.set_ylim(bottom=0)
    ax.set_xlim(t_rel[0], t_rel[-1])
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.legend(fontsize=FONT_SIZE - 1, loc="upper right")

    n_above = int(np.sum(dt_s > nominal_period_s * 2))
    ax.text(
        0.01, 0.97,
        f"n = {len(records):,}   >2× nominal: {n_above}   "
        f"mean Δt = {np.mean(dt_s) * 1e3:.1f} ms",
        transform=ax.transAxes, fontsize=FONT_SIZE - 2, va="top", color="0.4",
    )

    fig.tight_layout()
    return fig


def plot_timing_summary(
    records: List[Dict],
    nominal_rate_hz: float,
    session_id: Optional[str] = None,
) -> plt.Figure:
    timestamp_s, host_time, frame_ids = extract_arrays(records)
    nominal_period_s = 1.0 / nominal_rate_hz

    cam_dt_s     = np.diff(timestamp_s)
    cam_t        = timestamp_s[:-1]
    residual_us  = (cam_dt_s - nominal_period_s) * 1e6
    well_behaved = np.abs(cam_dt_s - nominal_period_s) < 1.0

    host_dt_s  = np.diff(host_time)
    host_t_rel = host_time[:-1] - host_time[0]

    n_frames   = len(records)
    n_dropped  = int(np.sum(np.diff(frame_ids) - 1))
    duration_s = timestamp_s[-1] - timestamp_s[0]

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 10),
        gridspec_kw={"hspace": 0.45, "height_ratios": [2, 1.5, 2]},
    )

    title = "Acquisition Timing Summary"
    if session_id:
        title += f"  —  {session_id}"
    fig.suptitle(title, fontsize=FONT_SIZE + 2, y=0.99)

    kw = dict(s=SCATTER_SIZE, color=SCATTER_COLOR,
              alpha=SCATTER_ALPHA, linewidths=0, rasterized=True)

    axes[0].scatter(cam_t, cam_dt_s, **kw)
    axes[0].axhline(nominal_period_s, color=REF_LINE_COLOR, linewidth=1.2, zorder=5)
    axes[0].set_ylabel(r"Camera $\Delta t$ (s)", fontsize=FONT_SIZE)
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlim(cam_t[0], cam_t[-1])
    axes[0].text(
        0.01, 0.97,
        f"n = {n_frames:,}   dropped = {n_dropped}   duration = {duration_s:.0f} s",
        transform=axes[0].transAxes, fontsize=FONT_SIZE - 2, va="top", color="0.4",
    )

    axes[1].scatter(cam_t[well_behaved], residual_us[well_behaved], **kw)
    axes[1].axhline(0, color=REF_LINE_COLOR, linewidth=1.2, zorder=5)
    axes[1].set_ylim(-10, 10)
    axes[1].set_ylabel(r"Camera jitter ($\mu$s)", fontsize=FONT_SIZE)
    axes[1].set_xlim(cam_t[0], cam_t[-1])
    r_std = float(np.std(residual_us[well_behaved]))
    axes[1].text(
        0.01, 0.97, f"std = {r_std:.2f} µs",
        transform=axes[1].transAxes, fontsize=FONT_SIZE - 2, va="top", color="0.4",
    )

    axes[2].scatter(host_t_rel, host_dt_s, **kw)
    axes[2].axhline(nominal_period_s, color=REF_LINE_COLOR, linewidth=1.2, zorder=5)
    axes[2].set_ylabel(r"Host $\Delta t$ (s)", fontsize=FONT_SIZE)
    axes[2].set_xlabel("Session Time (seconds)", fontsize=FONT_SIZE)
    axes[2].set_ylim(bottom=0)
    axes[2].set_xlim(host_t_rel[0], host_t_rel[-1])
    axes[2].text(
        0.01, 0.97,
        f"mean = {np.mean(host_dt_s)*1e3:.1f} ms   "
        f">2× nominal: {int(np.sum(host_dt_s > nominal_period_s * 2))}",
        transform=axes[2].transAxes, fontsize=FONT_SIZE - 2, va="top", color="0.4",
    )

    for ax in axes:
        ax.tick_params(labelsize=FONT_SIZE - 1)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate timing diagnostic plots from a session metadata file."
    )
    parser.add_argument("--meta", required=True,
                        help="Path to _meta.jsonl file from FrameWriter.")
    parser.add_argument("--rate", type=float, default=None,
                        help="Nominal frame rate Hz (estimated from data if omitted).")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save figures.")
    parser.add_argument("--format", default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="Output format (default: pdf).")
    parser.add_argument("--summary-only", action="store_true",
                        help="Produce combined summary figure only.")
    args = parser.parse_args()

    print(f"Loading: {args.meta}")
    records = load_metadata(args.meta)
    print(f"Loaded {len(records):,} frames.")

    if args.rate is not None:
        nominal_hz = args.rate
    else:
        ts, _, _ = extract_arrays(records)
        nominal_hz = 1.0 / float(np.median(np.diff(ts)))
        print(f"Estimated rate: {nominal_hz:.2f} Hz")

    session_id = os.path.basename(args.meta).split("_meta.jsonl")[0]
    os.makedirs(args.output_dir, exist_ok=True)

    def save(fig, name):
        path = os.path.join(args.output_dir,
                            f"{session_id}_{name}.{args.format}")
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    if args.summary_only:
        save(plot_timing_summary(records, nominal_hz, session_id), "timing_summary")
    else:
        save(plot_camera_timing(records, nominal_hz, session_id), "camera_timing")
        save(plot_host_timing(records, nominal_hz, session_id),   "host_timing")
        save(plot_timing_summary(records, nominal_hz, session_id),"timing_summary")


if __name__ == "__main__":
    main()