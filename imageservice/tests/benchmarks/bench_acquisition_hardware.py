"""
bench_acquisition_hardware.py - Sustained acquisition throughput benchmark.

Measures real-world acquisition performance on a connected Ximea camera
using a single 848x848 pixel ROI. Reports:

  - Actual achieved frame rate (from hardware timestamps)
  - Frame drop count and drop rate
  - Per-frame timing statistics (acquire, centroid, push)
  - Writer throughput (frames written vs acquired)
  - Timestamp jitter (inter-frame interval statistics)

This benchmark uses real hardware and synthetic image injection so that
the pixel data is deterministic, but all timing comes from the real
camera driver and transport layer.

Usage
-----
    python3 tests/benchmarks/bench_acquisition_hardware.py

    # Specify target frame rates to sweep
    python3 tests/benchmarks/bench_acquisition_hardware.py --rates 10 25 50 100

    # Specify camera serial number
    python3 tests/benchmarks/bench_acquisition_hardware.py --serial XXXXXXXX

    # Longer run for more stable statistics
    python3 tests/benchmarks/bench_acquisition_hardware.py --duration 30

    # Skip centroid computation (isolate camera+writer throughput only)
    python3 tests/benchmarks/bench_acquisition_hardware.py --no-centroid

    # Skip writing to disk
    python3 tests/benchmarks/bench_acquisition_hardware.py --no-write
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import pathlib as _pathlib
_project_root = str(_pathlib.Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_SERIAL_NUMBER: Optional[str] = None   # None = first available
SENSOR_WIDTH:  int = 4512
SENSOR_HEIGHT: int = 4512

ROI_WIDTH:    int = 848
ROI_HEIGHT:   int = 848
ROI_OFFSET_X: int = (SENSOR_WIDTH  - ROI_WIDTH)  // 2
ROI_OFFSET_Y: int = (SENSOR_HEIGHT - ROI_HEIGHT) // 2

EXPOSURE_US: int = 5_000   # 5 ms - leaves headroom at 100 Hz (10 ms period)

DEFAULT_RATES_HZ:    List[float] = [10.0, 25.0, 50.0, 100.0]
DEFAULT_DURATION_S:  float       = 10.0
OUTPUT_DIR:          str         = os.path.join(tempfile.gettempdir(),
                                                 "imageservice_bench")

logging.basicConfig(
    level=logging.WARNING,   # suppress info during benchmark
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
)
logger = logging.getLogger("bench_acquisition_hardware")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    """Results for a single (frame_rate, duration) benchmark run."""
    target_hz:         float
    duration_s:        float

    # Frame counts
    frames_acquired:   int
    frames_dropped:    int
    frames_written:    int

    # Actual rate (derived from hardware timestamps)
    actual_hz_mean:    float   # mean from 1/interframe intervals
    actual_hz_median:  float

    # Inter-frame interval statistics (microseconds)
    interframe_mean_us:   float
    interframe_std_us:    float
    interframe_p99_us:    float
    interframe_max_us:    float

    # Per-frame processing time (microseconds, from perf_counter_ns)
    t_acquire_mean_us:  float
    t_acquire_p99_us:   float
    t_centroid_mean_us: float
    t_push_mean_us:     float
    t_total_mean_us:    float   # acquire + centroid + push
    t_total_p99_us:     float

    # Drop rate
    @property
    def drop_rate_pct(self) -> float:
        total = self.frames_acquired + self.frames_dropped
        return 100.0 * self.frames_dropped / total if total > 0 else 0.0

    # Budget utilisation at target rate
    @property
    def budget_us(self) -> float:
        return 1_000_000.0 / self.target_hz

    @property
    def budget_utilisation_pct(self) -> float:
        return 100.0 * self.t_total_mean_us / self.budget_us


# ---------------------------------------------------------------------------
# Synthetic frame generator
# ---------------------------------------------------------------------------

def make_frame_sequence(n: int, height: int, width: int) -> np.ndarray:
    """
    Pre-generate n synthetic uint16 frames with a slow-drifting Gaussian PSF.
    Frames are cycled during the benchmark run.
    """
    rng = np.random.default_rng(42)
    stack = np.zeros((n, height, width), dtype=np.uint16)
    cy0, cx0 = height / 2.0, width / 2.0
    sigma = 4.0
    y_grid, x_grid = np.ogrid[:height, :width]
    for i in range(n):
        cx = cx0 + 5.0 * np.sin(2 * np.pi * i / n)
        cy = cy0 + 5.0 * np.cos(2 * np.pi * i / n)
        bg = rng.integers(5, 20, (height, width), dtype=np.uint16)
        psf = 3000 * np.exp(
            -((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2)
        )
        stack[i] = np.clip(bg.astype(np.float32) + psf, 0, 4095).astype(np.uint16)
    return stack


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    target_hz: float,
    duration_s: float,
    enable_centroid: bool,
    enable_write: bool,
    serial_number: Optional[str],
    frame_sequence: np.ndarray,
    output_dir: str,
) -> BenchResult:
    """Run a single sustained acquisition benchmark at target_hz."""

    from workers.camera import XimeaCamera, single_roi_config
    from workers.centroid import compute_pointing_error
    from workers.writer import FrameWriter

    cfg = single_roi_config(
        width=ROI_WIDTH,
        height=ROI_HEIGHT,
        offset_x=ROI_OFFSET_X,
        offset_y=ROI_OFFSET_Y,
        exposure_us=EXPOSURE_US,
        frame_rate_hz=target_hz,
        label="single_roi",
        serial_number=serial_number,
        transport_buffer_size=32,
    )

    n_seq = len(frame_sequence)
    seq_idx = [0]

    # Pre-allocate diagnostic arrays (max frames = rate * duration + 20% headroom)
    max_frames = int(target_hz * duration_s * 1.2) + 100
    t_acquire_arr  = np.zeros(max_frames, dtype=np.float32)
    t_centroid_arr = np.zeros(max_frames, dtype=np.float32)
    t_push_arr     = np.zeros(max_frames, dtype=np.float32)
    t_total_arr    = np.zeros(max_frames, dtype=np.float32)
    timestamps_ns  = np.zeros(max_frames, dtype=np.int64)

    frames_acquired = 0
    frames_dropped  = 0

    session_id = f"bench_{int(target_hz)}hz"
    os.makedirs(output_dir, exist_ok=True)

    writer = FrameWriter(
        data_dir=output_dir,
        session_id=session_id,
        roi_label="single_roi",
        frame_shape=(ROI_HEIGHT, ROI_WIDTH),
        buffer_n_frames=256,
        flush_timeout_s=10.0,
    ) if enable_write else None

    if writer:
        writer.start()

    def inject_frame(img, timeout=1000):
        """Call real driver then overwrite pixel buffer with synthetic data."""
        real_get_image(img, timeout=timeout)
        img._data = frame_sequence[seq_idx[0] % n_seq].copy()
        seq_idx[0] += 1

    deadline = time.monotonic() + duration_s

    with XimeaCamera(cfg) as cam:
        real_get_image = cam._cam.get_image
        cam._cam.get_image = inject_frame

        while time.monotonic() < deadline:
            t0 = time.perf_counter_ns()

            try:
                frame = cam.acquire_frame()
            except Exception as e:
                logger.warning("acquire_frame error: %s", e)
                continue

            t1 = time.perf_counter_ns()

            frames_dropped  += frame.nframes_dropped
            roi              = frame.rois["single_roi"]

            if enable_centroid:
                compute_pointing_error(
                    roi,
                    roi_offset_x=ROI_OFFSET_X,
                    roi_offset_y=ROI_OFFSET_Y,
                    sensor_width=SENSOR_WIDTH,
                    sensor_height=SENSOR_HEIGHT,
                    timestamp_ns=frame.timestamp_ns,
                    frame_id=frame.frame_id,
                )

            t2 = time.perf_counter_ns()

            if writer:
                writer.push(roi, frame)

            t3 = time.perf_counter_ns()

            if frames_acquired < max_frames:
                idx = frames_acquired
                t_acquire_arr[idx]  = (t1 - t0) / 1e3
                t_centroid_arr[idx] = (t2 - t1) / 1e3
                t_push_arr[idx]     = (t3 - t2) / 1e3
                t_total_arr[idx]    = (t3 - t0) / 1e3
                timestamps_ns[idx]  = frame.timestamp_ns

            frames_acquired += 1

    if writer:
        writer.stop()

    n = min(frames_acquired, max_frames)

    # Inter-frame intervals from hardware timestamps (nanoseconds → microseconds)
    ts = timestamps_ns[:n]
    interframe_us = np.diff(ts).astype(np.float64) / 1e3
    interframe_us = interframe_us[interframe_us > 0]   # drop any zero gaps

    # Trim timing arrays to actual frame count
    ta = t_acquire_arr[:n]
    tc = t_centroid_arr[:n]
    tp = t_push_arr[:n]
    tt = t_total_arr[:n]

    return BenchResult(
        target_hz=target_hz,
        duration_s=duration_s,
        frames_acquired=frames_acquired,
        frames_dropped=frames_dropped,
        frames_written=writer.stats.frames_written if writer else 0,
        actual_hz_mean=(
            float(np.mean(1e6 / interframe_us)) if len(interframe_us) > 0 else 0.0
        ),
        actual_hz_median=(
            float(np.median(1e6 / interframe_us)) if len(interframe_us) > 0 else 0.0
        ),
        interframe_mean_us=(
            float(np.mean(interframe_us)) if len(interframe_us) > 0 else 0.0
        ),
        interframe_std_us=(
            float(np.std(interframe_us)) if len(interframe_us) > 0 else 0.0
        ),
        interframe_p99_us=(
            float(np.percentile(interframe_us, 99)) if len(interframe_us) > 0 else 0.0
        ),
        interframe_max_us=(
            float(np.max(interframe_us)) if len(interframe_us) > 0 else 0.0
        ),
        t_acquire_mean_us=float(np.mean(ta)),
        t_acquire_p99_us=float(np.percentile(ta, 99)),
        t_centroid_mean_us=float(np.mean(tc)),
        t_push_mean_us=float(np.mean(tp)),
        t_total_mean_us=float(np.mean(tt)),
        t_total_p99_us=float(np.percentile(tt, 99)),
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_result(r: BenchResult) -> None:
    budget = r.budget_us
    drop_warn = " ⚠" if r.frames_dropped > 0 else ""

    print(f"\n{'─'*60}")
    print(f"  Target: {r.target_hz:.0f} Hz   "
          f"Duration: {r.duration_s:.0f}s   "
          f"ROI: {ROI_HEIGHT}×{ROI_WIDTH}px")
    print(f"{'─'*60}")

    print(f"  Frame counts")
    print(f"    Acquired:  {r.frames_acquired:>6d}")
    print(f"    Dropped:   {r.frames_dropped:>6d}{drop_warn}  "
          f"({r.drop_rate_pct:.2f}%)")
    if r.frames_written > 0:
        print(f"    Written:   {r.frames_written:>6d}")

    print(f"\n  Actual frame rate  (from hardware timestamps)")
    print(f"    Mean:    {r.actual_hz_mean:>8.2f} Hz")
    print(f"    Median:  {r.actual_hz_median:>8.2f} Hz")

    print(f"\n  Inter-frame interval  (budget: {budget:.0f} µs)")
    print(f"    Mean:    {r.interframe_mean_us:>8.1f} µs")
    print(f"    Std:     {r.interframe_std_us:>8.1f} µs")
    print(f"    p99:     {r.interframe_p99_us:>8.1f} µs  "
          f"({'OK' if r.interframe_p99_us < budget else 'OVER BUDGET ⚠'})")
    print(f"    Max:     {r.interframe_max_us:>8.1f} µs")

    print(f"\n  Processing time per frame")
    print(f"    acquire  mean: {r.t_acquire_mean_us:>7.1f} µs   "
          f"p99: {r.t_acquire_p99_us:>7.1f} µs")
    print(f"    centroid mean: {r.t_centroid_mean_us:>7.1f} µs")
    print(f"    push     mean: {r.t_push_mean_us:>7.1f} µs")
    print(f"    total    mean: {r.t_total_mean_us:>7.1f} µs   "
          f"p99: {r.t_total_p99_us:>7.1f} µs")
    print(f"    budget utilisation: {r.budget_utilisation_pct:.1f}%  "
          f"({'OK' if r.budget_utilisation_pct < 80 else 'HIGH ⚠'})")


def print_summary(results: List[BenchResult]) -> None:
    print(f"\n{'═'*60}")
    print(f"  Summary")
    print(f"{'═'*60}")
    print(f"  {'Rate':>6}  {'Acquired':>9}  {'Dropped':>8}  "
          f"{'Drop%':>6}  {'Actual Hz':>10}  {'p99 IFI':>9}  {'Budget%':>8}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*8}  "
          f"{'─'*6}  {'─'*10}  {'─'*9}  {'─'*8}")
    for r in results:
        drop_flag = " ⚠" if r.frames_dropped > 0 else "  "
        budget_flag = " ⚠" if r.budget_utilisation_pct >= 80 else "  "
        print(f"  {r.target_hz:>5.0f}Hz"
              f"  {r.frames_acquired:>9d}"
              f"  {r.frames_dropped:>7d}{drop_flag}"
              f"  {r.drop_rate_pct:>5.2f}%"
              f"  {r.actual_hz_mean:>9.2f}Hz"
              f"  {r.interframe_p99_us:>7.0f}µs"
              f"  {r.budget_utilisation_pct:>6.1f}%{budget_flag}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hardware acquisition throughput benchmark - 848×848 ROI."
    )
    parser.add_argument(
        "--rates", nargs="+", type=float,
        default=DEFAULT_RATES_HZ,
        help=f"Frame rates to benchmark (default: {DEFAULT_RATES_HZ})",
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_S,
        help=f"Duration per rate in seconds (default: {DEFAULT_DURATION_S})",
    )
    parser.add_argument(
        "--serial", default=None,
        help="Camera serial number",
    )
    parser.add_argument(
        "--no-centroid", action="store_true",
        help="Skip centroid computation (isolate camera+writer performance)",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="Skip writing frames to disk",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Output directory for frame files (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    serial = args.serial or CAMERA_SERIAL_NUMBER

    print(f"\nAcquisition hardware benchmark")
    print(f"  Camera SN:   {serial or 'first available'}")
    print(f"  ROI:         {ROI_HEIGHT} × {ROI_WIDTH} pixels")
    print(f"  Exposure:    {EXPOSURE_US} µs")
    print(f"  Duration:    {args.duration} s per rate")
    print(f"  Centroid:    {'disabled' if args.no_centroid else 'enabled'}")
    print(f"  Write:       {'disabled' if args.no_write else 'enabled'}")
    print(f"  Rates:       {args.rates} Hz")

    # Pre-generate synthetic frames once (reused across all rate benchmarks)
    print("\nGenerating synthetic frame sequence...", end=" ", flush=True)
    # Use 200 frames in the sequence so the pattern repeats smoothly
    frame_seq = make_frame_sequence(200, ROI_HEIGHT, ROI_WIDTH)
    print("done")

    results = []
    for rate in args.rates:
        print(f"\nRunning {rate:.0f} Hz benchmark "
              f"({args.duration:.0f}s)...", end=" ", flush=True)
        result = run_benchmark(
            target_hz=rate,
            duration_s=args.duration,
            enable_centroid=not args.no_centroid,
            enable_write=not args.no_write,
            serial_number=serial,
            frame_sequence=frame_seq,
            output_dir=args.output_dir,
        )
        results.append(result)
        print("done")
        print_result(result)

    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
