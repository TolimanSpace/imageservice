"""
compressor.py — Post-session compression pipeline.

Reads the raw .bin and .jsonl files written by FrameWriter, processes
them in batches of N frames, and produces netCDF4 files for downlink.

Pipeline (per batch)
--------------------
For each batch of N frames:

  1. Load N centre ROI frames and N frames per corner ROI from .bin files,
     guided by the .jsonl metadata file.

  2. Star finding — deconvolve each centre ROI frame and detect the
     positions of the two stars (find_stars() from star_finder.py).

  3. Coordinate transform — convert star positions from centre ROI-local
     to sensor coordinates, then to each corner ROI's local coordinates.

  4. Sidelobe crop — run crop_and_merge_corners() on each corner ROI frame using
     the transformed star positions, producing one merged strip per frame.

  5. Difference images — compute frame[i] - frame[i-1] for i in 1..N-1,
     for both the centre ROI stack and the sidelobe strip stack.

  6. Adaptive bit reduction:
       - Compute the range of all difference values across the batch.
       - If the range fits within int8 (±127), store as int8.
       - Otherwise store as int16 (±4095 for 12-bit source data).
       - The dtype chosen and any scale factor are recorded as netCDF
         variable attributes so the data can be reconstructed exactly.

  7. netCDF bundle — write one .nc file per batch containing:
       Centre ROI:     reference frame (uint16) + N-1 difference frames
       Sidelobe strip: reference frame (uint16) + N-1 difference frames
       Metadata:       per-frame timestamps, frame IDs, star positions,
                       dropped-frame counts, dtype used, session info.

Output filename
---------------
  <data_dir>/<session_id>_batch{batch_idx:04d}.nc

Usage
-----
Run from the command line after a session::

    python compressor.py \\
        --data-dir /data/imageservice \\
        --session-id obs_2026_177_001 \\
        --config /etc/imageservice/config.json \\
        --kernel deconvolution_constant.npy

Or call compress_session() programmatically::

    compress_session(system, session_params)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import netCDF4 as nc
import numpy as np

from config import SystemConfig, SessionConfig
from cropping import (
    DEFAULT_STRIP_WIDTH,
    crop_and_merge_corners,
)
from star_finder import (
    find_stars,
    load_deconvolution_kernel,
    roi_local_to_sensor,
    sensor_to_roi_local,
)
from camera import CameraMode


logger = logging.getLogger(__name__)

# Maximum value for 12-bit data — used for int8 range check
_MAX_12BIT: int = 4095
_INT8_MAX:  int = 127


# ---------------------------------------------------------------------------
# Session parameters (compression-specific, not in SessionConfig)
# ---------------------------------------------------------------------------

@dataclass
class CompressionParams:
    """
    Parameters for a compression run.

    Attributes
    ----------
    session_id : str
        Must match the session_id used during acquisition.
    batch_size_n : int
        Number of frames per netCDF batch. Should match
        SystemConfig.compression_batch_size_n.
    n_stars : int
        Number of stars to detect per frame. Default 2.
    kernel_path : str
        Path to the deconvolution kernel .npy file.
    sidelobe_offset : int
        Distance from star centre to spectrum in pixels (ROI-local).

    strip_width : int
        Sidelobe strip width in pixels.
    corner_roi_labels : list of str
        Labels of the four corner ROIs, in order. Must match the labels
        used in the .bin filenames written by FrameWriter.
    centre_roi_label : str
        Label of the centre ROI .bin file.
    """
    session_id: str
    batch_size_n: int
    n_stars: int
    kernel_path: str
    sidelobe_offset: int
    strip_width: int = DEFAULT_STRIP_WIDTH
    corner_roi_labels: List[str] | None = None
    centre_roi_label: str = "centre"
    corner_angles: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.corner_roi_labels is None:
            self.corner_roi_labels = [
                "top_left", "top_right", "bot_left", "bot_right"
            ]
        if self.corner_angles is None:
            # top_left/bot_right: spectrum runs at ~135° (anti-diagonal)
            # top_right/bot_left: spectrum runs at ~45° (main diagonal)
            # These are nominal values; measure precisely after optics
            # integration and pass the measured values as corner_angles.
            object.__setattr__(self, "corner_angles", {
                "top_left":  135.0,
                "top_right":  45.0,
                "bot_left":   45.0,
                "bot_right": 135.0,
            })


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compress_session(
    system: SystemConfig,
    session: "SessionConfig",
    params: CompressionParams,
) -> None:
    """
    Run the full post-session compression pipeline.

    Reads all .bin/.jsonl files for the session, processes them in
    batches of params.batch_size_n frames, and writes one .nc file
    per batch to system.data_dir.

    Parameters
    ----------
    system : SystemConfig
        Provides sensor dimensions, data paths, and compression settings.
    session : SessionConfig
        Provides camera mode and ROI geometry (mode, rois).
    params : CompressionParams
        Compression-specific parameters.
    """
    logger.info(
        "Starting compression: session=%s  batch_size=%d",
        params.session_id, params.batch_size_n,
    )

    kernel = load_deconvolution_kernel(params.kernel_path)

    # Load metadata (shared across all ROIs — use centre ROI meta file)
    meta_path = _meta_path(system.data_dir, params.session_id,
                           params.centre_roi_label)
    metadata = _load_metadata(meta_path)
    n_frames_total = len(metadata)
    logger.info("Total frames in session: %d", n_frames_total)

    if n_frames_total == 0:
        logger.warning("No frames found in metadata file. Nothing to compress.")
        return

    # Load all ROI frame stacks
    centre_stack = _load_frame_stack(
        system.data_dir, params.session_id,
        params.centre_roi_label,
        _centre_roi_shape(system, session),
        n_frames_total,
    )
    corner_stacks: Dict[str, np.ndarray] = {}
    for label in params.corner_roi_labels:
        shape = _corner_roi_shape(system, session, label)
        if shape is None:
            logger.warning("No shape found for corner ROI '%s' — skipping.", label)
            continue
        corner_stacks[label] = _load_frame_stack(
            system.data_dir, params.session_id,
            label, shape, n_frames_total,
        )

    # Get ROI offsets for coordinate transforms
    centre_offset_x, centre_offset_y = system.centre_roi_offset(session)
    corner_offsets = _get_corner_offsets(system, session)

    # Process in batches
    n_batches = (n_frames_total + params.batch_size_n - 1) // params.batch_size_n
    logger.info("Processing %d batch(es).", n_batches)

    for batch_idx in range(n_batches):
        start = batch_idx * params.batch_size_n
        end   = min(start + params.batch_size_n, n_frames_total)
        logger.info("Batch %d/%d: frames %d–%d", batch_idx + 1, n_batches,
                    start, end - 1)

        batch_meta   = metadata[start:end]
        batch_centre = centre_stack[start:end]
        batch_corners = {
            label: stack[start:end]
            for label, stack in corner_stacks.items()
        }

        _process_batch(
            system=system,
            session=session,
            params=params,
            batch_idx=batch_idx,
            batch_meta=batch_meta,
            batch_centre=batch_centre,
            batch_corners=batch_corners,
            kernel=kernel,
            centre_offset_x=centre_offset_x,
            centre_offset_y=centre_offset_y,
            corner_offsets=corner_offsets,
        )

    logger.info("Compression complete: %d batches written.", n_batches)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def _process_batch(
    system: SystemConfig,
    session: "SessionConfig",
    params: CompressionParams,
    batch_idx: int,
    batch_meta: List[dict],
    batch_centre: np.ndarray,
    batch_corners: Dict[str, np.ndarray],
    kernel: np.ndarray,
    centre_offset_x: int,
    centre_offset_y: int,
    corner_offsets: Dict[str, Tuple[int, int]],
) -> None:
    """Process one batch and write a netCDF file."""

    n = len(batch_meta)

    # ------------------------------------------------------------------
    # Star finding and sidelobe extraction
    # ------------------------------------------------------------------
    sidelobe_strips: List[Optional[np.ndarray]] = []

    for frame_idx in range(n):
        centre_frame = batch_centre[frame_idx]

        # Find stars in centre ROI (ROI-local coordinates)
        stars_local = find_stars(
            centre_frame, kernel,
            n_stars=params.n_stars,
        )

        # Convert to sensor coordinates
        stars_sensor = roi_local_to_sensor(
            stars_local["xs"], stars_local["ys"],
            centre_offset_x, centre_offset_y,
        )

        # Transform star positions into each corner ROI's local coordinates
        # and extract + merge all sidelobe strips for this frame.
        star_positions_per_corner = {
            label: sensor_to_roi_local(
                stars_sensor["xs"], stars_sensor["ys"],
                *corner_offsets.get(label, (0, 0)),
            )
            for label in batch_corners
        }

        corner_frames = {
            label: stack[frame_idx]
            for label, stack in batch_corners.items()
        }

        merged = crop_and_merge_corners(
            corner_frames,
            star_positions_per_corner,
            corner_angles=params.corner_angles,
            width=params.strip_width,
        )

        sidelobe_strips.append(merged)
        if merged is None:
            logger.warning(
                "Batch %d frame %d: no sidelobe strips extracted.",
                batch_idx, frame_idx,
            )

    # ------------------------------------------------------------------
    # Build sidelobe stack (fill missing frames with zeros)
    # ------------------------------------------------------------------
    valid_strips = [s for s in sidelobe_strips if s is not None]
    if not valid_strips:
        logger.error("Batch %d: no valid sidelobe strips. Skipping.", batch_idx)
        return

    strip_shape = valid_strips[0].shape
    sl_stack = np.zeros((n, *strip_shape), dtype=np.uint16)
    for i, strip in enumerate(sidelobe_strips):
        if strip is not None and strip.shape == strip_shape:
            sl_stack[i] = strip

    # ------------------------------------------------------------------
    # Difference images and adaptive bit reduction
    # ------------------------------------------------------------------
    centre_ref  = batch_centre[0]
    centre_diff, centre_dtype, centre_scale = _make_diffs(batch_centre)

    sl_ref  = sl_stack[0]
    sl_diff, sl_dtype, sl_scale = _make_diffs(sl_stack)

    # ------------------------------------------------------------------
    # Write netCDF
    # ------------------------------------------------------------------
    out_path = os.path.join(
        system.data_dir,
        f"{params.session_id}_batch{batch_idx:04d}.nc",
    )
    _write_netcdf(
        path=out_path,
        batch_meta=batch_meta,
        centre_ref=centre_ref,
        centre_diff=centre_diff,
        centre_dtype=centre_dtype,
        centre_scale=centre_scale,
        sl_ref=sl_ref,
        sl_diff=sl_diff,
        sl_dtype=sl_dtype,
        sl_scale=sl_scale,
        params=params,
    )
    logger.info("Written: %s", out_path)


# ---------------------------------------------------------------------------
# Difference image computation and adaptive bit reduction
# ---------------------------------------------------------------------------

def _make_diffs(
    stack: np.ndarray,
) -> Tuple[np.ndarray, str, float]:
    """
    Compute N-1 difference images and apply adaptive bit reduction.

    Parameters
    ----------
    stack : np.ndarray
        Shape (N, H, W), dtype uint16.

    Returns
    -------
    diffs : np.ndarray
        Shape (N-1, H, W). dtype is int8 or int16.
    dtype_name : str
        "int8" or "int16" — recorded in netCDF metadata.
    scale : float
        Scale factor applied before casting to int8, or 1.0 for int16.
        To reconstruct: original_diff = stored_value * scale.
    """
    # Widen to int32 before subtraction to avoid uint16 wraparound
    stack_i32 = stack.astype(np.int32)
    diffs_i32 = stack_i32[1:] - stack_i32[:-1]   # shape (N-1, H, W)

    diff_min = int(diffs_i32.min())
    diff_max = int(diffs_i32.max())
    diff_range = max(abs(diff_min), abs(diff_max))

    if diff_range <= _INT8_MAX:
        # Diffs fit in int8 without scaling — lossless
        return diffs_i32.astype(np.int8), "int8", 1.0
    else:
        # Store as int16 — fits full 12-bit difference range (±4095) losslessly
        logger.debug(
            "Diff range ±%d exceeds int8 — storing as int16.", diff_range
        )
        return diffs_i32.astype(np.int16), "int16", 1.0


# ---------------------------------------------------------------------------
# netCDF writer
# ---------------------------------------------------------------------------

def _write_netcdf(
    path: str,
    batch_meta: List[dict],
    centre_ref: np.ndarray,
    centre_diff: np.ndarray,
    centre_dtype: str,
    centre_scale: float,
    sl_ref: np.ndarray,
    sl_diff: np.ndarray,
    sl_dtype: str,
    sl_scale: float,
    params: CompressionParams,
) -> None:
    """Write one netCDF batch file."""

    n_diff = len(batch_meta) - 1

    with nc.Dataset(path, "w", format="NETCDF4") as ds:

        # Global attributes
        ds.session_id    = params.session_id
        ds.batch_size    = params.batch_size_n
        ds.n_stars       = params.n_stars
        ds.corner_angles = str(params.corner_angles)
        ds.strip_width   = params.strip_width
        ds.conventions   = "CF-1.8"

        # Dimensions
        ds.createDimension("centre_row",  centre_ref.shape[0])
        ds.createDimension("centre_col",  centre_ref.shape[1])
        ds.createDimension("sl_row",      sl_ref.shape[0])
        ds.createDimension("sl_col",      sl_ref.shape[1])
        ds.createDimension("diff_frame",  n_diff)

        # --- Centre ROI ---
        v = ds.createVariable(
            "centre_reference", "u2", ("centre_row", "centre_col"),
            zlib=True, complevel=system_deflate(params),
        )
        v.long_name   = "Centre ROI reference frame (frame 0 of batch)"
        v.units       = "ADU"
        v.valid_max   = np.uint16(_MAX_12BIT)
        v[:] = centre_ref

        diff_nc_dtype = "i1" if centre_dtype == "int8" else "i2"
        v = ds.createVariable(
            "centre_differences", diff_nc_dtype,
            ("diff_frame", "centre_row", "centre_col"),
            zlib=True, complevel=system_deflate(params),
        )
        v.long_name    = "Centre ROI difference images (frame[i] - frame[i-1])"
        v.units        = "ADU"
        v.dtype_used   = centre_dtype
        v.scale_factor = centre_scale
        v[:] = centre_diff

        # --- Sidelobe strip ---
        v = ds.createVariable(
            "sidelobe_reference", "u2", ("sl_row", "sl_col"),
            zlib=True, complevel=system_deflate(params),
        )
        v.long_name = "Merged sidelobe strip reference frame (frame 0 of batch)"
        v.units     = "ADU"
        v.valid_max = np.uint16(_MAX_12BIT)
        v[:] = sl_ref

        sl_nc_dtype = "i1" if sl_dtype == "int8" else "i2"
        v = ds.createVariable(
            "sidelobe_differences", sl_nc_dtype,
            ("diff_frame", "sl_row", "sl_col"),
            zlib=True, complevel=system_deflate(params),
        )
        v.long_name    = "Sidelobe strip difference images"
        v.units        = "ADU"
        v.dtype_used   = sl_dtype
        v.scale_factor = sl_scale
        v[:] = sl_diff

        # --- Per-frame metadata ---
        ds.createDimension("frame", len(batch_meta))

        ts = ds.createVariable("timestamp_ns", "i8", ("frame",))
        ts.long_name = "Camera hardware timestamp (nanoseconds, camera epoch)"
        ts[:]        = [m["timestamp_ns"] for m in batch_meta]

        fid = ds.createVariable("frame_id", "i8", ("frame",))
        fid.long_name = "Monotonic acquisition frame counter (acq_nframe)"
        fid[:]        = [m["frame_id"] for m in batch_meta]

        ht = ds.createVariable("host_time", "f8", ("frame",))
        ht.long_name  = "Host monotonic time at frame acquisition"
        ht.units      = "seconds"
        ht[:]         = [m["host_time"] for m in batch_meta]

        nd = ds.createVariable("nframes_dropped", "i4", ("frame",))
        nd.long_name  = "Frames dropped before this frame (from acq_nframe gap)"
        nd[:]         = [m["nframes_dropped"] for m in batch_meta]


def system_deflate(params: CompressionParams) -> int:
    """Return deflate level — extracted here to allow easy override in tests."""
    return 5   # default; could be passed via CompressionParams if needed


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_metadata(path: str) -> List[dict]:
    """Load a .jsonl metadata file into a list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_frame_stack(
    data_dir: str,
    session_id: str,
    roi_label: str,
    frame_shape: Tuple[int, int],
    n_frames: int,
) -> np.ndarray:
    """
    Load all frames for one ROI from its .bin file.

    Returns np.ndarray of shape (n_frames, height, width), dtype uint16.
    """
    path = os.path.join(data_dir, f"{session_id}_{roi_label}_frames.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Frame file not found: {path}")

    raw = np.fromfile(path, dtype=np.uint16)
    h, w = frame_shape
    expected = n_frames * h * w
    if raw.size != expected:
        raise ValueError(
            f"{path}: expected {expected} values ({n_frames}×{h}×{w}), "
            f"got {raw.size}."
        )
    return raw.reshape(n_frames, h, w)


def _meta_path(data_dir: str, session_id: str, roi_label: str) -> str:
    return os.path.join(data_dir, f"{session_id}_{roi_label}_meta.jsonl")


def _centre_roi_shape(
    system: SystemConfig,
    session: SessionConfig,
) -> Tuple[int, int]:
    """Return (height, width) of the centre ROI from SessionConfig."""
    if session.mode is CameraMode.SINGLE_ROI and session.rois:
        roi = session.rois[0]
        return (roi.height, roi.width)
    if session.mode is CameraMode.MULTI_ROI and session.rois:
        # Centre ROI is index 4 per make_rois() convention
        centre = session.rois[4]
        return (centre.height, centre.width)
    # FULL_FRAME: use sensor dimensions
    return (system.sensor_height, system.sensor_width)


def _corner_roi_shape(
    system: SystemConfig,
    session: SessionConfig,
    label: str,
) -> Optional[Tuple[int, int]]:
    """Return (height, width) of a named corner ROI from SessionConfig."""
    if session.mode is CameraMode.MULTI_ROI and session.rois:
        # Corner ROIs are indices 0, 2, 6, 8 per make_rois() convention
        corner_indices = {
            "top_left": 0, "top_right": 2,
            "bot_left": 6, "bot_right": 8,
        }
        idx = corner_indices.get(label)
        if idx is not None:
            roi = session.rois[idx]
            return (roi.height, roi.width)
    return None


def _get_corner_offsets(
    system: SystemConfig,
    session: SessionConfig,
) -> Dict[str, Tuple[int, int]]:
    """
    Return {roi_label: (offset_x, offset_y)} for the four corner ROIs.

    Derived from session.rois (set during acquisition).
    Returns an empty dict for non-MULTI_ROI modes.
    """
    if session.mode is not CameraMode.MULTI_ROI or not session.rois:
        return {}

    # Layout from make_rois():
    # index 0=top_left, 2=top_right, 6=bot_left, 8=bot_right
    corner_indices = {
        "top_left":  0,
        "top_right": 2,
        "bot_left":  6,
        "bot_right": 8,
    }
    return {
        label: (session.rois[idx].offset_x, session.rois[idx].offset_y)
        for label, idx in corner_indices.items()
        if idx < len(session.rois)
    }