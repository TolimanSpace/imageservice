"""
sim_converter.py - Convert simulated photon-count images to AcquiredFrame.

Bridges the simulation pipeline and the real acquisition pipeline by
converting simulator output (integer photon counts) into the same
AcquiredFrame objects that camera.py produces during real acquisition.
The resulting frames can be fed directly into:

    - writer.push(roi_arr, frame)
    - compute_pointing_error(frame.rois[label], ...)
    - compressor.compress_session(...)

Signal chain
------------
The conversion follows the physical signal chain of the real camera:

    photons
        × QE          -> electrons       (quantum efficiency, 0–1)
        / gain         -> ADU             (electrons per ADU)
        + bias         -> ADU with offset (ADC zero point)
        clip [0, 4095] -> 12-bit ADU
        cast uint16   -> raw camera output

All conversion is done in int64 to avoid overflow before clipping.

Noise is NOT added here - the simulator is expected to have already
included shot noise (Poisson) and read noise (Gaussian) in the photon
counts it provides.

Camera models
-------------
Physical constants are stored in CameraModel dataclasses, one per
supported camera. Add new models to CAMERA_MODELS.

    MC203MG-SY-UB: Ximea xiC series, Sony IMX342 sensor, 12-bit.
      QE:   0.65 (peak ~650 nm, Sony datasheet)
      gain: 0.37 e-/ADU (full well ~15000 e-, 12-bit range 4096 ADU)
      bias: 100  ADU (typical dark level at 12-bit)

    MQ013MG-E2: Ximea xiQ series, ON Semi AR0134 sensor, 12-bit.
      QE:   0.48 (peak ~525 nm)
      gain: 0.5  e-/ADU
      bias: 64   ADU

Usage
-----
Full-frame simulated image -> AcquiredFrame (ROIs cropped from session geometry):

    from workers.sim_converter import full_frame_to_acquired_frame

    photons = simulator.render()    # shape (sensor_H, sensor_W), int64
    frame   = full_frame_to_acquired_frame(
        photon_image=photons,
        session=session,
        frame_id=i,
        camera_model="MC203MG-SY-UB",
    )
    writer.push(frame.rois["roi"], frame)

Per-ROI simulated images -> AcquiredFrame:

    from workers.sim_converter import rois_to_acquired_frame

    roi_photons = {
        "centre":    simulator.render_centre(),
        "top_left":  simulator.render_corner("top_left"),
        ...
    }
    frame = rois_to_acquired_frame(
        roi_photons=roi_photons,
        session=session,
        frame_id=i,
        camera_model="MC203MG-SY-UB",
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from workers.camera import AcquiredFrame, CameraMode, RoiDefinition
from workers.config import SessionConfig

# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraModel:
    """
    Physical conversion constants for one camera model.

    Attributes
    ----------
    name : str
        Human-readable camera model name, e.g. "MC203MG-SY-UB".
    quantum_efficiency : float
        Fraction of incident photons converted to electrons (0–1).
    electrons_per_adu : float
        Conversion gain in electrons per ADU. Lower = more sensitive.
    bias_adu : int
        ADC zero-point offset in ADU added to every pixel.
    bit_depth : int
        ADC bit depth. Maximum output value = 2**bit_depth - 1.
        Must match camera.output_bit_depth in SystemConfig.
    full_well_electrons : int
        Approximate full-well capacity in electrons. Informational only
        (clipping is done at bit_depth level, not here).
    notes : str
        Free-text notes on the source of these constants.
    """
    name:                 str
    quantum_efficiency:   float
    electrons_per_adu:    float
    bias_adu:             int
    bit_depth:            int = 12
    full_well_electrons:  int = 0
    notes:                str = ""

    @property
    def max_adu(self) -> int:
        return (1 << self.bit_depth) - 1


# ---------------------------------------------------------------------------
# Camera model registry
# ---------------------------------------------------------------------------

CAMERA_MODELS: Dict[str, CameraModel] = {

    "MC203MG-SY-UB": CameraModel(
        name="MC203MG-SY-UB",
        quantum_efficiency=0.65,
        electrons_per_adu=0.37,
        bias_adu=100,
        bit_depth=12,
        full_well_electrons=15_000,
        notes=(
            "Ximea xiC series, Sony IMX342 sensor. "
            "QE peak ~650 nm (Sony datasheet). "
            "Gain from full-well capacity / ADC range: ~15000 e / 4096 ADU. "
            "Bias from typical dark-frame measurement at 12-bit."
        ),
    ),

    "MQ013MG-E2": CameraModel(
        name="MQ013MG-E2",
        quantum_efficiency=0.48,
        electrons_per_adu=0.50,
        bias_adu=64,
        bit_depth=12,
        full_well_electrons=10_000,
        notes=(
            "Ximea xiQ series, ON Semiconductor AR0134 sensor. "
            "QE peak ~525 nm. "
            "Gain and bias from ON Semi AR0134 datasheet."
        ),
    ),

    "MD120MU-SY": CameraModel(
        name="MD120MU-SY",
        quantum_efficiency=0.60,
        electrons_per_adu=0.25,
        bias_adu=100,
        bit_depth=12,
        full_well_electrons=12_000,
        notes=(
            "Ximea xiD series, Sony IMX304 sensor. "
            "QE peak ~600 nm. "
            "Gain and bias are estimates - update from datasheet."
        ),
    ),
}


def get_camera_model(name: str) -> CameraModel:
    """
    Look up a CameraModel by name.

    Parameters
    ----------
    name : str
        Camera model name, e.g. "MC203MG-SY-UB". Case-sensitive.

    Returns
    -------
    CameraModel

    Raises
    ------
    KeyError
        If the model name is not in CAMERA_MODELS.
    """
    if name not in CAMERA_MODELS:
        raise KeyError(
            f"Unknown camera model '{name}'. "
            f"Available: {list(CAMERA_MODELS.keys())}"
        )
    return CAMERA_MODELS[name]


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def photons_to_adu(
    photon_image: np.ndarray,
    model: CameraModel,
) -> np.ndarray:
    """
    Convert a photon-count image to a 12-bit ADU uint16 array.

    Signal chain:
        photons × QE -> electrons
        electrons / gain -> ADU (float)
        ADU + bias -> ADU with offset
        clip [0, max_adu] -> valid ADU range
        cast -> uint16

    Parameters
    ----------
    photon_image : np.ndarray
        Integer photon counts, any shape. Noise should already be
        included by the simulator.
    model : CameraModel
        Conversion constants for the camera.

    Returns
    -------
    np.ndarray
        uint16 array, same shape as photon_image, values in [0, max_adu].
    """
    # Work in int64 throughout to avoid overflow before clipping
    electrons = (photon_image.astype(np.int64) * model.quantum_efficiency)
    adu       = (electrons / model.electrons_per_adu).astype(np.int64)
    adu      += model.bias_adu
    adu       = np.clip(adu, 0, model.max_adu)
    return adu.astype(np.uint16)


# ---------------------------------------------------------------------------
# Full-frame -> AcquiredFrame
# ---------------------------------------------------------------------------

def full_frame_to_acquired_frame(
    photon_image: np.ndarray,
    session: SessionConfig,
    frame_id: int,
    camera_model: str,
    timestamp_ns: Optional[int] = None,
    host_time: Optional[float] = None,
    nframes_dropped: int = 0,
) -> AcquiredFrame:
    """
    Convert a full-frame simulated photon-count image to an AcquiredFrame.

    Applies the camera signal chain (QE -> gain -> bias -> clip -> uint16)
    and crops out the ROIs defined in session.rois, producing the same
    dict of {label: uint16_array} that camera.py would populate from
    real hardware.

    Use this when the simulator renders a complete sensor image and the
    pipeline should see the same per-ROI view that real acquisition
    produces.

    Parameters
    ----------
    photon_image : np.ndarray
        Shape (sensor_H, sensor_W), integer photon counts.
    session : SessionConfig
        Provides ROI geometry (rois list) and camera mode. For
        FULL_FRAME mode, the entire converted image is returned under
        the key "full_frame". For SINGLE_ROI and MULTI_ROI, each
        data-bearing ROI (is_data=True) is cropped and returned under
        its label.
    frame_id : int
        Monotonically increasing frame counter.
    camera_model : str
        Camera model name, e.g. "MC203MG-SY-UB".
    timestamp_ns : int or None
        Simulated hardware timestamp in nanoseconds. If None, uses
        time.monotonic_ns() at call time.
    host_time : float or None
        Host monotonic time. If None, uses time.monotonic().
    nframes_dropped : int
        Number of frames dropped before this one (for drop tracking).

    Returns
    -------
    AcquiredFrame
    """
    model = get_camera_model(camera_model)
    adu   = photons_to_adu(photon_image, model)

    if session.mode is CameraMode.FULL_FRAME:
        rois = {"full_frame": adu}

    else:
        # Crop each data-bearing ROI from the full-frame ADU image
        if session.rois is None:
            raise ValueError(
                "session.rois is None but mode is not FULL_FRAME. "
                "Cannot crop ROIs without geometry."
            )
        rois = _crop_rois(adu, [r for r in session.rois if r.is_data])

    return _make_frame(rois, session, frame_id,
                       timestamp_ns, host_time, nframes_dropped)


# ---------------------------------------------------------------------------
# Per-ROI -> AcquiredFrame
# ---------------------------------------------------------------------------

def rois_to_acquired_frame(
    roi_photons: Dict[str, np.ndarray],
    session: SessionConfig,
    frame_id: int,
    camera_model: str,
    timestamp_ns: Optional[int] = None,
    host_time: Optional[float] = None,
    nframes_dropped: int = 0,
) -> AcquiredFrame:
    """
    Convert per-ROI simulated photon-count images to an AcquiredFrame.

    Use this when the simulator renders each ROI separately (e.g. the
    simulator knows the optical layout and renders centre and corner ROIs
    individually).

    Parameters
    ----------
    roi_photons : dict
        {roi_label: photon_count_array}. Labels must match the ROI
        labels in session.rois (e.g. "centre", "top_left", etc.).
        Only data-bearing ROIs (is_data=True) need to be provided;
        extra keys are passed through unchanged.
    session : SessionConfig
        Provides the ROI label list for validation.
    frame_id : int
    camera_model : str
    timestamp_ns : int or None
    host_time : float or None
    nframes_dropped : int

    Returns
    -------
    AcquiredFrame
    """
    model = get_camera_model(camera_model)

    rois = {
        label: photons_to_adu(arr, model)
        for label, arr in roi_photons.items()
    }

    return _make_frame(rois, session, frame_id,
                       timestamp_ns, host_time, nframes_dropped)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crop_rois(
    adu_image: np.ndarray,
    roi_defs: list,
) -> Dict[str, np.ndarray]:
    """
    Crop a list of RoiDefinition regions from a full-frame ADU image.

    Returns {label: uint16_array} for each ROI.
    Raises ValueError if any ROI extends beyond the image boundary.
    """
    h, w = adu_image.shape
    rois = {}
    for roi in roi_defs:
        y0 = roi.offset_y
        y1 = roi.offset_y + roi.height
        x0 = roi.offset_x
        x1 = roi.offset_x + roi.width
        if y1 > h or x1 > w:
            raise ValueError(
                f"ROI '{roi.label}' ({x0}:{x1}, {y0}:{y1}) extends beyond "
                f"image boundary ({w}×{h})."
            )
        rois[roi.label] = adu_image[y0:y1, x0:x1].copy()
    return rois


def _make_frame(
    rois: Dict[str, np.ndarray],
    session: SessionConfig,
    frame_id: int,
    timestamp_ns: Optional[int],
    host_time: Optional[float],
    nframes_dropped: int,
) -> AcquiredFrame:
    """Construct an AcquiredFrame from already-converted ROI arrays."""
    return AcquiredFrame(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns if timestamp_ns is not None
                     else time.monotonic_ns(),
        host_time=host_time if host_time is not None
                  else time.monotonic(),
        mode=session.mode,
        rois=rois,
        nframes_dropped=nframes_dropped,
    )
