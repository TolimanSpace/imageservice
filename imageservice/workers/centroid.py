from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CentroidResult:
    """
    Centroid of the centre ROI in ROI-local pixel coordinates

    Attributes
    ----------
    x : float
        Column coordinate within the centre ROI. Sub-pixel precision.
        (0.0, 0.0) is the centre of the top-left pixel of the ROI
    y : float
        Row coordinate within the centre ROI. Sub-pixel precision.
    total_intensity : float
        Sum of all pixel values. Potentially useful as a signal strength
        indicator for quality control
    peak_value : int
        Maximum pixel value in the ROI. QA for saturation.
    """
    x: float
    y: float
    total_intensity: float
    peak_value: int

@dataclass(frozen=True)
class PointingError:
    """
    Deviation of the system centroid from the sensor centre.

    This is the primary output of the centroid module and the quantity
    consumed by the pointing controller.

    Attributes
    ----------
    dx : float
        Horizontal deviation in pixels. Postive = star is right of centre
    dy : float
        Vertical deviation in pixels. Positive = star is below centre
        (rows increase downward in NumPy convention).
    centroid : Centroid Result
        The underlying ROI-local centroid from which the error is derived.
        Carried along so the controller can gate on signal quality via
        centroid.total_intensity or centroid.peak_value
    x_sensor : float
        Absolute centroid position in sensor pixels coordinates (column)
    y_sensor : float
        Absolute centroid poistion in sensor pixel coordinates (row)
    timestamp_ns : int
        Camera hardware timestamp in nanoseconds, copied directly from
        AcquiredFrame.timestamp_ns. This is recorded at the start of the
        exposure. Pointing controller can use to account for pipeline
        latency between exposure and delivery. Epoch is camera-internal
        and arbitrary. Needs to be synchronised to an external reference
        or used as a relative measure.
    frame_id : int
        The acq_nframe value from the source AcquiredFrame. Allows the
        pointing controller to detect if a pointing error update has
        been skipped (e.g. due to no-signal) or dropped frames.
    """
    dx: float
    dy: float
    centroid: CentroidResult
    x_sensor: float
    y_sensor: float
    timestamp_ns: int
    frame_id: int

    def as_tuple(self) -> tuple:
        "Return (dx, dy) as a plain tuple"
        return (self.dx, self.dy)

#-----------------------------------
# Centroid calculation
#-----------------------------------

def compute_centroid(
    image: np.ndarray,
    min_total_intensity: float = 1.0
) -> Optional[CentroidResult]:
    """
    Compute the intensity-weighted centroid of a 2D image array

    Uses the full ROI: every pixel contribues with a weight equal to its
    intensity value. Result is in ROI-local pixel coordinates.

    Parameters
    ----------
    image : np.ndarray
        2D array of shape (height, width). Expected dtyp is uint16, with
        12-bit data in a 16-bit container, values in [0, 4095], but any
        numeric dtype is accepted.
    min_total_intensity : float
        Minimum summed intensity to return a result. If the total signal
        is below this threshold the ROI is considered to have no usable
        signal and None is returned. The defautl of 1.0 rejects only an
        all-zero case, but should be increased to a meaninful level in
        production.

    Returns
    -------
    CentroidResult or None
        None if total intensity is below the min_total_intensity

    Raises
    ------
    ValueError
        If Image is not a 2D array

    Notes
    -----
    The weighted centroid formula is computed using 1D reductions rather
    than a full-meshgrid for efficiency:
        col_sums = image.sum(axis=0)
        row_sums = image.sum(axis=1)
        x = dot(col_indices, col_sums) / total
        y = dot(row_indices, row_sums) / total
    """
    if image.ndim != 2:
        raise ValueError(
            f"Image must be 2D, got shape {image.shape}"
        )

    n_rows, n_cols = image.shape

    # Float64 to avoid uint16 overflow on large ROI
    total = image.sum(dtype=np.float64)

    if total < min_total_intensity:
        logger.debug(
            "compute_centroid: total intensity %.1f below threshold %.1f.",
            total, min_total_intensity
        )
        return None

    col_sums = image.sum(axis=0, dtype=np.float64)
    row_sums = image.sum(axis=1, dtype=np.float64)

    cols = np.arange(n_cols, dtype=np.float64)
    rows = np.arange(n_rows, dtype=np.float64)

    x = np.dot(cols, col_sums) / total
    y = np.dot(rows, row_sums) / total

    return CentroidResult(
        x = x,
        y = y,
        total_intensity = total,
        peak_value = int(image.max())
    )

#-------------------------------
# Pointing error computation
#-------------------------------

def compute_pointing_error(
    centre_roi: np.ndarray,
    roi_offset_x: int,
    roi_offset_y: int,
    sensor_width: int,
    sensor_height: int,
    timestamp_ns: int,
    frame_id: int,
    min_total_intensity: float = 1.0,
) -> Optional[PointingError]:
    """
    Compute the pointing error from the centre ROI image.

    Combines centroid computation, ROI-to-sensor coordinate conversion, and
    deviation from sensor centre into a single call suitable for the acquisition loop.

    Parameters
    ----------
    centre_roi : np.ndarray
        2D uint16 array for the centre ROI, as found in AcquiredFrame.rois[CENTRE_ROI_LABEL].
    roi_offset_x : int
        x-offset of the centre ROI on the sensor in pixels. From:
        CameraConfig.rois[4].offset_x (MULTI_ROI) or
        CameraConfig.rois[0].offset_x (SINGLE_ROI)
    roi_offset_y : int
        y-offset of the centre ROI on the sensor in pixels
    sensor_width : int
        Full sensor width in pixels. The centre is at sensor_width / 2
    sensor_height : int
        Full sensor height in pixels. The centre is at sensor_height / 2
    timestamp_ns : int
        Camera hardware timestamp in nanoseconds. From AcquiredFrame.timestamp_ns.
        Passed through directly without modification
    frame_id : int
        Acquisition frame counter. From AcquiredFrame.frame_id.
        Passed through directly without modificiation
    min_total_intensity : float
        Passed through to compute_centroid(), which returns None if the ROI has
        insufficient signal.

    Returns
    -------
    PointingError or None
        None if the centre ROI has insufficient signal
    """

    centroid = compute_centroid(centre_roi, min_total_intensity)

    # Insufficient flux
    if centroid is None:
        return None

    # Convert ROI-local centroid to sensor coordinators
    x_sensor = centroid.x + roi_offset_x
    y_sensor = centroid.y + roi_offset_y

    # Deviation from sensor centre
    dx = x_sensor - sensor_width / 2.0
    dy = y_sensor - sensor_height / 2.0

    return PointingError(
        dx = dx,
        dy = dy,
        centroid = centroid,
        x_sensor = x_sensor,
        y_sensor = y_sensor,
        timestamp_ns = timestamp_ns,
        frame_id = frame_id,
    )
