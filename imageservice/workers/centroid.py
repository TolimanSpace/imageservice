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
    n_rows: int
    n_cols: int

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
