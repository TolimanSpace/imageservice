"""
star_finder.py — Star detection in the centre ROI via deconvolution.

Provides find_stars(), which deconvolves the centre ROI image using a
pre-computed kernel and returns the pixel coordinates of the N brightest 
peaks. Used by the compression pipeline to determine star positions 
for sidelobe strip extraction.

Deconvolution kernel
--------------------
The deconvolution constant is a pre-computed array stored as a .npy file. 
It encodes the inverse filter for the system PSF and is applied as a 
pointwise multiplication in the frequency domain:

    result = IFFT( FFT(image / max(image)) * kernel )

The kernel file path is supplied via SystemConfig (or passed directly
to load_deconvolution_kernel()). The kernel must match the shape of
the centre ROI exactly — if the ROI size changes, a new kernel is
needed.

The kernel is loaded once at the start of a compression session and
passed into find_stars() explicitly.

Star positions
--------------
find_stars() returns star positions in the LOCAL coordinate system of
the image passed in (i.e. pixel indices within the centre ROI array).
The caller is responsible for converting to sensor coordinates:

    x_sensor = x_local + centre_roi_offset_x
    y_sensor = y_local + centre_roi_offset_y

and then to corner ROI-local coordinates for sidelobe extraction:

    x_corner_local = x_sensor - corner_roi_offset_x
    y_corner_local = y_sensor - corner_roi_offset_y
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from scipy.fft import fft2, fftshift, ifft2
from scipy.ndimage import maximum_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel loading
# ---------------------------------------------------------------------------

def load_deconvolution_kernel(path: str) -> np.ndarray:
    """
    Load the pre-computed deconvolution kernel from a .npy file.

    The kernel should be a complex array of the same shape as the centre
    ROI (e.g. 128×128). It is applied pointwise in the frequency domain.

    Parameters
    ----------
    path : str
        Path to the .npy kernel file.

    Returns
    -------
    np.ndarray
        Complex kernel array.

    Raises
    ------
    FileNotFoundError
        If the kernel file does not exist.
    ValueError
        If the loaded array is not 2-D.
    """
    logger.info("Loading deconvolution kernel from: %s", path)
    kernel = np.load(path)
    if kernel.ndim != 2:
        raise ValueError(
            f"Deconvolution kernel must be 2-D, got shape {kernel.shape}"
        )
    logger.info("Kernel loaded: shape=%s  dtype=%s", kernel.shape, kernel.dtype)
    return kernel


# ---------------------------------------------------------------------------
# Deconvolution
# ---------------------------------------------------------------------------

def deconvolve_image(
    image: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Apply the deconvolution kernel to a centre ROI image.

    Normalises the image by its maximum value before transforming, to
    make the result independent of absolute flux level.

    Parameters
    ----------
    image : np.ndarray
        2-D uint16 centre ROI array.
    kernel : np.ndarray
        Pre-computed frequency-domain filter kernel (complex, same shape
        as image). Load with load_deconvolution_kernel().

    Returns
    -------
    np.ndarray
        Real-valued deconvolved image, same shape as input.

    Raises
    ------
    ValueError
        If image and kernel shapes do not match.
    """
    if image.shape != kernel.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match kernel shape "
            f"{kernel.shape}. A new kernel is needed if the ROI size changed."
        )

    max_val = np.nanmax(image)
    if max_val == 0:
        logger.debug("deconvolve_image: image is all zeros, returning zeros.")
        return np.zeros_like(image, dtype=np.float32)

    f_image = fft2(image.astype(np.float32) / max_val)
    result = fftshift(np.real(ifft2(f_image * kernel)))
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Star finding
# ---------------------------------------------------------------------------

def find_stars(
    image: np.ndarray,
    kernel: np.ndarray,
    n_stars: int = 2,
    threshold: float = 0.1,
    filter_size: int = 5,
) -> Dict[str, List[float]]:
    """
    Find the N brightest stars in a centre ROI image.

    Deconvolves the image with the supplied kernel, finds local maxima
    above a relative threshold, and returns the pixel coordinates of
    the N brightest candidates.

    Parameters
    ----------
    image : np.ndarray
        2-D uint16 centre ROI array.
    kernel : np.ndarray
        Pre-computed deconvolution kernel. Load once per session with
        load_deconvolution_kernel() and pass in on every call.
    n_stars : int
        Number of stars to return. Default 2 (binary star system).
    threshold : float
        Fraction of the deconvolved image peak below which candidates
        are rejected. Default 0.1 (10% of peak).
    filter_size : int
        Neighbourhood size for local maximum detection. Larger values
        suppress closely spaced false peaks.

    Returns
    -------
    dict with keys:
        "xs" : List[float]  — column coordinates (ROI-local)
        "ys" : List[float]  — row coordinates (ROI-local)

    If fewer than n_stars candidates are found, the returned lists will
    be shorter than n_stars. The caller should handle this case.

    Notes
    -----
    Coordinates are returned as floats to match the interface expected
    by crop_sidelobes(), though the underlying detection is at integer
    pixel resolution. Sub-pixel refinement (e.g. fitting a Gaussian to
    each peak) could be added here if needed.
    """
    deconvolved = deconvolve_image(image, kernel)

    peak = np.nanmax(deconvolved)
    if peak <= 0:
        logger.warning("find_stars: deconvolved image has no positive values.")
        return {"xs": [], "ys": []}

    local_max = deconvolved == maximum_filter(deconvolved, size=filter_size)
    thresh_mask = deconvolved > threshold * peak
    candidates = np.argwhere(local_max & thresh_mask)   # shape (N, 2) — (row, col)

    if len(candidates) == 0:
        logger.warning(
            "find_stars: no candidates above threshold=%.2f.", threshold
        )
        return {"xs": [], "ys": []}

    # Sort by deconvolved brightness, descending
    sorted_candidates = sorted(
        candidates,
        key=lambda c: deconvolved[c[0], c[1]],
        reverse=True,
    )

    xs: List[float] = []
    ys: List[float] = []
    for candidate in sorted_candidates[:n_stars]:
        ys.append(float(candidate[0]))
        xs.append(float(candidate[1]))

    if len(xs) < n_stars:
        logger.warning(
            "find_stars: found %d star(s), expected %d. "
            "Consider lowering threshold (currently %.2f).",
            len(xs), n_stars, threshold,
        )

    logger.debug(
        "find_stars: found %d star(s) at xs=%s ys=%s",
        len(xs), xs, ys,
    )
    return {"xs": xs, "ys": ys}


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def roi_local_to_sensor(
    xs: List[float],
    ys: List[float],
    roi_offset_x: int,
    roi_offset_y: int,
) -> Dict[str, List[float]]:
    """
    Convert star positions from centre ROI-local to sensor coordinates.

    Parameters
    ----------
    xs, ys : list of float
        Star positions in centre ROI-local pixel coordinates.
    roi_offset_x, roi_offset_y : int
        Centre ROI offset from the sensor origin (from SystemConfig).

    Returns
    -------
    dict with keys "xs", "ys" in sensor coordinates.
    """
    return {
        "xs": [x + roi_offset_x for x in xs],
        "ys": [y + roi_offset_y for y in ys],
    }


def sensor_to_roi_local(
    xs: List[float],
    ys: List[float],
    roi_offset_x: int,
    roi_offset_y: int,
) -> Dict[str, List[float]]:
    """
    Convert star positions from sensor coordinates to a corner ROI's
    local coordinate system.

    Parameters
    ----------
    xs, ys : list of float
        Star positions in sensor pixel coordinates.
    roi_offset_x, roi_offset_y : int
        The corner ROI's offset from the sensor origin.

    Returns
    -------
    dict with keys "xs", "ys" in ROI-local coordinates.
    """
    return {
        "xs": [x - roi_offset_x for x in xs],
        "ys": [y - roi_offset_y for y in ys],
    }