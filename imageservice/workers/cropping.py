"""
cropping.py - ROI cropping and sidelobe strip extraction.

Provides two main operations

 1. crop_centre()   -   Extract a square sub-image centred on a given pixel position.
                        Mostly legacy from a non-ROI implementation to get the core
                        image, but may still be useful.

 2. crop_sidelobes() -  Extract narrow strips (~6 pixels wide) along the diagonal
                        sidelobe spectra in each corner region of interest, and
                        merge them into a single rectangular array. Each star
                        produces two sidelobe axes (at angle_degrees and 
                        angle_degrees+90 degrees), each with two arms, giving 
                        four strips per star. For two stars, eight strips are
                        merged into one array.

Coordinate conventions
----------------------
All position arguments (x_pos, y_pos, x_poss, y_poss) are in the local coordinate
systems of the image array passed in - i.e. column and row indices within that array,
not sensor coordinates. The caller is responsible for transforming sensor coordinates
to region-of-interest local coordinates before calling these functions.

Conversion between sensor and region of interest coordinates are:

    x_local = x_sensor - roi_offset_x
    y_local = y_sensor - roi_offset_y

where roi_offset_x/y are the region of interest's offsets given in the SessionConfig.

Sidelobe geometry
-----------------

"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Dict

import numpy as np

logger = logging.getLogger(__name__)

# Default values
# Sidelobe offset - how far the middle of a sidelobe is from its star
DEFAULT_SIDELOBE_OFFSET = 1625

# Default strip with in pixels along the perpenticular to the dispersion axis
DEFAULT_STRIP_WIDTH = 6

# Default strip length in pixels along the dispersion axis
DEFAULT_STRIP_LENGTH = 360

# Default dispersion axis angle
DEFAULT_ANGLE_DEGREES: Dict[str, float] = {
    "top_left": 135.0,
    "top_right": 45.0,
    "bot_left": 45.0,
    "bot_right": 135.0
}


#-------------------------------
# Region of interest crop
#-------------------------------

def crop_centre(
    image: np.ndarray,
    x_pos: float,
    y_pos: float,
    size: int=128,
) -> np.ndarray:
    """
    Return a square sub image centred at (x_pos, y_pos)

    Parameters
    ----------
    image : np.ndarray
        2-D array of flux values
    x_pos : float
        Column coordinater of the centre point
    y_pos : float
        Row coordinate of the centre point
    size : int
        Side length of the output square in pixels (default 128)

    Returns
    -------
    np.ndarray
        Cropped subimage of shape (size, size) or smaller if
        requested region extends beyond the image boundary
    """
    # Round positions
    x_pos = round(x_pos)
    y_pos = round(y_pos)

    # Calculate dimensions of the cutout
    half_size = size // 2
    start_x = max(x_pos - half_size, 0)
    start_y = max(y_pos - half_size, 0)
    end_x = min(x_pos + half_size + (size % 2), image.shape[1])
    end_y = min(y_pos + half_size + (size % 2), image.shape[0])

    return image[start_y:end_y, start_x:end_x]


#------------------------------------------
# Rectangular area crop (internal helper)
#------------------------------------------

def _crop_areas(
    image: np.ndarray,
    x_poss: np.ndarray,
    y_poss: np.ndarray,
    size: Tuple[int, int],
) -> np.ndarray:
    """
    Extract multiple rectangular sub-images from a single image

    Args:
        image (numpy.ndarray): ndarray of flux values
        x_poss (list): list of X coordinates of centre points
        y_poss (list): list of Y coordinates of centre points
        size (tuple): size of each image (default (480, 360))

    Returns
    -------
    np.ndarray of shape (n_positions, height, width)
    """
    h, w = size

    output = []

    for x_pos, y_pos in zip(x_poss, y_poss):
        # Round positions
        x_pos = round(x_pos)
        y_pos = round(y_pos)

        # Calculate dimensions of the cutout
        hx, hy = w //2, h //2
        start_x = max(x_pos - hx, 0)
        start_y = max(y_pos - hy, 0)
        end_x = min(x_pos + hx + (w % 2), image.shape[1])
        end_y = min(y_pos + hy + (h % 2), image.shape[0])

        # Extract cutout
        output.append(image[start_y:end_y, start_x:end_x])

    return np.asarray(output)

#-----------------------
# Sidelobe cropping
#-----------------------

def crop_sidelobe_strip(
    image: np.ndarray,
    x_star: float,
    y_star: float,
    angle_degrees: float,
    width: int = DEFAULT_STRIP_WIDTH,
) -> np.ndarray:
    """
    Extract a narrow strip along the dispersion axis in a corner region of interest

    The mask selects all pixels within width/2 perpendicular pixels of the line
    through (x_star, y_star) at angle_degrees:

        |(x-x_star)*sin(angle_degrees) - (y-y_star)*cos(angle_degrees)| <= width/2

    Parameters
    ----------
    image : np.ndarray
        2-D uint16 corner region of interest array
    x_star : float
        Star column position in local image coordinates
    y_star : float
        Star row position in local image coordinates
    angle_degrees : float
        Dispersion axis angle from the column axis in degrees.
    width : int
        Strip width in pixels (perpendicular to dispersion axis)
    """
    angle_rad = np.deg2rad(angle_degrees)
    cot_a = 1.0 / np.tan(angle_rad)

    y_grid, x_grid = np.indices(image.shape)
    xx = x_grid - x_star
    yy = y_grid - y_star

    mask = np.abs(xx - yy * cot_a) <= width / 2.0

    if not mask.any():
        return np.empty((0,width), dtype=image.dtype)

    # Find image columns with exactly 'width' masked pixels
    col_counts = mask.sum(axis=0)
    complete_cols = np.where(col_counts == width)[0]

    if complete_cols.size == 0:
        return np.empty((0,width), dtype=image.dtype)

    # Extract masked pixels for complete cols
    sub_mask = mask[:, complete_cols]
    sub_image = image[:, complete_cols]

    pixels_col_major = sub_image.flatten(order='F')
    mask_col_major = sub_mask.flatten(order='F')

    return pixels_col_major[mask_col_major].reshape(len(complete_cols), width)


def crop_and_merge_corners(
    corner_rois: Dict[str, np.ndarray],
    star_positions_per_corner: Dict[str, Dict[str, List[float]]],
    corner_angles: Optional[Dict[str, float]] = None,
    width: int = DEFAULT_STRIP_WIDTH,
) -> Optional[np.ndarray]:
    """
    Extract strips from all corner regions of interest for all stars and merge

    For a two-star system with four corner regions of interest, this produces
    eight strips, merged horizontally into a single array.

    Parameters
    ----------

    Returns
    -------
    np.ndarray
    """
    angles = corner_angles if corner_angles is not None else DEFAULT_ANGLE_DEGREES

    strips = []

    for label, roi in corner_rois.items():
        angle_deg = angles.get(label, 45.0)
        stars = star_positions_per_corner.get(label, {"xs": [], "ys": []})
        for x_s, y_s in zip(stars["xs"], stars["ys"]):
            strip = crop_sidelobe_strip(
                roi, x_s, y_s,
                angle_degrees = angle_deg,
                width = width
            )
            if strip.shape[0] > 0:
                strips.append(strip)
            else:
                logger.warning(
                    "crop_and_merge_corners: empty strip for label=%s "
                    "star=(%.1f, %.1f).", label, x_s, y_s,
                )

    if not strips:
        logger.warning("crop_and_merge_corners: no valid strips extracted.")
        return None

    min_rows = min(s.shape[0] for s in strips)
    return np.concatenate([s[:min_rows] for s in strips], axis=1)

#-----------------
# Old algorithm
#-----------------

def crop_sidelobes(
    image: np.ndarray,
    x_poss: List[float],
    y_poss: List[float],
    centroid_data: Dict[str, float],
    sidelobe_offset: int = DEFAULT_SIDELOBE_OFFSET,
    angle_degrees: float = 45.0,
    width: int = DEFAULT_STRIP_WIDTH,
    length: int = DEFAULT_STRIP_LENGTH,
) -> Optional[np.ndarray]:
    """
    Return a sub image of sidelobes for centred at (x_poss, y_poss) from an image

    This is the new version of this function

    Args:
        image (numpy.ndarray): ndarray of flux values
        x_poss (list): list of X coordinates of the stars
        y_poss (list): list of Y coordinates of the stars
        centroid_data (dict): dictionary with image centroid information
        angle_degrees (float): angle of sidelobes from x axis in degrees (default 45)
        width (int): width of crop in pixels (default 6)
        length (int): length of crop around each sidelobe in pixels (default 360)

    Returns:
        numpy.ndarray: crops around each sidelobe combined into a single rectangular array of size width x 8*length
    """

    angles_rad = np.array([
        np.deg2rad(angle_degrees),
        np.deg2rad(angle_degrees+90.0)
    ])

    size=(480, 360)

    # Four arm directions: +45, +135, -45, -135
    sl_x = (sidelobe_offset * np.cos(angles_rad)).astype(int)
    sl_y = (sidelobe_offset * np.sin(angles_rad)).astype(int)
    sl_x = np.concatenate((sl_x, - sl_x))
    sl_y = np.concatenate((sl_y, - sl_y))
    angles_rad_4 = np.concatenate((angles_rad, angles_rad))

    # Build coordinate grids
    y_grid, x_grid = np.indices(image.shape)

    yy = (y_grid - np.broadcast_to(y_poss, (*image.shape, 2)).transpose((2,0,1))).transpose((1,2,0))
    xx = (x_grid - np.broadcast_to(x_poss, (*image.shape, 2)).transpose((2,0,1))).transpose((1,2,0))

    crop_im = _crop_areas(image, centroid_data['x'] + sl_x, centroid_data['y'] + sl_y, size)
    crop_x = _crop_areas(xx, centroid_data['x'] + sl_x, centroid_data['y'] + sl_y, size)
    crop_y = _crop_areas(yy, centroid_data['x'] + sl_x, centroid_data['y'] + sl_y, size)

    crop_im = np.transpose(crop_im, axes=(1,2,0))
    crop_x = np.transpose(crop_x, axes=(1,2,3,0))
    crop_y = np.transpose(crop_y, axes=(1,2,3,0))

    tan = np.tan(angles_rad)
    cot = 1/tan

    mask = np.abs(crop_x - crop_y*cot) <= width/2

    crop_im = np.tile(crop_im, (2,1,1,1)).transpose((3,0,2,1))

    result = crop_im[mask.T]

    try:
        result = result.reshape(result.shape[0]//width, width)
    except:
        pass

    return result

