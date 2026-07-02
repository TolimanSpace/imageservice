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
from typing import List, Optional, Tuple

import numpy as np

#TODO: move this global variable for where the sidelobes are relative to the centre to a place where it can be adjusted
SIDELOBE_OFFSET = 1625

def crop_centre(image,x_pos,y_pos,size=128):
    """
    Return a sub image centred at (x_pos, y_pos) from an image

    Args:
        image (numpy.ndarray): ndarray of flux values
        x_pos (float): X coordinate of the centre point
        y_pos (float): Y coordinate of the centre point
        size (int): size of each side of the box (default 128)

    Returns:
        cropped (numpy.ndarray): cropped image
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

    # Extract cutout
    cropped = image[start_y:end_y, start_x:end_x]

    return cropped


def crop_areas(image, x_poss, y_poss, size = (480, 360)):
    """
    Return a set of sub images from an image

    Args:
        image (numpy.ndarray): ndarray of flux values
        x_poss (list): list of X coordinates of centre points
        y_poss (list): list of Y coordinates of centre points
        size (tuple): size of each image (default (480, 360))

    Returns:
        numpy.ndarray: cropped images in a single array
    """

    output = []

    for x_pos, y_pos in zip(x_poss, y_poss):
        # Round positions
        x_pos = round(x_pos)
        y_pos = round(y_pos)

        # Calculate dimensions of the cutout
        half_size_x = size[1] // 2
        half_size_y = size[0] // 2
        start_x = max(x_pos - half_size_x, 0)
        start_y = max(y_pos - half_size_y, 0)
        end_x = min(x_pos + half_size_x + (size[1] % 2), image.shape[1])
        end_y = min(y_pos + half_size_y + (size[0] % 2), image.shape[0])

        # Extract cutout
        output.append(image[start_y:end_y, start_x:end_x])

    return np.asarray(output)


def crop_sidelobes_old(image, x_poss, y_poss, angle_degrees = 45, width = 6, length=360):
    """
    Return a sub image of sidelobes for centred at (x_poss, y_poss) from an image

    This is an old version of this function that takes too long to run

    Args:
        image (numpy.ndarray): ndarray of flux values
        x_poss (list): list of X coordinates of the stars
        y_poss (list): list of Y coordinates of the stars
        angle_degrees (float): angle of sidelobes from x axis in degrees (default 45)
        width (int): width of crop in pixels (default 6)
        length (int): length of crop around each sidelobe in pixels (default 360)

    Returns:
        numpy.ndarray: crops around each sidelobe combined into a single rectangular array of size 4*width x 2*length
    """
    
    output = []
    
    angles_radians = [np.deg2rad(angle_degrees), np.deg2rad(angle_degrees+90)]

    for x_pos, y_pos in zip(x_poss, y_poss):
        y, x = np.indices(image.shape)
        y = y - y_pos
        x = x - x_pos

        for angle in angles_radians:
            sidelobe_x = np.abs(SIDELOBE_OFFSET * np.cos(angle))
            sidelobe_y = np.abs(SIDELOBE_OFFSET * np.sin(angle))

            if np.abs(angle-np.pi/2) <= np.pi/4:
                mask = (np.abs(x - y / np.tan(angle)) <= width/2) & (np.abs(np.abs(y)-sidelobe_y) <= length/2)
            else:
                mask = (np.abs(y - x * np.tan(angle)) <= width/2) & (np.abs(np.abs(x)-sidelobe_x) <= length/2)

            output.append(image[mask].reshape((image[mask].shape[0]//width,width)))

    result = np.block([a for a in output])

    return result

def crop_sidelobes(image, x_poss, y_poss, centroid_data, angle_degrees = 45, width = 6, length=360):
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

    angles_radians = np.array([np.deg2rad(angle_degrees), np.deg2rad(angle_degrees+90)])

    sidelobes_x = (SIDELOBE_OFFSET * np.cos(angles_radians)).astype(int)
    sidelobes_y = (SIDELOBE_OFFSET * np.sin(angles_radians)).astype(int)

    sidelobes_x = np.concatenate((sidelobes_x, - sidelobes_x))
    sidelobes_y = np.concatenate((sidelobes_y, - sidelobes_y))
    angles_radians = np.concatenate((angles_radians, angles_radians))

    y, x = np.indices(image.shape)
    yy = (y - np.broadcast_to(y_poss, (*image.shape, 2)).transpose((2,0,1))).transpose((1,2,0))
    xx = (x - np.broadcast_to(x_poss, (*image.shape, 2)).transpose((2,0,1))).transpose((1,2,0))

    crop_im = crop_areas(image, centroid_data['x'] + sidelobes_x, centroid_data['y'] + sidelobes_y)
    crop_x = crop_areas(xx, centroid_data['x'] + sidelobes_x, centroid_data['y'] + sidelobes_y)
    crop_y = crop_areas(yy, centroid_data['x'] + sidelobes_x, centroid_data['y'] + sidelobes_y)

    crop_im = np.transpose(crop_im, axes=(1,2,0))
    crop_x = np.transpose(crop_x, axes=(1,2,3,0))
    crop_y = np.transpose(crop_y, axes=(1,2,3,0))

    tan = np.tan(angles_radians)
    cot = 1/tan

    mask = np.abs(crop_x - crop_y*cot) <= width/2

    crop_im = np.tile(crop_im, (2,1,1,1)).transpose((3,0,2,1))

    result = crop_im[mask.T]

    try:
        result = result.reshape(result.shape[0]//width, width)
    except:
        pass

    return result

