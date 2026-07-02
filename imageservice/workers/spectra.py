"""
spectra.py - Functions for computing 1D spectra on board the spacecraft.

"""

import numpy as np
from scipy.optimize import curve_fit

def sidelobe_crosssection(x, bkgd, amp_A, x_A, amp_B, x_B, sigma):
    """
    Function that returns a sum of two Gaussians of the same width

    Args:
        x: array of x values
        bkgd: constant
        amp_A: amplitude of A component
        x_A: position of centre of A component
        amp_B: amplitude of B component
        x_B: position of centre of B component
        sigma: RMS width of the Gaussians

    Returns:
        result (np.ndarray)
    """
    return bkgd + amp_A * np.exp(-(x - x_A)**2 / (2 * sigma**2)) + amp_B * np.exp(-(x - x_B)**2 / (2 * sigma**2))


def gaussian_area(amp,sigma):
    return np.sqrt(2*np.pi) * amp * sigma


def fit_spectra(image, x_dist, y_dist, b_offset=0, width=60, length = 500):
    """
    Routine to fit spectra

    Args:
        image (numpy.ndarray): Input array of flux values.
        x_dist (numpy.ndarray): Distance of each pixel to the positive/negative sidelobe axis
        y_dist (numpy.ndarray): Distance of each pixel to the negative/positive sidelobe axis
        b_offset (float): Expected projected separation of B component from A component
        width (int): Distance from sidelobes to include in fit of Gaussians, in pixels
        length (int): Length along sidelobes to include, in pixels

    Returns:
        res (np.ndarray): Output array with spectra, with shape (2xlength, 4)
        a_pos (float): Fitted position of A component along sidelobe axis
        b_pos (float): Fitted position of B component along sidelobe axis
        sigma (float): Fitted width of Gaussian

    """

    mask = np.abs(y_dist) <= width

    xaxis = x_dist[mask]
    yaxis = y_dist[mask]
    values = image[mask]

    xmask = ((xaxis > -SIDELOBE_OFFSET - length/2) & (xaxis <= -SIDELOBE_OFFSET + length/2)) | ((xaxis > SIDELOBE_OFFSET - length/2) & (xaxis <= SIDELOBE_OFFSET + length/2))

    # Perform global fit to set positions, widths
    params, covs = curve_fit(sidelobe_crosssection, yaxis[xmask], values[xmask], p0 = [np.median(image), 30, 0, 10, b_offset, 1])

    a_pos = params[2]
    b_pos = params[4]
    sigma = params[5]

    bins = np.arange(int(np.min(xaxis)), int(np.max(xaxis)))
    idx = np.digitize(xaxis, bins+0.5)

    bins_mask = ((bins > -SIDELOBE_OFFSET - length/2) & (bins <= -SIDELOBE_OFFSET + length/2)) | ((bins > SIDELOBE_OFFSET - length/2) & (bins <= SIDELOBE_OFFSET + length/2))

    res = []

    for i in np.unique(idx)[:-1][bins_mask]:
        popt, pcov = curve_fit(lambda x, bkgd, amp_A, amp_B: sidelobe_crosssection(x, bkgd, amp_A, a_pos, amp_B, b_pos, sigma), yaxis[idx == i], values[idx == i])
        res.append(popt)

    return np.array(res), a_pos, b_pos, sigma


def spectra_1D(image, x_poss, y_poss, angle_degrees = 45, width = 60, length=500):
    """
    One dimensional spectra of the sidelobes

    Args:
        image (numpy.ndarray): Input array of flux values.
        x_poss (list): list of floats of x positions of stars
        y_poss (list): list of floats of y positions of stars
        angle_degrees (float): Angle of sidelobes from x axis in degrees (default 45)
        width (int): Distance from sidelobes to include in fit of Gaussians, in pixels
        length (int): Length along sidelobes to include, in pixels

    Returns:
        spectra (numpy.ndarray): Output array with spectra, with shape (2xlength, 4)
    """

    angle_radians = np.deg2rad(angle_degrees)
    
    # Calculate distances from sidelobe axes of the primary
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])

    xv, yv = np.meshgrid(x,y)

    sin_angle = np.sin(angle_radians)
    cos_angle = np.cos(angle_radians)
    
    dist_1 = (xv - x_poss[0]) * sin_angle - (yv - y_poss[0]) * cos_angle
    dist_2 = (xv - x_poss[0]) * cos_angle + (yv - y_poss[0]) * sin_angle

    # Calculate how far the B component is expected from the centre
    b_offset_1 = (x_poss[1] - x_poss[0]) * sin_angle + (y_poss[1] - y_poss[0]) * cos_angle
    b_offset_2 = (x_poss[1] - x_poss[0]) * cos_angle - (y_poss[1] - y_poss[0]) * sin_angle

    # Fit spectra
    res_1, a_pos_1, b_pos_1, sigma_1 = fit_spectra(image, dist_1, dist_2, b_offset = b_offset_1, width=width, length=length)
    res_2, a_pos_2, b_pos_2, sigma_2 = fit_spectra(image, dist_2, dist_1, b_offset = b_offset_2, width=width, length=length)

    # Calculate area
    spectra = np.block([gaussian_area(res_1[:,1:], sigma_1), gaussian_area(res_2[:,1:], sigma_2)])
    
    return spectra

