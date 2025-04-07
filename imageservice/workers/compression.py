import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime


SIDELOBE_OFFSET = 1625


def crop_centre(image,x_pos,y_pos,size=128):
    """
    Return a sub image centred at (x_pos, y_pos) from an image

    Parameters:
    - image: ndarray of flux values
    - x_pos: X coordinate of the centre point
    - y_pos: Y coordinate of the centre point
    - size: size of each side of the box (default 128)

    Returns:
    - cropped: ndarray
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

    Parameters:
    - image: ndarray of flux values
    - x_poss: list of X coordinates of the stars
    - y_poss: list of Y coordinates of the stars
    - angle_degrees: angle of sidelobes from x axis in degrees (default 45)
    - width: width of crop in pixels (default 6)
    - length: length of crop around each sidelobe in pixels (default 360)

    Returns:
    - ndarray: crops around each sidelobe combined into a single rectangular array of size 4*width x 2*length
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

def sidelobe_crosssection(x, bkgd, amp_A, x_A, amp_B, x_B, sigma):
    """
    Function that returns a sum of two Gaussians of the same width

    Parameters:
    - x: array of x values
    - bkgd: constant
    - amp_A: amplitude of A component
    - x_A: position of centre of A component
    - amp_B: amplitude of B component
    - x_B: position of centre of B component
    - sigma: RMS width of the Gaussians

    Returns:
    - result array, the same leng
    
    """
    return bkgd + amp_A * np.exp(-(x - x_A)**2 / (2 * sigma**2)) + amp_B * np.exp(-(x - x_B)**2 / (2 * sigma**2))


def gaussian_area(amp,sigma):
    return np.sqrt(2*np.pi) * amp * sigma


def fit_spectra(image, x_dist, y_dist, b_offset=0, width=60, length = 500):
    """
    Routine to fit spectra

    Parameters:
    ----------
    image : ndarray 
        Input array of flux values.
    x_dist: ndarray
        Distance of each pixel to the positive/negative sidelobe axis
    y_dist: ndarray
        Distance of each pixel to the negative/positive sidelobe axis
    b_offset: float
        Expected projected separation of B component from A component
    width: int
        Distance from sidelobes to include in fit of Gaussians, in pixels
    length: int
        Length along sidelobes to include, in pixels

    Returns:
    res: ndarray
        Output array with spectra, with shape (2xlength, 4)
    a_pos: float
        Fitted position of A component along sidelobe axis
    b_pos: float
        Fitted position of B component along sidelobe axis
    sigma: float
        Fitted width of Gaussian

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

    Parameters
    ----------
    image : ndarray 
        Input array of flux values.
    x_poss: list of floats
        x positions of stars
    y_poss: list of floats
        y positions of stars
    angle_degrees: float
        Angle of sidelobes from x axis in degrees (default 45)
    width: int
        Distance from sidelobes to include in fit of Gaussians, in pixels
    length: int
        Length along sidelobes to include, in pixels

    Returns:
    spectra: ndarray
        Output array with spectra, with shape (2xlength, 4)
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

