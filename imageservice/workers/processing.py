import logging
import os
import numpy as np
from scipy.ndimage import center_of_mass, maximum_filter
from scipy.fft import fft2, ifft2, fftshift
from datetime import datetime
_logger = logging.getLogger(__name__)

DECONVOLVE_CONSTANT_FILEPATH = "./deconvolution_constant.npy"
DECONVOLVE_CONSTANT = np.load(DECONVOLVE_CONSTANT_FILEPATH)

def find_centroid(frame):
    t = str(datetime.now())
    y, x = center_of_mass(frame)
    return {"x": x, "y": y, "t": t}

def deconvolve_image(image):
    f_image = fft2(image/np.nanmax(image))
    result = fftshift(np.real(ifft2(f_image*DECONVOLVE_CONSTANT)))
    return result

def find_stars(image, n_stars = 2, threshold = 0.1, size = 5):
    deconvolved_image = deconvolve_image(image)
    local_max = (deconvolved_image == maximum_filter(deconvolved_image, size = size))
    thresh = threshold  * np.nanmax(deconvolved_image)
    candidates = np.argwhere(local_max & (deconvolved_image > thresh))
    sorted_candidates = sorted(candidates, key = lambda c: deconvolved_image[c[0], c[1]], reverse = True)
    xs = []
    ys = []
    for candidate in sorted_candidates[:n_stars]:
        xs.append(candidate[1])
        ys.append(candidate[0])

    return {"xs": xs, "ys": ys}