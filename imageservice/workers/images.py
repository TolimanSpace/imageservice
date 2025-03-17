import logging
import ast
import numpy as np
from astropy.io import fits
from .processing import find_centroid, find_stars
from .compression import crop_centre, crop_sidelobes

_logger = logging.getLogger(__name__)


def create_fits(frame):

    imageData = frame["frame"]

    # Create header
    header = fits.Header()
    header['SEQNUM'] = frame["i"]
    header['CAMTIME'] = frame["camtime"]
    header['COMTIME'] = frame["comptime"]
    header["EXPOSURE"] = frame["exposure"]
    header["PXLFMT"] = frame["pxlfmt"]
    header["XOFF"] = frame["xoff"]
    header["YOFF"] = frame["yoff"]
    header["XPAD"] = frame["xpad"]
    header["YPAD"] = frame["ypad"]

    # Create compressed image HDU        
    hdu = fits.CompImageHDU(imageData, header)

    # Write to disk
    filename = f'images/raw/frame_{frame["camtime"]}.fits'
    hdu.writeto(filename, overwrite=True)

    # Log status
    _logger.info(f"FITS file written to {filename}")

    return filename


def compress_image(raw_filename):

    with fits.open(raw_filename) as raw_hdul:

        raw_image = raw_hdul[1].data
        raw_image_header = raw_hdul[1].header

        centroid_data = find_centroid(raw_image)
        core = crop_centre(raw_image, centroid_data["x"], centroid_data["y"])
        star_poss = find_stars(core)
        x_poss = np.round(star_poss['xs'] + centroid_data['x'] - core.shape[1]//2)
        y_poss = np.round(star_poss['ys'] + centroid_data['y'] - core.shape[0]//2)
        sidelobes = crop_sidelobes(raw_image, x_poss, y_poss)

        core_hdu = fits.CompImageHDU(core, name="CORE")
        sidelobes_hdu = fits.CompImageHDU(sidelobes, name="SIDELOBES")

        # Create primary header
        header = fits.Header()
        for key in raw_image_header:
            header[key] = raw_image_header[key]

        header["CENTR_X"] = np.round(centroid_data['x'])
        header["CENTR_Y"] = np.round(centroid_data['y'])
        header["STAR_1_X"] = x_poss[0]
        header["STAR_1_Y"] = y_poss[0]
        header["STAR_2_X"] = x_poss[1]
        header["STAR_2_Y"] = y_poss[1]

        primary_hdu = fits.PrimaryHDU(header=header)

        hdul = fits.HDUList([primary_hdu, core_hdu, sidelobes_hdu])

        # Write to disk
        filename = f'images/compressed/frame_proc_{header["CAMTIME"]}.fits.gz'
        hdul.writeto(filename, overwrite=True)

        # Log status
        _logger.info(f"FITS file written to {filename}")

    return True

def dump_data(frame):

    metadata_dict = {
        'SEQNUM': frame["i"],
        'CAMTIME': frame["camtime"],
        'COMTIME': frame["comptime"],
        'EXPOSURE': frame["exposure"],
        'PXLFMT': frame["pxlfmt"],
        'XOFF': frame["xoff"],
        'YOFF': frame["yoff"],
        'XPAD': frame["xpad"],
        'YPAD': frame["ypad"],
    }

    metadata_filename = f'images/raw/frame_{frame["camtime"]}.txt'
    with open(metadata_filename, 'a') as file:
            file.write(f"{metadata_dict}")

    imageData = frame["frame"]
    filename = f'images/raw/frame_{frame["camtime"]}.npy'
    np.save(filename, imageData)

    return filename


def compress_dump(raw_filename):

    raw_image = np.load(raw_filename)

    metadata_file = raw_filename.strip(".npy")+".txt"

    with open(metadata_file, "r") as file:
        for line in file:
            raw_image_header = ast.literal_eval(line.strip())

    centroid_data = find_centroid(raw_image)
    core = crop_centre(raw_image, centroid_data["x"], centroid_data["y"])
    star_poss = find_stars(core)
    x_poss = np.round(star_poss['xs'] + centroid_data['x'] - core.shape[1]//2)
    y_poss = np.round(star_poss['ys'] + centroid_data['y'] - core.shape[0]//2)
    sidelobes = crop_sidelobes(raw_image, x_poss, y_poss)

    core_hdu = fits.CompImageHDU(core, name="CORE")
    sidelobes_hdu = fits.CompImageHDU(sidelobes, name="SIDELOBES")

    # Create primary header
    header = fits.Header()
    for key in raw_image_header:
        header[key] = raw_image_header[key]

    header["CENTR_X"] = np.round(centroid_data['x'])
    header["CENTR_Y"] = np.round(centroid_data['y'])
    header["STAR_1_X"] = x_poss[0]
    header["STAR_1_Y"] = y_poss[0]
    header["STAR_2_X"] = x_poss[1]
    header["STAR_2_Y"] = y_poss[1]

    primary_hdu = fits.PrimaryHDU(header=header)

    hdul = fits.HDUList([primary_hdu, core_hdu, sidelobes_hdu])

    # Write to disk
    filename = f'images/compressed/frame_proc_{header["CAMTIME"]}.fits.gz'
    hdul.writeto(filename, overwrite=True)

    # Log status
    _logger.info(f"FITS file written to {filename}")

    return True