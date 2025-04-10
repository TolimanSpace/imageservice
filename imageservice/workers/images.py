import logging
import ast
import numpy as np
import xarray as xr
from astropy.io import fits
from .processing import find_centroid, find_stars
from .compression import crop_centre, crop_sidelobes #, crop_sidelobes_old

_logger = logging.getLogger(__name__)


def crop_image_with_metadata(raw_filename):

    # Load raw image
    raw_image = np.load(raw_filename)

    # Load raw image metadata
    metadata_file = raw_filename.strip(".npy")+".txt"

    with open(metadata_file, "r") as file:
        for line in file:
            header = ast.literal_eval(line.strip())

    # Crop core and sidelobes from raw image
    centroid_data = find_centroid(raw_image)
    core = crop_centre(raw_image, centroid_data["x"], centroid_data["y"])
    star_poss = find_stars(core)
    x_poss = np.round(star_poss['xs'] + centroid_data['x'] - core.shape[1]//2)
    y_poss = np.round(star_poss['ys'] + centroid_data['y'] - core.shape[0]//2)
    sidelobes = crop_sidelobes(raw_image, x_poss, y_poss, centroid_data)

    # Convert CAMTIME to a string to avoid truncation/error when creating netCDF file
    header["CAMTIME"] = str(header["CAMTIME"])

    # Add position information to header
    header["CENTR_X"] = int(np.round(centroid_data['x']))
    header["CENTR_Y"] = int(np.round(centroid_data['y']))
    header["STAR_1_X"] = int(x_poss[0])
    header["STAR_1_Y"] = int(y_poss[0])
    header["STAR_2_X"] = int(x_poss[1])
    header["STAR_2_Y"] = int(y_poss[1])

    return core.astype(np.int16), sidelobes.astype(np.int16), header


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

    # imageData = frame["frame"]
    # filename = f'images/raw/frame_{frame["camtime"]}.npy'
    # np.save(filename, imageData)
    filename = frame["rawfile"]

    return filename


def compress_dump(raw_filename):

    core, sidelobes, raw_image_header = crop_image_with_metadata(raw_filename)

    # Create primary header
    header = fits.Header()
    for key in raw_image_header:
        header[key] = raw_image_header[key]

    primary_hdu = fits.PrimaryHDU(header=header)
    core_hdu = fits.CompImageHDU(core, name="CORE")
    sidelobes_hdu = fits.CompImageHDU(sidelobes, name="SIDELOBES")

    hdul = fits.HDUList([primary_hdu, core_hdu, sidelobes_hdu])

    # Write to disk
    filename = f'images/compressed/frame_proc_{header["CAMTIME"]}.fits.gz'
    hdul.writeto(filename, overwrite=True)

    # Log status
    _logger.info(f"FITS file written to {filename}")

    return True

def compress_netcdf(raw_filename):

    core, sidelobes, header = crop_image_with_metadata(raw_filename)

    # Create xarray Dataset
    core_array = xr.DataArray(core, dims=("y", "x"))
    sidelobe_array = xr.DataArray(sidelobes)

    dataset = xr.Dataset({
        "core": core_array,
        "sidelobes": sidelobe_array
    })

    dataset.attrs = header

    # Set compression encoding
    dataset["core"].encoding = {"zlib": True, "complevel": 9}
    dataset["sidelobes"].encoding = {"zlib": True, "complevel": 9}

    # Write to disk
    filename = f'images/compressed/frame_proc_{header["CAMTIME"]}.nc'
    dataset.to_netcdf(filename)

    # Log status
    _logger.info(f"NetCDF file written to {filename}")

    return True


def compress_netcdf_bulk(raw_filenames):

    # Get reference image
    ref_core, ref_sidelobes, header = crop_image_with_metadata(raw_filenames[0])

    diff_cores = []
    diff_sidelobes = []
    header = [header]

    # Get differences from reference for remaining images
    for filename in raw_filenames[1:]:

        core, sidelobes, metadata = crop_image_with_metadata(filename)

        diff_cores.append(core - ref_core)
        diff_sidelobes.append(sidelobes - ref_sidelobes)
        header.append(metadata)

    diff_cores = np.asarray(diff_cores)
    diff_sidelobes = np.asarray(diff_sidelobes)

    # Convert to int8 if safe to do so
    if np.max(np.abs(diff_cores)) <= 127:
        diff_cores = diff_cores.astype(np.int8)

    if np.max(np.abs(diff_sidelobes)) <= 127:
        diff_sidelobes = diff_sidelobes.astype(np.int8)

    # Create xarray Dataset
    ref_core_array = xr.DataArray(ref_core, dims=("y", "x"))
    diff_core_array = xr.DataArray(diff_cores, dims=("i", "y", "x"))
    ref_side_array = xr.DataArray(ref_sidelobes, dims=("a","b"))
    diff_side_array = xr.DataArray(diff_sidelobes, dims=("i", "a", "b"))

    dataset = xr.Dataset({
        "ref_core": ref_core_array,
        "ref_sidelobes": ref_side_array,
        "diff_core": diff_core_array,
        "diff_sidelobes": diff_side_array
    })

    # Convert header list to dict
    header = {str(i)+'_'+k: v for i, d in enumerate(header) for k, v in d.items()}

    dataset.attrs = header

    # Set compression encoding
    dataset["ref_core"].encoding = {"zlib": True, "complevel": 9}
    dataset["ref_sidelobes"].encoding = {"zlib": True, "complevel": 9}

    # Write to disk
    filename = f'images/compressed/frame_proc_{header["0_CAMTIME"]}.nc'
    dataset.to_netcdf(filename)

    # Log status
    _logger.info(f"NetCDF file written to {filename}")


    return True


