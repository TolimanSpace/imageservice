import logging
from astropy.io import fits

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

    return True