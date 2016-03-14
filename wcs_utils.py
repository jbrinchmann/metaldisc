import numpy as np
from astropy import wcs

def calc_rel_coords(ra, dec, ra_centre, dec_centre):
    """Calculates the relative sky coordinates about a central ra, dec

    Parameters
    ----------
    ra : array of floats
        right ascention [deg]
    dec : array of floats
        declination [deg]
    ra_centre : float
        right ascention of centre [deg]
    dec_centre : float
        declination of centre [deg]

    Returns
    -------
    x : array of floats
        x-axis distances from galaxy centre [arcsec]
    y : array of floats
        y-axis distances from galaxy centre [arcsec]

    """

    x = ra - ra_centre
    y = dec - dec_centre

    x -= np.round(x/360.) * 360. # wrap to interval (-180,180)

    x *= np.cos(np.radians(dec_centre)) * 3600. #deg to arcsec
    y *= 3600. #deg to arcsec

    return x, y 


def get_pixel_coord(hdr, ra_centre, dec_centre, oversample=1, x_offset=0., y_offset=0.):
    """Get relative image coordinates about a central ra, dec

    Notes
    -----
    x_offset=-0.5, y_offset=-0.5 yields coordinates of pixels' lower-left corners

    Parameters
    ----------
    hdr : astropy.io.fits.Header instance
        FITS header including WCS info
    ra_centre : float
        right ascention of centre [deg]
    dec_centre : float
        declination of centre [deg]
    oversample : [optional]int
        factor by which to oversample grid
    x_offset : [optional]float
        x-axis pixel coord offsets
    y_offset : [optional]float
        y-axis pixel coord offsets

    Returns
    -------
    x : array of floats
        x coords of relative pixel centres [arcsec]
    y : array of floats
        y coords of relative pixel centres [arcsec]

    """
    #get pixel centres [image coords]
    n_x = hdr['NAXIS1'] * oversample
    n_y = hdr['NAXIS2'] * oversample
    pix_x = np.tile(np.arange(n_x, dtype=float), n_y)
    pix_y = np.repeat(np.arange(n_y, dtype=float), n_x)

    pix_x = pix_x / oversample - 0.5 + 0.5/oversample
    pix_y = pix_y / oversample - 0.5 + 0.5/oversample
    
    pix_x += x_offset / oversample
    pix_y += y_offset / oversample

    #convert pixel centres to sky coords
    wcsobj = wcs.WCS(hdr)
    coords = np.column_stack([pix_x, pix_y])
    new_coords = wcsobj.all_pix2world(coords, 0) #[[ra], [dec]]

    #get pixel distances from centre [arcsec]
    x, y = calc_rel_coords(new_coords[:,0], new_coords[:,1],
                           ra_centre, dec_centre)
    
    x = x.reshape([n_y, n_x])
    y = y.reshape([n_y, n_x])

    return x, y


def get_pixel_area(hdr, ra_centre, dec_centre, oversample=1):
    """Set pixel areas

    Parameters
    ----------
    hdr : astropy.io.fits.Header instance
        FITS header including WCS info
    ra_centre : float
        right ascention of centre [deg]
    dec_centre : float
        declination of centre [deg]
    oversample : [optional]int
        factor by which to oversample grid

    Returns
    -------
    area : array of floats
        pixel areas [arcsec^2]

    """

    #get pixel corners [arcsec]  (clockwise)
    A_x, A_y = get_pixel_coord(hdr, ra_centre, dec_centre,
                               oversample=oversample,
                               x_offset=-0.5, y_offset=-0.5)
    B_x, B_y = get_pixel_coord(hdr, ra_centre, dec_centre,
                               oversample=oversample,
                               x_offset=-0.5, y_offset=+0.5)
    C_x, C_y = get_pixel_coord(hdr, ra_centre, dec_centre,
                               oversample=oversample,
                               x_offset=+0.5, y_offset=+0.5)
    D_x, D_y = get_pixel_coord(hdr, ra_centre, dec_centre,
                               oversample=oversample,
                               x_offset=+0.5, y_offset=-0.5)

    #get vectors between opposite corners
    AC_x = C_x - A_x
    AC_y = C_y - A_y

    BD_x = D_x - B_x
    BD_y = D_y - B_y

    area = 0.5 * np.abs((AC_x * BD_y) - (BD_x * AC_y))

    return area

