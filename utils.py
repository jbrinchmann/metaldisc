def separation_to_radec(x, y, ra_centre, dec_centre):
    """Convert separations to ra and dec coords

    Parameters
    ----------
    x : float (array-like)
        x-position [arcsec]
    y : float (array-like)
        y-position [arcsec]
    ra_centre : float
        Right ascention ad x=0 [deg]
    dec_centre : float
        Declination at y=0 [deg]
        
    Returns
    -------
    ra : float (array-like)
        on sky right ascention [deg]
    dec : float (array-like)
        on sky declination [deg]

    """

    x /= np.cos(np.radians(dec_centre))
    x /= 3600.
    y /= 3600.
    ra = x + ra_centre
    dec = y + dec_centre
    
    return ra, dec
