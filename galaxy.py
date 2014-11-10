import numpy as np

class Galaxy(object):
    """Galaxy model producing undegraded 2D emission line maps"""
    def __init__(self, z, xy_sampling=0.05, xy_max=10.):
        """Initialize galaxy

        Args:
            z (float): Redshift of galaxy

        Kwargs:
            xy_sampling (float): spatial sampling of model (arcsec)
            xy_max (float): max limit of spatial sampling (arcsec)

        Raises:
            ValueError
        """
    
        #Check redshift is positive
        if z < 0.:
            t = ("Illegal input: z (redshift) cannot be less than 0. "
                 "Value given z={0:.3d}".format(z))
            raise ValueError(t)

        #Check spatial sampling is positive
        if xy_sampling <= 0.:
            t = ("Illegal input: xy_sampling (spatial sampling) cannot be less "
                 "than 0. Value given z={0:.3d}".format(xy_sampling))
            raise ValueError(t)

        #Check spatial max limit is positive
        if xy_max <= 0.:
            t = ("Illegal input: xy_max (spatial limit) cannot be less "
                 "than 0. Value given z={0:.3d}".format(xy_sampling))
            raise ValueError(t)
    
        if xy_sampling >= xy_max:
            t = ("Illegal input: xy_max should be greater than xy_sampling")
            raise ValueError(t)

        self.z = z
        self.xy_sampling = xy_sampling
        self.xy_max = xy_max

        x = np.arange(0., xy_max, xy_sampling)
        self.x_grid = np.array([0])

#        self.kpc_per_arcsec = self._kpc_per_arcsec(z)

    def get_bin
