import numpy as np

def create_xy_grid(x_sampling, y_sampling, x_max, y_max):
    """Creates regualarly sampled grid centred on (0, 0)

    Args:
        x_sampling (float): spacing between samples in the x dirction
        y_sampling (float): spacing between samples in the y dirction
        x_max (float): defines limits of grid in x direction [-x_max, x_max]
        y_max (float): defines limits of grid in y_dircetion [-y_max, y_max]

    Returns:
        x_grid (array): grid of x coords
        y_grid (array): grid of y coords
    """
    x = np.arange(0., x_max, x_sampling)
    x = np.concatenate([-x[::-1], x[1:]]) #make symmetric about 0

    y = np.arange(0., y_max, y_sampling)
    y = np.concatenate([-y[::-1], y[1:]]) #make symmetric about 0

    x_grid, y_grid = np.meshgrid(x, y)
    
    return x_grid, y_grid


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

        grids = create_xy_grid(xy_sampling, xy_sampling, xy_max, xy_max)
        self.x_grid = grids[0]
        self.y_grid = grids[1]


#    def get_bin
