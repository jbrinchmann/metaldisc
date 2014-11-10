class Galaxy(object):
    """Galaxy model producing undegraded 2D emission line maps"""
    def __init__(self, z, R_e):
        """Initialize galaxy

        Args:
            z (float): Redshift of galaxy
            R_e (float): Scale radius of galaxy (kpc)

        """
        self.z = z
        self.R_e = R_e
