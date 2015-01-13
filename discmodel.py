import numpy as np

class ExponetialDisc(object):
    """Exponetial disc flux model"""
    def __init__(self, galaxy):
        """Initialize expoential flux model

        Args:
            galaxy (Galaxy instance): parent galaxy 
        """
        self.galaxy = galaxy

    def calc_flux(self, params):
        """Calculate flux profile of galaxy"""
        r = self.galaxy.r_grid
        R_s = params['R_s']
        profile = np.exp(-r/R)
        return profile
         



#exp model needs
#scale radius in kpc?(redshift) convert
#normaliziation
#pixel coords + area ?
