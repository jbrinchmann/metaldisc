import warnings

import numpy as np

class SFRProfile(object):
    def __init__(self, galaxy):
        self.galaxy = galaxy

    def required_params(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class ExponentialSFRProfile(SFRProfile):

    def required_params(self):
        params = ('z', 'r0', 'SFRDensity_centre', 'SFRDensity_thres')
        return params

    def __call__(self, params):
        z = params['z']
        r0 = params['r0']
        SFRDensity_centre = params['SFRDensity_centre']
        SFR_thres = params['SFRDensity_thres']
        
        pix_area_kpc = self.galaxy.pix_area * self.galaxy.kpc_per_arcsec(z)**2.
        norm = Sigma_SFR_centre * pix_area_kpc # M_sol/yr
        SFR = norm * np.exp(-self.galaxy.r / r0) # exp profile
        
        SFR_at_edge = norm * np.exp(-self.galaxy.xy_max / r0) # exp profile
        if SFR_at_edge >= SFR_thres:
            warnings.warn("Galaxy truncated by grid and therefor not circular. "
                          "Consider increasing xy_max, or SFRDensity_thres")
        
        SFR[SFR <= SFR_thres] = np.nan

        return SFR

