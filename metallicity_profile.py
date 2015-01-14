
class MetallicityProfile(object):
    
    def params(self):
        """
        raise NotImplementedError


    def _linear_metallicity_profile(self, r_s, Z_in, Z_out):
        """Linear Metallicity_Profile

        Args:
            r_s - scale radius of disc (arcsec)
            Z_in - metallicity at centre of galaxy (12 + log(O/H))
            Z_out - metallicity at one scale radius from centre (12 + log(O/H))
        """
        Z = (Z_out - Z_in) * (self.r_grid / r_s) + Z_in
        return Z
