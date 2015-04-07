import numpy as np

class ExpDisc(object):
    def __init__(self, total_SFR, r_d):
#        self._galaxy = None

        self.total_SFR = total_SFR
        self.r_d = r_d

#    @property
#    def galaxy(self):
#        return self._galaxy
#
#    @galaxy.setter
#    def galaxy(self, value):
#        if self._galaxy is None:
#            self._galaxy = value
#        else:
#            raise Exception("Galaxy cannot be set, please delete first")
#    @galaxy.deleter
#    def galaxy(self, value):
#        self._galaxy = None

    def calc_central_SFD(self, r_max):
        r_d = self.r_d
        norm = 2. * np.pi * r_d * (r_d - np.exp(-r_max/r_d) * (r_d + r_max))

        central_SFD = self.total_SFR / norm
        return central_SFD
        

    def __call__(self, r, r_max):
        
        central_SFD = self.calc_central_SFD(r_max)
        sf_density = central_SFD * np.exp(-r / self.r_d)

        return sf_density

