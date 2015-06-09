import numpy as np

class LinearExtinction(object):
    def __init__(self, r_d, tauV_in, tauV_out):
        self._galaxy = None

        self.r_d = r_d
        self.tauV_in = tauV_in
        self.tauV_out = tauV_out

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

    def __call__(self, r):
        tauV = (self.tauV_out - self.tauV_in) * (r / self.r_d) + self.tauV_in
        return tauV
