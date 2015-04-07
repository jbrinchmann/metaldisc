import numpy as np

class LinearMetallicity(object):
    def __init__(self, r_d, Z_in, Z_out):
        self._galaxy = None

        self.r_d = r_d
        self.Z_in = Z_in
        self.Z_out = Z_out

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
        Z = (self.Z_out - self.Z_in) * (r / self.r_d) + self.Z_in
        return Z
