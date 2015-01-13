import unittest
import numpy as np
from numpy import testing as npt

from galaxy import *
from discmodel import *

class TestExponetialDisc(unittest.TestCase):

    def test_init_galaxy(self):
        """galaxy object parent stored"""
        galaxy = Galaxy(0.5)
        expDisc = ExponetialDisc(galaxy)

        self.assertIs(expDisc.galaxy, galaxy)
    
    def test_sensible_fluxes(self):
        """flux non-negative"""
        galaxy = Galaxy(0.5)
        expDisc = ExponetialDisc(galaxy)
        flux = expDisc.calc_flux()
        assert np.all(flux >= 0.)
        
    def test_TODO(self):
        """add central SFR/Halpha scale parameter to flux"""
        raise Exception

if __name__ == '__main__':
    unittest.main()
