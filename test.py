import unittest
import numpy as np
from numpy import testing as npt

from galaxy import Galaxy

class TestGalaxy(unittest.TestCase):

    def test_init_z(self):
        """Redshift input validated and stored"""
        #Is value stored
        galaxy = Galaxy(0.5)
        self.assertEqual(galaxy.z, 0.5)

        #Is z >= 0
        with self.assertRaises(ValueError):
            galaxy = Galaxy(-0.1)

    def test_init_xy_sampling(self):
        """xy spatial sampling input validated and stored"""
        #is value stored
        galaxy = Galaxy(1, xy_sampling=0.01)
        self.assertEqual(galaxy.xy_sampling, 0.01)

        #is xy_sampling > 0
        with self.assertRaises(ValueError):
            galaxy = Galaxy(1., xy_sampling=0.)

    def test_init_xy_max(self):
        """xy max spatial limit input validated and stored"""
        #is value stored
        galaxy = Galaxy(1, xy_max=10.)
        self.assertEqual(galaxy.xy_max, 10.)

        #is xy_max > 0
        with self.assertRaises(ValueError):
            galaxy = Galaxy(1., xy_max=0.)

    def test_init_xy_gridding(self):
        """xy grid constructed"""
        #is xy_sampling < xy_max
        with self.assertRaises(ValueError):
            galaxy = Galaxy(1., xy_sampling=2., xy_max=1.)
        
        galaxy = Galaxy(1., xy_sampling=0.5, xy_max=1.5)

        expect = np.array([[-0.5, 0., 0.5],
                           [-0.5, 0., 0.5],
                           [-0.5, 0., 0.5]])
        npt.assert_array_equal(galaxy.x_grid, expect)

        expect = np.array([[0.5, 0.5, 0.5],
                           [0., 0., 0.],
                           [-0.5, -0.5, -0.5]])
        npt.assert_array_equal(galaxy.y_grid, expect)






if __name__ == '__main__':
    unittest.main()
