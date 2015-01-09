import unittest
import numpy as np
from numpy import testing as npt

from galaxy import *

class TestGalaxy(unittest.TestCase):

    def test_init_z(self):
        """Redshift input validated"""
        #Is z >= 0
        galaxy = Galaxy(0.5, 1.)
        with self.assertRaises(ValueError):
            galaxy = Galaxy(-0.1, 1)

    def test_init_R_s(self):
        """Redshift input validated"""
        #Is value stored
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
        galaxy = Galaxy(1., xy_max=10.)
        self.assertEqual(galaxy.xy_max, 10.)

        #is xy_max > 0
        with self.assertRaises(ValueError):
            galaxy = Galaxy(1., xy_max=0.)

    def test_init_xy_gridding(self):
        """xy grid constructed"""
        #is xy_sampling < xy_max
        with self.assertRaises(ValueError):
            galaxy = Galaxy(1., xy_sampling=2., xy_max=1.)
        
        galaxy = Galaxy(1., xy_sampling=0.1, xy_max=0.2)

        expect_x = np.array([[-0.1, 0., 0.1],
                             [-0.1, 0., 0.1],
                             [-0.1, 0., 0.1]])

        expect_y = np.array([[-0.1, -0.1, -0.1],
                             [0., 0., 0.],
                             [0.1, 0.1, 0.1]])

        npt.assert_array_equal(galaxy.x_grid, expect_x)
        npt.assert_array_equal(galaxy.y_grid, expect_y)

    def test_init_radius(self):
        """radius of grid calculated"""
        expect_r = np.sqrt(np.array([[2., 1., 2.],
                                     [1., 0., 1.],
                                     [2., 1., 2.]]))

        galaxy = Galaxy(1., xy_sampling=1., xy_max=2.)
        out_r = galaxy.r_grid

        npt.assert_array_equal(out_r, expect_r)
        
    def test_create_xy_grid(self):
        """xy grid constructed"""
        expect_x = np.array([[-0.5, 0., 0.5],
                             [-0.5, 0., 0.5],
                             [-0.5, 0., 0.5]])

        expect_y = np.array([[-0.3, -0.3, -0.3],
                             [0., 0., 0.],
                             [0.3, 0.3, 0.3]])

        out_x, out_y = create_xy_grid(0.5, 0.3, 1., 0.5)

        npt.assert_array_equal(out_x, expect_x)
        npt.assert_array_equal(out_y, expect_y)




if __name__ == '__main__':
    unittest.main()
