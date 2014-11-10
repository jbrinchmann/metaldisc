import unittest

from galaxy import Galaxy

class TestGalaxy(unittest.TestCase):

    def setUp(self):
        self.galaxy = Galaxy(0.5, 15.)

    def test_initialize_z(self):
        self.assertEqual(self.galaxy.z, 0.5)

    def test_initialize_R_e(self):
        self.assertEqual(self.galaxy.R_e, 15.)


if __name__ == '__main__':
    unittest.main()
