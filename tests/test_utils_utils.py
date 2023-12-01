import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import unittest
import math
import utils.utils


class TestEuclideanDistanceCalculator(unittest.TestCase):
    """
    A class for unit testing the EuclideanDistanceCalculator class.
    """

    def test_distance_calculation(self):
        # Test case 1
        vector1 = (1, 2, 3)
        vector2 = (4, 5, 6)
        self.assertAlmostEqual(utils.utils.calculate_distance(vector1, vector2), math.sqrt(27))

        # Test case 2
        vector3 = (0, 0, 0)
        vector4 = (0, 0, 0)
        self.assertAlmostEqual(utils.utils.calculate_distance(vector3, vector4), 0.0)

        # Test case 3
        vector5 = (1, 2, 3, 4)
        vector6 = (5, 6, 7, 8)
        self.assertAlmostEqual(utils.utils.calculate_distance(vector5, vector6), math.sqrt(64))


if __name__ == '__main__':
    unittest.main()