import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import unittest
import numpy as np
import utils.arrays as ua

class TestMapValuesWithDict(unittest.TestCase):

    def test_mapping(self):
        # Test the mapping function with a simple dictionary
        input_array = np.array([[1, 2, 3],
                                [2, 3, 1],
                                [3, 1, 2]])
        mapping_dict = {1: 0, 2: 0, 3: 1}
        expected_result = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [1, 0, 0]])
        mapped_result = ua.map_values_with_dict(input_array, mapping_dict)
        self.assertTrue(np.array_equal(mapped_result, expected_result))


    def test_empty_array(self):
        # Test the mapping function with an empty array
        input_array = np.array([])
        mapping_dict = {1: 0, 2: 0, 3: 1}
        expected_result = np.array([])
        mapped_result = ua.map_values_with_dict(input_array, mapping_dict)
        self.assertTrue(np.array_equal(mapped_result, expected_result))


    def test_mapping_with_empty_dict(self):
        # Test the mapping function with an empty mapping dictionary
        input_array = np.array([[1, 2, 3],
                                [2, 3, 1],
                                [3, 1, 2]])
        mapping_dict = {}
        expected_result = np.copy(input_array)
        mapped_result = ua.map_values_with_dict(input_array, mapping_dict)
        self.assertTrue(np.array_equal(mapped_result, expected_result))


    def test_mapping_with_scalar(self):
        # Test the mapping function with an empty mapping dictionary
        input_array = np.array([1])
        mapping_dict = {1:0}
        expected_result = np.array([0])
        mapped_result = ua.map_values_with_dict(input_array, mapping_dict)
        self.assertTrue(np.array_equal(mapped_result, expected_result))



if __name__ == '__main__':
    unittest.main()