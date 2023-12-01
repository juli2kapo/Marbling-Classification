import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import unittest
import utils.images
import numpy as np


class TestImageSegmentation(unittest.TestCase):
    """
    A class for unit testing the ImageSegmentation class.
    """

    def test_segmentation(self):
        # Test case with a small image
        image = np.array([[[1, 2, 3], [4, 5, 6]],
                          [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)

        vector1 = (1, 2, 3)
        vector2 = (7, 8, 9)

        segmentation_mask = utils.images.binary_segmentation_by_distance(image, vector1, vector2)

        expected_mask = np.array([[255, 0],
                                  [0, 0]], dtype=np.uint8)

        np.testing.assert_array_equal(segmentation_mask, expected_mask)


    def test_connected_components(self):
        # Create a synthetic segmentation mask for testing
        segmentation_mask = np.zeros((10, 10), dtype=np.uint8)
        segmentation_mask[1:5, 1:5] = 255  # Component 1
        segmentation_mask[7:9, 7:9] = 255  # Component 2
        segmentation_mask[6:8, 2] = 255    # Component 3 (area = 2)

        # Find connected components with area >= 4
        components = utils.images.find_connected_components(segmentation_mask, thresh_area=4)

        # Check the number of components
        self.assertEqual(len(components), 2)

        # Check the sizes of the components
        self.assertEqual(np.count_nonzero(components[0]), 16)  # Size of Component 1
        self.assertEqual(np.count_nonzero(components[1]), 4)   # Size of Component 2


    def test_analyze_components_properties(self):
        # Create a synthetic segmentation mask for testing
        segmentation_mask = np.zeros((10, 10), dtype=np.uint8)
        
        # Component 1: 4x4 square
        segmentation_mask[1:5, 1:5] = 255  

        # Component 2: One pixel
        segmentation_mask[1, 8] = 255

        # Component 3: 2x3 rectangle
        segmentation_mask[7:9, 7:10] = 255  

        # Component 4: 'C' shape
        segmentation_mask[6:9, 1:3] = 255   #  255, 255, 255
        segmentation_mask[6, 3] = 255       #  255, 255, 0
        segmentation_mask[8, 3] = 255       #  255, 255, 255


        # Find connected components
        components = utils.images.find_connected_components(segmentation_mask)

        # Analyze area and perimeter of each connected component
        components_properties = utils.images.analyze_components_shape_properties(components)

        # Check the results
        self.assertEqual(len(components_properties), 4)

        # Check the properties of Component 1
        self.assertEqual(components_properties[0]["area"], 16)
        self.assertEqual(components_properties[0]["perimeter"], 12) 
        self.assertEqual(components_properties[0]["solidity"], 1) 

        # Check the properties of Component 2
        self.assertEqual(components_properties[1]["area"], 1)
        self.assertEqual(components_properties[1]["perimeter"], 0) 
        self.assertEqual(components_properties[1]["solidity"], 0) 

        # Check the properties of Component 3
        self.assertEqual(components_properties[2]["area"], 8)
        self.assertEqual(components_properties[2]["perimeter"], 8.828427076339722) 
        self.assertEqual(components_properties[2]["solidity"], 0.75) 

        # Check the properties of Component 4
        self.assertEqual(components_properties[3]["area"], 6)
        self.assertEqual(components_properties[3]["perimeter"], 6) 
        self.assertEqual(components_properties[3]["solidity"], 1) 


if __name__ == '__main__':
    unittest.main()