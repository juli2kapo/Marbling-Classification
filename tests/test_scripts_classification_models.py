import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import unittest
import scripts.classification_models as scm
import numpy as np

class TestLBPImageClassifier(unittest.TestCase):
    """
    A class for unit testing the LBPImageClassifier class.
    """

    def test_compute_lbp_histogram(self):
        # Set up the classifier with test parameters
        classifier = scm.LBPImageDescriptor(p=24, radius=8)
        
        # Define the path to the test image
        test_image_path_1 = 'data/marbling_dataset_v2/L0/45.png'
        test_image_path_2 = 'data/marbling_dataset_v2/L7/50.png'
        
        hist_true_1 = [0.05709569354756442,0.02971306500944679,0.02608562966865765,0.019940490640093258,0.014107463638939022,0.011173610581283562,0.009890483357486256,0.009155286137364558,0.008142656003989387,0.008336859043266816,0.008905596515436433,0.009918226648811604,0.010341311841523147,0.01145797931736837,0.008787687527303707,0.008336859043266816,0.006706940677902671,0.007816672330916558,0.007525367772000412,0.00916915778302723,0.010812947794054048,0.015376719217073653,0.022042044957988303,0.03296596591734374,0.04604692777724492,0.5901483572499531]
        hist_true_2 = [0.05916430743404853,0.028258362168322695,0.029359337317737863,0.021023382615023005,0.015413652091812379,0.013316556569116816,0.010275768061208253,0.009227220299860471,0.009646639404399584,0.009017510747590916,0.009174792911793083,0.00857187794901811,0.007523330187670328,0.010747614553814754,0.006055363321783434,0.006238859180019296,0.0051378840306041266,0.005662157911278017,0.0030407885079085646,0.00411554996329004,0.0038534130229530947,0.0065796372024573245,0.012556359442139675,0.03292439970632032,0.03551955541565607,0.6375956799815519]

        # Check if the compute_lbp_histogram method returns a valid histogram
        hist_test_1 = classifier.compute_lbp_histogram(test_image_path_1)
        hist_test_2 = classifier.compute_lbp_histogram(test_image_path_2)

        self.assertTrue(isinstance(hist_test_1, np.ndarray))
        self.assertEqual(len(hist_test_1), classifier.p + 2)
        np.testing.assert_array_equal(hist_true_1, hist_test_1)

        self.assertTrue(isinstance(hist_test_2, np.ndarray))
        self.assertEqual(len(hist_test_2), classifier.p + 2)
        np.testing.assert_array_equal(hist_true_2, hist_test_2)
        

    def test_compute_lbp_histogram_with_invalid_image_path(self):
        # Check if the compute_lbp_histogram method raises an exception for an invalid image path
        with self.assertRaises(Exception):
            self.classifier.compute_lbp_histogram('invalid/image/path.jpg')

if __name__ == '__main__':
    unittest.main()
