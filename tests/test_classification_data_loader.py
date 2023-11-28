import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import unittest
import torch
import data_loaders.classification_data_loader as cdl

PATH_TO_DATASET = "data/unit_test_data"
DATASET = "test"
CLASESS = ["cat", "dog"]

class TestClassificationDataset(unittest.TestCase):
    def test_init(self):
        # Create a test dataset
        dataset = cdl.ClassificationDataset(path_to_data=PATH_TO_DATASET, dataset=DATASET, classes=CLASESS)

        # Check the attributes
        self.assertEqual(dataset.path_to_data, PATH_TO_DATASET)
        self.assertEqual(dataset.dataset, DATASET)
        self.assertEqual(dataset.classes, CLASESS)

        # Check the lengths of the image paths and labels lists
        self.assertEqual(len(dataset.images_path), 6)
        self.assertEqual(len(dataset.labels), 6)

    def test_load_data(self):
        # Create a test dataset
        dataset = cdl.ClassificationDataset(path_to_data=PATH_TO_DATASET, dataset=DATASET, classes=CLASESS)

        # Check the contents of the image paths and labels lists
        self.assertEqual(dataset.images_path[0], "data/unit_test_data/cat/02.webp")
        self.assertEqual(dataset.images_path[3], "data/unit_test_data/dog/02.webp")
        self.assertEqual(dataset.labels[0], 0)
        self.assertEqual(dataset.labels[3], 1)


    def test_get_class(self):
        dataset = cdl.ClassificationDataset(path_to_data=PATH_TO_DATASET, dataset=DATASET, classes=CLASESS)
        self.assertEqual(dataset.get_class(0), 'cat')
        self.assertEqual(dataset.get_class(3), 'dog')

    def test_dataset_length(self):
        dataset = cdl.ClassificationDataset(path_to_data=PATH_TO_DATASET, dataset=DATASET, classes=CLASESS)
        self.assertEqual(len(dataset), 6) 

    def test_getitems_batch(self):
        dataset = cdl.ClassificationDataset(path_to_data=PATH_TO_DATASET, dataset=DATASET, classes=CLASESS)

        # Assuming at least two images in the dataset for this test
        indices = [0, 1, 3]
        images_path, images, labels = dataset.__getitems__(indices)

        self.assertEqual(len(images_path), 3)
        self.assertEqual(len(images), 3)
        self.assertEqual(len(labels), 3)
        self.assertIsInstance(images_path[0], str)
        self.assertIsInstance(images[0], torch.Tensor)
        self.assertIsInstance(labels[0], int)

if __name__ == '__main__':
    unittest.main()
