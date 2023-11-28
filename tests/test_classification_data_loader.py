import unittest
from data_loaders.data_loaders import ClassificationDataset


class TestClassificationDataset(unittest.TestCase):
    def test_dataset_creation(self):
        dataset = ClassificationDataset(path_to_data='path/to/dataset', dataset='test_dataset', classes=['class1', 'class2'])
        self.assertEqual(len(dataset), 0)  # Assuming an empty dataset for simplicity

    def test_get_class(self):
        dataset = ClassificationDataset(path_to_data='path/to/dataset', dataset='test_dataset', classes=['class1', 'class2'])
        self.assertEqual(dataset.get_class(0), 'class1')

    def test_dataset_length(self):
        dataset = ClassificationDataset(path_to_data='path/to/dataset', dataset='test_dataset', classes=['class1', 'class2'])
        self.assertEqual(len(dataset), 0)  # Assuming an empty dataset for simplicity


if __name__ == '__main__':
    unittest.main()
