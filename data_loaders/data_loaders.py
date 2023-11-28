import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ClassificationDataset(Dataset):
    """
    A custom dataset class for image classification.

    Attributes:
        path_to_data (str): The path to the directory containing the dataset.
        dataset (str): The name of the dataset.
        classes (list): A list containing the names of classes in the dataset.

    Methods:
        __init__: Initializes the dataset with the provided path, dataset name, and class names.
        load_images: Loads images into memory.
        get_class: Returns the class label for a given image index.
        __len__: Returns the total number of images in the dataset.
        __getitem__: Returns a tuple containing the image and its class label.

    Example:
        dataset = ClassificationDataset(path_to_data='path/to/dataset', dataset='example_dataset', classes=['cat', 'dog'])
    """

    def __init__(self, path_to_data, dataset, classes):
        """
        Initialize the ClassificationDataset.

        Args:
            path_to_data (str): The path to the directory containing the dataset.
            dataset (str): The name of the dataset.
            classes (list): A list containing the names of classes in the dataset.
        """
        self.path_to_data = path_to_data
        self.dataset = dataset
        self.classes = classes
        self.images, self.labels = self.load_images()

    def load_images(self):
        """
        Load images into memory.

        Returns:
            tuple: A tuple containing two lists - one for images and one for labels.
        """
        images = []
        labels = []
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.path_to_data, class_name)
            for img_filename in os.listdir(class_path):
                img_path = os.path.join(class_path, img_filename)
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                labels.append(class_idx)

        return images, labels

    def get_class(self, index):
        """
        Get the class label for a given image index.

        Args:
            index (int): Index of the image.

        Returns:
            str: The class label.
        """
        return self.classes[self.labels[index]]

    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
            int: The total number of images.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Return a tuple containing the image and its class label.

        Args:
            index (int): Index of the image.

        Returns:
            tuple: A tuple containing the image and its class label.
        """
        img = self.images[index]
        label = self.labels[index]

        # Apply any additional transformations if needed
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img = transform(img)

        return img, label

