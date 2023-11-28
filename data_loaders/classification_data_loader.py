# This file contains the definition of the ClassificationDataset class, which
# loads a dataset of images for image classification problems.

# Import necessary libraries
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
        images_path (list): A list containing the path of images in the dataset.
        images (list) : A list containing the images in the dataset.
        labels (list) : A list containing the index of classes in the dataset.

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
        self.images_path, self.images, self.labels = self.load_images()

    def load_images(self):
        """
        Load images into memory.

        Returns:
            tuple: A tuple containing two lists - one for images and one for labels.
        """
        images_path = [] 
        images = []
        labels = []
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.path_to_data, class_name)
            for img_filename in os.listdir(class_path):
                img_path = os.path.join(class_path, img_filename)
                img = Image.open(img_path).convert('RGB')
                images_path.append(img_path)
                images.append(img)
                labels.append(class_idx)

        return images_path, images, labels

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
            tuple: A tuple containing the image and its path and class label.
        """
        path = self.images_path[index]
        img = self.images[index]
        label = self.labels[index]

        # Apply any additional transformations if needed
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img = transform(img)

        return path, img, label

    def __getitems__(self, indices):
        """
        Return a batch of images and their class labels.

        Args:
            indices (list): List of indices for the images.

        Returns:
            tuple: A tuple containing a batch of images and their corresponding path and class labels.
        """
        images_path = [self.images_path[i] for i in indices]
        images = [self.images[i] for i in indices]
        labels = [self.labels[i] for i in indices]

        # Apply any additional transformations if needed
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        images = [transform(img) for img in images]

        return images_path, images, labels