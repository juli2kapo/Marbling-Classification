import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import argparse
import cv2
import numpy as np
from skimage import feature

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable


class mlp_class(nn.Module):
    def __init__(self, input_net, output_net):
        super().__init__()
        self.hidden1 = nn.Linear(input_net, 150)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(150, 150)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(150, 50)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(50, output_net)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.output(x)
        return x


class NeuralNetworkClassifier:
    """
    A classifier that uses a PyTorch neural network model for image classification.

    Parameters:
    - model_path: Path to the PyTorch model file.

    Methods:
    - load_model(): Loads the PyTorch model from the specified path.
    - classify(lbp_histogram): Classifies an image using the provided LBP histogram.

    Attributes:
    - model: PyTorch neural network model.
    """

    def __init__(self, model_path, input_model, output_model):
        """
        Initialize the Neural Network Classifier.

        Args:
        - model_path (str): Path to the PyTorch model file.
        - input_model (int): Model input size.
        - output_model (int): Model output size.
        - output_model (int): Path to the PyTorch model file.
        """
        self.model_path = model_path
        self.input_model = input_model 
        self.output_model = output_model 
        self.model = self.load_model()

    def load_model(self):
        """
        Load the PyTorch model from the specified path.

        Returns:
        - model: PyTorch neural network model.
        """
        model = mlp_class(self.input_model, self.output_model) 
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        return model

    def classify(self, lbp_histogram):
        """
        Classify an image using the provided LBP histogram.

        Args:
        - lbp_histogram (numpy.ndarray): LBP histogram for the input image.

        Returns:
        - prediction (int): Predicted class (0 or 1).
        """
        # Convert the LBP histogram to a PyTorch tensor
        input_tensor = torch.FloatTensor(lbp_histogram).unsqueeze(0)

        # Make the input tensor a variable
        input_variable = Variable(input_tensor)

        # Forward pass through the neural network
        output = self.model(input_variable)

        # Get the predicted class (0 or 1)
        _, prediction = torch.max(output.data, 1)

        # model.eval()
        # y_pred = model(X_test)

        # y_pred_np = torch.argmax(y_pred, 1).detach().numpy()
        # y_test_np = torch.argmax(y_test, 1).detach().numpy()

        # cm = confusion_matrix(y_test_np, y_pred_np)
        # acc = accuracy_score(y_test_np, y_pred_np)
        # prec = precision_score(y_test_np, y_pred_np, average='macro')
        # rec = recall_score(y_test_np, y_pred_np, average='macro')
        # f1 = f1_score(y_test_np, y_pred_np, average='macro')


        return prediction.item()


class LBPImageClassifier:
    """
    A simple image classifier using Local Binary Patterns (LBP).

    Parameters:
    - p: Number of neighbors for LBP computation.
    - radius: Radius for LBP computation.
    - eps: Bias for the histogram.

    Methods:
    - compute_lbp_histogram(image): Computes the LBP histogram for an image (as a NumPy array or path to the image).

    Attributes:
    - p: Number of neighbors for LBP computation.
    - radius: Radius for LBP computation.
    - eps: Bias for the histogram.
    """

    def __init__(self, p=8, radius=1, eps=1e-7):
        """
        Initialize the LBPImageClassifier.

        Args:
        - p (int): Number of neighbors for LBP computation.
        - radius (int): Radius for LBP computation.
        - eps (float): Bias for the histogram.
        """
        self.p = p
        self.radius = radius
        self.eps = eps


    def compute_lbp_histogram(self, image):
        """
        Compute the LBP histogram for a given image.

        Args:
        - image (numpy.ndarray or str): Input image as a NumPy array or path to the image.

        Returns:
        - histogram (numpy.ndarray): LBP histogram for the input image.
        """

        if isinstance(image, str):  # If input is a path to an image
            # Read the image
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.p,
            self.radius, method="uniform")
        (histogram, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.p + 3),
            range=(0, self.p + 2))
        # normalize the histogram
        histogram = histogram.astype("float")
        histogram /= (histogram.sum() + self.eps)
        
        # return the histogram of Local Binary Patterns
        return histogram


def get_image_paths(input_dir):
    """
    Recursively retrieves paths of all image files within the specified directory.

    Args:
    - input_dir (str): Path to the input directory.

    Returns:
    - image_paths (list): List of paths to image files.
    """
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='LBP Image Classifier')
    parser.add_argument('-i', '--image_path', type=str, help='Path to the input image', required=True)
    parser.add_argument('-m', '--model_path', type=str, help='Path to the classification model', required=True)
    parser.add_argument('-o', '--output_model', type=int, help='Model output size', required=True)
    parser.add_argument('-p', '--p', type=int, default=8, help='Number of neighbors for LBP computation')
    parser.add_argument('-r', '--radius', type=int, default=1, help='Radius for LBP computation')

    args = parser.parse_args()

    # Validate image path
    if not os.path.isfile(args.image_path):
        # Validate input directory
        if not os.path.isdir(args.input_dir):
            print(f"Error: The specified input directory '{args.image_path}' is not valid or does not exist.")
            return
        else:
            print(f"Error: The specified image path '{args.image_path}' is not valid or does not exist.")
            return

    # Validate model path
    if not os.path.isfile(args.model_path):
        print(f"Error: The specified model path '{args.model_path}' is not valid or does not exist.")
        return

    # Initialize the classifier
    classifier = LBPImageClassifier(p=args.p, radius=args.radius)

    # Compute LBP histogram for the input image
    lbp_histogram = classifier.compute_lbp_histogram(args.image_path)

    # Print the computed histogram
    print("LBP Histogram:", lbp_histogram)

    # Initialize the neural network classifier
    neural_network_classifier = NeuralNetworkClassifier(model_path=args.model_path, input_model=args.p+2, output_model=args.output_model)

    # Classify the image based on the LBP histogram
    prediction = neural_network_classifier.classify(lbp_histogram)

    # Print the classification result
    print("Classification result:", prediction)


if __name__ == '__main__':
    main()

