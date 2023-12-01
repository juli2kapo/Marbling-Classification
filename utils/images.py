import numpy as np
import cv2
from utils.utils import calculate_distance
from skimage import feature


def binary_segmentation_by_distance(image, vector1, vector2):
    """
    Segment the input image based on the Euclidean distance between each pixel and the provided vectors.  If 
    the pixel's distance to vector1 is less than the distance to vector2, it assign the value 255; otherwise, 
    the value 0.

    Args:
    image (numpy.ndarray): The input image.
    vector1 (list or tuple): The first vector.
    vector2 (list or tuple): The second vector.

    Returns:
    numpy.ndarray: The segmented image mask.
    """

    # Calculate distances for each pixel
    distances_v1 = np.apply_along_axis(lambda pixel: calculate_distance(pixel, vector1), axis=2, arr=image)
    distances_v2 = np.apply_along_axis(lambda pixel: calculate_distance(pixel, vector2), axis=2, arr=image)

    # Create a mask based on the distances
    segmentation_mask = np.where(distances_v1 < distances_v2, 255, 0).astype(np.uint8)

    return segmentation_mask


def find_connected_components(segmentation_mask, thresh_area=0):
    """
    Find connected components in a binary segmentation mask.

    Args:
    segmentation_mask (numpy.ndarray): Binary segmentation mask.
    thresh_area (int): Threshold area. Components with area less than thresh_area will be discarded.

    Returns:
    list: List of connected components, where each component is represented by a NumPy array of pixel coordinates.
    """

    # Ensure the input is binary
    if segmentation_mask.dtype != np.uint8:
        raise ValueError("Input mask must be a binary image (dtype=np.uint8).")

    # Find connected components
    num_labels, labels = cv2.connectedComponents(segmentation_mask, connectivity=8)

    # Extract connected components as NumPy arrays
    connected_components = []
    for label in range(1, num_labels):  # Exclude background label 0
        component_mask = np.zeros(labels.shape, dtype=np.uint8)
        component_mask[labels == label] = 255
        area = np.count_nonzero(component_mask)
        # Check if the area is greater than thresh_area
        if area >= thresh_area:
            connected_components.append(component_mask)

    return connected_components


def analyze_components_shape_properties(connected_components, thresh_area=0):
    """
    Analyze the area, perimeter and solidity of each connected component.

    Args:
    connected_components (list): list of connected components to analyze.
    thresh_area (int): Threshold area. Connected components with area less than thresh_area will be discarded.

    Returns:
    list: List of dictionaries, where each dictionary contains the area and perimeter of a connected component.
    """
    if not connected_components:
        raise ValueError("No connected components found. Call find_connected_components first.")

    components_properties = []
    for component in connected_components:

        # Compute area
        area = np.count_nonzero(component)

        # Check if the area is greater than thresh_area
        if area >= thresh_area:

            # compute perimeter
            contours, _ = cv2.findContours(component, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(contours[0], closed=True)

            # Compute solidity
            inner_area = cv2.contourArea(contours[0])
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            solidity = 0
            if hull_area > 0:
                solidity = float(inner_area)/hull_area

            component_properties = {"area": area, "perimeter": perimeter, "solidity": solidity}
            components_properties.append(component_properties)

    return components_properties



class LocalBinaryPatterns:
    # From https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
            
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist