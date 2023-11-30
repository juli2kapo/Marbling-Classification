import math
import cv2 
import numpy as np

def calculate_distance(vector1, vector2):
    """
    Calculate the Euclidean distance between two vectors.

    Args:
    vector1 (list or tuple): The first vector.
    vector2 (list or tuple): The second vector.

    Returns:
    float: The Euclidean distance between the two vectors.
    """
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimensionality.")

    squared_diff = sum((x - y)**2 for x, y in zip(vector1, vector2))
    distance = math.sqrt(squared_diff)
    
    return distance

def binarySegmentationWithCalibration(imgtocalibrate,calibration):
    
    calibration = cv2.resize(fill, (imgtocalibrate.shape[1], imgtocalibrate.shape[0]))
    img_combined = np.vstack([imgtocalibrate, calibration])

    img_reshaped = img_combined.reshape((-1, 3))
    img_float = np.float32(img_reshaped)


    # Define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels_cv2, centers_cv2 = cv2.kmeans(img_float, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape labels to have the same dimensions as the original image
    labels_reshaped_cv2 = labels_cv2.reshape(img_combined.shape[0], img_combined.shape[1])
    
    # Create masks for each cluster
    masks = [(labels_reshaped_cv2 == i) for i in range(2)]
    masks = [mask[:imgtocalibrate.shape[0], :] for mask in masks]
    segmented_parts_cv2 = [cv2.bitwise_and(imgtocalibrate, imgtocalibrate, mask=masks[i].astype(np.uint8)*255) for i in range(2)]
    

    avg_intensities = [np.mean(segment) for segment in segmented_parts_cv2]

    sorted_indices = np.argsort(avg_intensities)[::1]  # Sort in descending order

    masks = [masks[i] for i in sorted_indices]
    mask_cv2 = masks[0]
    
    return mask_cv2.astype(np.uint8)*255

