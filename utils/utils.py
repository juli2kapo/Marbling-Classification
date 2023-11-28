import math


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



