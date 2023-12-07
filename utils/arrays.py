import numpy as np


def map_values_with_dict(array, mapping_dict):
    """
    Maps values in the array to new values based on a user-defined mapping dictionary.

    Args:
    - array (numpy.ndarray): Input array.
    - mapping_dict (dict): Dictionary specifying the mapping of values.

    Returns:
    - mapped_array (numpy.ndarray): Array with mapped values.
    """
    if len(array)==0:
        mapped_array = np.array([])
        return mapped_array

    if len(mapping_dict)==0:
        mapped_array = array
        return mapped_array

    # Apply mapping using the numpy 'vectorize' function and the provided dictionary
    mapped_array = np.vectorize(mapping_dict.get)(array)
    return mapped_array
