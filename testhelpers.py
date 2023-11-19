import numpy as np


def array_equal(output, expected, roundFactor=2):
    """
    Compares two arrays for equality, considering a specified round factor.

    Args:
        output (ndarray): The output array to compare.
        expected (ndarray): The expected array to compare against.
        roundFactor (int, optional): The round factor for rounding the output array. Defaults to 2.

    Returns:
        bool: True if the arrays are equal, False otherwise.

    """
    output = np.round(output, roundFactor)
    return np.array_equal(output, expected)
