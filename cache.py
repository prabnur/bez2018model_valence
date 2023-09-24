import numpy as np
import os

def save(path, array):
    # Extract the directory name from the file path
    directory = os.path.dirname(path)
    
    # Create the directory recursively if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save the numpy array to the path
    np.save(path, array)