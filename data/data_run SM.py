# Required imports
import ee
from typing import Iterable
from data_functions_SM import get_all_data
import os
import sys
#sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
#from params import MODEL_TARGET, POLYGON, DATA_DATE, FEATURE_BANDS

import numpy as np
""" Provides the setpoint values according to which the data will be collected. """

# Get the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the parent directory
parent_dir = os.path.join(current_dir, '..')

# Convert the relative path to an absolute path
parent_dir = os.path.abspath(parent_dir)

# Append the parent directory to the Python path
sys.path.append(parent_dir)

# Now you can import from the parent directory
from utils import auth_ee

# Initialise the Earth Engine module.
ee.Initialize()

# Defining the main year around which data will be collected
f_date = '2017'

# Select the feature bands
feature_bands = ["B4", "B8"]

#Testing purpose
#coordinates = [-0.173979, 51.441938, -0.166597, 51.446512]
coordinates = [-0.117588, 51.532101, -0.10952, 51.536906]

#Top Left: [-0.173979, 51.441938]
#Top Right: [-0.173979, 51.446512]
#Bottom Left: [-0.166597, 51.446512]
#Bottom Right: [-0.166597, 51.441938]

#coordinates = #correct format for ee API!

# Running the function get_coordinates to test the script, returns NDVI ndarray
NDVI_all = get_all_data(coordinates, f_date, feature_bands)
#print(NDVI_all)
#print(NDVI_all[0])

#print(type(NDVI_all))

# assuming all arrays in NDVI_all have the same shape
NDVI_array = np.stack(NDVI_all, axis=0)  # adjust axis as necessary
#print(NDVI_array.shape)
#print(NDVI_array[0])
#print(NDVI_array[1])
#print(NDVI_array[2])


#print(type(NDVI_array))

# Reshape the array to have shape (N, 2500) and find unique slices
unique_slices = np.unique(NDVI_array.reshape(NDVI_array.shape[0], -1), axis=0)
# Count of unique slices
num_unique_slices = unique_slices.shape[0]
# Compute the standard deviation across the N dimension for each element in the 50x50 slices
std_devs = np.std(NDVI_array, axis=0)
# Summary
print(f"Number of unique slices: {num_unique_slices}")
print(f"Standard deviation array shape: {std_devs.shape}")
print(f"Min standard deviation: {np.min(std_devs)}")
print(f"Max standard deviation: {np.max(std_devs)}")
#if __name__ == "__main__":
#    if MODEL_TARGET == "gcs":
#        print("saving model to cloud")
#    else:
#        print("saving model locally")
