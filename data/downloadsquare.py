# Required imports
# Required imports
import ee
import io
import requests
import numpy as np
from typing import Iterable
import sys
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
#import params
# Import additional libraries
import math
from itertools import product
import ee
from typing import Iterable
from data_functions_SM import get_all_data, get_input_image_mean
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
date = 2017

# Select the feature bands
feature_bands = ["B4", "B8"]

#Testing purpose
#coordinates = [-0.173979, 51.441938, -0.166597, 51.446512]

#1 x 1 coords
#coordinates = [-0.119047, 51.513891, -0.113382, 51.516935]

#2 x 2 coords
#coordinates = [-0.116129, 51.538935, -0.102911, 51.545608]

#n x n coords
square = [-0.140934, 51.487877, -0.112438, 51.504976]

#Top Left: [-0.173979, 51.441938]
#Top Right: [-0.173979, 51.446512]
#Bottom Left: [-0.166597, 51.446512]
#Bottom Right: [-0.166597, 51.441938]

def get_input_image_large(year: int, feature_bands, square, type):
    if type == "image":
        collection = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")  # Sentinel-2 images
            .filterDate(f"{int(year)}-1-1", f"{int(year)}-12-31")  # filter by year
            .filterBounds(square)
            .select(feature_bands)  # select all bands starting with B
            .sort('system:time_start')
        )

        # Logging the size of the collection
        print(f'Size of the collection: {collection.size().getInfo()}')

        # Logging the first image info
        first_image = collection.first()
        print(f'First image info: {first_image.getInfo()}')
        return first_image

large_image = get_input_image_large(date, feature_bands, square, 'image')
print(large_image)
print(large_image.shape())



#coordinates = #correct format for ee API!

# Running the function get_coordinates to test the script, returns NDVI ndarray
#NDVI_all = get_all_data(coordinates, f_date, feature_bands)
#print(NDVI_all)
#print(NDVI_all[0])

#print(type(NDVI_all))

# assuming all arrays in NDVI_all have the same shape
#NDVI_array = np.stack(NDVI_all, axis=0)  # adjust axis as necessary
#print(NDVI_array.shape)
#print(NDVI_array)
#print(NDVI_array[1])
#print(NDVI_array[2])


#print(type(NDVI_array))

# Reshape the array to have shape (N, 2500) and find unique slices
#unique_slices = np.unique(NDVI_array.reshape(NDVI_array.shape[0], -1), axis=0)
# Count of unique slices
##num_unique_slices = unique_slices.shape[0]
# Compute the standard deviation across the N dimension for each element in the 50x50 slices
std_devs = np.std(NDVI_array, axis=0)
# Summary
#print(f"Number of unique slices: {num_unique_slices}")
#print(f"Standard deviation array shape: {std_devs.shape}")
#print(f"Min standard deviation: {np.min(std_devs)}")
#print(f"Max standard deviation: {np.max(std_devs)}")
#if __name__ == "__main__":
#    if MODEL_TARGET == "gcs":
#        print("saving model to cloud")
#    else:
#        print("saving model locally")
