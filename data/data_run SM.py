# Required imports
import ee
from typing import Iterable
from data_functions_SM import get_all_data
import sys
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
from params import MODEL_TARGET, POLYGON, DATA_DATE, FEATURE_BANDS
from utils import auth_ee
import numpy as np
""" Provides the setpoint values according to which the data will be collected. """

# Initialise the Earth Engine module.
ee.Initialize()

# Defining the main year around which data will be collected
f_date = '2017'

# Select the feature bands
feature_bands = ["B4", "B8"]

#Testing purpose
coordinates = [-0.173979, 51.441938, -0.166597, 51.446512]

#Top Left: [-0.173979, 51.441938]
#Top Right: [-0.173979, 51.446512]
#Bottom Left: [-0.166597, 51.446512]
#Bottom Right: [-0.166597, 51.441938]

#coordinates = #correct format for ee API!

# Running the function get_coordinates to test the script, returns NDVI ndarray
NDVI_all = get_all_data(coordinates, f_date, feature_bands)
print(NDVI_all[0])

# assuming all arrays in NDVI_all have the same shape
NDVI_array = np.stack(NDVI_all, axis=0)  # adjust axis as necessary
print(NDVI_array.shape)

#if __name__ == "__main__":
#    if MODEL_TARGET == "gcs":
#        print("saving model to cloud")
#    else:
#        print("saving model locally")
