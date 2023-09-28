# Required imports
import ee
from typing import Iterable
from data_functions_SM import get_all_data
import streamlit as st
import sys
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
#from params import MODEL_TARGET, POLYGON, DATA_DATE, FEATURE_BANDS
from utils import auth_ee
""" Provides the setpoint values according to which the data will be collected. """

# Initialise the Earth Engine module.
auth_ee(st.secrets['client_email'], st.secrets['private_key'])


# Defining the main year around which data will be collected
f_date = '2017'

# # Defining the target ImageCollection, filtered by the main year
# target = (ee.ImageCollection("MODIS/061/MCD12Q1")
#           .filterDate(f_date)
#           .sort('system:time_start'))  # Sort by time to get earliest image

# Oversimplified North America region.
#polygon = [[[-145.7, 63.2], [-118.1, 22.3], [-78.2, 5.6], [-52.9, 47.6]]]

# Global polygon, while minimising the amount of water
#polygon = [[[-180, -60], [180, -60], [180, 85], [-180, 85], [-180, -60]]]

#4 x 4 coords
coordinates = [-0.140934, 51.487877, -0.112438, 51.504976]

# Select the feature bands
feature_bands = ["B4", "B8"]

# Running the function get_coordinates to test the script
result = get_all_data(coordinates, f_date, feature_bands)
print(result)

#if __name__ == "__main__":
#    if MODEL_TARGET == "gcs":
#        print("saving model to cloud")
#    else:
#        print("saving model locally")
