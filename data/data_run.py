# Required imports
import ee
from typing import Iterable
from data.data_functions import get_data, get_target_image, get_coordinates_felix

""" Provides the setpoint values according to which the data will be collected. """

# Initialise the Earth Engine module.
ee.Initialize()

# Defining the main year around which data will be collected
f_date = '2017'

# Defining the target ImageCollection, filtered by the main year
target = (ee.ImageCollection("MODIS/061/MCD12Q1")
          .filterDate(f_date)
          .sort('system:time_start'))  # Sort by time to get earliest image

# Oversimplified North America region.
polygon = [[[-145.7, 63.2], [-118.1, 22.3], [-78.2, 5.6], [-52.9, 47.6]]]

# Global polygon, while minimising the amount of water
#polygon = [[[-180, -60], [180, -60], [180, 85], [-180, 85], [-180, -60]]]

# Select the feature bands
feature_bands = ["B4", "B8"]

# Running the function get_coordinates to test the script
get_data(get_coordinates_felix(polygon, target), int(f_date), feature_bands, get_target_image(target))
