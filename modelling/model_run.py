# Required imports
import ee
import numpy as np
from modelling.model_functions import baseline, evaluate, majority_pool
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

# Getting data to evaluate the model
train_f, train_t, test_f, test_t = get_data(get_coordinates_felix(polygon, target), int(f_date), feature_bands, get_target_image(target))

# Test the evaluation function (using train rather than test just because it's more datat to check, in end will need to use test)
evaluate(train_t, baseline(train_f))

# Test print shape of returned data
#print(test_f.shape)
#print(test_t.shape)

# Test majority pool function
#result = majority_pool(test_f)
#print(result.shape)  # Should output (N, 3, 3)
#print(result)
