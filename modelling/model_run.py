# Required imports
#import params
import os
import sys

import ee
import numpy as np
from model_functions import baseline, evaluate_model, majority_pool, process_and_expand, train_cnn


""" Provides the setpoint values according to which the data will be collected. """
# Get the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the parent directory
parent_dir = os.path.join(current_dir, '..')

# Convert the relative path to an absolute path
parent_dir = os.path.abspath(parent_dir)

# Append the parent directory to the Python path
sys.path.append(parent_dir)

import params as params
from data.data_functions_MVP import get_data
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
train_f, train_t, test_f, test_t = get_data(params.POLYGON, params.DATA_DATE, params.FEATURE_BANDS) #get_target_image(target))

processed_train_f = process_and_expand(train_f)
processed_train_t = process_and_expand(train_t)
processed_test_f = process_and_expand(test_f)
processed_test_t = process_and_expand(test_t)

#Train and predict CNN Model
model = train_cnn(processed_train_f, processed_train_t)
y_pred = model.predict(processed_test_f)

#Save model as h5 format
h5model = model.save("model.h5")

# Test the evaluation function (using train rather than test just because it's more datat to check, in end will need to use test)
loss, f1_score = model.evaluate(processed_test_f, processed_test_t)
print("Test Loss:", loss)
print("Test F1 Score:", f1_score)

# Test print shape of returned data
print(test_f.shape)
print(test_t.shape)



# Test majority pool function
#result = majority_pool(test_f)
#print(result.shape)  # Should output (N, 3, 3)
#print(result)
