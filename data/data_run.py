# Required imports
import ee
from typing import Iterable
from data_functions_fg import get_data, get_target_image, get_coordinates_felix
import sys
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
from params import MODEL_TARGET, POLYGON, DATA_DATE, FEATURE_BANDS

""" Provides the setpoint values according to which the data will be collected. """

# Initialise the Earth Engine module.
ee.Initialize()


# Running the function get_coordinates to test the script
get_data(POLYGON, DATA_DATE, FEATURE_BANDS)

#if __name__ == "__main__":
#    if MODEL_TARGET == "gcs":
#        print("saving model to cloud")
#    else:
#        print("saving model locally")
