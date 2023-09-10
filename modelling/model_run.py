# Required imports
import ee
from modelling.model_functions import baseline, train_cnn, predict_cnn, evaluate
from data.data_functions import get_data

# Initialise the Earth Engine module.
ee.Initialize()
