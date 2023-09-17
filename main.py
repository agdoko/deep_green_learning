# This is where the main api flow goes when running on CloudRun
import ee
from flask import Flask, jsonify, request
import data.data_functions as dat # import the data functions
import modelling.model_functions as mod # import the model functions
from utils import auth_ee

app = Flask(__name__)

@app.route('/', methods=['POST'])
def main_func():
    # HARDCODED / necessary? Initialise the Earth Engine module.
    print('Starting')
    auth_ee()

    data = request.get_json()   # Get the data from the request. year + point
                                # year: string of the requested year
                                # point: list of two coordinates (lat, lon)

    # HARDCODED Defining the target ImageCollection, filtered by the main year
    target = (ee.ImageCollection("MODIS/061/MCD12Q1")
          .filterDate(data['year'])
          .sort('system:time_start'))

    # HARDCODED Select the feature bands
    feature_bands = ["B4", "B8"]

    # Getting data to evaluate the model
    train_f, train_t, test_f, test_t = dat.get_data(data["point"], int(data["year"]), feature_bands, dat.get_target_image(target))

    # Test the evaluation function (using train rather than test just because it's more datat to check, in end will need to use test)
    result = mod.evaluate(train_t, mod.baseline(train_f))

    return jsonify({"f1": result})
