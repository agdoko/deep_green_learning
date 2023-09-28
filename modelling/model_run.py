# Required imports
import ee
from google.cloud import storage
from model_functions import baseline, evaluate_model, majority_pool, process_and_expand, train_cnn
from get_data_from_gcs import get_data
import sys
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
import params
import os
""" Provides the setpoint values according to which the data will be collected. """
# Get the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize Earth Engine
ee.Initialize()

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket(params.BUCKET)

# List all objects (files) in the specified GCS folder
targets = bucket.list_blobs(prefix='Targets_npy/')
features = bucket.list_blobs(prefix='Features_npy/')

# Getting data to evaluate the model
train_f, train_t, test_f, test_t = get_data(targets,features)

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
