import ee
import numpy as np
from google.cloud import storage
import os
import sys
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
import params
from io import BytesIO

# Initialize Earth Engine
ee.Initialize()

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket(params.BUCKET)

# List all objects (files) in the specified GCS folder
targets = bucket.list_blobs(prefix='Targets_npy/')
features = bucket.list_blobs(prefix='Features_npy/')



def get_data(targets, features):

    stacked_feature_list = []
    stacked_target_list = []



    # Loop through each object in the folder
    for blob in features:

        # Skip if it's not a GeoTIFF file
        if not blob.name.endswith('.npy'):
            continue


        # Get the .npy file as bytes from GCS
        npy_bytes = blob.download_as_bytes()

        # Load the data from the .npy bytes
        loaded_data = np.load(BytesIO(npy_bytes))
        stacked_feature_list.append(loaded_data)


    for blob in targets:

        # Skip if it's not a GeoTIFF file
        if not blob.name.endswith('.npy'):
            continue

        # Get the .npy file as bytes from GCS
        npy_bytes = blob.download_as_bytes()

        # Load the data from the .npy bytes
        loaded_data = np.load(BytesIO(npy_bytes))
        stacked_target_list.append(loaded_data)


    target_stacked_array = np.stack(stacked_target_list, axis=0)
    feature_stacked_array = np.stack(stacked_feature_list, axis=0)

    depth = target_stacked_array.shape[0]
    split_index = int(depth * 0.8)  # 80% for training

    # Train-test split
    train_target = target_stacked_array[:split_index]
    test_target = target_stacked_array[split_index:]

    train_feature = feature_stacked_array[:split_index]
    test_feature = feature_stacked_array[split_index:]

    print(train_feature.shape, train_target.shape, test_feature.shape, test_target.shape)

    return train_feature, train_target, test_feature, test_target
