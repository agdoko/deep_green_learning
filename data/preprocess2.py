import ee
import numpy as np
from google.cloud import storage
import os
import sys
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
import params
import rasterio
from rasterio.io import MemoryFile
import re

# Initialize Earth Engine
ee.Initialize()

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket(params.BUCKET)

# Specify the GCS folder path you want to process
targets = 'Targets/'
features = 'Features/'

# List all objects (files) in the specified GCS folder
blobs_t = bucket.list_blobs(prefix=targets)
blobs_f = bucket.list_blobs(prefix=features)
#output_path = 'Targets_npy/target_image_P1.npy'

points_list = [] # creating a new list to store the points where feature image is available
pattern = r'P(\d+)'

for blob in blobs_f:
    # Get the object's name (path relative to the folder)
    object_name = blob.name

    # Use re.search to find the pattern in the string
    match = re.search(pattern, object_name)

    # Check if a match was found
    if match:
    # Extract the matched portion (P followed by the number)
        points_list.append(match.group(0))


    # Skip if it's not a GeoTIFF file
    if not object_name.endswith('.tif'):
        continue

    # Specify the output path for the numpy file
    output_path = f'Features_npy/{os.path.splitext(os.path.basename(object_name))[0]}.npy'
    #print(output_path)
    # Use rasterio to open the image from GCS
    with rasterio.open(f'gs://{params.BUCKET}/{object_name}', 'r') as src:
        image_data = src.read()
        # You can perform operations on image_data if needed
        clean_data = np.nan_to_num(image_data, nan=0)
        reshaped_array = np.reshape(clean_data, (50,50))
        image_array = reshaped_array[ :, :, np.newaxis]
        #print(image_array.shape)

        # Save the image_data as a numpy file
        with open('feature.npy', 'wb') as ndarray_file:
            np.save(ndarray_file, image_array)

            # Upload the numpy file to GCS
            blob = bucket.blob(output_path)
            blob.upload_from_filename('feature.npy')

        # Remove the temporary numpy file
        os.remove('feature.npy')


# Loop through each object in the folder
for blob in blobs_t:
    # Get the object's name (path relative to the folder)
    object_name = blob.name

    match = re.search(pattern, object_name)

    # Check if a match was found
    if match.group(0) in points_list:

        # Skip if it's not a GeoTIFF file
        if not object_name.endswith('.tif'):
            continue

        # Specify the output path for the numpy file
        output_path = f'Targets_npy/{os.path.splitext(os.path.basename(object_name))[0]}.npy'
        #print(output_path)
        # Use rasterio to open the image from GCS
        with rasterio.open(f'gs://{params.BUCKET}/{object_name}', 'r') as src:
            image_data = src.read()
            # You can perform operations on image_data if needed
            clean_data = np.nan_to_num(image_data, nan=0)

            reshaped_array = np.reshape(clean_data, (1,1))
            image_array = reshaped_array[ :, :, np.newaxis]
            #print(image_array.shape)

            # Save the image_data as a numpy file
            with open('target.npy', 'wb') as ndarray_file:
                np.save(ndarray_file, image_array)

                # Upload the numpy file to GCS
                blob = bucket.blob(output_path)
                blob.upload_from_filename('target.npy')

            # Remove the temporary numpy file
            os.remove('target.npy')
