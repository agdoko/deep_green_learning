import ee
import numpy as np
from google.cloud import storage
import os
import sys
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
import params
import rasterio
from rasterio.io import MemoryFile

# Initialize Earth Engine
ee.Initialize()

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket(params.BUCKET)

# Specify the GCS path of the converted image
gcs_path = f'gs://{params.BUCKET}/Targets_tf/target_image_P1.tif'
output_path = 'Features_npy/feature_image_P1.npy'

# Use rasterio to open the image from GCS
# Open an existing GeoTIFF file for writing
with rasterio.open(gcs_path, 'r') as src:
    # Iterate over the bands you want to write to
    image_data = src.read()
    print(image_data.shape)
    #with open('feature_image_P1.npy', 'wb') as ndarray_file:
    #    np.save(ndarray_file, image_data)
    #    blob = bucket.blob(output_path)
    #    blob.upload_from_filename('feature_image_P1.npy')

# Remove the temporary file
#os.remove('feature_image_P1.npy')
