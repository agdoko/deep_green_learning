import ee
import numpy as np
from google.cloud import storage
import sys
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
import params
import rasterio
from rasterio.io import MemoryFile

# Initialize Earth Engine
ee.Initialize()

# Initialize Google Cloud Storage client
storage_client = storage.Client()


# Specify the GCS path of the converted image
gcs_path = f'gs://{params.BUCKET}/Features_tf/feature_image_P1.tif'

# Use rasterio to open the image from GCS
with MemoryFile() as memfile:
    with memfile.open(driver='GTiff', width=1, height=1, count=1, dtype='float32', crs='EPSG:4326') as dataset:
        dataset.write_band(1, rasterio.open(gcs_path).read(1))
        ndvi_array = dataset.read(1)
        print(ndvi_array)
