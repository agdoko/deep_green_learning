import ee
import sys
from data_functions_fg import get_coordinates_felix, get_target_image, get_input_image
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
import params as params
from google.cloud import storage

storage_client = storage.Client()

bucket = storage_client.get_bucket(params.BUCKET)
targets = 'Targets_nc/'
features = 'Features_nc/'

# List all objects (files) in the specified GCS folder
blobs_t = bucket.list_blobs(prefix=targets)
blobs_f = bucket.list_blobs(prefix=features)
# Print the list of objects
for blob in blobs_t:
    #print(f'Object Name: {blob.name}')
    blob.delete()
for blob in blobs_f:
    #print(f'Object Name: {blob.name}')
    blob.delete()

# Initialize Earth Engine
ee.Initialize()

def custom_ndvi(image):
    # Select the red and near-infrared bands
    band_red = image.select('B4')  # Replace 'B4' with your red band
    band_nir = image.select('B8')  # Replace 'B8' with your NIR band

    # Calculate NDVI with a small constant added to the denominator to avoid division by zero
    constant = 1e-10
    ndvi = band_nir.subtract(band_red).divide(band_nir.add(band_red).add(constant))

    # Rename the output band to 'NDVI' for clarity
    ndvi = ndvi.rename('NDVI')

    return ndvi

target_dict = get_coordinates_felix(params.POLYGON, get_target_image(params.DATA_DATE))
    # Loop over each year from year
for point in target_dict:
    # Get the picked point and create a 500m x 500m square around it
    picked_point = ee.Geometry.Point(target_dict[point])
    square = picked_point.buffer(10 * 50 / 2, 1).bounds(1)

    # Define image collection features
    image_collection_features = get_input_image(params.DATA_DATE,params.FEATURE_BANDS, square, "collection")

    # Get the first image
    image_features = get_input_image(params.DATA_DATE,params.FEATURE_BANDS, square, "image")
    NDVI = custom_ndvi(image_features)


    img_target = get_target_image(params.DATA_DATE)
    c_img_target = img_target.clip(square)

    #projection_f = image_features.select(0).projection().getInfo()
    #crs_f = projection_f['crs']
    #crs_f_transform = projection_f['transform']


    export_features = {
        'image': NDVI,
        'description': 'features_tf',
        'bucket': params.BUCKET,
        'fileNamePrefix': f'Features_nc/feature_image_{point}',
        'fileFormat': 'GeoTIFF',
        'dimensions': [50,50],  # Set the scale (e.g., 500 meters)
        #'region': square  # Set the export region
    }

    export_targets = {
            'image': c_img_target,
            'description': 'target_tf',
            'bucket': params.BUCKET,
            'fileNamePrefix': f'Targets_nc/target_image_{point}',  # Adjust the export file name
            'fileFormat': 'GeoTIFF', # Use the desired format
            'dimensions': [1,1],  # Set the scale (e.g., 500 meters)
            #'region': square
        }

    task_f = ee.batch.Export.image.toCloudStorage(**export_features)
    task_f.start()
    task_t = ee.batch.Export.image.toCloudStorage(**export_targets)
    task_t.start()

#picked_point = ee.Geometry.Point(next(iter(get_coordinates_felix(params.POLYGON,target).items()))[1])
#square = picked_point.buffer(250).bounds()

# Define export parameters


# Start the export task
