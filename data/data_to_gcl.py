import ee
import sys
from data_functions_fg import get_coordinates_felix, get_target_image, get_input_image
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
import params
from google.cloud import storage

storage_client = storage.Client()

bucket = storage_client.get_bucket(params.BUCKET)

blobs = bucket.list_blobs()

# Print the list of objects
for blob in blobs:
    #print(f'Object Name: {blob.name}')
    blob.delete()


# Initialize Earth Engine
ee.Initialize()

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

    img_target = get_target_image(params.DATA_DATE)
    c_img_target = img_target.clip(square)

    export_features = {
        'image': image_features,
        'description': 'export_feartures',
        'bucket': params.BUCKET,
        'fileNamePrefix': f'Features/feature_image_{point}',
        'dimensions': [50,50],  # Set the scale (e.g., 500 meters)
        #'region': square  # Set the export region
    }

    export_targets = {
        'image': c_img_target,
        'description': 'export_targets',
        'bucket': params.BUCKET,
        'fileNamePrefix': f'Targets/target_image_{point}',
        'dimensions': [1,1],  # Set the scale (e.g., 500 meters)
        #'region': square  # Set the export region
    }

    task_f = ee.batch.Export.image.toCloudStorage(**export_features)
    task_f.start()
    task_t = ee.batch.Export.image.toCloudStorage(**export_targets)
    task_t.start()

#picked_point = ee.Geometry.Point(next(iter(get_coordinates_felix(params.POLYGON,target).items()))[1])
#square = picked_point.buffer(250).bounds()

# Define export parameters


# Start the export task
