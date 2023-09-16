import ee
import sys
from data_functions import get_coordinates_felix
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
import params
# Initialize Earth Engine
ee.Initialize()

# Define the Image Collection
target = ee.ImageCollection(params.TARGET) \
    .filterDate(params.DATA_DATE) \
    .sort('system:time_start')

picked_point = ee.Geometry.Point(next(iter(get_coordinates_felix(params.POLYGON,target).items()))[1])
square = picked_point.buffer(250).bounds()

# Define export parameters
export_config = {
    'image': target.first(),
    'description': 'test_export_target',
    'bucket': params.BUCKET,
    'fileNamePrefix': 'target_image_test',
    'scale': 500,  # Set the scale (e.g., 500 meters)
    'region': square  # Set the export region
}

# Start the export task
task = ee.batch.Export.image.toCloudStorage(**export_config)
task.start()
