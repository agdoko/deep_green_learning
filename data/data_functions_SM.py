# Required imports
import ee
import pyproj
import io
import requests
import numpy as np
from typing import Iterable
import sys
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
#import params
# Import additional libraries
import math
from itertools import product


""" Defines the functions used to get the data for initial model training. """

def get_input_image(year: int, feature_bands, square, type):
    if type == "image":
        return (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")  # Sentinel-2 images
            .filterDate(f"{int(year)}-1-1", f"{int(year)+3}-12-31")  # filter by year
            .filterBounds(square)
            #.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))  # filter cloudy images
            #.map(mask_sentinel2_clouds)  # mask/hide cloudy pixels
            .select(feature_bands)  # select all bands starting with B
            #.median()  # median of all non-cloudy pixels
            #.unmask(default_value)  # default value for masked pixels
            #.float()  # convert to float32
            .sort('system:time_start')
            .first()
        )
    elif type =="collection":

        return (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")  # Sentinel-2 images
            .filterDate(f"{int(year)}-1-1", f"{int(year)+3}-12-31")  # filter by year
            .filterBounds(square)
            #.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))  # filter cloudy images
            #.map(mask_sentinel2_clouds)  # mask/hide cloudy pixels
            .select(feature_bands)  # select all bands starting with B
            #.median()  # median of all non-cloudy pixels
            #.unmask(default_value)  # default value for masked pixels
            #.float()  # convert to float32
            .sort('system:time_start')
        )

    else:
        raise ValueError("Invalid parameter value. Use 'image' or 'collection'.")

def get_input_image_mean(year: int, feature_bands, square, type):
    if type == "image":
        return (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")  # Sentinel-2 images
            .filterDate(f"{int(year)}-1-1", f"{int(year)}-12-31")  # filter by year
            .filterBounds(square)
            #.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))  # filter cloudy images
            #.map(mask_sentinel2_clouds)  # mask/hide cloudy pixels
            .select(feature_bands)  # select all bands starting with B
            #.median()  # median of all non-cloudy pixels
            #.unmask(default_value)  # default value for masked pixels
            #.float()  # convert to float32
            .sort('system:time_start')
            .median()
        )
    elif type =="collection":

        return (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")  # Sentinel-2 images
            .filterDate(f"{int(year)}-1-1", f"{int(year)+3}-12-31")  # filter by year
            .filterBounds(square)
            #.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))  # filter cloudy images
            #.map(mask_sentinel2_clouds)  # mask/hide cloudy pixels
            .select(feature_bands)  # select all bands starting with B
            #.median()  # median of all non-cloudy pixels
            #.unmask(default_value)  # default value for masked pixels
            #.float()  # convert to float32
            .sort('system:time_start')
        )

    else:
        raise ValueError("Invalid parameter value. Use 'image' or 'collection'.")

# Extracting the right patch from the images
def get_patch(
    image: ee.Image, patch_size: int) -> np.ndarray:

    # YOUR CONVERSION HERE
    url = image.getDownloadURL(
        {
            "dimensions": [patch_size, patch_size],
            "format": "NPY"
        }
    )
    return url

# Taking the coordinates and getting the features data for the target points
def get_data(coordinates, year, feature_bands):
    """ Get the feature and target data, both as ndarrays. """

    ### FEATURES ###

    # Initialize an empty list to hold the images and skipped points
    #stacked_feature_list = []
    #skipped_points = []

    # Debugging counter for featuress
    #features_counter = 0

    #TO DO WE NEED THIS TO MATCH COORDINATES 4x, ACCOUNT FOR THE ORDER
    # ee.Geometry.Rectangle([xMin, yMin, xMax, yMax])

    user_rectangle = ee.Geometry.Rectangle(coordinates)

    image_feature = get_input_image(year,feature_bands, user_rectangle, "image") # this is an ee.image object


        # Get the image as a numpy array
        # TO DO  IT WILL NO LONGER PASS PATCH SIZE 50, BUT INSTEAD coordinates variable 1
    url = get_patch(image_feature,50)
    response = requests.get(url)
    image_array_features= np.load(io.BytesIO(response.content), allow_pickle=True)

        # Creating the NDVI array - NDVI is an index used for detecting forest in the academic literature
        # Extract B4 (Red) and B8 (NIR)
    B4 = image_array_features['B4'].astype(float)
    B8 = image_array_features['B8'].astype(float)

        # Calculate NDVI - basically the normalised difference between Red and NIR bands
    NDVI = (B8 - B4) / (B8 + B4 + 1e-10)  # adding a small constant to avoid division by zero
    return NDVI

def get_all_data(coordinates, year, feature_bands):
   # Get the UTM zone number for the center of the area of interest
    utm_zone = int((coordinates[0] + 180) / 6) + 1

    # Define the projection transformation
    project_latlon_to_utm = pyproj.Transformer.from_crs(
        crs_from='epsg:4326',  # WGS 84
        crs_to=f'epsg:326{utm_zone}',  # UTM Zone
        always_xy=True  # x, y order; longitude, latitude
    ).transform
    project_utm_to_latlon = pyproj.Transformer.from_crs(
        crs_from=f'epsg:326{utm_zone}',  # UTM Zone
        crs_to='epsg:4326',  # WGS 84
        always_xy=True  # x, y order; longitude, latitude
    ).transform

    # Convert the geographic coordinates to UTM coordinates
    coordinates_utm = project_latlon_to_utm(coordinates[0], coordinates[1]), \
                      project_latlon_to_utm(coordinates[2], coordinates[3])

    # Determine the dimensions of the specified area in meters
    width_m = abs(coordinates_utm[1][0] - coordinates_utm[0][0])
    height_m = abs(coordinates_utm[1][1] - coordinates_utm[0][1])

    # Determine the number of images required along the width and the height
    images_width = math.ceil(width_m / (50 * 10))  # 50 pixels, 10m per pixel
    images_height = math.ceil(height_m / (50 * 10))  # 50 pixels, 10m per pixel

    # Split the specified area into grids
    grid_coordinates_utm = [
        (
            coordinates_utm[0][0] + i * (50 * 10),  # x-coordinate (in meters)
            coordinates_utm[0][1] + j * (50 * 10),  # y-coordinate (in meters)
            coordinates_utm[0][0] + (i + 1) * (50 * 10),  # x-coordinate (in meters)
            coordinates_utm[0][1] + (j + 1) * (50 * 10)  # y-coordinate (in meters)
        )
        for i, j in product(range(images_width), range(images_height))
    ]

    # Convert the UTM grid coordinates back to geographic coordinates
    grid_coordinates = [
        (
            project_utm_to_latlon(utm_coords[0], utm_coords[1]) +
            project_utm_to_latlon(utm_coords[2], utm_coords[3])
        )
        for utm_coords in grid_coordinates_utm
    ]
    #print(grid_coordinates)

    # Download the images and compute NDVI for each grid cell
    all_ndvi = []
    for grid_coord in grid_coordinates:
        print(grid_coord)
        user_rectangle = ee.Geometry.Rectangle(grid_coord)
        #print(user_rectangle.coordinates())
        image_feature = get_input_image_mean(year, feature_bands, user_rectangle, "image")  # assuming this function exists
        #print(image_feature.getInfo())
        # Get the image as a numpy array
        url = get_patch(image_feature, 50)  # assuming this function exists
        response = requests.get(url)
        image_array_features = np.load(io.BytesIO(response.content), allow_pickle=True)

        # Extract B4 (Red) and B8 (NIR)
        B4 = image_array_features['B4'].astype(float)
        B8 = image_array_features['B8'].astype(float)

        # Calculate NDVI
        NDVI = (B8 - B4) / (B8 + B4 + 1e-10)  # adding a small constant to avoid division by zero
        all_ndvi.append(NDVI)

    # Optionally, you may want to stitch the NDVI arrays together to create a single composite NDVI array
    # ...

    return all_ndvi  # or return the composite NDVI array

# ... Your existing code ...
