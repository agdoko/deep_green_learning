# Required imports
import ee
import os
import io
import requests
import numpy as np
from typing import Iterable
import sys
from sklearn.model_selection import train_test_split
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')


# Get the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the parent directory
parent_dir = os.path.join(current_dir, '..')

# Convert the relative path to an absolute path
parent_dir = os.path.abspath(parent_dir)

# Append the parent directory to the Python path
sys.path.append(parent_dir)
""" Defines the functions used to get the data for initial model training. """

import params as params

# Remaps the target land classification from mutli-class to binary
def get_target_image(year) -> ee.Image:
    """ Buckets multi-class land cover classifications into 2 classes:
    1 = forest
    0 = non-forest """
    # Remap the ESA classifications into the Dynamic World classifications
    fromValues = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    toValues = [1, 1, 1, 1, 1, 0, 0, 0,0, 0, 0,0,0,0,0,0,0]
    return (
        ee.ImageCollection(params.TARGET)
        .filterDate(year)
        .sort('system:time_start')  # Sort by time to get earliest image
        .first()
        .select("LC_Type1")
        .remap(fromValues, toValues)
        .rename("landcover")
        .unmask(0)  # fill missing values with 0 (water)
        .byte()  # 9 classifications fit into an unsinged 8-bit integer
    )

def get_input_image(year: int, feature_bands, square, type):
    if type == "image":
        return (
            ee.ImageCollection(params.FEATURES)  # Sentinel-2 images
            .filterDate(f"{int(year)}-1-1", f"{int(year)+3}-12-31")  # filter by year
            .filterBounds(square)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))  # filter cloudy images
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
            ee.ImageCollection(params.FEATURES)  # Sentinel-2 images
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


# Getting the target sample points, equally stratified across both classes
def sample_points(
    region: ee.Geometry, image: ee.Image, points_per_class: int, scale: int
) -> Iterable[tuple[float, float]]:
    """ Applies the stratified sampling algorithm to the given target image."""
    points = image.stratifiedSample(
        points_per_class,
        region=region,
        scale=scale,
        geometries=True,
    )
    for point in points.toList(points.size()).getInfo():
        yield point["geometry"]["coordinates"]

# OBSOLETE
# Getting the coordinates for the target points using Ana's random 100 approach
def get_coordinates(points):
    """ Returns a dictionary of square pixel coordinates for the target points. """
    target_dict = {}
    numb = 1

    # Iterating through the global image to generate stratified sampling coordinates
    for point in points:
        target_dict[f"P{numb}"] = list(point)
        numb +=1

    return target_dict

# TO DO - make this more robust. it works for about 80 points but can break past 100
# Getting the coordinates for the target points using Felix' stratified approach
def get_coordinates_felix(polygon, target):
    """ Returns a dictionary of square pixel coordinates for the target points. """
    # Defining the region of interest
    region = ee.Geometry.Polygon(polygon)

    # Getting the target image and creating a dictionary to store the coordinates
    target_dict = {}
    numb = 1

    # Iterating through the global image to generate stratified sampling coordinates
    for point in sample_points(region, target, points_per_class=200, scale=500):
        target_dict[f"P{numb}"] = point
        numb +=1

    return target_dict

# Extracting the right patch from the images

def get_patch(
    image: ee.Image, patch_size: int) -> np.ndarray:

    url = image.getDownloadURL(
        {

            "dimensions": [patch_size, patch_size],
            "format": "NPY"
        }
    )
    return url

# Taking the coordinates and getting the features data for the target points

def get_data(polygon, year, feature_bands):
    """ Get the feature and target data, both as ndarrays. """

    ### FEATURES ###

    # Initialize an empty list to hold the images and skipped points
    stacked_feature_list = []
    skipped_points = []

    # Debugging counter for featuress
    features_counter = 0
    target_dict = get_coordinates_felix(polygon, get_target_image(year))
    # Loop over each year from year
    for point in target_dict:
        # Get the picked point and create a 500m x 500m square around it
        picked_point = ee.Geometry.Point(target_dict[point])
        square = picked_point.buffer(10 * 50 / 2, 1).bounds(1)

        # Define image collection features
        image_collection_features = get_input_image(year,feature_bands, square, "collection")

        # Check size of the image collection
        count = image_collection_features.size().getInfo()
        if count == 0:
            print(f"Skipping point: {point}")
            skipped_points.append(point)
            continue

        # Get the first image
        image_features = get_input_image(year,feature_bands, square, "image")

        # Clip the gotten image to the 500m x 500m square
        #c_img_features = image_features.clip(square)


        # Get the image as a numpy array

        url = get_patch(image_features,50)
        response = requests.get(url)
        image_array_features= np.load(io.BytesIO(response.content), allow_pickle=True)

        # Creating the NDVI array - NDVI is an index used for detecting forest in the academic literature
        # Extract B4 (Red) and B8 (NIR)
        B4 = image_array_features['B4'].astype(float)
        B8 = image_array_features['B8'].astype(float)

        # Calculate NDVI - basically the normalised difference between Red and NIR bands
        NDVI = (B8 - B4) / (B8 + B4 + 1e-10)  # adding a small constant to avoid division by zero

        # Append the numpy array to the list
        stacked_feature_list.append(NDVI)
        print(stacked_feature_list[-1]) # Uncomment to see the numpy array
        print(f"Appending feature {features_counter} with shape {NDVI.shape}") # Uncomment to see the shape of the numpy array
        features_counter += 1

    # Apply cropping to the numpy array to ensure consistent shape
    #cropped_arrays_features = []

    #for arr in stacked_feature_list:
    #    cropped_features = arr[:50, :50]
    #    cropped_arrays_features.append(cropped_features)

    feature_stacked_array = np.stack(stacked_feature_list, axis=0)

    ### TARGETS ###

    # Account for the fact that some points were skipped in feature dataset, and we must maintain matching target points that remain
    new_target_dict = {k: v for k, v in target_dict.items() if k not in skipped_points}

    # Initialize an empty list to hold the images and skipped points
    stacked_target_list = []

    # Debugging counter for targets
    target_counter = 0

    # Loop over each year from year
    for point in new_target_dict:
        # Get the picked point and create a 500m x 500m square around it
        picked_point = ee.Geometry.Point(new_target_dict[point])
        square = picked_point.buffer(500 * 1 / 2, 1).bounds(1)

        # Clip the gotten image to the 500m x 500m square

        img_target = get_target_image(year)
        c_img_target = img_target.clip(square)

        # Get the image as a numpy array
        url = get_patch(c_img_target,1)
        response = requests.get(url)
        image_array_targets= np.load(io.BytesIO(response.content), allow_pickle=True)

        # Append the numpy array to the list
        stacked_target_list.append(image_array_targets)
        print(stacked_target_list[-1]) # Uncomment to see the numpy array
        print(f"Appending target {target_counter} with shape {image_array_targets.shape}") # Uncomment to see the shape of the numpy array
        target_counter += 1

    # Apply cropping to the numpy array to ensure consistent shape
    #cropped_arrays_targets = []

    #for arr in stacked_target_list:
    #    cropped_targets = arr[:3, :3]
    #    cropped_arrays_targets.append(cropped_targets)

    target_stacked_array = np.stack(stacked_target_list, axis=0)
    target_stacked_array= target_stacked_array[:,0,0]

    print(feature_stacked_array.shape)
    print(target_stacked_array.shape)
    ### TRAINING AND TEST DATASETS ###

    train_feature, test_feature, train_target, test_target = train_test_split(
    feature_stacked_array,
    target_stacked_array,
    test_size=0.2,  # 20% for testing
    stratify=target_stacked_array,
    random_state=42  # Set a random seed for reproducibility
    )

    print(train_feature.shape, train_target.shape, test_feature.shape, test_target.shape)
    # Separating the feature and target arrays into training and test datasets using 80/20 split



    print(train_feature.shape, train_target.shape, test_feature.shape, test_target.shape)
    return train_feature, train_target, test_feature, test_target
