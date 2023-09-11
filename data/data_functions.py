# Required imports
import ee
import io
import requests
import numpy as np
from typing import Iterable

""" Defines the functions used to get the data for initial model training. """

# Remaps the target land classification from mutli-class to binary
def get_target_image(target) -> ee.Image:
    """ Buckets multi-class land cover classifications into 2 classes:
    1 = forest
    0 = non-forest """
    # Remap the ESA classifications into the Dynamic World classifications
    fromValues = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    toValues = [1, 1, 1, 1, 1, 0, 0, 0,0, 0, 0,0,0,0,0,0,0]
    return (
        target.first()
        .select("LC_Type1")
        .remap(fromValues, toValues)
        .rename("landcover")
        .unmask(0)  # fill missing values with 0 (water)
        .byte()  # 9 classifications fit into an unsinged 8-bit integer
    )

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
    labels_image = get_target_image(target)
    target_dict = {}
    numb = 1

    # Iterating through the global image to generate stratified sampling coordinates
    for point in sample_points(region, labels_image, points_per_class=2, scale=500):
        target_dict[f"P{numb}"] = point
        numb +=1

    return target_dict

# Extracting the coordinates

# Taking the coordinates and getting the features data for the target points
def get_data(target_dict, year, feature_bands, target):
    """ Get the feature and target data, both as ndarrays. """

    ### FEATURES ###

    # Initialize an empty list to hold the images and skipped points
    stacked_feature_list = []
    skipped_points = []

    # Debugging counter for featuress
    features_counter = 0

    # Loop over each year from year
    for point in target_dict:
        # Get the picked point and create a 500m x 500m square around it
        picked_point = ee.Geometry.Point(target_dict[point])
        square = picked_point.buffer(250).bounds()

        # Define image collection features
        image_collection_features = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterDate(f"{year}-1-1", f"{year+3}-12-31")
            .filterBounds(square)
            .select(feature_bands)
            .sort('system:time_start'))

        # Check size of the image collection
        count = image_collection_features.size().getInfo()
        if count == 0:
            print(f"Skipping point: {point}")
            skipped_points.append(point)
            continue

        # Get the first image
        image_features = image_collection_features.first()

        # Clip the gotten image to the 500m x 500m square
        c_img_features = image_features.clip(square)

        # Get the download url for the clipped image
        url = c_img_features.getDownloadUrl({
            'scale': 10, # Because the feature satellite images are 10m x 10m per pixel in resolution
            'format': 'NPY' # numpy
            })

        # Get the image as a numpy array
        image_array_features = requests.get(url)
        image_array_features = np.load(io.BytesIO(image_array_features.content))

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
    cropped_arrays_features = []

    for arr in stacked_feature_list:
        cropped_features = arr[:50, :50]
        cropped_arrays_features.append(cropped_features)

    feature_stacked_array = np.stack(cropped_arrays_features, axis=0)

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
        picked_point = ee.Geometry.Point(target_dict[point])
        square = picked_point.buffer(250).bounds()

        # Clip the gotten image to the 500m x 500m square
        c_img_target = target.clip(square)

        # Get the download url for the clipped image
        url = c_img_target.getDownloadUrl({
            'scale': 500, # Because the target satellite images are 500m x 500m per pixel in resolution
            'format': 'NPY' # numpy
            })

        # Get the image as a numpy array
        image_array_targets = requests.get(url)
        image_array_targets = np.load(io.BytesIO(image_array_targets.content))

        # Append the numpy array to the list
        stacked_target_list.append(image_array_targets)
        print(stacked_target_list[-1]) # Uncomment to see the numpy array
        print(f"Appending target {target_counter} with shape {image_array_targets.shape}") # Uncomment to see the shape of the numpy array
        target_counter += 1

    # Apply cropping to the numpy array to ensure consistent shape
    cropped_arrays_targets = []

    for arr in stacked_target_list:
        cropped_targets = arr[:3, :3]
        cropped_arrays_targets.append(cropped_targets)

    target_stacked_array = np.stack(cropped_arrays_targets, axis=0)

    ### TRAINING AND TEST DATASETS ###

    # Separating the feature and target arrays into training and test datasets using 80/20 split

    depth = target_stacked_array.shape[0]
    split_index = int(depth * 0.8)  # 80% for training

    # Train-test split
    train_target = target_stacked_array[:split_index]
    test_target = target_stacked_array[split_index:]

    train_feature = feature_stacked_array[:split_index]
    test_feature = feature_stacked_array[split_index:]

    return train_feature, train_target, test_feature, test_target
