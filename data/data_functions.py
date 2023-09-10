# Required imports
import ee
from typing import Iterable

""" Defines the functions used to get the data for initial model training. """

# Definining the target image collection
def get_target_image(target) -> ee.Image:
    """ Buckets land cover classifications into 2 classes:
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

# Getting the coordinates for the target points
def get_coordinates(polygon, target):
    """ Returns a dictionary of coordinates for the target points. """
    # Defining the region of interest
    region = ee.Geometry.Polygon(polygon)

    # Getting the target image and creating a dictionary to store the coordinates
    labels_image = get_target_image(target)
    target_dict = {}
    numb = 1

    # Iterating through the global image to generate stratified sampling coordinates
    for point in sample_points(region, labels_image, points_per_class=5, scale=500):
        target_dict[f"P{numb}"] = point
        numb +=1

    return target_dict

# Taking the coordinates and getting the features data for the target points

