# Required imports
import ee
from typing import Iterable

""" # Authentication test
ee.Initialize()
print(ee.Image("NASA/NASADEM_HGT/001").get("title").getInfo())
 """

# Initialise the Earth Engine module.
ee.Initialize()

# Defining the main year around which data will be collected
f_date = '2017'

# Defining the target ImageCollection, filtered by the main year
target = (ee.ImageCollection("MODIS/061/MCD12Q1")
          .filterDate(f_date)
          .sort('system:time_start'))  # Sort by time to get earliest image

# Definining the target image collection
def get_target_image() -> ee.Image:
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

# Oversimplified North America region.
#polygon = [[[-145.7, 63.2], [-118.1, 22.3], [-78.2, 5.6], [-52.9, 47.6]]]

# Global polygon, while minimising the amount of water
polygon = [[[-180, -60], [180, -60], [180, 85], [-180, 85], [-180, -60]]]

def get_coordinates(polygon):
    """ Returns a dictionary of coordinates for the target points. """
    # Defining the region of interest
    region = ee.Geometry.Polygon(polygon)

    # Getting the target image and creating a dictionary to store the coordinates
    labels_image = get_target_image()
    target_dict = {}
    numb = 1

    # Iterating through the global image to generate stratified sampling coordinates
    for point in sample_points(region, labels_image, points_per_class=5, scale=500):
        target_dict[f"P{numb}"] = point
        numb +=1

    # Printing the target dictionary at end to check script ran successfully
    print(target_dict)




# Running the function get_coordinates to test the script
get_coordinates(polygon)
