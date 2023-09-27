import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
#from deep_green_learning.data.data_functions_SM import get_data
import sys
import os
import ee
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow
from tensorflow import keras
from PIL import Image
#from tensorflow import predict
from folium.utilities import image_to_url
from st_files_connection import FilesConnection
import json
import pyproj
from pyproj import Transformer
from tensorflow.keras.models import load_model
from matplotlib.colors import LinearSegmentedColormap

# Get the path to the parent directory
#parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
parent_dir = os.path.abspath(os.path.dirname(__file__))

# Construct the path to the .h5 file
model_path = os.path.join(parent_dir, 'model_mvp.h5')

# Append the parent directory to the Python path
sys.path.append(parent_dir)

from data.data_functions_SM import get_all_data
#from modelling.model_functions import load_model
from utils import auth_ee

# Initialize coordinates variable at the top-level
map_coordinates = None

# ... rest of your code ...

#json_file_name = 'authentication_keys/semiotic_garden_key.json'
#conn = st.experimental_connection('gcs', type=FilesConnection)
#json_cred = conn.read(json_file_name, input_format="json", ttl=600)

#print(type(st.secrets)) MAYBE SHOW THIS TO NURIA!!!
#print(json.dumps(st.secrets))

# Write JSON file
#json_cred = json.dumps(st.secrets)
#print(json_cred)

# Set the title of your Streamlit app
st.title("Forest Detection App")

# Add introductory text
st.write(
    "Welcome to the Forest Detection App! This app allows you to select an area on the map "
    "and detect whether there is a forest present or not. You can also provide a date for "
    "the satellite image you'd like to analyze."
)

#@st.cache_data(persist=True)

def create_map(center):
    m = folium.Map(
        location=center,
        zoom_start=12,
        control_scale=True,
        tiles="OpenStreetMap",
        attr='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
    )
    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "poly": False,
            "circle": False,
            "polygon": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        }).add_to(m)
    return m

st.title("Folium Map with Rectangle")

c1, c2 = st.columns(2)

initial_center = [51.5328, -0.0769]
with c1:
    original_map = create_map(initial_center)
    output = st_folium(original_map)

with c2:
    info = output.get("all_drawings")
    if info:
        map_coordinates = info[0]["geometry"]["coordinates"][0]

        A1 = map_coordinates[0]
        A2 = map_coordinates[1]
        B1 = map_coordinates[2]
        B2 = map_coordinates[3]

        st.write("Coordinates of the rectangle")

        st.write(f"Bottom Left: {A1}")
        st.write(f"Top Left: {A2}")
        st.write(f"Top Right: {B1}")
        st.write(f"Bottom Right: {B2}")

    else:
        st.write("Please draw a rectangle on the map.")


# Add a date input for satellite image analysis
selected_date = st.date_input("Select a date for satellite image analysis")



# Add a button to initiate analysis
if st.button("Analyze"):
    if map_coordinates is not None:
        # Initialise the Earth Engine module.
        auth_ee(st.secrets['client_email'], st.secrets['private_key'])
        # Forest detection logic
        # Ensure coordinates are in the format expected by ee
        #e.g. ee.Geometry.Rectangle(minLng, minLat, maxLng, maxLat)/(xMin, yMin, xMax, yMax)
        ee_coordinates = [A1[0], A1[1], B1[0], B1[1]]
        feature_bands = ["B4", "B8"]
        #year = '2017'

        NDVI = get_all_data(ee_coordinates, selected_date.year, feature_bands)

    # assuming all arrays in NDVI_all have the same shape
        NDVI_array = np.stack(NDVI, axis=0)  # adjust axis as necessary
        #print(NDVI_array.shape)

        #print(NDVI.shape)
        #NDVI_expanded = np.expand_dims(NDVI, axis=0)
        #NDVI_expanded = np.expand_dims(NDVI_expanded, axis=-1)
        st.write(f"Analyzing satellite image for {selected_date.year}...")

        model = load_model(model_path)
        #print(NDVI_expanded.dtype)
        #print(NDVI_expanded.shape)
        #print(NDVI_expanded)
        #print(model.summary())

        y_pred = model.predict(NDVI_array)
        print(y_pred)
        print(y_pred.shape)

        # print(y_pred.dtype)
        # print(y_pred.shape)
        # print(y_pred)

        print(ee_coordinates)
        print(map_coordinates)

        # Get the UTM zone number for the center of the area of interest
        utm_zone = int((ee_coordinates[0] + 180) / 6) + 1

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
        coordinates_utm = project_latlon_to_utm(ee_coordinates[0], ee_coordinates[1]), \
                      project_latlon_to_utm(ee_coordinates[2], ee_coordinates[3])

        # Determine the dimensions of the specified area in meters
        width_m = abs(coordinates_utm[1][0] - coordinates_utm[0][0])
        height_m = abs(coordinates_utm[1][1] - coordinates_utm[0][1])

        print(width_m)
        print(height_m)

        # Reshape based on natural breaks for square shape
        total_patches = NDVI_array.shape[0]
        side_length = int(np.sqrt(total_patches))

        reshaped_NDVI = NDVI_array.reshape((side_length, side_length, 50, 50))

        # Stitch the images
        stitched_NDVI_rows = [np.concatenate(reshaped_NDVI[i, :, :, :], axis=1) for i in range(side_length)]
        stitched_NDVI = np.concatenate(stitched_NDVI_rows, axis=0)


        # # Determine the number of images required along the width and the height
        # images_width = int(np.ceil(width_m / (50 * 10)))
        # images_height = int(np.ceil(height_m / (50 * 10)))

        # # Reshape based on computed images_width and images_height
        # reshaped_NDVI = NDVI_array.reshape((images_height, images_width, 50, 50))

        # # Stitch the images
        # stitched_NDVI = np.concatenate(np.concatenate(reshaped_NDVI, axis=2), axis=1)

        # Custom colormap
        colors = [(1, 1, 1), (0, 0.5, 0)]  # White to deep green
        n_bins = 100
        cmap_name = 'white_to_green'
        deg_colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        # # Visualization
        # fig, ax = plt.subplots(figsize=(10, 10))
        # im = ax.imshow(stitched_NDVI, cmap=deg_colormap, vmin=-1, vmax=1)
        # ax.set_title(f'All NDVI Arrays Stitched Together for {selected_date.year}')
        # ax.set_aspect('equal', 'box')  # Make it square
        # fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
        # st.pyplot(fig)

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(stitched_NDVI, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(f'All NDVI Arrays Stitched Together for {selected_date.year}')
        ax.set_aspect('equal', 'box')  # Make it square
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
        st.pyplot(fig)

    else:
        st.write("Please draw a rectangle on the map.")


print('Got to end of code :)')
