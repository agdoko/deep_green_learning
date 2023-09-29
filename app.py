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
model_path = os.path.join(parent_dir, 'model.h5')

# Append the parent directory to the Python path
sys.path.append(parent_dir)

from data.data_functions_SM import get_all_data
from modelling.model_functions import baseline
from utils import auth_ee

# Initialize coordinates variable at the top-level
map_coordinates = None

# Set the title of your Streamlit app
st.title("Forest Detection App")

# Add introductory text
st.write(
    "Welcome to the Forest Detection App! This app allows you to select an area on the map "
    "and detect whether there is a forest present or not. You can also provide a date for "
    "the satellite image you'd like to analyze."
)

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

        # selected year from user
        NDVI = get_all_data(ee_coordinates, selected_date.year, feature_bands)

        # year prior to selected year
        NDVI_prior = get_all_data(ee_coordinates, str(int(selected_date.year) - 1), feature_bands)

        # assuming all arrays in NDVI_all have the same shape
        NDVI_array = np.stack(NDVI, axis=0)  # adjust axis as necessary
        print(NDVI_array)
        print(NDVI_array.shape)

        # baseline predictions current year
        NDVI_baseline = baseline(NDVI_array)
        print(NDVI_baseline)
        print(NDVI_baseline.shape)

        #prior year model prediction
        NDVI_prior_array = np.stack(NDVI_prior, axis=0)  # adjust axis as necessary

        # baseline predictions prior year
        NDVI_baseline_prior = baseline(NDVI_prior_array)
        print(NDVI_baseline_prior)
        print(NDVI_baseline_prior.shape)

        st.write(f"Analyzing satellite image for {selected_date.year}...")

        model = load_model(model_path)

        # prediction for current year
        y_pred = model.predict(np.expand_dims(NDVI_array, axis=-1))
        # prediction for year prior
        y_pred_prior = model.predict(np.expand_dims(NDVI_prior_array, axis=-1))
        print(y_pred)
        print(y_pred.shape)

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

        # Reshape based on natural breaks for square shape current year
        total_patches = NDVI_array.shape[0]
        side_length = 1

        while (side_length + 1) ** 2 <= total_patches:
            side_length += 1

        closest_square = side_length ** 2

        NDVI_array = NDVI_array[:closest_square, :, :]

        # reshape current year
        reshaped_NDVI = NDVI_array.reshape((side_length, side_length, 50, 50), order='F')
        reshaped_NDVI = np.flip(reshaped_NDVI, axis=0)
        print(reshaped_NDVI.shape)

        # Stitch the images
        stitched_NDVI_rows = [np.concatenate(reshaped_NDVI[i, :, :, :], axis=1) for i in range(side_length)]
        print(stitched_NDVI_rows[0].shape)
        #print(stitched_NDVI_rows.shape)
        stitched_NDVI = np.concatenate(stitched_NDVI_rows, axis=0)
        print(stitched_NDVI.shape)

        # Visualization of features
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(stitched_NDVI, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(f'All NDVI Arrays Stitched Together for {selected_date.year}')
        ax.set_aspect('equal', 'box')  # Make it square
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
        st.pyplot(fig)

        # Just like we truncate the features for visualisation, we need to do the same with the predictions
        y_pred = y_pred[:closest_square, :, :, :]
        y_pred_prior = y_pred[:closest_square, :, :, :]
        NDVI_baseline = np.expand_dims(NDVI_baseline[:closest_square, :, :], axis=-1)
        NDVI_baseline_prior = np.expand_dims(NDVI_baseline_prior[:closest_square, :, :], axis=-1)

        # Assuming y_pred and y_pred_prior have shape (num_patches,)
        # Create 50x50 blocks for each predicted value
        y_pred_blocks = np.array([np.full((50, 50), val) for val in y_pred])
        y_pred_prior_blocks = np.array([np.full((50, 50), val) for val in y_pred_prior])
        NDVI_baseline_blocks = np.array([np.full((50, 50), val) for val in NDVI_baseline])
        NDVI_baseline_prior_blocks = np.array([np.full((50, 50), val) for val in NDVI_baseline_prior])

        # Reshape and stitch the prediction blocks
        def reshape_and_stitch(y_blocks):
            reshaped = y_blocks.reshape((side_length, side_length, 50, 50), order='F')
            reshaped = np.flip(reshaped, axis=0)
            #reshaped[[0, -1], :] = reshaped[[-1, 0], :]
            stitched_rows = [np.concatenate(reshaped[i, :, :, :], axis=1) for i in range(side_length)]
            return np.concatenate(stitched_rows, axis=0)

        stitched_pred = reshape_and_stitch(y_pred_blocks)
        stitched_pred_prior = reshape_and_stitch(y_pred_prior_blocks)
        stitched_baseline = reshape_and_stitch(NDVI_baseline_blocks)
        stitched_baseline_prior = reshape_and_stitch(NDVI_baseline_prior_blocks)

        # Visualization for current year
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(stitched_pred, cmap='Greens', vmin=0, vmax=1)
        ax.set_title(f'Predictions for {selected_date.year}')
        ax.set_aspect('equal', 'box')
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
        st.pyplot(fig)

        # Visualization for prior year
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(stitched_pred_prior, cmap='Greens', vmin=0, vmax=1)
        ax.set_title(f'Predictions for {selected_date.year - 1}')
        ax.set_aspect('equal', 'box')
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
        st.pyplot(fig)

        # Baseline Visualization for current year
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(stitched_baseline['ndvi'].astype(float), cmap='Greens', vmin=0, vmax=1)
        ax.set_title(f'Baseline for {selected_date.year}')
        ax.set_aspect('equal', 'box')
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
        st.pyplot(fig)

        # Baseline Visualization for prior year
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(stitched_baseline_prior['ndvi'].astype(float), cmap='Greens', vmin=0, vmax=1)
        ax.set_title(f'Baseline for {selected_date.year - 1}')
        ax.set_aspect('equal', 'box')
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
        st.pyplot(fig)

    else:
        st.write("Please draw a rectangle on the map.")


print('Got to end of code :)')
