import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
#from deep_green_learning.data.data_functions_SM import get_data
import sys
import os
import ee
import numpy as np
import tensorflow
from tensorflow import keras
#from tensorflow import predict



# Get the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Construct the path to the .h5 file
model_path = os.path.join(parent_dir, 'model.h5')

# Append the parent directory to the Python path
sys.path.append(parent_dir)

from data.data_functions_SM import get_data
from modelling.model_functions import load_model


# ... rest of your code ...


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
    output = st_folium(create_map(initial_center))

with c2:
    info = output.get("all_drawings")
    if info:
        coordinates = info[0]["geometry"]["coordinates"][0]

        A1 = coordinates[0]
        A2 = coordinates[1]
        B1 = coordinates[2]
        B2 = coordinates[3]

        st.write("Coordinates of the rectangle")

        st.write(f"Top Left: {A1}")
        st.write(f"Top Right: {A2}")
        st.write(f"Bottom Left: {B1}")
        st.write(f"Bottom Right: {B2}")


# Add a date input for satellite image analysis
selected_date = st.date_input("Select a date for satellite image analysis")



# Add a button to initiate analysis
if st.button("Analyze"):
    # Initialise the Earth Engine module.
    ee.Initialize()
    # Forest detection logic
    # Ensure coordinates are in the format expected by ee
    coordinates = [A1[0], A1[1], B1[0], B1[1]]
    feature_bands = ["B4", "B8"]
    year = '2017'

    NDVI = get_data(coordinates, selected_date.year, feature_bands)
    print(NDVI.shape)
    NDVI_expanded = np.expand_dims(NDVI, axis=0)
    NDVI_expanded = np.expand_dims(NDVI_expanded, axis=-1)
    st.write(f"Analyzing satellite image for {selected_date.year}...")

    model = load_model(model_path)
    print(NDVI_expanded.shape)
    #print(model.summary())

    y_pred = model.predict(NDVI_expanded)
    print(y_pred)
    print(y_pred.shape)
    # You can add code here to analyze the selected area for the presence of a forest.
