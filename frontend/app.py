import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

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
    # Replace this section with your forest detection logic
    st.write(f"Analyzing satellite image for {selected_date}...")
    # You can add code here to analyze the selected area for the presence of a forest.
