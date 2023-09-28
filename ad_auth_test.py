# Testing that authentification was successful
from utils import auth_ee
import streamlit as st
from st_files_connection import FilesConnection

json_file_name = 'authentication_keys/semiotic_garden_key.json'
conn = st.experimental_connection('gcs', type=FilesConnection)
json_cred = conn.read(json_file_name, input_format="json", ttl=600)

if __name__ == '__main__':
    auth_ee(json_cred)
