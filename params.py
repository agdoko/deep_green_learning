import os

MODEL_TARGET = os.environ.get("MODEL_TARGET")
BUCKET = os.environ.get("BUCKET_NAME")
DATA_DATE = "2017"
TARGET="MODIS/061/MCD12Q1"
FEATURES="COPERNICUS/S2_HARMONIZED"
FEATURE_BANDS = ["B4", "B8"]
POLYGON=[[[-145.7, 63.2], [-118.1, 22.3], [-78.2, 5.6], [-52.9, 47.6]]]

# Setting up Protocol Buffers Environment Variable if it's not already set
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"