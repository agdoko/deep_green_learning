# Deep Green Learning 🌲

Detect deforestation from satellite imagery using deep learning. Draw a rectangle anywhere on Earth, and the app compares forest cover between two time periods using a CNN trained on Sentinel-2 data.

Built as a capstone project at **Le Wagon Data Science Bootcamp** (London).

## How It Works

1. **Draw** a rectangle on the interactive map
2. **Select** a date for analysis
3. **Compare** CNN predictions for that year vs. the previous year
4. **Visualize** NDVI and forest cover change

The app divides your selected region into 50×50 pixel patches (500m × 500m at 10m resolution), fetches Sentinel-2 imagery via Google Earth Engine, and runs inference on each patch.

## Technical Details

### Data Pipeline
- **Source**: Sentinel-2 imagery (`COPERNICUS/S2_HARMONIZED`) via Google Earth Engine API
- **Bands**: B4 (Red) + B8 (NIR) → NDVI calculation
- **Labels**: MODIS land cover data (`MODIS/061/MCD12Q1`)
- **Patch size**: 50×50 pixels at 10m resolution

### CNN Architecture
```
Input (50×50×1 NDVI)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(256) → Dense(128) → Dense(1, sigmoid)
    ↓
Output: Forest probability per patch
```

### Stack
- **ML**: TensorFlow/Keras, scikit-learn
- **Data**: Google Earth Engine API, NumPy
- **Frontend**: Streamlit + Folium (interactive maps)
- **Backend**: Flask + Gunicorn
- **Deployment**: Docker, GCP Cloud Run
- **Geo**: pyproj (WGS84 ↔ UTM transforms)

## Project Structure

```
├── app.py              # Streamlit application
├── main.py             # Flask API for Cloud Run
├── model.h5            # Trained CNN weights
├── data/               # Earth Engine data functions
│   ├── data_functions_SM.py
│   └── preprocess.py
├── modelling/          # Model training & inference
│   └── model_functions.py
├── notebooks/          # Exploratory analysis
├── Dockerfile          # Container config
└── requirements.txt    # Dependencies
```

## Quick Start

```bash
# Clone
git clone https://github.com/agdoko/deep_green_learning.git
cd deep_green_learning

# Install dependencies
pip install -r requirements.txt

# Set up Earth Engine authentication (requires GCP service account)
# Add credentials to .env or Streamlit secrets

# Run
streamlit run app.py
```

## Team

Built by a team of 4 at Le Wagon London.
