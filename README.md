# Solar Panel Segmentation & Temporal Analysis (TFM GIS)

This project implements a methodology for the automatic detection of photovoltaic solar panels using historical PNOA (Plan Nacional de Ortopotografía Aérea) orthophotos and convolutional neural networks (U-Net). It also enables temporal analysis of the installed surface area within a specific study area.

## Features

- **Automated Download**: Fetch historical mosaics via WMS (IGN) for different years.
- **Deep Learning Segmentation**: Inference using a U-Net model (EfficientNetB1) to segment solar panels.
- **Spatial Analysis**: Calculation of surface area in hectares, growth deltas, and temporal evolution rates.
- **Geospatial Visualization**: Results exported in georeferenced GeoTIFF format (EPSG:25830) and PNG previews.
- **Centralized Configuration**: Global paths and parameters managed via environment variables.

## Project Structure

```text
tfm_gis/
├── data/                  # Input data, models, and intermediate results
├── output/                # Final mosaics, GeoTIFFs, and evolution plots
├── src/                   # Processing scripts
│   ├── 01_download_pnoa.py      # Step 1: WMS tiles download
│   ├── 02_unet_inference.py     # Step 2: Segmentation with U-Net
│   ├── 03_calculate_solar_area.py # Step 3: Area and evolution analysis
│   └── 04_merge_tiles.py        # Step 4: Mosaic and GeoTIFF creation
├── unet/                  # Model training
│   └── unet_solar_panels.ipynb  # Notebook with the training process
├── .env                   # Environment variables (private)
├── .env.example           # Environment variables template
├── requirements.txt       # Project dependencies
└── README.md              # Documentation
```

## Installation and Setup

1. **Clone the repository** and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the environment**:
   Copy the `.env.example` file to `.env` and adjust the paths for your system:
   ```bash
   # Edit .env
   BASE_DIR=C:\Your\Path\To\Project\tfm_gis
   CRS=EPSG:25830
   WMS_URL=https://www.ign.es/wms/pnoa-historico
   ```

## Workflow

Scripts are numbered to follow the logical execution order:

1.  **Download**: Run `01_download_pnoa.py` to download aerial images of the area of interest (AOI).
2.  **Segmentation**: Run `02_unet_inference.py` to generate detection masks for the downloaded images.
3.  **Analysis**: Run `03_calculate_solar_area.py` to obtain the temporal evolution table and the growth plot.
4.  **Export**: Run `04_merge_tiles.py` to merge tiles into a single GeoTIFF compatible with QGIS/ArcGIS.

## Technologies Used

- **Deep Learning**: TensorFlow, Keras, Segmentation Models.
- **GIS & Remote Sensing**: Rasterio, GeoPandas, PyProj, OWSLib.
- **Image Processing**: OpenCV, Pillow (PIL).
- **Data Analysis**: Pandas, NumPy, Matplotlib.
- **Model Training**: Jupyter Notebooks.
