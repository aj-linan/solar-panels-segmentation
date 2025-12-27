# Solar Panel Segmentation & Temporal Analysis

This project provides a complete pipeline for the automatic detection of photovoltaic solar panels using high-resolution PNOA (Plan Nacional de Ortopotografía Aérea) orthophotos. It leverages Deep Learning (U-Net) to identify solar farms and enables spatio-temporal analysis of their expansion over time.

## Key Features

- **Automated Data Acquisition**: Download historical mosaics (2016, 2019, 2022) directly via IGN WMS services.
- **Deep Learning Inference**: Segmentation using a U-Net model with an EfficientNetB1 backbone.
- **Temporal Analysis**: Calculation of solar surface area (m²/ha) and growth rates across different periods.
- **Rigorous Validation**: Tools to compare predictions against manually annotated Ground Truth (IoU, Dice Score).
- **Advanced Visualizations**:
    - **Evolution Maps**: Color-coded expansion tracking.
    - **Error Diagnostics**: Spatial Confusion Matrix overlays (TP, FP, FN).
    - **SIG Ready**: Export of georeferenced GeoTIFF mosaics (EPSG:25830).

## Project Structure

```text
├── data/
│   ├── figures/           # Evolution maps, growth charts, and visual comparisons
│   ├── metrics/           # CSV reports (IoU scores, area statistics)
│   ├── models/            # Pre-trained U-Net weights (.h5)
│   ├── pnoa-historic/     # Raw imagery tiles (JPEG) and Ground Truth masks
│   └── pnoa-segmentation/ # Output: UNet binary masks filtered by threshold
├──src/                    # Pipeline Core Scripts
│   ├── 01_download_pnoa.py      # Automated WMS tile downloader
│   ├── 02_unet_inference.py     # Segmentation inference engine
│   ├── 03_calculate_solar_area.py # Solar surface & growth analysis
│   └── 04_merge_tiles.py        # Mosaicing and GeoTIFF export
├── utils/                 # Validation & Visual Toolbox
│   ├── generate_masks.py    # Tool: Convert XML (CVAT) → PNG Masks
│   ├── calculate_metrics.py # Tool: IoU & Dice Score Calculator
│   ├── calculate_area.py    # Tool: Ground Truth Area Statistics
│   ├── error_map.py         # Viz: Spatial Confusion Matrix (TP/FP/FN)
│   ├── evolution_map.py     # Viz: Temporal Expansion Map
│   └── comparison_grid.py   # Viz: Side-by-side Validation Grid
├── unet/                  # Reference Notebook: Model Training (unet_solar_panels.ipynb)
├── env.example            # Template for environment variables (copy to .env)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Setup & Configuration

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   Rename `env.example` to `.env` and configure:
   - `BASE_DIR`: Absolute path to the repository.
   - `CRS`: Coordinate system (default: `EPSG:25830`).
   - `WMS_URL`: IGN historical PNOA service URL.

## Workflow Execution

### 1. Data Processing
Run the numbered scripts in sequence:
- `python src/01_download_pnoa.py`: Fetch JPEG tiles for the study area.
- `python src/02_unet_inference.py`: Apply the model to generate binary masks.
- `python src/03_calculate_solar_area.py`: Generate growth summary CSVs and charts.

### 2. Validation Suite
If Ground Truth is available:
- `python utils/generate_masks.py`: Prepare GT from XML annotations.
- `python utils/calculate_metrics.py`: Quantify model performance.
- `python utils/error_map.py`: See where the model succeeds or fails spatially.

### 3. Spatial Synthesis
- `python utils/evolution_map.py`: Create the final expansion map.
- `python src/04_merge_tiles.py`: Generate georeferenced mosaics for GIS software.

## Data Sources

- **PNOA Imagery**: Provided by the Spanish [IGN](https://www.ign.es/wms/pnoa-historico). Access via WMS.
- **Solar Dataset (Kaggle)**: [PNOA 2022 Aerial Imagery - Photovoltaic Segmentation](https://www.kaggle.com/datasets/mrhendley/pnoa-2022-aerial-imagery-photovoltaic-segmentation).
- **Standard**: Follows [PV03 dataset](https://www.kaggle.com/datasets/salimhammadi07/solar-panel-detection-and-identification) guidelines for solar panel semantic segmentation.

## Tech Stack
- **Deep Learning**: TensorFlow, Keras.
- **Geospatial**: Rasterio, GeoPandas, Shapely.
- **Analysis**: OpenCV, Pandas, NumPy, Matplotlib.
