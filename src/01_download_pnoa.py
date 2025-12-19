import geopandas as gpd
from owslib.wms import WebMapService
from pyproj import Transformer
from pathlib import Path
import math
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Script: 01_download_pnoa.py
# Description: Downloads historical PNOA orthophotos via WMS in tiled format.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Global settings from .env
BASE_DIR = Path(os.getenv("BASE_DIR", "."))
WMS_URL = os.getenv("WMS_URL", "https://www.ign.es/wms/pnoa-historico")
CRS_DEF = os.getenv("CRS", "EPSG:25830")

# Download specific settings
LAYER_NAME = "PNOA2016"
FORMAT = "image/jpeg"
TILE_SIZE = 512
RESOLUTION = 0.6

# Input/Output paths
SHAPEFILE_PATH = BASE_DIR / "data" / "aoi" / "aoi.shp"
OUTPUT_DIR = BASE_DIR / "data" / "pnoa-historic"
OUTPUT_CSV = BASE_DIR / "data" / "aoi" / "aoi_bbox.csv"

# AOI selection
USE_SHAPEFILE = True
BBOX_MANUAL = (-5.73, 37.19, -5.67, 37.23)

def download_pnoa():
    # -----------------------------------------------------------------------------
    # AOI BOUNDS CALCULATION
    # -----------------------------------------------------------------------------
    if USE_SHAPEFILE:
        gdf = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
        minx_aoi, miny_aoi, maxx_aoi, maxy_aoi = gdf.total_bounds
    else:
        minx_aoi, miny_aoi, maxx_aoi, maxy_aoi = BBOX_MANUAL

    # Transform AOI to target CRS
    transformer = Transformer.from_crs("EPSG:4326", CRS_DEF, always_xy=True)
    minx_dest, miny_dest = transformer.transform(minx_aoi, miny_aoi)
    maxx_dest, maxy_dest = transformer.transform(maxx_aoi, maxy_aoi)

    # -----------------------------------------------------------------------------
    # TILE GRID CALCULATION
    # -----------------------------------------------------------------------------
    width_m = maxx_dest - minx_dest
    height_m = maxy_dest - miny_dest
    tile_m = TILE_SIZE * RESOLUTION

    n_tiles_x = math.ceil(width_m / tile_m)
    n_tiles_y = math.ceil(height_m / tile_m)

    # Center the grid
    total_width = n_tiles_x * tile_m
    total_height = n_tiles_y * tile_m
    center_x = (minx_dest + maxx_dest) / 2
    center_y = (miny_dest + maxy_dest) / 2
    
    minx_final = center_x - total_width/2
    miny_final = center_y - total_height/2
    maxx_final = minx_final + total_width
    maxy_final = miny_final + total_height

    # -----------------------------------------------------------------------------
    # SAVE AOI BBOX
    # -----------------------------------------------------------------------------
    bbox_info = {
        "minx": [minx_final],
        "miny": [miny_final],
        "maxx": [maxx_final],
        "maxy": [maxy_final]
    }
    output_csv_path = Path(OUTPUT_CSV)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bbox_info).to_csv(output_csv_path, index=False)
    print(f"AOI BBOX saved to {OUTPUT_CSV}")

    # -----------------------------------------------------------------------------
    # DOWNLOAD JPEG TILES
    # -----------------------------------------------------------------------------
    layer_output_dir = Path(OUTPUT_DIR) / LAYER_NAME
    layer_output_dir.mkdir(parents=True, exist_ok=True)

    wms_service = WebMapService(WMS_URL, version="1.3.0", timeout=120)

    for i in range(n_tiles_x):
        for j in range(n_tiles_y):
            tile_minx = minx_final + i * tile_m
            tile_miny = miny_final + j * tile_m
            tile_maxx = tile_minx + tile_m
            tile_maxy = tile_miny + tile_m

            tile_bbox = (tile_minx, tile_miny, tile_maxx, tile_maxy)
            tile_file_path = layer_output_dir / f"tile_{i:02d}_{j:02d}.jpeg"

            img = wms_service.getmap(
                layers=[LAYER_NAME],
                styles=["default"],
                srs=CRS_DEF,
                bbox=tile_bbox,
                size=(TILE_SIZE, TILE_SIZE),
                format=FORMAT,
                timeout=120
            )

            with open(tile_file_path, "wb") as f:
                f.write(img.read())

    print(f"Process finished. All tiles for {LAYER_NAME} downloaded.")

def main():
    download_pnoa()

if __name__ == "__main__":
    main()
