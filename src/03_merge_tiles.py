import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import re
import rasterio
from rasterio.transform import from_bounds
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Script: 03_merge_tiles.py
# Description: Unified script to stitch tiles into GeoTIFF and/or PNG mosaics.
# Supports multiple folders, georeferencing, and scaling for previews.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Global settings from .env
BASE_DIR = Path(os.getenv("BASE_DIR", "."))
CRS = os.getenv("CRS", "EPSG:25830")  # Coordinate Reference System

# AOI and Georeferencing
BBOX_CSV = BASE_DIR / "data" / "aoi" / "aoi_bbox.csv"

# Input folders and suffixes to process
# Format: (folder_path, suffix, output_base_name)
INPUT_BASE_DIR = BASE_DIR / "data" / "pnoa-segmentation"
LAYERS_TO_STITCH = [
    # (INPUT_BASE_DIR / "PNOA2016_09", "_mask", "PNOA2016_09_mask"),
    (INPUT_BASE_DIR / "PNOA2016_05", "_overlay", "PNOA2016_05_overlay"),
    # (INPUT_BASE_DIR / "PNOA2019_09", "_mask", "PNOA2019_09_mask"),
    (INPUT_BASE_DIR / "PNOA2019_05", "_overlay", "PNOA2019_05_overlay"),
    # (INPUT_BASE_DIR / "PNOA2022_09", "_mask", "PNOA2022_09_mask"),
    (INPUT_BASE_DIR / "PNOA2022_05", "_overlay", "PNOA2022_05_overlay"),
]

# Output settings
OUTPUT_GEOTIFF_DIR = BASE_DIR / "data" / "raster"
OUTPUT_PNG_DIR = INPUT_BASE_DIR 

# Export options: "geotiff", "png" (can include both)
EXPORT_TYPES = ["geotiff","png"]

# PNG specific settings
MAX_PNG_SIZE = 3400  # Maximum dimension (width or height) for the PNG export

# -----------------------------------------------------------------------------
# SCRIPT LOGIC
# -----------------------------------------------------------------------------

# Fix for PROJ_LIB conflict
try:
    import rasterio
    os.environ["PROJ_LIB"] = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
except ImportError:
    pass

def load_bbox(csv_path):
    """Reads the study area bounds from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"BBOX CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return {
        "minx": df.iloc[0]['minx'],
        "miny": df.iloc[0]['miny'],
        "maxx": df.iloc[0]['maxx'],
        "maxy": df.iloc[0]['maxy']
    }

def stitch_layer(folder, suffix, output_name, bbox):
    """Stitches tiles and exports them according to configuration."""
    if not folder.exists():
        print(f"Warning: Folder {folder} does not exist. Skipping.")
        return

    print(f"\nProcessing: {output_name} (Suffix: {suffix}) in {folder.name}")
    
    # Identify tile files
    files = [f for f in os.listdir(folder) if f.endswith(f"{suffix}.png")]
    if not files:
        print(f"No files with suffix '{suffix}' found in {folder}")
        return

    tiles_info = []
    max_i, max_j = 0, 0
    
    for f in files:
        # Expected pattern: tile_XX_YY_suffix.png
        match = re.search(r'tile_(\d+)_(\d+)', f)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            tiles_info.append((i, j, f))
            max_i, max_j = max(max_i, i), max(max_j, j)

    if not tiles_info:
        print("No valid tiles identified from files.")
        return

    # Load metadata from first tile
    first_tile_path = folder / tiles_info[0][2]
    first_tile = cv2.imread(str(first_tile_path), cv2.IMREAD_UNCHANGED)
    tile_h, tile_w = first_tile.shape[:2]
    channels = 1 if len(first_tile.shape) == 2 else first_tile.shape[2]
    dtype = first_tile.dtype
    
    full_width = (max_i + 1) * tile_w
    full_height = (max_j + 1) * tile_h
    print(f"  Grid: {max_i+1}x{max_j+1}, Total size: {full_width}x{full_height}")

    # Initialize full mosaic buffer
    buffer_shape = (full_height, full_width, channels) if channels > 1 else (full_height, full_width)
    full_img = np.zeros(buffer_shape, dtype=dtype)

    # Stitch tiles
    for i, j, f in tiles_info:
        tile_path = folder / f
        img_tile = cv2.imread(str(tile_path), cv2.IMREAD_UNCHANGED)
        
        # Mapping: j=0 is bottom in UTM/WMS, top in image coords
        row_start = (max_j - j) * tile_h
        col_start = i * tile_w
        full_img[row_start:row_start + tile_h, col_start:col_start + tile_w] = img_tile

    # Export GeoTIFF
    if "geotiff" in EXPORT_TYPES:
        OUTPUT_GEOTIFF_DIR.mkdir(parents=True, exist_ok=True)
        tif_path = OUTPUT_GEOTIFF_DIR / f"{output_name}.tif"
        
        transform = from_bounds(bbox['minx'], bbox['miny'], bbox['maxx'], bbox['maxy'], full_width, full_height)
        
        with rasterio.open(
            tif_path, 'w', driver='GTiff', height=full_height, width=full_width,
            count=channels, dtype=dtype, crs=CRS, transform=transform
        ) as dst:
            if channels > 1:
                dst.write(np.transpose(full_img, (2, 0, 1)))
            else:
                dst.write(full_img, 1)
        print(f"  [GeoTIFF] Saved: {tif_path}")

    # Export PNG
    if "png" in EXPORT_TYPES:
        OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
        png_path = OUTPUT_PNG_DIR / f"{output_name}.png"
        
        # Convert to RGB if needed (OpenCV is BGR or Grayscale)
        if channels == 1:
            img_to_save = full_img
        elif channels == 3:
            img_to_save = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
        elif channels == 4:
            img_to_save = cv2.cvtColor(full_img, cv2.COLOR_BGRA2RGBA)
        else:
            img_to_save = full_img

        # Rescale if exceeds max size
        h, w = full_img.shape[:2]
        scale = min(MAX_PNG_SIZE / w, MAX_PNG_SIZE / h)
        
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            # Use PIL for high quality resizing or CV2
            img_pil = Image.fromarray(img_to_save)
            img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_resized.save(png_path, optimize=True)
            print(f"  [PNG] Resized to {new_w}x{new_h} and saved: {png_path}")
        else:
            Image.fromarray(img_to_save).save(png_path, optimize=True)
            print(f"  [PNG] Saved: {png_path}")

def main():
    try:
        bbox = load_bbox(BBOX_CSV)
    except Exception as e:
        print(f"Error loading BBOX: {e}")
        return

    for folder_path, suffix, output_name in LAYERS_TO_STITCH:
        stitch_layer(folder_path, suffix, output_name, bbox)

if __name__ == "__main__":
    main()
