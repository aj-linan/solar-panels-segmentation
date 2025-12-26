import os
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import shape
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# =============================================================================
# Script: evolution_map.py
# Description: Generates a multi-temporal spatial evolution map of solar 
#              installations by classifying detections into logical expansion 
#              periods (2016-2019 and 2019-2022).
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = Path(os.getenv("BASE_DIR", "."))
CRS = os.getenv("CRS", "EPSG:25830")

# Input Directories (Threshold 0.9 as requested)
DIR_2019 = BASE_DIR / "data" / "pnoa-segmentation" / "PNOA2019_09"
DIR_2022 = BASE_DIR / "data" / "pnoa-segmentation" / "PNOA2022_09"
BBOX_PATH = BASE_DIR / "data" / "aoi" / "aoi_bbox.csv"

# Output configuration
OUTPUT_MAP = BASE_DIR / "data" / "figures" / "photovoltaic_evolution_map.png"

# Geographical constants
PIXEL_RES = 0.6
TILE_SIZE = 512
TILE_M = TILE_SIZE * PIXEL_RES  # 307.2 meters

# Visual Settings
COLORS = {
    '2016-2019': '#F39C12', # Intermediate Orange
    '2019-2022': '#C0392B'  # Intense Red
}

def vectorize_mask(mask, transform):
    """
    Converts a binary mask into a list of Shapely geometries using Rasterio.
    """
    shapes = features.shapes(mask, mask=(mask > 0), transform=transform)
    return [shape(geom) for geom, val in shapes]

def generate_evolution_data():
    """
    Processes all multi-temporal tiles and classifies them into temporal categories.
    """
    print("Loading AOI bounds...")
    bbox = pd.read_csv(BBOX_PATH).iloc[0]
    minx_final = bbox['minx']
    maxy_final = bbox['maxy'] # We use maxy because WMS tiles start from top-down usually? 
    # Wait, check 01_download_pnoa logic: 
    # tile_miny = miny_final + j * tile_m
    # So j=0 is bottom.
    miny_final = bbox['miny']

    all_2019 = []
    all_new_2022 = []

    print("Processing tiles and vectorizing expansion...")
    
    # We use 2022 as the base for search
    if not DIR_2022.exists():
        print(f"Error: Directory {DIR_2022} not found.")
        return None, None

    mask_files = [f for f in os.listdir(DIR_2022) if f.endswith("_mask.png")]
    
    for filename in mask_files:
        match = re.search(r'tile_(\d+)_(\d+)', filename)
        if not match: continue
        i, j = int(match.group(1)), int(match.group(2))
        
        # Calculate Tile Transform
        # X starts from minx_final
        # Y starts from miny_final + j*tile_m (base)
        # But for Rasterio transform from_origin, we need (minx, maxy)
        tile_minx = minx_final + i * TILE_M
        tile_maxy = miny_final + (j + 1) * TILE_M
        transform = rasterio.transform.from_origin(tile_minx, tile_maxy, PIXEL_RES, PIXEL_RES)

        # Load 2022 Mask
        path_2022 = DIR_2022 / filename
        mask_22 = cv2.imread(str(path_2022), cv2.IMREAD_GRAYSCALE)
        
        # Load 2019 Mask (if exists)
        path_2019 = DIR_2019 / filename
        if path_2019.exists():
            mask_19 = cv2.imread(str(path_2019), cv2.IMREAD_GRAYSCALE)
        else:
            mask_19 = np.zeros_like(mask_22)

        if mask_22 is None: continue

        # Logic for classes
        # Class 1: Presence in 2019
        c1_mask = mask_19
        # Class 2: New in 2022 (Existed in 22 but not in 19)
        c2_mask = cv2.bitwise_and(mask_22, cv2.bitwise_not(mask_19))

        # Vectorize
        if np.any(c1_mask > 0):
            all_2019.extend(vectorize_mask(c1_mask, transform))
        if np.any(c2_mask > 0):
            all_new_2022.extend(vectorize_mask(c2_mask, transform))

    # Create GeoDataFrames
    gdf_19 = gpd.GeoDataFrame({'geometry': all_2019}, crs=CRS)
    gdf_new = gpd.GeoDataFrame({'geometry': all_new_2022}, crs=CRS)
    
    return gdf_19, gdf_new

def load_base_map():
    """Reads and stitches 2022 imagery tiles and brightens them for better contrast."""
    print("Loading PNOA 2022 background imagery...")
    DIR_PNOA = BASE_DIR / "data" / "pnoa-historic" / "PNOA2022"
    
    tiles = [f for f in os.listdir(DIR_PNOA) if f.startswith("tile_") and f.endswith(".jpeg")]
    ids_i = [int(re.search(r'tile_(\d+)', f).group(1)) for f in tiles]
    ids_j = [int(re.search(r'tile_\d+_(\d+)', f).group(1)) for f in tiles]
    
    ni, nj = max(ids_i) + 1, max(ids_j) + 1
    full_img = np.zeros((nj * TILE_SIZE, ni * TILE_SIZE, 3), dtype=np.uint8)
    
    for f in tiles:
        match = re.search(r'tile_(\d+)_(\d+)', f)
        i, j = int(match.group(1)), int(match.group(2))
        img = cv2.imread(str(DIR_PNOA / f))
        if img is not None:
            y_pos = (nj - 1 - j) * TILE_SIZE
            x_pos = i * TILE_SIZE
            full_img[y_pos:y_pos+TILE_SIZE, x_pos:x_pos+TILE_SIZE] = img
            
    # Brighten the image (Scale up by 1.2 and add offset)
    bright_img = cv2.convertScaleAbs(full_img, alpha=1.2, beta=30)
    return bright_img

def plot_evolution_map(gdf_19, gdf_new):
    """
    Creates a refined cartographical output with high visibility and professional finish.
    """
    print("Generating cartographic output...")
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Plot Base Map
    base_img = load_base_map()
    bbox = pd.read_csv(BBOX_PATH).iloc[0]
    extent = [bbox['minx'], bbox['maxx'], bbox['miny'], bbox['maxy']]
    ax.imshow(base_img, extent=extent, origin='upper', alpha=0.9)
    
    # Plot layers
    if not gdf_19.empty:
        gdf_19.plot(ax=ax, color=COLORS['2016-2019'], label='Expansion 2016-2019', 
                    edgecolor='white', linewidth=0.5, alpha=0.75)
    if not gdf_new.empty:
        gdf_new.plot(ax=ax, color=COLORS['2019-2022'], label='Expansion 2019-2022', 
                    edgecolor='white', linewidth=0.5, alpha=0.75)

    # Formatting
    ax.set_title("Spatio-Temporal Evolution of Solar PV Installations", fontsize=20, fontweight='bold', pad=30)
    ax.set_xlabel("Easting (m) [ETRS89 / UTM 30N]", fontsize=12, labelpad=15)
    ax.set_ylabel("Northing (m)", fontsize=12, labelpad=15)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.4, color='white')

    # Professional Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['2016-2019'], 
               markersize=15, alpha=0.8, label='Expansion 2016-2019'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['2019-2022'], 
               markersize=15, alpha=0.8, label='Expansion 2019-2022')
    ]
    ax.legend(handles=legend_elements, loc='upper right', title="Temporal Classification", 
               frameon=True, shadow=True, facecolor='white', framealpha=0.95, fontsize=11, title_fontsize=12)

    # Scale Bar
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width_m = xlim[1] - xlim[0]
    scale_len = 1000
    
    sb_x = xlim[0] + (width_m * 0.05)
    sb_y = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot([sb_x, sb_x + scale_len], [sb_y, sb_y], color='black', lw=3)
    ax.plot([sb_x, sb_x], [sb_y-30, sb_y+30], color='black', lw=2)
    ax.plot([sb_x + scale_len/2, sb_x + scale_len/2], [sb_y-20, sb_y+20], color='black', lw=1.5)
    ax.plot([sb_x + scale_len, sb_x + scale_len], [sb_y-30, sb_y+30], color='black', lw=2)
    ax.text(sb_x + scale_len/2, sb_y + (ylim[1] - ylim[0]) * 0.015, 
            f"{scale_len} m", ha='center', fontsize=11, fontweight='bold')

    # Advanced Multi-part North Arrow (Classic Star Style)
    nx = xlim[1] - (width_m * 0.08)
    ny = ylim[1] - (ylim[1] - ylim[0]) * 0.12
    a_size = width_m * 0.04
    # Star points
    ax.fill([nx, nx + a_size/3, nx, nx - a_size/3], [ny, ny - a_size, ny - a_size*1.3, ny - a_size], 'black', alpha=0.8)
    ax.fill([nx, nx + a_size, nx, nx - a_size], [ny - a_size*0.6, ny - a_size*0.6, ny - a_size*0.6, ny - a_size*0.6], 'black', alpha=0.8) # Horizontal cross
    ax.text(nx, ny + a_size*0.2, "N", ha='center', va='bottom', fontsize=22, fontweight='heavy', 
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Footer/Citation with more spacing
    plt.figtext(0.5, 0.02, "Imagery: PNOA 2022 (IGN). Analysis: AI-based pixel segmentation (U-Net). Projection: ETRS89 / UTM zone 30N.", 
                ha='center', fontsize=10, style='italic', color='#444444')

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Tight layout but with manual bottom adjustment for footer
    plt.subplots_adjust(bottom=0.1)
    
    plt.savefig(OUTPUT_MAP, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Success! Final cartographic map saved to: {OUTPUT_MAP}")

def main():
    gdf_19, gdf_new = generate_evolution_data()
    if gdf_19 is not None:
        plot_evolution_map(gdf_19, gdf_new)

if __name__ == "__main__":
    main()
