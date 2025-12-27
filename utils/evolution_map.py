import os
import cv2
import numpy as np
import pandas as pd
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

# Input Directories
DIR_2019 = BASE_DIR / "data" / "pnoa-segmentation" / "PNOA2019_05"
DIR_2022 = BASE_DIR / "data" / "pnoa-segmentation" / "PNOA2022_05"
DIR_PNOA = BASE_DIR / "data" / "pnoa-historic" / "PNOA2022"

# Output configuration
OUTPUT_MAP = BASE_DIR / "data" / "figures" / "photovoltaic_evolution_map.png"

# Geographical constants
TILE_SIZE = 512

# Visualization settings
MAX_PNG_SIZE = 3400  # Max dimension for PNG export

# Visual Settings (BGR for OpenCV)
COLORS_BGR = {
    '2016-2019': (18, 156, 243), # Orange (#F39C12)
    '2019-2022': (43, 57, 192)   # Red (#C0392B)
}

# -----------------------------------------------------------------------------
# VISUALIZATION UTILS
# -----------------------------------------------------------------------------
def draw_legend(img):
    """
    Adds a visual legend to the top-right of the image.
    Colors (BGR): Orange (2016-2019), Red (2019-2022)
    """
    h, w = img.shape[:2]
    
    # Legend settings (Slightly reduced to fit box)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 5
    line_type = cv2.LINE_AA
    
    # Legend box dimensions
    box_w = 1000
    box_h = 320
    margin = 60
    
    # Top-right corner of the legend box
    x1 = w - box_w - margin
    y1 = margin
    
    # Draw background semi-transparent rectangle
    try:
        sub_img = img[y1:y1+box_h, x1:x1+box_w]
        black_rect = np.zeros_like(sub_img)
        img[y1:y1+box_h, x1:x1+box_w] = cv2.addWeighted(black_rect, 0.7, sub_img, 0.3, 0)
    except Exception as e:
        print(f"Warning: Could not draw legend background: {e}")
    
    # Drawing items
    items = [
        ("Expansion 2016 - 2019", COLORS_BGR['2016-2019']),
        ("Expansion 2019 - 2022", COLORS_BGR['2019-2022'])
    ]
    
    for idx, (label, color) in enumerate(items):
        pos_y = y1 + 110 + idx * 120
        # Draw color square (Larger)
        cv2.rectangle(img, (x1 + 40, pos_y - 70), (x1 + 120, pos_y + 10), color, -1)
        # Draw white border
        cv2.rectangle(img, (x1 + 40, pos_y - 70), (x1 + 120, pos_y + 10), (255, 255, 255), 4)
        # Draw text
        cv2.putText(img, label, (x1 + 150, pos_y), font, font_scale, (255, 255, 255), thickness, line_type)

# -----------------------------------------------------------------------------
# CORE PROCESSING
# -----------------------------------------------------------------------------
def generate_evolution_image():
    """
    Stitches images and masks, blends them, and saves the final result.
    """
    print("Loading PNOA 2022 background and masks...")
    
    if not DIR_PNOA.exists():
        print(f"Error: Background directory {DIR_PNOA} not found.")
        return

    # 1. Prepare grid
    tiles = [f for f in os.listdir(DIR_PNOA) if f.startswith("tile_") and f.endswith(".jpeg")]
    if not tiles:
        print("Error: No files found in TILES_DIR.")
        return

    ids_i = [int(re.search(r'tile_(\d+)', f).group(1)) for f in tiles]
    ids_j = [int(re.search(r'tile_\d+_(\d+)', f).group(1)) for f in tiles]
    
    ni, nj = max(ids_i) + 1, max(ids_j) + 1
    
    # Initialize large layers
    full_img = np.zeros((nj * TILE_SIZE, ni * TILE_SIZE, 3), dtype=np.uint8)
    mask_19_full = np.zeros((nj * TILE_SIZE, ni * TILE_SIZE), dtype=np.uint8)
    mask_22_new_full = np.zeros((nj * TILE_SIZE, ni * TILE_SIZE), dtype=np.uint8)

    print(f"Stitching layers for a {ni}x{nj} grid...")
    for f in tiles:
        match = re.search(r'tile_(\d+)_(\d+)', f)
        if not match: continue
        i, j = int(match.group(1)), int(match.group(2))
        
        y_pos = (nj - 1 - j) * TILE_SIZE
        x_pos = i * TILE_SIZE
        
        # Load background
        img = cv2.imread(str(DIR_PNOA / f))
        if img is not None:
            full_img[y_pos:y_pos+TILE_SIZE, x_pos:x_pos+TILE_SIZE] = img
            
        # Load masks
        mask_name = f.replace(".jpeg", "_mask.png")
        
        # Load 2022 mask
        path_22 = DIR_2022 / mask_name
        m22 = cv2.imread(str(path_22), cv2.IMREAD_GRAYSCALE) if path_22.exists() else None
        
        # Load 2019 mask
        path_19 = DIR_2019 / mask_name
        m19 = cv2.imread(str(path_19), cv2.IMREAD_GRAYSCALE) if path_19.exists() else None
        
        if m22 is not None:
            if m19 is None:
                m19 = np.zeros_like(m22)
            
            # Layer 1: Existed in 2019
            mask_19_full[y_pos:y_pos+TILE_SIZE, x_pos:x_pos+TILE_SIZE] = m19
            
            # Layer 2: New in 2022 (Present in 22, absent in 19)
            m22_new = cv2.bitwise_and(m22, cv2.bitwise_not(m19))
            mask_22_new_full[y_pos:y_pos+TILE_SIZE, x_pos:x_pos+TILE_SIZE] = m22_new

    # Brighten background
    full_img = cv2.convertScaleAbs(full_img, alpha=1.2, beta=15)
    
    # Create color overlays
    overlay_19 = np.zeros_like(full_img)
    overlay_19[mask_19_full > 0] = COLORS_BGR['2016-2019']
    
    overlay_22 = np.zeros_like(full_img)
    overlay_22[mask_22_new_full > 0] = COLORS_BGR['2019-2022']
    
    # Combine layers with alpha blending
    res_img = full_img.copy()
    
    # Blend Period 1
    mask_19_idx = mask_19_full > 0
    if np.any(mask_19_idx):
        alpha = 0.6
        res_img[mask_19_idx] = cv2.addWeighted(full_img, 1-alpha, overlay_19, alpha, 0)[mask_19_idx]
    
    # Blend Period 2
    mask_22_idx = mask_22_new_full > 0
    if np.any(mask_22_idx):
        alpha = 0.6
        res_img[mask_22_idx] = cv2.addWeighted(full_img, 1-alpha, overlay_22, alpha, 0)[mask_22_idx]

    # Resize before adding legend (to keep legend scale consistent with error map)
    h, w = res_img.shape[:2]
    scale = min(MAX_PNG_SIZE / w, MAX_PNG_SIZE / h)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"Resizing final map from {w}x{h} to {new_w}x{new_h}...")
        res_img = cv2.resize(res_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Draw Legend
    draw_legend(res_img)
    
    # Save Output
    OUTPUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_MAP), res_img)
    print(f"Success! Evolution image saved: {OUTPUT_MAP.name}")

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    generate_evolution_image()
