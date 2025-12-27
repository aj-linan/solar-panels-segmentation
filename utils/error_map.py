import os
import cv2
import numpy as np
import re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Script: error_map.py
# Description: Generates a spatial error map (Confusion Matrix visualization)
#              comparing predictions (TP, FP, FN) against ground truth.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = Path(os.getenv("BASE_DIR", "."))

# Input Directories
PRED_DIR = BASE_DIR / "data" / "pnoa-segmentation" / "PNOA2022_09"
REF_DIR = BASE_DIR / "data" / "pnoa-historic" / "PNOA2022_masks"
TILES_DIR = BASE_DIR / "data" / "pnoa-historic" / "PNOA2022"

# Output configuration
OUTPUT_FULL_MAP = BASE_DIR / "data" / "figures" / "pnoa2022_error_map_09.png"

# Visualization settings
MAX_PNG_SIZE = 3400  # Max dimension for PNG export
# -----------------------------------------------------------------------------
# VISUALIZATION UTILS
# -----------------------------------------------------------------------------
def draw_legend(img):
    """
    Adds a visual legend to the bottom-right of the image.
    Colors (BGR): Green (TP), Red (FP), Blue (FN)
    """
    h, w = img.shape[:2]
    
    # Legend settings (Increased for better visibility)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 4
    line_type = cv2.LINE_AA
    
    # Legend box dimensions
    box_w = 700
    box_h = 280
    margin = 40
    
    # Top-left corner of the legend box (Top-Right position)
    x1 = w - box_w - margin
    y1 = margin
    
    # Draw background semi-transparent rectangle (localized to the box)
    y2 = y1 + box_h
    x2 = x1 + box_w
    
    sub_img = img[y1:y2, x1:x2]
    # Create a dark background for the box area
    black_rect = np.zeros_like(sub_img)
    # Blend: 70% black + 30% original content
    img[y1:y2, x1:x2] = cv2.addWeighted(black_rect, 0.7, sub_img, 0.3, 0)
    
    # Drawing items (Label, Color)
    items = [
        ("True Positive", (0, 255, 0)),
        ("False Positive", (0, 0, 255)),
        ("False Negative", (255, 0, 0))
    ]
    
    for idx, (label, color) in enumerate(items):
        pos_y = y1 + 75 + idx * 80
        # Draw color square
        cv2.rectangle(img, (x1 + 30, pos_y - 45), (x1 + 80, pos_y + 10), color, -1)
        # Draw white border for visibility
        cv2.rectangle(img, (x1 + 30, pos_y - 45), (x1 + 80, pos_y + 10), (250, 250, 250), 3)
        # Draw text
        cv2.putText(img, label, (x1 + 110, pos_y), font, font_scale, (255, 255, 255), thickness, line_type)
# -----------------------------------------------------------------------------
# CORE PROCESSING
# -----------------------------------------------------------------------------
def run_batch_error_map():
    """
    Processes all tiles, generates error overlays in memory, and stitches them into a full map.
    """
    print(f"Starting batch error map generation...")
    
    if not PRED_DIR.exists() or not REF_DIR.exists() or not TILES_DIR.exists():
        print("Error: One or more data directories missing.")
        return

    # 1. Gather all predicted masks
    pred_files = [f for f in os.listdir(PRED_DIR) if f.endswith("_mask.png")]
    
    max_i, max_j = 0, 0
    tile_h, tile_w = 0, 0

    processed_data = [] # List of (i, j, error_overlay)

    for filename in pred_files:
        # Match tile coordinates
        match = re.search(r'tile_(\d+)_(\d+)', filename)
        if not match: continue
        i, j = int(match.group(1)), int(match.group(2))
        
        # Paths
        pred_path = PRED_DIR / filename
        ref_path = REF_DIR / filename
        # Original tiles are .jpeg in PNOA2022
        tile_path = TILES_DIR / filename.replace("_mask.png", ".jpeg")
        
        if not ref_path.exists() or not tile_path.exists():
            continue
            
        # Load images
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        ref_mask = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        original_tile = cv2.imread(str(tile_path))
        
        if pred_mask is None or ref_mask is None or original_tile is None:
            continue
            
        if tile_h == 0:
            tile_h, tile_w = original_tile.shape[:2]
            
        # Boolean masks for error calculation
        p_bool = pred_mask > 0
        r_bool = ref_mask > 0
        
        # Calculate categories
        tp = np.logical_and(p_bool, r_bool)
        fp = np.logical_and(p_bool, np.logical_not(r_bool))
        fn = np.logical_and(np.logical_not(p_bool), r_bool)

        # Create a color mask (BGR for OpenCV)
        color_mask = np.zeros_like(original_tile)
        color_mask[tp] = [0, 255, 0]   # Green (Match)
        color_mask[fp] = [0, 0, 255]   # Red (Extra prediction)
        color_mask[fn] = [255, 0, 0]   # Blue (Missed panel)

        # Blend with 50% transparency
        combined_mask = np.logical_or(tp, np.logical_or(fp, fn))
        
        error_overlay = original_tile.copy()
        blended = cv2.addWeighted(original_tile, 0.5, color_mask, 0.5, 0)
        
        # Apply blended colors only over affected pixels
        error_overlay[combined_mask] = blended[combined_mask]

        processed_data.append((i, j, error_overlay))
        max_i = max(max_i, i)
        max_j = max(max_j, j)

    if not processed_data:
        print("No tiles were processed successfully.")
        return

    # 2. Stitching process
    full_width = (max_i + 1) * tile_w
    full_height = (max_j + 1) * tile_h
    
    print(f"Stitching tiles into a {full_width}x{full_height} mosaic...")
    full_img = np.zeros((full_height, full_width, 3), dtype=np.uint8)
    
    for i, j, overlay in processed_data:
        # Tile indexing: j-axis usually decreases from top to bottom
        row_start = (max_j - j) * tile_h
        col_start = i * tile_w
        full_img[row_start:row_start + tile_h, col_start:col_start + tile_w] = overlay

    # 3. Resizing for output
    h, w = full_img.shape[:2]
    scale = min(MAX_PNG_SIZE / w, MAX_PNG_SIZE / h)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"Resizing final map from {w}x{h} to {new_w}x{new_h}...")
        full_img = cv2.resize(full_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 4. Add Legend
    draw_legend(full_img)

    # Ensure output directory exists
    OUTPUT_FULL_MAP.parent.mkdir(parents=True, exist_ok=True)
    
    # Save image
    cv2.imwrite(str(OUTPUT_FULL_MAP), full_img)
    print(f"Successfully saved full error map to: {OUTPUT_FULL_MAP}")

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_batch_error_map()
