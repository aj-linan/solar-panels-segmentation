import os
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Script: calculate_area.py
# Description: Calculates the real (Ground Truth) area of solar panels in 2022
#              based on the high-quality manual masks.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = Path(os.getenv("BASE_DIR", "."))
GT_DIR = BASE_DIR / "data" / "pnoa-historic" / "PNOA2022_masks"
PIXEL_RESOLUTION = 0.6  # meters per pixel (PNOA standard)

def calculate_real_area():
    """
    Scans the ground truth masks and calculates the cumulative solar area.
    """
    if not GT_DIR.exists():
        print(f"Error: Ground truth directory not found at: {GT_DIR}")
        return

    print(f"Reading Ground Truth masks from: {GT_DIR.name}...")
    
    pixel_area_m2 = PIXEL_RESOLUTION ** 2
    total_white_pixels = 0
    mask_files = [f for f in os.listdir(GT_DIR) if f.endswith("_mask.png")]
    
    if not mask_files:
        print("No masks found in the directory. Please check the GT_DIR path.")
        return

    for filename in mask_files:
        path = GT_DIR / filename
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Count pixels with value > 127 (white)
            white_pixels = np.count_nonzero(img > 127)
            total_white_pixels += white_pixels
            
    # Unit conversions
    area_m2 = total_white_pixels * pixel_area_m2
    area_ha = area_m2 / 10000
    
    # Professional Output
    print("\n" + "="*45)
    print("      REAL SOLAR PANEL AREA (GT 2022)")
    print("="*45)
    print(f"Total tiles analysed:   {len(mask_files)}")
    print(f"Total Solar Area (m2):  {area_m2:,.2f} m²")
    print(f"Total Solar Area (ha):  {area_ha:.4f} ha")
    print("="*45)
    print(f"Note: Based on PIXEL_RES = {PIXEL_RESOLUTION}m")

if __name__ == "__main__":
    calculate_real_area()
