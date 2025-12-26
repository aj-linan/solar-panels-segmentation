import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Script: calculate_metrics.py
# Description: Compares predicted masks with ground truth masks to calculate
# IoU and Dice scores.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = Path(os.getenv("BASE_DIR", "."))

# Paths (Update these based on your actual data structure)
# Metrics (Update these based on your actual data structure)
# Predicted masks from UNet inference
PRED_DIR = BASE_DIR / "data" / "pnoa-segmentation" / "PNOA2022_05"
# Ground truth masks generated from XML annotations
GT_DIR = BASE_DIR / "data" / "pnoa-historic" / "PNOA2022_masks"

OUTPUT_CSV = BASE_DIR / "data" / "metrics" / "pnoa_2022_metrics_05.csv"

# Post-processing settings
MIN_AREA_THRESHOLD = 20 # Minimum pixels for a white blob to be kept

# -----------------------------------------------------------------------------
# METRIC FUNCTIONS
# -----------------------------------------------------------------------------
def apply_postprocessing(mask, min_area):
    """
    Filters out white pixel blobs smaller than min_area.
    """
    # Ensure binary 0/255
    binary = (mask > 0).astype(np.uint8) * 255
    
    # Use connected components to find blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create empty mask for result
    refined_mask = np.zeros_like(binary)
    
    # Iterate through each component (skip index 0 which is background)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            # Keep this blob
            refined_mask[labels == i] = 255
            
    return refined_mask

def calculate_metrics(y_true, y_pred):
    """
    Calculates IoU and Dice Score for binary masks.
    Expects numpy arrays with 0 and 255 (or 0 and 1).
    """
    # Ensure binary
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    
    # Avoid division by zero for IoU
    if union == 0:
        iou = 1.0 if y_true.sum() == 0 else 0.0
    else:
        iou = intersection / union

    # Dice = 2 * Intersection / (Sum of pixels in both)
    sum_pixels = y_true.sum() + y_pred.sum()
    if sum_pixels == 0:
        dice = 1.0 if y_true.sum() == 0 else 0.0
    else:
        dice = (2. * intersection) / sum_pixels

    return iou, dice

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def run_evaluation():
    print(f"Comparing predictions in: {PRED_DIR}")
    print(f"To ground truth in:      {GT_DIR}")

    if not PRED_DIR.exists():
        print(f"Error: Predicted masks directory not found: {PRED_DIR}")
        return
    if not GT_DIR.exists():
        print(f"Error: Ground truth directory not found: {GT_DIR}")
        return

    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Get list of files in predicted directory
    pred_files = [f for f in os.listdir(PRED_DIR) if f.endswith("_mask.png")]
    
    results = []
    
    for filename in pred_files:
        gt_path = GT_DIR / filename
        pred_path = PRED_DIR / filename

        if not gt_path.exists():
            print(f"Warning: Match not found for {filename} in ground truth directory.")
            continue

        # Load images
        img_gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        img_pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

        if img_gt is None or img_pred is None:
            print(f"Error reading {filename}. Skipping.")
            continue

        # Apply Post-processing (noise removal)
        img_pred_refined = apply_postprocessing(img_pred, MIN_AREA_THRESHOLD)

        iou, dice = calculate_metrics(img_gt, img_pred_refined)
        
        results.append({
            "filename": filename,
            "iou": iou,
            "dice": dice
        })

    if not results:
        print("No matches found to evaluate.")
        return

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate global means
    mean_iou = df["iou"].mean()
    mean_dice = df["dice"].mean()
    
    # Add summary row
    summary_row = {
        "filename": "TOTAL_MEAN",
        "iou": mean_iou,
        "dice": mean_dice
    }
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Print summary
    print("\n" + "="*30)
    print("GLOBAL METRICS SUMMARY")
    print("="*30)
    print(f"Processed files: {len(results)}")
    print(f"Mean IoU:       {mean_iou:.4f}")
    print(f"Mean Dice:      {mean_dice:.4f}")
    print("="*30)
    print(f"Details saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_evaluation()
