import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from dotenv import load_dotenv
import re
from pathlib import Path

# Load environment variables
load_dotenv()

# =============================================================================
# Script: generate_masks.py
# Description: Parses XML annotations from CVAT and generates binary masks and
#              visual overlays for solar panel ground truth.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION & PATHS
# -----------------------------------------------------------------------------
BASE_DIR = Path(os.getenv("BASE_DIR", "."))

# Input/Output paths
XML_PATH = BASE_DIR / "data" / "aoi" / "pnoa_2022_aoi_annotations.xml"
MASKS_DIR = BASE_DIR / "data" / "pnoa-historic" / "PNOA_2022_masks"
TILES_DIR = BASE_DIR / "data" / "pnoa-historic" / "PNOA2022"
OVERLAYS_DIR = BASE_DIR / "data" / "pnoa-historic" / "PNOA_2022_overlays"
FULL_OVERLAY_PATH = BASE_DIR / "data" / "pnoa-historic" / "PNOA_2022_full_overlay.png"

# Visualization settings
MAX_PNG_SIZE = 3400
# -----------------------------------------------------------------------------
# PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------
def decode_rle_mask(rle_str, width, height):
    """Decodes CVAT RLE mask format."""
    counts = [int(x) for x in rle_str.split(',')]
    mask = np.zeros(width * height, dtype=np.uint8)
    
    current_pos = 0
    val = 0 # Start with background (0)
    for count in counts:
        mask[current_pos : current_pos + count] = val
        current_pos += count
        val = 255 if val == 0 else 0
        
    return mask.reshape((height, width))
# -----------------------------------------------------------------------------
# MAIN MASK GENERATION
# -----------------------------------------------------------------------------
def generate_masks():
    print(f"Reading annotations from: {XML_PATH}")
    if not os.path.exists(XML_PATH):
        print(f"Error: XML file not found at {XML_PATH}")
        return

    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(OVERLAYS_DIR, exist_ok=True)

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    count = 0
    for image in root.iter("image"):
        img_name = image.attrib["name"]
        width = int(image.attrib["width"])
        height = int(image.attrib["height"])

        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        has_annotations = False

        # --- POLYGONS ---
        for polygon in image.iter("polygon"):
            has_annotations = True
            points = polygon.attrib["points"]
            pts = []
            for p in points.split(";"):
                x, y = map(float, p.split(","))
                pts.append([int(round(x)), int(round(y))])
            pts = np.array([pts], dtype=np.int32)
            cv2.fillPoly(mask, pts, 255)

        # --- BOXES ---
        for box in image.iter("box"):
            has_annotations = True
            xtl = int(float(box.attrib["xtl"]))
            ytl = int(float(box.attrib["ytl"]))
            xbr = int(float(box.attrib["xbr"]))
            ybr = int(float(box.attrib["ybr"]))
            cv2.rectangle(mask, (xtl, ytl), (xbr, ybr), 255, -1)

        # --- RLE MASKS ---
        for rle_element in image.iter("mask"):
            has_annotations = True
            rle_str = rle_element.attrib["rle"]
            m_left = int(rle_element.attrib["left"])
            m_top = int(rle_element.attrib["top"])
            m_width = int(rle_element.attrib["width"])
            m_height = int(rle_element.attrib["height"])
            
            # Decode the small mask segment
            decoded_segment = decode_rle_mask(rle_str, m_width, m_height)
            
            # Paste the segment into the full mask
            # We use bitwise_or to combine with existing annotations
            y_end = min(m_top + m_height, height)
            x_end = min(m_left + m_width, width)
            
            # Adjust decoded_segment if it goes out of bounds (shouldn't happen in valid CVAT XML)
            segment_to_paste = decoded_segment[0 : y_end - m_top, 0 : x_end - m_left]
            
            mask[m_top:y_end, m_left:x_end] = cv2.bitwise_or(
                mask[m_top:y_end, m_left:x_end], 
                segment_to_paste
            )

        # Output filenames
        base_name = os.path.splitext(img_name)[0]
        mask_filename = f"{base_name}_mask.png"
        mask_path = os.path.join(MASKS_DIR, mask_filename)
        
        # Save Mask
        Image.fromarray(mask).save(mask_path)
        
        # --- OVERLAY GENERATION ---
        # Read original image
        # Assuming original image name matches tile name in TILES_DIR
        # Try both .jpeg and .png extensions just in case, or use the one from XML logic if possible
        # The XML 'name' usually has the extension, e.g., tile_00_00.jpeg
        tile_path = os.path.join(TILES_DIR, img_name)
        
        if os.path.exists(tile_path):
            img_bgr = cv2.imread(tile_path)
            if img_bgr is not None:
                # Create a green overlay: (B, G, R) = (0, 255, 0)
                # We only want to color the parts where mask == 255
                
                # Check shapes
                if img_bgr.shape[:2] != mask.shape:
                    img_bgr = cv2.resize(img_bgr, (width, height))
                
                # Create colored mask
                zeros = np.zeros_like(mask)
                # Green channel gets 255 where mask is 255
                green_mask = cv2.merge([zeros, mask, zeros]) 
                
                # Overlay
                # alpha 0.5
                overlay = cv2.addWeighted(img_bgr, 1, green_mask, 0.5, 0)
                
                # Save overlay
                overlay_filename = f"{base_name}_overlay.png"
                overlay_path = os.path.join(OVERLAYS_DIR, overlay_filename)
                cv2.imwrite(overlay_path, overlay)
            else:
                print(f"Warning: Could not read image at {tile_path}")
        else:
            print(f"Warning: Original image not found at {tile_path}")

        count += 1

    print(f"Finished! Generated {count} masks in: {MASKS_DIR}")
    print(f"Generating stitched overlay...")
    stitch_overlays()
# -----------------------------------------------------------------------------
# MOSAIC GENERATION
# -----------------------------------------------------------------------------
def stitch_overlays():
    """Stitches generated overlays into a single mosaic."""
    if not os.path.exists(OVERLAYS_DIR):
        print(f"Overlay directory {OVERLAYS_DIR} does not exist.")
        return

    files = [f for f in os.listdir(OVERLAYS_DIR) if f.endswith("_overlay.png")]
    if not files:
        print("No overlay files to stitch.")
        return

    # Parse grid
    tiles_info = []
    max_i, max_j = 0, 0
    
    for f in files:
        # Expected pattern: tile_XX_YY_overlay.png
        match = re.search(r'tile_(\d+)_(\d+)', f)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            tiles_info.append((i, j, f))
            max_i, max_j = max(max_i, i), max(max_j, j)

    if not tiles_info:
        print("No valid tiles identified.")
        return

    # Load first tile to get sizing
    first_tile_path = os.path.join(OVERLAYS_DIR, tiles_info[0][2])
    first_tile = cv2.imread(first_tile_path)
    if first_tile is None:
        print("Failed to read first tile.")
        return
        
    tile_h, tile_w, channels = first_tile.shape
    
    full_width = (max_i + 1) * tile_w
    full_height = (max_j + 1) * tile_h
    
    print(f"Stitching grid: {max_i+1}x{max_j+1}, Total size: {full_width}x{full_height}")
    
    # Create empty mosaic (black background)
    full_img = np.zeros((full_height, full_width, channels), dtype=np.uint8)

    for i, j, f in tiles_info:
        tile_path = os.path.join(OVERLAYS_DIR, f)
        img_tile = cv2.imread(tile_path)
        
        if img_tile is None:
            continue
            
        # Place in grid
        # In this dataset, logic usually is simple grid placement. 
        # Check merge_tiles.py: row_start = (max_j - j) * tile_h => This implies j=0 is bottom
        # Let's double check if we can assume standard logical order or if we need that inversion.
        # User said "Generate and do the same as merge_tiles".
        # merge_tiles.py has: row_start = (max_j - j) * tile_h
        # So we should respect that coordinate system (Cartesian-ish where Y grows up?)
        # BUT usually image tiles named _00_00 starting from top-left.
        # Let's look at `04_merge_tiles.py` again.
        # "Mapping: j=0 is bottom in UTM/WMS, top in image coords"
        # "row_start = (max_j - j) * tile_h"
        # I will COPY that logic to be safe and consistent.
        
        row_start = (max_j - j) * tile_h
        col_start = i * tile_w
        
        # Ensure sizes match (handle edge cases if any, though normally should be exact)
        h_t, w_t = img_tile.shape[:2]
        
        full_img[row_start:row_start + h_t, col_start:col_start + w_t] = img_tile

    # Resize if needed
    h, w = full_img.shape[:2]
    scale = min(MAX_PNG_SIZE / w, MAX_PNG_SIZE / h)
    
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"Resizing stitch from {w}x{h} to {new_w}x{new_h} (MAX_PNG_SIZE={MAX_PNG_SIZE})")
        
        # CV2 resize
        img_resized = cv2.resize(full_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(FULL_OVERLAY_PATH, img_resized)
    else:
        cv2.imwrite(FULL_OVERLAY_PATH, full_img)
        
    print(f"Saved stitched overlay to: {FULL_OVERLAY_PATH}")

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    generate_masks()
