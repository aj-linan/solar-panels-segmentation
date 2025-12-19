import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Script: 03_calculate_solar_area.py
# Description: Calculates solar panel surface area and temporal evolution metrics.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Global settings from .env
BASE_DIR = Path(os.getenv("BASE_DIR", "."))

# Data folders to analyze
DATA_DIR = BASE_DIR / "data" / "pnoa-segmentation"
TARGET_YEARS = ["PNOA2016_09", "PNOA2019_09", "PNOA2022_09"]
TARGET_FOLDERS = [DATA_DIR / year for year in TARGET_YEARS]

# Analysis settings
PIXEL_RESOLUTION = 0.6  # meters per pixel
FILE_SUFFIX = '_mask.png'

# Output settings
OUTPUT_PLOT_PATH = BASE_DIR / "output" / "solar_evolution.png"

# -----------------------------------------------------------------------------
# CORE FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_solar_area(folders, pixel_resolution=0.6):
    """
    Calculates total area of solar panels and total study area across multiple folders.
    """
    pixel_area_m2 = pixel_resolution ** 2
    results = []
    
    for folder in folders:
        total_white_pixels = 0
        total_pixels = 0
        if not folder.exists():
            print(f"Warning: Folder {folder} does not exist.")
            continue
            
        files = [f for f in os.listdir(folder) if f.endswith(FILE_SUFFIX)]
        print(f"Processing {len(files)} mask images in {folder.name}...")
        
        for file in files:
            file_path = folder / file
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                total_pixels += img.size
                _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                white_pixels = np.count_nonzero(binary == 255)
                total_white_pixels += white_pixels
            else:
                print(f"Error: Could not read image at {file_path}")
                
        area_m2 = total_white_pixels * pixel_area_m2
        area_ha = area_m2 / 10000
        study_area_ha = (total_pixels * pixel_area_m2) / 10000
        
        # Extract year from folder name
        year_str = folder.name.replace("PNOA", "")
        results.append({
            'year': int(year_str),
            'area_ha': area_ha,
            'study_area_ha': study_area_ha,
            'percentage_coverage': (total_white_pixels / total_pixels * 100) if total_pixels > 0 else 0
        })
    
    # Sort results by year
    results.sort(key=lambda x: x['year'])
    
    # Calculate Deltas and Growth Rates
    for i in range(len(results)):
        if i == 0:
            results[i]['delta_ha'] = 0.0
            results[i]['growth_rate_pct'] = 0.0
        else:
            prev_area = results[i-1]['area_ha']
            curr_area = results[i]['area_ha']
            delta = curr_area - prev_area
            results[i]['delta_ha'] = delta
            results[i]['growth_rate_pct'] = (delta / prev_area * 100) if prev_area > 0 else 0
            
    return results

def plot_evolution(results, output_path):
    """
    Generates a visual plot for area evolution and growth rate.
    """
    years = [str(r['year']) for r in results]
    areas = [r['area_ha'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for Total Area
    bars = ax1.bar(years, areas, color='skyblue', label='Total Area (ha)', alpha=0.7)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total Solar Area (ha)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    # Line chart for Growth Rate
    ax2 = ax1.twinx()
    growth_rates = [r['growth_rate_pct'] for r in results]
    ax2.plot(years[1:], growth_rates[1:], color='red', marker='o', linewidth=2, label='Growth Rate (%)')
    ax2.set_ylabel('Growth Rate (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Annotate growth rate values
    for i, rate in enumerate(growth_rates):
        if i > 0:
            ax2.annotate(f'{rate:.1f}%', (years[i], growth_rates[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center', color='red')

    plt.title('Evolution of Solar Photovoltaic Surface')
    fig.tight_layout()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\nEvolution plot saved to: {output_path}")

def main():
    results = calculate_solar_area(TARGET_FOLDERS, PIXEL_RESOLUTION)
    
    # Table Output
    header = f"{'Year':<6} | {'Area (ha)':<12} | {'Delta (ha)':<12} | {'Growth (%)':<12} | {'% Coverage':<12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    
    for r in results:
        delta_str = f"{r['delta_ha']:+12.2f}" if r['delta_ha'] != 0 else f"{'-':>12}"
        growth_str = f"{r['growth_rate_pct']:12.2f}%" if r['growth_rate_pct'] != 0 else f"{'-':>12}"
        print(f"{r['year']:<6} | {r['area_ha']:12.2f} | {delta_str} | {growth_str} | {r['percentage_coverage']:12.4f}%")
    
    print("=" * len(header))
    
    # Plot Generation
    try:
        plot_evolution(results, OUTPUT_PLOT_PATH)
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()
