import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Script: 03_calculate_solar_area.py
# Description: Calculates solar panel surface area and temporal evolution metrics.
#              Refactored to handle multiple thresholds and years.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = Path(os.getenv("BASE_DIR", "."))

# Data directories
SEG_DIR = BASE_DIR / "data" / "pnoa-segmentation"
YEARS = [2016, 2019, 2022]
THRESHOLDS = ["05", "07", "09"]

# Analysis settings
PIXEL_RESOLUTION = 0.6  # meters per pixel
FILE_SUFFIX = '_mask.png'

# Output paths
FIGURES_DIR = BASE_DIR / "data" / "figures"
METRICS_DIR = BASE_DIR / "data" / "metrics"

# -----------------------------------------------------------------------------
# CORE FUNCTIONS
# -----------------------------------------------------------------------------

def calculate_folder_area(folder, pixel_resolution=0.6):
    """
    Calculates the solar area and study area for a single folder.
    """
    pixel_area_m2 = pixel_resolution ** 2
    total_white_pixels = 0
    total_pixels = 0
    
    if not folder.exists():
        return None
        
    files = [f for f in os.listdir(folder) if f.endswith(FILE_SUFFIX)]
    if not files:
        return None

    for file in files:
        file_path = folder / file
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            total_pixels += img.size
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            white_pixels = np.count_nonzero(binary == 255)
            total_white_pixels += white_pixels
                
    area_m2 = total_white_pixels * pixel_area_m2
    area_ha = area_m2 / 10000
    study_area_ha = (total_pixels * pixel_area_m2) / 10000
    coverage_pct = (total_white_pixels / total_pixels * 100) if total_pixels > 0 else 0
    
    return {
        'total_area_ha': area_ha,
        'study_area_ha': study_area_ha,
        'coverage_pct': coverage_pct
    }

def process_all_data():
    """
    Iterates through all year/threshold combinations and aggregates results.
    """
    all_results = []
    
    for threshold in THRESHOLDS:
        print(f"\nProcessing Threshold: {threshold}")
        threshold_results = []
        
        for year in YEARS:
            folder_name = f"PNOA{year}_{threshold}"
            folder_path = SEG_DIR / folder_name
            
            print(f"  Analysing {folder_name}...")
            stats = calculate_folder_area(folder_path, PIXEL_RESOLUTION)
            
            if stats:
                threshold_results.append({
                    'year': year,
                    'threshold': f"{int(threshold)/10}",
                    'area_ha': stats['total_area_ha'],
                    'study_area_ha': stats['study_area_ha'],
                    'coverage_pct': stats['coverage_pct']
                })
        
        # Sort by year
        threshold_results.sort(key=lambda x: x['year'])
        
        # Calculate Deltas and Growth Rates within this threshold
        for i in range(len(threshold_results)):
            if i == 0:
                threshold_results[i]['delta_ha'] = 0.0
                threshold_results[i]['growth_rate_pct'] = 0.0
            else:
                prev_area = threshold_results[i-1]['area_ha']
                curr_area = threshold_results[i]['area_ha']
                delta = curr_area - prev_area
                threshold_results[i]['delta_ha'] = delta
                threshold_results[i]['growth_rate_pct'] = (delta / prev_area * 100) if prev_area > 0 else 0
        
        all_results.extend(threshold_results)
        
    return pd.DataFrame(all_results)

def plot_evolution_by_threshold(df, threshold, output_path):
    """
    Generates an evolution plot for a specific threshold.
    """
    data = df[df['threshold'] == f"{int(threshold)/10}"].sort_values('year')
    if data.empty: return

    years = data['year'].astype(str).tolist()
    areas = data['area_ha'].tolist()
    growth_rates = data['growth_rate_pct'].tolist()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for Total Area
    bars = ax1.bar(years, areas, color='skyblue', label='Total Area (ha)', alpha=0.7)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total Solar Area (ha)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Line chart for Growth Rate
    ax2 = ax1.twinx()
    ax2.plot(years, growth_rates, color='red', marker='o', linewidth=2, label='Growth Rate (%)')
    ax2.set_ylabel('Growth Rate (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Annotate growth rate values
    for i, rate in enumerate(growth_rates):
        if i > 0: # Skip 2016 (baseline)
            ax2.annotate(f'{rate:.1f}%', (years[i], growth_rates[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center', color='red', fontweight='bold')

    # plt.title(f'Solar Surface Evolution (Threshold 0.{threshold})', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    fig.tight_layout()
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  - Individual figure saved: {output_path.name}")

def plot_combined_evolution(df, output_path):
    """
    Generates a combined plot showing all thresholds in one figure using grouped bars.
    """
    if df.empty: return

    years = sorted(df['year'].unique())
    x = np.arange(len(years))
    width = 0.25 # width of bars

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    colors_area = ['#AED6F1', '#5DADE2', '#2E86C1'] # Different blues for thresholds
    colors_line = ['#FFC300', '#FF5733', '#C70039'] # Hot colors for growth rates (Reds/Yellows)
    
    threshold_labels = sorted(df['threshold'].unique())

    for i, thresh in enumerate(threshold_labels):
        thresh_data = df[df['threshold'] == thresh].sort_values('year')
        
        # Bars (Area)
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, thresh_data['area_ha'], width, 
                       label=f'Area (T={thresh})', color=colors_area[i], alpha=0.8, edgecolor='white')
        
        # Add labels on top of bars (Staggered vertically)
        for bar in bars:
            height = bar.get_height()
            # Vary vertical offset based on threshold index i to avoid collision
            v_offset = 0.5 + (i * 2.5) if height > 100 else 0.5 + (i * 1.5)
            ax1.text(bar.get_x() + bar.get_width()/2., height + v_offset,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1B4F72')

        # Lines (Growth Rate)
        # We now plot from the beginning (index 0) to show the full trend
        x_indices = [years.index(y) for y in thresh_data['year']]
        
        ax2.plot(x_indices, thresh_data['growth_rate_pct'], 
                 color=colors_line[i], marker='o', markersize=8, linewidth=2, 
                 label=f'Growth % (T={thresh})')

        # Add growth labels (Staggered to avoid collision)
        for idx, row in thresh_data.iterrows():
            if row['year'] == years[0]: continue # Skip baseline year label (2016)
            
            x_pos = years.index(row['year'])
            # Shift label position based on threshold index i
            y_offset = 12 + (i * 16)
            ax2.annotate(f"{row['growth_rate_pct']:.0f}%", 
                        (x_pos, row['growth_rate_pct']),
                        textcoords="offset points", xytext=(0, y_offset), ha='center', 
                        color=colors_line[i], fontweight='bold', fontsize=10)

    # Formatting Ax1 (Area)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Total Solar Area (ha)', fontsize=12, fontweight='bold', color='#1B4F72')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(y) for y in years])
    ax1.set_ylim(0, df['area_ha'].max() * 1.2)
    
    # Formatting Ax2 (Growth)
    ax2.set_ylabel('Growth Rate (%)', fontsize=12, fontweight='bold', color='#7B241C')
    ax2.set_ylim(0, df['growth_rate_pct'].max() * 1.5)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=2, frameon=True, shadow=True)

    # plt.title('Evolution of Solar Area Across All Thresholds', fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout()
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  - Grouped figure saved: {output_path.name}")

def main():
    print("Starting Multi-Threshold Solar Area Analysis...")
    
    # 1. Calculate Area for all years and thresholds
    df_results = process_all_data()
    
    if df_results.empty:
        print("Error: No data found to process.")
        return

    # 2. Save detailed CSV
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / "solar_area_summary.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSummary table saved to: {csv_path}")

    # 3. Generate Individual Figures (Threshold specific)
    print("\nGenerating Figures:")
    for threshold in THRESHOLDS:
        output_file = FIGURES_DIR / f"solar_evolution_T{threshold}.png"
        plot_evolution_by_threshold(df_results, threshold, output_file)
    
    # 4. Generate Combined Figure (Grouped Bars)
    combined_file = FIGURES_DIR / "solar_evolution_combined.png"
    plot_combined_evolution(df_results, combined_file)

    # 5. Display Summary Table in console
    print("\n" + "="*80)
    print(f"{'Year':<6} | {'Thresh':<8} | {'Area (ha)':<10} | {'Delta (ha)':<10} | {'Growth (%)':<12} | {'Coverage %':<12}")
    print("-" * 80)
    for _, r in df_results.iterrows():
        delta = f"{r['delta_ha']:+10.2f}" if r['delta_ha'] != 0 else f"{'-':>10}"
        growth = f"{r['growth_rate_pct']:11.2f}%" if r['growth_rate_pct'] != 0 else f"{'-':>12}"
        print(f"{int(r['year']):<6} | {r['threshold']:<8} | {r['area_ha']:10.2f} | {delta} | {growth} | {r['coverage_pct']:11.4f}%")
    print("="*80)

if __name__ == "__main__":
    main()
