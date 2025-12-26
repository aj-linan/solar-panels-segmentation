import os
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')

def visualize_segmentation_grid():
    # Configuration
    years = ['2016', '2019', '2022']
    thresholds = ['05', '07', '09']
    tiles = ['tile_02_05', 'tile_08_09', 'tile_10_04', 'tile_11_13', 'tile_18_01']
    
    # Grid setup: 5 rows (5 tiles), 10 columns (3 years * 3 thresholds + 1 GT)
    num_rows = len(tiles)
    num_cols = len(years) * len(thresholds) + 1
    
    # Adjust figsize for horizontal layout (Width x Height)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(38, 20))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for row, tile_name in enumerate(tiles):
        current_col = 0
        # Prediction columns
        for year in years:
            for threshold in thresholds:
                folder_name = f"PNOA{year}_{threshold}"
                folder_path = os.path.join(BASE_DIR, 'data', 'pnoa-segmentation', folder_name)
                
                img_name = f"{tile_name}_overlay.png"
                img_path = os.path.join(folder_path, img_name)
                
                ax = axes[row, current_col]
                
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16)
                
                # Column labels (only on the first row)
                if row == 0:
                    t_val = float(threshold) / 10
                    ax.set_title(f"{year}\nT={t_val}", fontsize=28, fontweight='bold', pad=15)
                
                ax.set_xticks([])
                ax.set_yticks([])
                current_col += 1
        
        # --- GROUND TRUTH COLUMN (Last Column) ---
        ax = axes[row, current_col]
        gt_img_path = os.path.join(BASE_DIR, 'data', 'pnoa-historic', 'PNOA2022_overlays', f"{tile_name}_overlay.png")
        
        if os.path.exists(gt_img_path):
            img = Image.open(gt_img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'GT N/A', ha='center', va='center', fontsize=16)
        
        if row == 0:
            ax.set_title("Ground Truth\n(2022)", fontsize=28, fontweight='bold', pad=15)
        
        ax.set_xticks([])
        ax.set_yticks([])
            
    # Save the result
    output_dir = os.path.join(BASE_DIR, 'data', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'segmentation_comparison.png')
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Visualization saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    visualize_segmentation_grid()
