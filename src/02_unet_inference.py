import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Framework configuration
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm

# =============================================================================
# Script: 02_unet_inference.py
# Description: Performs solar panel segmentation using a trained U-Net model.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Global settings from .env
BASE_DIR = Path(os.getenv("BASE_DIR", "."))

# Model and inference settings
MODEL_PATH = BASE_DIR / "data" / "models" / "unet_efficientnetb1.h5"
INPUT_DIR = BASE_DIR / "data" / "pnoa-historic" / "PNOA2016"
OUTPUT_DIR = BASE_DIR / "data" / "pnoa-segmentation" / "PNOA2016_09"

IMAGE_SIZE = (512, 512)
THRESHOLD = 0.5
OVERLAY_COLOR = (0, 1, 0)  # Green

# -----------------------------------------------------------------------------
# CUSTOM OBJECTS FOR THE MODEL
# -----------------------------------------------------------------------------
def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.keras.backend.sum(y_true, axis=[1, 2, 3]) + tf.keras.backend.sum(y_pred, axis=[1, 2, 3])
    return tf.keras.backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)

bce = tf.keras.losses.BinaryCrossentropy()
def combined_loss(y_true, y_pred):
    dice_loss_val = 1 - dice_coef(y_true, y_pred)
    return 0.5 * dice_loss_val + 0.5 * bce(y_true, y_pred)

class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    """Temporary solution for 'groups' error in Keras 3 / MobileNet compatibility"""
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

CUSTOM_OBJECTS = {
    'dice_coef': dice_coef,
    'combined_loss': combined_loss,
    'iou_score': sm.metrics.iou_score,
    'DepthwiseConv2D': CustomDepthwiseConv2D
}

# -----------------------------------------------------------------------------
# PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------
def load_and_preprocess(file_path, resize=True):
    """Loads an image and prepares it for model input."""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0

    if resize:
        img = tf.image.resize(img, IMAGE_SIZE)

    return tf.expand_dims(img, axis=0)

def save_mask(mask_tensor, filename, output_dir):
    """Generates and saves the resulting binary mask."""
    binary_mask = (mask_tensor.squeeze() > THRESHOLD)
    final_mask = (binary_mask * 255).astype(np.uint8)
    
    os.makedirs(output_dir, exist_ok=True)
    output_filename = filename.rsplit('.', 1)[0] + '_mask.png'
    output_path = os.path.join(output_dir, output_filename)
    
    tf.keras.preprocessing.image.save_img(output_path, np.expand_dims(final_mask, axis=-1), scale=False)
    print(f"  -> Mask saved: {output_path}")
    
    return binary_mask

def save_overlay(original_path, binary_mask, filename, output_dir):
    """Saves an image combining the original with the colored detection."""
    original_img = load_img(original_path, target_size=IMAGE_SIZE)
    original_img = np.array(original_img)
    
    color_mask = binary_mask[:, :, np.newaxis] * np.array(OVERLAY_COLOR) * 255
    color_mask = color_mask.astype(np.uint8)
    
    alpha = 0.5
    overlay = original_img.copy()
    overlay[binary_mask] = (original_img[binary_mask] * (1 - alpha) + color_mask[binary_mask] * alpha).astype(np.uint8)
    
    output_filename = filename.rsplit('.', 1)[0] + '_overlay.png'
    output_path = os.path.join(output_dir, output_filename)
    tf.keras.preprocessing.image.save_img(output_path, overlay, scale=False)
    print(f"  -> Overlay saved: {output_path}")

def run_inference():
    """Main inference loop."""
    # Load model
    model = load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
    print("Model loaded successfully.")

    # List image files
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in image_files:
        file_path = os.path.join(INPUT_DIR, filename)
        input_tensor = load_and_preprocess(file_path)
        
        prediction = model.predict(input_tensor)
        
        binary_mask = save_mask(prediction, filename, OUTPUT_DIR)
        save_overlay(file_path, binary_mask, filename, OUTPUT_DIR)

    print(f"\nInference finished. Results saved in: {OUTPUT_DIR}")

def main():
    run_inference()

if __name__ == "__main__":
    main()
