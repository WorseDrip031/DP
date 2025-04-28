from PIL import Image
import numpy as np
from pathlib import Path
import tifffile

Image.MAX_IMAGE_PIXELS = None

# --- Settings ---
input_path = Path("input/segmented_binary_mask.tif")
output_path = Path("output/segmented_binary_mask_upsampled.tif")
TARGET_WIDTH = 46080   # <- define your desired output width
TARGET_HEIGHT = 90720  # <- define your desired output height

# --- Ensure output directory exists ---
output_path.parent.mkdir(parents=True, exist_ok=True)

# --- Load image using PIL ---
mask = Image.open(input_path)

# --- Convert image to NumPy array (binary values) ---
binary_mask = np.array(mask)

# --- Ensure it's binary (0 and 255 values) ---
binary_mask = (binary_mask > 0).astype(np.uint8) * 255

# --- Resize using nearest neighbor to preserve labels (no interpolation) ---
upscaled_mask = Image.fromarray(binary_mask).resize((TARGET_WIDTH, TARGET_HEIGHT), Image.NEAREST)

# --- Convert back to NumPy array for tifffile ---
upscaled_array = np.array(upscaled_mask)

# --- Save the upscaled mask using tifffile with Deflate compression ---
tifffile.imwrite(output_path, upscaled_array, compression="deflate")
print(f"Upscaled binary mask saved to: {output_path}")