from PIL import Image
import numpy as np
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None

# --- Settings ---
input_path = Path("input/downsampled/Gleason_Mask.tif")
output_path = Path("output/downsampled/Gleason_Mask_downsampled.tif")
MAX_SIZE = 2000

# --- Ensure output directory exists ---
output_path.parent.mkdir(parents=True, exist_ok=True)

# --- Load image using PIL ---
mask = Image.open(input_path)

# --- Convert image to NumPy array (binary values) ---
binary_mask = np.array(mask)

# --- Ensure it's binary (0 and 255 values) ---
binary_mask = (binary_mask > 0).astype(np.uint8) * 255

# --- Get original size ---
height, width = binary_mask.shape

# --- Compute new size keeping aspect ratio ---
scale = min(MAX_SIZE / width, MAX_SIZE / height, 1.0)
new_size = (int(width * scale), int(height * scale))  # (width, height)

# --- Resize using nearest neighbor to preserve labels (no interpolation) ---
downscaled_mask = Image.fromarray(binary_mask).resize(new_size, Image.NEAREST)

# --- Save the downscaled mask as a TIFF file ---
downscaled_mask.save(output_path)
print(f"Downscaled binary mask saved to: {output_path}")