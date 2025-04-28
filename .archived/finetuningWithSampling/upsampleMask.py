import numpy as np
from PIL import Image
from pathlib import Path


def upsample_mask(image_path:Path,
                  output_path:Path,
                  original_height:int,
                  original_width:int):
    
    if output_path.exists():
        return
    
    # Open Image
    Image.MAX_IMAGE_PIXELS = None
    mask = Image.open(image_path)

    # Convert image to NumPy array
    binary_mask = np.array(mask)
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    # Resize using nearest neighbor to preserve labels (no interpolation)
    upscaled_mask = Image.fromarray(binary_mask).resize((original_width, original_height), Image.NEAREST)
    upscaled_mask.save(output_path)
