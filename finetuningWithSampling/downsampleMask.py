import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple


def downsample_mask(image_path:Path,
                    output_path:Path,
                    max_size:int
                    ) -> Tuple[int, int]:
    
    if output_path.exists():
        Image.MAX_IMAGE_PIXELS = None
        mask = Image.open(image_path)
        binary_mask = np.array(mask)
        height, width = binary_mask.shape
        return height, width
    
    # Open Image
    Image.MAX_IMAGE_PIXELS = None
    mask = Image.open(image_path)

    # Convert image to NumPy array
    binary_mask = np.array(mask)
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    # Compute new size keeping aspect ratio
    height, width = binary_mask.shape
    scale = min(max_size / width, max_size / height, 1.0)
    new_size = (int(width * scale), int(height * scale))

    # Resize using nearest neighbor to preserve labels (no interpolation)
    downscaled_mask = Image.fromarray(binary_mask).resize(new_size, Image.NEAREST)
    downscaled_mask.save(output_path)

    return height, width
