import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple


def downsample_masks(masks_folder:Path,
                     output_folder:Path,
                     downsample_size:int
                     ) -> Tuple[int, int]:
    
    print("Mask downsampling started...")

    height, width = 0, 0

    mask_paths = list(masks_folder.glob("*.tif"))
    
    for mask_path in mask_paths:

        output_mask_path = output_folder / f"{mask_path.stem}_downsampled.tif"

        if output_mask_path.exists():
            Image.MAX_IMAGE_PIXELS = None
            mask = Image.open(mask_path)
            binary_mask = np.array(mask)
            height, width = binary_mask.shape
            continue

        # Open Image
        Image.MAX_IMAGE_PIXELS = None
        mask = Image.open(mask_path)

        # Convert image to NumPy array
        binary_mask = np.array(mask)
        binary_mask = (binary_mask > 0).astype(np.uint8) * 255

        # Compute new size keeping aspect ratio
        height, width = binary_mask.shape
        scale = min(downsample_size / width, downsample_size / height, 1.0)
        new_size = (int(width * scale), int(height * scale))

        # Resize using nearest neighbor to preserve labels (no interpolation)
        downscaled_mask = Image.fromarray(binary_mask).resize(new_size, Image.NEAREST)
        downscaled_mask.save(output_mask_path)
            
    print("Mask downsampling complete...")

    return height, width