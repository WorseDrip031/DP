import numpy as np
from PIL import Image
from pathlib import Path


def upsample_masks(masks_folder:Path,
                   finetuned_masks_folder:Path,
                   original_height:int,
                   original_width:int):
    
    print("Mask upsampling started...")

    mask_paths = list(masks_folder.glob("*_finetuned.tif"))

    for mask_path in mask_paths:

        original_name = mask_path.stem
        base_name = original_name.replace("_downsampled_finetuned", "")
        new_filename = f"{base_name}.tif"
        output_mask_path = finetuned_masks_folder / new_filename

        if output_mask_path.exists():
            continue

        # Open Image
        Image.MAX_IMAGE_PIXELS = None
        mask = Image.open(mask_path)

        # Convert image to NumPy array
        binary_mask = np.array(mask)
        binary_mask = (binary_mask > 0).astype(np.uint8) * 255

        # Resize using nearest neighbor to preserve labels (no interpolation)
        upscaled_mask = Image.fromarray(binary_mask).resize((original_width, original_height), Image.NEAREST)
        upscaled_mask.save(output_mask_path)

    print("Mask upsampling complete...")