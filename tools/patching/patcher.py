import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path


def create_and_analyse_patches(wsi_file_path:Path,
                               segmented_wsi:Image.Image,
                               scale_factor:float,
                               patch_size:int,
                               overlap_percentage:float,
                               tissue_coverage:float,
                               patch_folder:Path):
    
    print("Patch creation analysis started...")

    if any(os.scandir(patch_folder)):
        print(f"Patch creation complete for: {wsi_file_path.stem}")
        return

    # Open WSI
    Image.MAX_IMAGE_PIXELS = None
    wsi = Image.open(wsi_file_path)

    # Determine required metadata
    width, height = wsi.size
    step = int(patch_size * (1 - overlap_percentage))

    for y in tqdm(range(0, height - patch_size + 1, step), desc=f"Patch creation for: {wsi_file_path.stem}", unit="row"):
        for x in range(0, width - patch_size + 1, step):

            # Define the box to crop the patch (left, upper, right, lower)
            box = (x, y, x + patch_size, y + patch_size)
            patch = wsi.crop(box)

            # Define the downsampled box coordinates
            x1 = int(x * scale_factor)
            y1 = int(y * scale_factor)
            x2 = int((x+patch_size) * scale_factor)
            y2 = int((y+patch_size) * scale_factor)

            # Analyse the segmentation region
            segmented_wsi_np = np.array(segmented_wsi)
            segmented_region = segmented_wsi_np[y1:y2, x1:x2]
            tissue_pixels = np.sum(segmented_region == 255)
            crop_height, crop_width = segmented_region.shape
            total_pixels = crop_height * crop_width
            
            # Determine if patch contains enough tissue to process further
            process_patch = "n"
            if tissue_pixels / total_pixels >= tissue_coverage:
                process_patch = "p"

            patch.save(patch_folder / f"{y}_{x}_{process_patch}.png")

    wsi.close()
