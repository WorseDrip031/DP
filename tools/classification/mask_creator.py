import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import tifffile


def create_mask(mask_path:Path,
                mask_type:str,
                width:int,
                height:int,
                patches_folder:Path,
                classified_patches_folder:Path):
    
    # Retrieve required metadata
    inference_folder = classified_patches_folder.parent.parent
    parts = (inference_folder.stem).split("-")
    patch_size = int(parts[1])

    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)

    patch_files = list(patches_folder.glob("*.png"))
    for patch_file in tqdm(patch_files, desc=f"Creating {mask_type} mask"):
        parts = patch_file.stem.split('_')
        y, x, patch_type = int(parts[0]), int(parts[1]), parts[2]

        # Ignore 'n' type patches
        if patch_type == 'n':
            continue

        # Find the corresponding classified patch
        classified_matches = list(classified_patches_folder.glob(f"{y}_{x}_*.png"))
        classified_file = classified_matches[0]
        classified_parts = classified_file.stem.split('_')            
        classified_type = classified_parts[2]
        if classified_type == mask_type:
            mask[y:y+patch_size, x:x+patch_size] = 255  # Binary mask set to 255
    
    tifffile.imwrite(mask_path, mask, compression="lzw")


def create_masks(wsi_file_path:Path,
                 patches_folder:Path,
                 classified_patches_folder:Path,
                 masks_folder:Path):
    
    print("Mask creation analysis started...")

    gleason_mask_path = masks_folder / "Gleason_Mask.tif"
    normal_mask_path = masks_folder / "Normal_Mask.tif"
    stroma_mask_path = masks_folder / "Stroma_Mask.tif"

    if all(file.exists() for file in [gleason_mask_path, normal_mask_path, stroma_mask_path]):
        print("Mask creation complete...")
        return

    # Open image
    Image.MAX_IMAGE_PIXELS = None
    wsi = Image.open(wsi_file_path)
    width, height = wsi.size
    wsi.close()


    if not gleason_mask_path.exists():
        create_mask(gleason_mask_path, "gleason", width, height, patches_folder, classified_patches_folder)
    print(f"Mask creation complete for {gleason_mask_path.stem}")

    if not normal_mask_path.exists():
        create_mask(normal_mask_path, "normal", width, height, patches_folder, classified_patches_folder)
    print(f"Mask creation complete for {normal_mask_path.stem}")

    if not stroma_mask_path.exists():
        create_mask(stroma_mask_path, "stroma", width, height, patches_folder, classified_patches_folder)
    print(f"Mask creation complete for {stroma_mask_path.stem}")