import shutil
import tifffile
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path


def preprocess_mask(subfolder:Path,
                    destination_folder:Path):
    
    target_folder = destination_folder / subfolder.name
    target_folder.mkdir(parents=True, exist_ok=True)

    for mask_name in ["Normal_Mask.tif", "Stroma_Mask.tif"]:
        source_mask = subfolder / mask_name
        if source_mask.exists():
            shutil.copy(source_mask, target_folder / mask_name)

    gleason_masks = []
    for mask_name in ["G3_Mask.tif", "G4_Mask.tif", "G5_Mask.tif"]:
        mask_path = subfolder / mask_name
        if mask_path.exists():
            Image.MAX_IMAGE_PIXELS = None
            mask = Image.open(mask_path)
            mask_array = np.array(mask) > 0
            mask.close()
            gleason_masks.append(mask_array)

    if gleason_masks:
        union_mask = np.any(gleason_masks, axis=0).astype(np.uint8) * 255
        save_path = target_folder / "Gleason_Mask.tif"
        tifffile.imwrite(save_path, union_mask, compression="deflate")


def preprocess_masks(original_folder:Path,
                     destination_folder:Path):
    
    destination_folder.mkdir(parents=True, exist_ok=True)
    subfolders = [f for f in original_folder.iterdir() if f.is_dir()]

    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        
        preprocess_mask(subfolder, destination_folder)
