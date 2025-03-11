import json
from PIL import Image
from pathlib import Path
from typing import Tuple, List


def load_downsampled_wsi_and_json(downsampled_wsi_path:Path,
                                  json_file_path:Path
                                  ) -> Tuple[Image.Image, float]:
    
    downsampled_wsi = Image.open(downsampled_wsi_path)
    scale_factor = 1.0
    with open(json_file_path, "r") as f:
        data = json.load(f)
    scale_factor = data["scale_factor"]

    return downsampled_wsi, scale_factor


def downsample(image_path:Path,
               max_size:int
               )-> Tuple[Image.Image, float]:
    
    # Open image
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_path)

    # Determine new size
    width, height = image.size
    scale_factor = max_size / max(width, height)
    new_size = (int(width * scale_factor), int(height * scale_factor))

    # Downsample
    print(f"Dowsnampling started for: {image_path.stem}")
    downsampled_image = image.resize(new_size, Image.LANCZOS)
    image.close()

    return downsampled_image, scale_factor


def downsample_wsi_and_masks(wsi_file_path:Path,
                             visualizations_folder:Path,
                             max_size:int=2000
                             ) -> Tuple[Image.Image, float]:

    print("Downasmpling analysis started...")

    downsampled_wsi_path = visualizations_folder / "downsampled.png"
    json_file_path = visualizations_folder / "downsampled.json"

    # If downsampled image already exists, simply load into memory
    if downsampled_wsi_path.exists():
        downsampled_wsi, scale_factor = load_downsampled_wsi_and_json(downsampled_wsi_path, json_file_path)
    else:
        downsampled_wsi, scale_factor = downsample(wsi_file_path, max_size)
        downsampled_wsi.save(downsampled_wsi_path)
        data = {
            "scale_factor": scale_factor
        }
        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=1)

    print(f"Downsampling completed for: {wsi_file_path.stem}")

    masks_folder = wsi_file_path.parent / wsi_file_path.stem
    mask_paths: List[Path] = [
        masks_folder / "G3_Mask.tif",
        masks_folder / "G4_Mask.tif",
        masks_folder / "G5_Mask.tif",
        masks_folder / "Normal_Mask.tif",
        masks_folder / "Stroma_Mask.tif",
    ]

    for mask_path in mask_paths:
        if mask_path.exists():
            downsampled_mask_path = visualizations_folder / f"downsampled_{mask_path.stem}.png"
            if not downsampled_mask_path.exists():
                downsampled_mask, _ = downsample(mask_path, max_size)
                downsampled_mask.save(downsampled_mask_path)
            print(f"Downsampling complete for: {mask_path.stem}")

    return downsampled_wsi, scale_factor