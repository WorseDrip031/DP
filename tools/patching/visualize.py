import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import List, Tuple


def load_scale_factor(json_file_path:Path) -> float:
    with open(json_file_path, "r") as f:
        data = json.load(f)
    scale_factor = data["scale_factor"]
    return scale_factor


def visualize_regions_to_process(inference_folder:Path,
                                 patch_size:int,
                                 overlap_percentage:float):
    
    print("Regions to process analysis started...")

    regions_to_process_path = inference_folder / "visualizations" / "regions_to_process.png"
    if regions_to_process_path.exists():
        print("Regions to process visualization complete...")
        return

    # Step 1: Determine all required metadata
    max_x, max_y = 0, 0
    patches: List[Tuple[int, int, str, str]] = []

    for filename in os.listdir(inference_folder / "patches"):
        if filename.endswith(".png"):
            parts = filename[:-4].split("_")
            y, x, patch_type = int(parts[0]), int(parts[1]), parts[2]
            patches.append((y, x, patch_type, filename))
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    overlap_size = int(patch_size * overlap_percentage)

    wsi_width = max_x + patch_size
    wsi_height = max_y + patch_size

    map_width = wsi_width // overlap_size
    map_height = wsi_height // overlap_size

    downsampled_wsi_path = inference_folder / "visualizations" / "downsampled.png"
    downsampled_wsi = Image.open(downsampled_wsi_path)
    scale_factor = load_scale_factor(inference_folder / "visualizations" / "downsampled.json")
    downsampled_overlap_size = int(overlap_size * scale_factor)

    visual_width, visual_height = downsampled_wsi.size

    # Step 2: Analyse patches
    regions = np.zeros((map_height, map_width), dtype=np.uint8)

    for y, x, patch_type, filename in tqdm(patches, desc="Processing patches"):
        if patch_type == 'p':
            m_y = y // overlap_size
            m_x = x // overlap_size
            regions[m_y:m_y+2, m_x:m_x+2] += 1

    # Step 3: Create visualizations
    downsampled_wsi = downsampled_wsi.convert("RGBA")

    max_value = np.max(regions) + 1
    normalized_regions = (regions / max_value * 255).astype(np.uint8)

    mask_height = len(normalized_regions)*downsampled_overlap_size
    mask_width = len(normalized_regions[0])*downsampled_overlap_size
    
    mask = Image.new("RGBA", (mask_width, mask_height), (0, 0, 0, 0))

    for i in tqdm(range(len(normalized_regions)), desc="Generating mask"):
        for j in range(len(normalized_regions[i])):
            if normalized_regions[i, j] > 0:
                green_patch = Image.new("RGBA", (downsampled_overlap_size, downsampled_overlap_size), (0, 255, 0, normalized_regions[i][j]))
                mask.paste(green_patch, (j*downsampled_overlap_size, i*downsampled_overlap_size), green_patch)

    mask = mask.resize((visual_width, visual_height), resample=Image.LANCZOS)

    image = Image.alpha_composite(downsampled_wsi, mask)
    image.save(regions_to_process_path)

    print("Regions to process visualization complete...")