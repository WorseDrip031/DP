import os
import json
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import List, Tuple


def load_scale_factor(json_file_path:Path) -> float:
    with open(json_file_path, "r") as f:
        data = json.load(f)
    scale_factor = data["scale_factor"]
    return scale_factor


def load_patches_and_metadata(inference_folder:Path,
                              classified_patches_folder:Path
                              ) -> Tuple[List[Tuple[int, int, str, str]], int, int, int]:

    max_x, max_y = 0, 0
    patches: List[Tuple[int, int, str, str]] = []

    for filename in os.listdir(classified_patches_folder):
        if filename.endswith(".png"):
            parts = filename[:-4].split("_")
            y, x, patch_type = int(parts[0]), int(parts[1]), parts[2]
            patches.append((y, x, patch_type, filename))
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    parts = str(inference_folder.stem).split("-")
    patch_size = int(parts[1])
    overlap_percentage = float(parts[2])
    overlap_size = int(patch_size * overlap_percentage)

    wsi_width = max_x + patch_size
    wsi_height = max_y + patch_size

    map_width = wsi_width // overlap_size
    map_height = wsi_height // overlap_size

    return patches, map_width, map_height, overlap_size


def visualize_single_mode(mode:str,
                          regions:NDArray[np.uint8],
                          visual_width:int,
                          visual_height:int,
                          downsampled_overlap_size:int,
                          downsampled_wsi:Image.Image,
                          output_folder:Path):

    max_value = np.max(regions) + 1
    normalized_regions = (regions / max_value * 255).astype(np.uint8)

    mask_height = len(normalized_regions)*downsampled_overlap_size
    mask_width = len(normalized_regions[0])*downsampled_overlap_size

    mask = Image.new("RGBA", (mask_width, mask_height), (0, 0, 0, 0))

    for i in tqdm(range(len(normalized_regions)), desc=f"Generating {mode} mask"):
        for j in range(len(normalized_regions[i])):
            if normalized_regions[i, j] > 0:
                green_patch = Image.new("RGBA", (downsampled_overlap_size, downsampled_overlap_size), (0, 255, 0, normalized_regions[i][j]))
                mask.paste(green_patch, (j*downsampled_overlap_size, i*downsampled_overlap_size), green_patch)

    mask = mask.resize((visual_width, visual_height), resample=Image.LANCZOS)

    image = Image.alpha_composite(downsampled_wsi, mask)
    image.save((output_folder / f"{mode}.png"))


def visualize_regions_downsampled(classified_patches_folder:Path,
                                  modes:List[str]):
    
    print("Visualization analysis started...")
    
    inference_folder = classified_patches_folder.parent.parent
    patches_folder = inference_folder / "patches"
    output_folder = inference_folder / "visualizations" / classified_patches_folder.stem
    output_folder.mkdir(parents=True, exist_ok=True)

    # possible modes: normal, stroma, gleason
    if all((output_folder / f"{mode}.png").exists() for mode in modes):
        print("Visualization complete...")
        return

    # Step 1: Determine all required metadata
    patches, map_width, map_height, overlap_size = load_patches_and_metadata(inference_folder, patches_folder)

    downsampled_wsi_path = inference_folder / "visualizations" / "downsampled.png"
    downsampled_wsi = Image.open(downsampled_wsi_path)
    scale_factor = load_scale_factor(inference_folder / "visualizations" / "downsampled.json")
    downsampled_overlap_size = int(overlap_size * scale_factor)

    visual_width, visual_height = downsampled_wsi.size

    # Step 2: Analyse classified patches
    normal_regions = np.zeros((map_height, map_width), dtype=np.uint8)
    stroma_regions = np.zeros((map_height, map_width), dtype=np.uint8)
    gleason_regions = np.zeros((map_height, map_width), dtype=np.uint8)

    for y, x, patch_type, _ in tqdm(patches, desc="Processing classified patches"):
        if patch_type != 'n':

            matching_files = list(classified_patches_folder.glob(f"{y}_{x}_*.png"))
            classified_patch_file = matching_files[0]
            classified_patch_type = classified_patch_file.stem.split("_")[-1]

            m_y = y // overlap_size
            m_x = x // overlap_size
            if classified_patch_type == "normal":
                normal_regions[m_y:m_y+2, m_x:m_x+2] += 1
            if classified_patch_type == "stroma":
                stroma_regions[m_y:m_y+2, m_x:m_x+2] += 1
            if classified_patch_type == "gleason":
                gleason_regions[m_y:m_y+2, m_x:m_x+2] += 1

    # Step 3: Create visualizations
    downsampled_wsi = downsampled_wsi.convert("RGBA")

    if "normal" in modes:
        visualize_single_mode("normal",
                              normal_regions,
                              visual_width,
                              visual_height,
                              downsampled_overlap_size,
                              downsampled_wsi,
                              output_folder)
    if "stroma" in modes:
        visualize_single_mode("stroma",
                              stroma_regions,
                              visual_width,
                              visual_height,
                              downsampled_overlap_size,
                              downsampled_wsi,
                              output_folder)
    if "gleason" in modes:
        visualize_single_mode("gleason",
                              gleason_regions,
                              visual_width,
                              visual_height,
                              downsampled_overlap_size,
                              downsampled_wsi,
                              output_folder)
        
    print("Visualization complete...")