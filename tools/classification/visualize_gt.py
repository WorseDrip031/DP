import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List


def visualize_gt(image_path:Path,
                 mask_paths:List[Path],
                 output_path:Path):
    
    image = cv2.imread(str(image_path))

    # Load and combine all masks
    mask_union = None
    for mask_path in mask_paths:
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask) > 0
            # Union of masks (pixel-wise maximum)
            if mask_union is None:
                mask_union = mask_array.astype(np.uint8)
            else:
                mask_union = np.maximum(mask_union, mask_array.astype(np.uint8))

    # Convert union mask to uint8 (0 and 255)
    mask_union = (mask_union * 255).astype(np.uint8)

    # Find contours in the union mask
    contours, _ = cv2.findContours(mask_union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image (red color, thickness 2)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(str(output_path), image)


def visualize_with_groundtruth(classified_regions_folder:Path,
                               modes:List[str]):
    
    visualizations_folder = classified_regions_folder.parent

    if "normal" in modes:
        normal_path = classified_regions_folder / "normal.png"
        normal_mask_paths = [
            visualizations_folder / "downsampled_Normal_Mask.png",
        ]
        if any(mask_path.exists() for mask_path in normal_mask_paths):
            normal_with_gt_path = classified_regions_folder / "normal_with_gt.png"
            if not normal_with_gt_path.exists():
                visualize_gt(normal_path, normal_mask_paths, normal_with_gt_path)

    if "stroma" in modes:
        stroma_path = classified_regions_folder / "stroma.png"
        stroma_mask_paths = [
            visualizations_folder / "downsampled_Stroma_Mask.png",
        ]
        if any(mask_path.exists() for mask_path in stroma_mask_paths):
            stroma_with_gt_path = classified_regions_folder / "stroma_with_gt.png"
            if not stroma_with_gt_path.exists():
                visualize_gt(stroma_path, stroma_mask_paths, stroma_with_gt_path)

    if "gleason" in modes:
        gleason_path = classified_regions_folder / "gleason.png"
        gleason_mask_paths = [
            visualizations_folder / "downsampled_G3_Mask.png",
            visualizations_folder / "downsampled_G4_Mask.png",
            visualizations_folder / "downsampled_G5_Mask.png",
        ]
        if any(mask_path.exists() for mask_path in gleason_mask_paths):
            gleason_with_gt_path = classified_regions_folder / "gleason_with_gt.png"
            if not gleason_with_gt_path.exists():
                visualize_gt(gleason_path, gleason_mask_paths, gleason_with_gt_path)

    print("Visualization with gt complete...")