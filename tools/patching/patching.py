from pathlib import Path

from .downsample import downsample_wsi_and_masks
from .segmentation import segment_tissue
from .patcher import create_and_analyse_patches
from .visualize import visualize_regions_to_process


def preprocess_patches(base_inference_folder: Path,
                       wsi_file_path: Path,
                       patch_size: int,
                       overlap_percentage: float,
                       tissue_coverage: float
                       ) -> Path:
    
    # Step 1: Prepare a folder for WSI processing
    print(f"Creating inference folder for: {wsi_file_path.stem}")
    inference_folder = base_inference_folder / f"{wsi_file_path.stem}-{patch_size}-{overlap_percentage}-{tissue_coverage}"
    visualizations_folder = inference_folder / "visualizations"
    visualizations_folder.mkdir(parents=True, exist_ok=True)

    # Step 2: Downsample the WSI for faster segmentation
    dowsnampled_wsi, scale_factor = downsample_wsi_and_masks(wsi_file_path, visualizations_folder)

    # Step 3: Segment the tissue on the downsampled image
    segmented_wsi_path = visualizations_folder / f"segmented.png"
    segmented_wsi = segment_tissue(dowsnampled_wsi, segmented_wsi_path)

    # Step 4: Create patches and determine, if further processing is required
    patch_folder = inference_folder / "patches"
    patch_folder.mkdir(parents=True, exist_ok=True)
    create_and_analyse_patches(wsi_file_path, segmented_wsi, scale_factor, patch_size, 
                               overlap_percentage, tissue_coverage, patch_folder)
    
    # Step 5: Visualize regions for further processing + create downsampled version
    visualize_regions_to_process(inference_folder, patch_size, overlap_percentage)

    return inference_folder