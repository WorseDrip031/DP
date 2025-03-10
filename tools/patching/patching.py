from PIL import Image
from pathlib import Path

from .downsample import downsample_image
from .segmentation import segment_tissue
from .patcher import create_and_analyse_patches
from .visualize import visualize_regions

def preprocess_patches(base_inference_folder: Path,
                       wsi_file_path: Path,
                       patch_size: int,
                       overlap_percentage: float,
                       tissue_coverage: float
                       ) -> Path:
    
    # Step 1: Open the WSI and prepare a folder for its processing
    Image.MAX_IMAGE_PIXELS = None
    wsi = Image.open(wsi_file_path)
    inference_folder = base_inference_folder / f"{wsi_file_path.stem}-{patch_size}-{overlap_percentage}-{tissue_coverage}"
    inference_folder.mkdir(parents=True, exist_ok=True)
    visualizations_folder = inference_folder / "visualizations"

    # Step 2: Downsample the WSI for faster segmentation
    downsampled_wsi_path = visualizations_folder / f"downsampled.png"
    dowsnampled_wsi, scale_factor = downsample_image(wsi, downsampled_wsi_path)

    # Step 3: Segment the tissue on the downsampled image
    segmented_wsi_path = visualizations_folder / f"segmented.png"
    segmented_wsi = segment_tissue(dowsnampled_wsi, segmented_wsi_path)

    # Step 4: Create patches and determine, if further processing is required
    patch_folder = inference_folder / "patches"
    patch_folder.mkdir(parents=True, exist_ok=True)
    create_and_analyse_patches(wsi, segmented_wsi, scale_factor, patch_size, 
                               overlap_percentage, tissue_coverage, patch_folder)
    
    # Step 5: Visualize regions for further processing + create downsampled version
    #regions_image = visualize_regions(inference_folder, patch_size, overlap_percentage)
    #downsampled_regions_path = inference_folder / "regions_to_process_downsampled.png"
    #downsample_image(regions_image, downsampled_regions_path, save_scale_factor=False)

    return inference_folder