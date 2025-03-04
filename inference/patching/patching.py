from PIL import Image
from pathlib import Path

from .downsample import downsample_wsi
from .segmentation import segment_tissue
from .patch_creator_analyser import create_and_analyse_patches
from .visualize import visualize_blocks_to_process

def preprocess_patches(wsi_file_path, patch_size=512, overlap_percentage=0.5, tissue_coverage=0.5):
    
    # Required to open WSI images of huge sizes
    Image.MAX_IMAGE_PIXELS = None
    wsi = Image.open(wsi_file_path)
    temp_folder = Path(f"inference/patching/temp/{wsi_file_path.stem}")
    temp_folder.mkdir(parents=True, exist_ok=True)

    # Downsample WSI
    downsampled_wsi_path = temp_folder / f"downsampled.png"
    dowsnampled_wsi, scale_factor = downsample_wsi(wsi, downsampled_wsi_path)

    # Segment tissue
    segmented_wsi_path = temp_folder / f"segmented.png"
    segmented_wsi = segment_tissue(dowsnampled_wsi, segmented_wsi_path)

    # Create and analyse patches
    patch_folder = temp_folder / "patches"
    patch_folder.mkdir(parents=True, exist_ok=True)
    create_and_analyse_patches(wsi, segmented_wsi, scale_factor, patch_size, 
                               overlap_percentage, tissue_coverage, patch_folder)
    
    # Visualize Blocks to Process
    visualize_blocks_to_process(temp_folder, patch_size, overlap_percentage)

    return patch_folder