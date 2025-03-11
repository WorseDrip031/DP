from pathlib import Path

from .model_loader import load_model
from .classifyer import classify_patches
from .visualize_downscaled import visualize_regions_downsampled
from .visualize_gt import visualize_with_groundtruth
from .mask_creator import create_masks

def analyse_patches(wsi_file_path:Path,
                    inference_folder:Path,
                    model_name:str,
                    model_config_path:Path,
                    model_checkpoint_path:Path):
    
    # Step 1: Load model into memory
    print(f"Model loading started for: {model_name}")
    model = load_model(model_config_path, model_checkpoint_path)
    print(f"Model loading complete for: {model_name}")

    # Step 2: Calssify patches into classes
    classified_patches_folder = inference_folder / "classified_patches" / model_name
    classified_patches_folder.mkdir(parents=True, exist_ok=True)
    patches_folder = inference_folder / "patches"
    classify_patches(model, patches_folder, classified_patches_folder)

    # Step 3: Visualize classified regions
    modes = ["normal", "stroma", "gleason"]
    visualize_regions_downsampled(classified_patches_folder, modes)

    # Step 4: Visualize classified regions with groundtruth
    classified_regions_folder = inference_folder / "visualizations" / model_name
    visualize_with_groundtruth(classified_regions_folder, modes)

    # Step 5: Create binary masks for the 3 classes
    masks_folder = inference_folder / "masks" / model_name
    masks_folder.mkdir(parents=True, exist_ok=True)
    create_masks(wsi_file_path, patches_folder, classified_patches_folder, masks_folder)