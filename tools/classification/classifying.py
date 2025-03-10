from pathlib import Path

from .model_loader import load_model
from .classifyer import classify_patches
from .visualize_downscaled import visualize_regions_downsampled

def analyse_patches(inference_folder:Path,
                    model_name:str,
                    model_config_path:Path,
                    model_checkpoint_path:Path):
    
    # Step 1: Load model into memory
    model = load_model(model_config_path, model_checkpoint_path)

    # Step 2: Calssify patches into classes
    classified_patches_folder = inference_folder / "classified_patches" / model_name
    classified_patches_folder.mkdir(parents=True, exist_ok=True)
    patches_folder = inference_folder / "patches"
    classify_patches(model, patches_folder, classified_patches_folder)

    # Visualize regions
    modes = ["normal", "stroma", "gleason"]
    visualize_regions_downsampled(classified_patches_folder, modes)