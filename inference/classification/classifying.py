from .model_loader import load_model
from .classifyer import classify_patches
from .visualize import visualize_regions

def analyse_patches(model_name, patch_folder, model_config_path, model_checkpoint_path):
    
    # Load model into memory
    model = load_model(model_config_path, model_checkpoint_path)

    # Calssify patches into classes
    classified_patches_folder = patch_folder.parent / f"classified_patches-{model_name}"
    classified_patches_folder.mkdir(parents=True, exist_ok=True)
    classify_patches(model, patch_folder, classified_patches_folder)

    # Visualize regions
    visualize_regions(classified_patches_folder, 512, 0.5, ["all", "normal", "stroma", "gleason"])