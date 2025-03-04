from pathlib import Path

from inference.patching import preprocess_patches
from inference.classification import analyse_patches

DATASET_BASEPATH=Path(".scratch/data")

wsi_file_path = DATASET_BASEPATH / "AGGC-2022-Unprepared" / "train" / "Subset1_Train_071.tiff"

patch_folder = preprocess_patches(wsi_file_path)

MODELS_BASEPATH=Path(".scratch/models")

model_name = "0001 - Pretrained ResNet18 Grouped"
model_config_path = MODELS_BASEPATH / model_name / "config.yaml"
model_checkpoint_path = MODELS_BASEPATH / model_name / "checkpoints" / "checkpoint-0002.pt"

analyse_patches(model_name, patch_folder, model_config_path, model_checkpoint_path)