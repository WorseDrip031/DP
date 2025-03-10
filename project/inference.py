from pathlib import Path

from tools.patching import preprocess_patches
from tools.classification import analyse_patches

DATASET_BASEPATH = Path(".scratch/data/AGGC-2022-Unprepared")
MODELS_BASEPATH = Path(".scratch/experiments")
INFERENCE_BASEPATH = Path(".scratch/inference")

########################################
#  Stage 1: Pre-processing - Patching  #
########################################

wsi_file_path = DATASET_BASEPATH / "test" / "Subset1_Test_025.tiff"     # Path to the WSI to be analysed
patch_size = 512                                                        # Size of the created patches
overlap_percentage = 0.5                                                # Percentage of overlap between patches
tissue_coverage = 0.5                                                   # Minimal tissue coverage of patch for processing

inference_folder = preprocess_patches(INFERENCE_BASEPATH, wsi_file_path, patch_size, overlap_percentage, tissue_coverage)


###################################
#  Stage 2: Patch Classification  #
###################################

model_name = "0001 - Pretrained ResNet18 Grouped"
model_config_path = MODELS_BASEPATH / model_name / "config.yaml"
model_checkpoint_path = MODELS_BASEPATH / model_name / "checkpoints" / "checkpoint-0002.pt"

analyse_patches(inference_folder, model_name, model_config_path, model_checkpoint_path)