from pathlib import Path
from typing import List, Tuple

from tools.patching import preprocess_patches
from tools.classification import analyse_patches
from tools.postprocessing import postprocess_predictions

DATASET_BASEPATH = Path(".scratch/data/AGGC-2022-Unprepared")
MODELS_BASEPATH = Path(".scratch/experiments")
INFERENCE_BASEPATH = Path(".scratch/inference")


#############################
#  Stage 1 Hyperparameters  #
#############################

patch_size = 512                                                        # Size of the created patches
overlap_percentage = 0.5                                                # Percentage of overlap between patches
tissue_coverage = 0.5                                                   # Minimal tissue coverage of patch for processing

wsi_file_paths = [
    #DATASET_BASEPATH / "test" / "Subset1_Test_001.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_002.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_003.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_004.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_005.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_006.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_007.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_008.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_009.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_010.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_011.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_012.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_013.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_014.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_015.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_016.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_017.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_018.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_019.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_020.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_021.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_022.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_023.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_024.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_025.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_026.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_027.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_028.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_029.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_030.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_031.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_032.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_033.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_034.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_035.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_036.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_037.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_038.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_039.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_040.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_041.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_042.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_043.tiff",
    DATASET_BASEPATH / "test" / "Subset1_Test_044.tiff",
    #DATASET_BASEPATH / "test" / "Subset1_Test_045.tiff",
]


##############################
#  Stage 2: Hyperparameters  #
##############################

model_paths:List[Tuple[str, Path, Path]] = []

model_name = "0005 - Pretrained Frozen ViT Grouped Downscale"
model_config_path = MODELS_BASEPATH / model_name / "config.yaml"
model_checkpoint_path = MODELS_BASEPATH / model_name / "checkpoints" / "checkpoint-0001.pt"
model_paths.append((model_name, model_config_path, model_checkpoint_path))

model_name = "0007 - EVA02 Grouped Pretrained Frozne Downscale"
model_config_path = MODELS_BASEPATH / model_name / "config.yaml"
model_checkpoint_path = MODELS_BASEPATH / model_name / "checkpoints" / "checkpoint-0005.pt"
model_paths.append((model_name, model_config_path, model_checkpoint_path))


#####################
#  Batch Inference  #
#####################

for wsi_path in wsi_file_paths:

    print()
    print("########################################")
    print("#  Stage 1: Pre-processing - Patching  #")
    print("########################################")
    print()

    inference_folder = preprocess_patches(INFERENCE_BASEPATH, wsi_path, patch_size, overlap_percentage, tissue_coverage)

    print()
    print("###################################")
    print("#  Stage 2: Patch Classification  #")
    print("###################################")
    print()

    masks_folders:List[Path] = []

    for m_paths in model_paths:

        masks_folder = analyse_patches(wsi_path, inference_folder, m_paths[0], m_paths[1], m_paths[2])
        masks_folders.append(masks_folder)

    print()
    print("##########################")
    print("#  Stage 3: Fine Tuning  #")
    print("##########################")
    print()

    for mask_folder in masks_folders:

        postprocess_predictions(wsi_path, inference_folder, mask_folder)