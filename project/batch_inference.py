from pathlib import Path
from typing import List

from tools.patching import preprocess_patches

DATASET_BASEPATH = Path(".scratch/data/AGGC-2022-Unprepared")
MODELS_BASEPATH = Path(".scratch/models")
INFERENCE_BASEPATH = Path(".scratch/inference")

########################################
#  Stage 1: Pre-processing - Patching  #
########################################

patch_size = 512                                                        # Size of the created patches
overlap_percentage = 0.5                                                # Percentage of overlap between patches
tissue_coverage = 0.5                                                   # Minimal tissue coverage of patch for processing

wsi_file_paths:List[Path] = []

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_001.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_011.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_022.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_033.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_044.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_055.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_066.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_071.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_075.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_080.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_085.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "val" / "Subset1_Val_002.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "val" / "Subset1_Val_008.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "val" / "Subset1_Val_012.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "val" / "Subset1_Val_018.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "test" / "Subset1_Test_005.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "test" / "Subset1_Test_015.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "test" / "Subset1_Test_025.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "test" / "Subset1_Test_035.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "test" / "Subset1_Test_045.tiff"
wsi_file_paths.append(wsi_file_path)

for wsi_path in wsi_file_paths:
    inference_folder = preprocess_patches(INFERENCE_BASEPATH, wsi_path, patch_size, overlap_percentage, tissue_coverage)