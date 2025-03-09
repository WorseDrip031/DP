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

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_002.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_003.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_004.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_005.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_006.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_007.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_008.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_009.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_010.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_011.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_012.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_013.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_014.tiff"
wsi_file_paths.append(wsi_file_path)

wsi_file_path = DATASET_BASEPATH / "train" / "Subset1_Train_015.tiff"
wsi_file_paths.append(wsi_file_path)

for wsi_path in wsi_file_paths:
    inference_folder = preprocess_patches(INFERENCE_BASEPATH, wsi_path, patch_size, overlap_percentage, tissue_coverage)