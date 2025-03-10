from pathlib import Path
from typing import List, Tuple

from tools.patching import preprocess_patches
from tools.classification import analyse_patches

DATASET_BASEPATH = Path(".scratch/data/AGGC-2022-Unprepared")
MODELS_BASEPATH = Path(".scratch/experiments")
INFERENCE_BASEPATH = Path(".scratch/inference")

print("########################################")
print("#  Stage 1: Pre-processing - Patching  #")
print("########################################")

########################################
#  Stage 1: Pre-processing - Patching  #
########################################

patch_size = 512                                                        # Size of the created patches
overlap_percentage = 0.5                                                # Percentage of overlap between patches
tissue_coverage = 0.5                                                   # Minimal tissue coverage of patch for processing
wsi_file_paths:List[Path] = []

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

inference_folders:List[Path] = []
for wsi_path in wsi_file_paths:
    inference_folder = preprocess_patches(INFERENCE_BASEPATH, wsi_path, patch_size, overlap_percentage, tissue_coverage)
    inference_folders.append(inference_folder)


print("###################################")
print("#  Stage 2: Patch Classification  #")
print("###################################")

###################################
#  Stage 2: Patch Classification  #
###################################

model_paths:List[Tuple[str, Path, Path]] = []

model_name = "0001 - Pretrained ResNet18 Grouped"
model_config_path = MODELS_BASEPATH / model_name / "config.yaml"
model_checkpoint_path = MODELS_BASEPATH / model_name / "checkpoints" / "checkpoint-0002.pt"
model_paths.append((model_name, model_config_path, model_checkpoint_path))

model_name = "0005 - Pretrained Frozen ViT Grouped Downscale"
model_config_path = MODELS_BASEPATH / model_name / "config.yaml"
model_checkpoint_path = MODELS_BASEPATH / model_name / "checkpoints" / "checkpoint-0001.pt"
model_paths.append((model_name, model_config_path, model_checkpoint_path))

model_name = "0007 - EVA02 Grouped Pretrained Frozne Downscale"
model_config_path = MODELS_BASEPATH / model_name / "config.yaml"
model_checkpoint_path = MODELS_BASEPATH / model_name / "checkpoints" / "checkpoint-0005.pt"
model_paths.append((model_name, model_config_path, model_checkpoint_path))

for inference_folder in inference_folders:
    for m_paths in model_paths:
        analyse_patches(inference_folder, m_paths[0], m_paths[1], m_paths[2])