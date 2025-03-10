from pathlib import Path
from typing import List, Tuple

from tools.classification import analyse_patches

DATASET_BASEPATH = Path(".scratch/data/AGGC-2022-Unprepared")
MODELS_BASEPATH = Path(".scratch/experiments")
INFERENCE_BASEPATH = Path(".scratch/inference")

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

for inference_folder in INFERENCE_BASEPATH.iterdir():
    if inference_folder.is_dir():
        for m_paths in model_paths:
            analyse_patches(inference_folder, m_paths[0], m_paths[1], m_paths[2])