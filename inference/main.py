from pathlib import Path

from inference.patching import preprocess_patches

DATASET_BASEPATH=Path(".scratch/data")

wsi_file_path = DATASET_BASEPATH / "AGGC-2022-Unprepared" / "train" / "Subset1_Train_071.tiff"

preprocess_patches(wsi_file_path)