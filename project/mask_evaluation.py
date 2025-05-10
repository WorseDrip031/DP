from pathlib import Path

from tools.evaluation import preprocess_masks
from tools.evaluation import evaluate_masks


base_folder = Path(".scratch") / "data" / "AGGC-2022-Masks"
original_folder = base_folder / "Test"
destination_folder = base_folder / "Refined"
predicted_folder = Path(".scratch") / "inference"

preprocess_masks(original_folder, destination_folder)
evaluate_masks(destination_folder, predicted_folder, True)
