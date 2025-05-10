from pathlib import Path

from tools.classification import load_model
from tools.evaluation import prepare_datamodule
from tools.evaluation import evaluate_model


MODELS_BASEPATH = Path(".scratch/experiments")
model_name = "0007 - EVA02 Grouped Pretrained Frozne Downscale"
model_config_path = MODELS_BASEPATH / model_name / "config.yaml"
model_checkpoint_path = MODELS_BASEPATH / model_name / "checkpoints" / "checkpoint-0005.pt"

model = load_model(model_config_path, model_checkpoint_path)
datamodule = prepare_datamodule(model_config_path)
evaluate_model(model, datamodule)