import yaml
from pathlib import Path
from argparse import Namespace

from tools.training import AGGC2022ClassificationDatamodule


def load_config(config_path:Path
                ) -> Namespace:

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

        # Retrieve the required variables
        model_architecture = config.get("model_architecture", "N/A")
        use_augmentations = config.get("use_augmentations", "N/A")
        batch_size = config.get("batch_size", 0)
        num_workers = config.get("num_workers", 0)
        gleason_handling = config.get("gleason_handling", "N/A")
        vit_technique = config.get("vit_technique", "N/A")

        # Create a dictionary with the retrieved variables
        hyperparameters = Namespace()
        hyperparameters.model_architecture = model_architecture
        hyperparameters.use_augmentations = use_augmentations
        hyperparameters.batch_size = batch_size
        hyperparameters.num_workers = num_workers
        hyperparameters.gleason_handling = gleason_handling
        hyperparameters.vit_technique = vit_technique

        return hyperparameters


def prepare_datamodule(config_path:Path) -> AGGC2022ClassificationDatamodule:
    config = load_config(config_path)
    a = 5
    datamodule = AGGC2022ClassificationDatamodule(config)
    return datamodule