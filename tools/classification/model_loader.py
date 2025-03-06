import yaml
import torch
import torch.nn as nn
from pathlib import Path

from tools.training import ResNet18Model, ViTModel, EVA02Model


def load_hyperparameters(model_config_path:Path
                         ) -> dict[str, str]:
    
    with open(model_config_path, 'r') as file:
        config = yaml.safe_load(file)

        # Retrieve the required variables
        gleason_handling = config.get("gleason_handling", "N/A")
        model_architecture = config.get("model_architecture", "N/A")
        use_pretrained_model = config.get("use_pretrained_model", "N/A")
        use_frozen_model = config.get("use_frozen_model", "N/A")

        # Create a dictionary with the retrieved variables
        hyperparameters = {
            "gleason_handling": gleason_handling,
            "model_architecture": model_architecture,
            "use_pretrained_model": use_pretrained_model,
            "use_frozen_model": use_frozen_model,
        }

        return hyperparameters


def load_model(model_config_path:Path, 
               model_checkpoint_path:Path
               ) -> nn.Module:
    
    print("Loading model hyperparameters...")

    hyperparameters = load_hyperparameters(model_config_path)
    gleason_handling = hyperparameters["gleason_handling"]
    model_architecture = hyperparameters["model_architecture"]
    use_pretrained_model = hyperparameters["use_pretrained_model"]
    use_frozen_model = hyperparameters["use_frozen_model"]

    # Determine the number of classes
    if gleason_handling == "Grouped":
        num_classes = 3
    else:
        num_classes = 5

    # Determine if to use pretrained model
    use_pretrained = True
    if use_pretrained_model == "No":
        use_pretrained = False

    # Determine if to use frozen model
    use_frozen = True
    if use_frozen_model == "No":
        use_frozen = False

    print("Loading model...")

    # Determine model architecture
    if model_architecture == "ResNet18":
        model = ResNet18Model(num_classes, use_pretrained)
    elif model_architecture == "ViT":
        model = ViTModel(num_classes, use_pretrained, use_frozen)
    elif model_architecture == "EVA02":
        model = EVA02Model(num_classes, use_pretrained, use_frozen)
    else:
        raise Exception(f"Invalid architecture: {model_architecture}")
    
    checkpoint = torch.load( # type: ignore
        model_checkpoint_path,
        map_location=torch.device("cpu")
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model