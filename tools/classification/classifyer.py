import cv2
import os
from tqdm import tqdm
import torchvision.transforms as TF
from typing import Callable
import torch.nn as nn
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from tools.training import ResNet18Model, ViTModel, EVA02Model


class NumpyToTensor(Callable):
    def __call__(self, x):
        x = np.transpose(x, axes=(2,0,1))   # HWC -> CHW
        x = torch.from_numpy(x) / 255.0     # <0;255>UINT8 -> <0;1>
        return x.float()   


def classify_patches(model:nn.Module,
                     original_folder:Path,
                     new_folder:Path):

    if any(os.scandir(new_folder)):
        print("Patch classification analysis complete...")
        return

    if isinstance(model, ResNet18Model):
        image_transform = TF.Compose([
            NumpyToTensor()
        ])
    elif isinstance(model, ViTModel):
        image_transform = TF.Compose([
            NumpyToTensor(),
            TF.Resize(size=(224,224))
        ])
    elif isinstance(model, EVA02Model):
        image_transform = TF.Compose([
            NumpyToTensor(),
            TF.Resize(size=(448,448))
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    patch_list = list(original_folder.glob("*.png"))
    for patch_path in tqdm(patch_list, desc="Classifying patches", unit="file"):
        if not patch_path.stem.endswith("n"):
            patch = cv2.imread(patch_path, cv2.IMREAD_COLOR)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = image_transform(patch)
            patch = patch.unsqueeze(0).to(device)

            prediction_logits = model(patch)
            predicted_class = torch.argmax(prediction_logits, dim=1)
            one_hot_vector = torch.nn.functional.one_hot(predicted_class, num_classes=prediction_logits.shape[1])  
            one_hot_vector = one_hot_vector.squeeze(0).tolist()
            
            patch_image = Image.open(patch_path)
            clean_stem = patch_path.stem.removesuffix("_p")

            if one_hot_vector == [1, 0, 0]:
                new_name = f"{clean_stem}_normal.png"
            elif one_hot_vector == [0, 1, 0]:
                new_name = f"{clean_stem}_stroma.png"
            elif one_hot_vector == [0, 0, 1]:
                new_name = f"{clean_stem}_gleason.png"

            patch_image.save(new_folder / new_name)