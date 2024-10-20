import torch
import torchvision.transforms as TF
from torch.utils.data import DataLoader

from tools.dataset import create_animals_dataset, NumpyToTensor, create_aggc_classification_dataset


class DataModule:
    def __init__(self):

        self.input_transform = TF.Compose([
            NumpyToTensor(),
            TF.Resize(size=(224,224))
        ])

        self.dataset_train = create_animals_dataset(
            is_train=True,
            image_transform=self.input_transform
        )
        self.dataset_val = create_animals_dataset(
            is_train=False,
            image_transform=self.input_transform
        )
       
        
    def setup(self, cfg):
        self.dataloader_train = torch.utils.data.dataloader.DataLoader(
            dataset=self.dataset_train, 
            batch_size=cfg.batch_size, 
            shuffle=True,
            num_workers=cfg.num_workers
        )
        self.dataloader_val = torch.utils.data.dataloader.DataLoader(
            dataset=self.dataset_val, 
            batch_size=cfg.batch_size, 
            shuffle=False,
            num_workers=cfg.num_workers
        )





class AGGC2022ClassificationDatamodule:
    def __init__(self, cfg):

        self.augmentation_transform = None
        self.input_transform = TF.Compose([
            NumpyToTensor()
        ])

        if cfg.use_augmentations == "Yes":
            self.augmentation_transform = TF.Compose([
                TF.RandomHorizontalFlip(p=0.5),
                TF.RandomVerticalFlip(p=0.5),
                TF.RandomApply([TF.RandomRotation(degrees=90)], p=0.5)
            ])

        self.dataset_train = create_aggc_classification_dataset(
            type="train",
            image_transform=self.input_transform,
            augmentation_transform=self.augmentation_transform
        )
        self.dataset_val = create_aggc_classification_dataset(
            type="val",
            image_transform=self.input_transform,
            augmentation_transform=None
        )
        self.dataset_val = create_aggc_classification_dataset(
            type="test",
            image_transform=self.input_transform,
            augmentation_transform=None
        )

        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )

        self.dataloader_val = DataLoader(
            self.dataset_val,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        self.dataloader_test = DataLoader(
            self.dataset_test,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )