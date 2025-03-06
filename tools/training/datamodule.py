import torchvision.transforms as TF
from torch.utils.data import DataLoader

from .dataset import NumpyToTensor, create_aggc_classification_dataset


class AGGC2022ClassificationDatamodule:
    def __init__(self, cfg):

        self.augmentation_transform = None

        if (cfg.model_architecture == "ViT" or cfg.model_architecture == "EVA02") and cfg.vit_technique == "Downscale":
            if cfg.model_architecture == "EVA02":
                self.input_transform = TF.Compose([
                    NumpyToTensor(),
                    TF.Resize(size=(448,448))
                ])
            else:
                self.input_transform = TF.Compose([
                    NumpyToTensor(),
                    TF.Resize(size=(224,224))
                ])
        else:
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
            cfg,
            type="train",
            image_transform=self.input_transform,
            augmentation_transform=self.augmentation_transform
        )
        self.dataset_val = create_aggc_classification_dataset(
            cfg,
            type="val",
            image_transform=self.input_transform,
            augmentation_transform=None
        )
        self.dataset_test = create_aggc_classification_dataset(
            cfg,
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