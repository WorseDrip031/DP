import torch
import torchvision.transforms as TF

from tools.dataset import create_animals_dataset, NumpyToTensor


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