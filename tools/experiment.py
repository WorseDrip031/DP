from pathlib import Path
import yaml
import torch
import os
import re
from torchsummary import summary
from argparse import Namespace

from tools.datamodule import DataModule, AGGC2022ClassificationDatamodule
from tools.model import SimpleConvModel, PretrainedConvModel, ResNet18Model, ResNet50Model, ViTModel, EVA02Model
from tools.trainer import Trainer
import tools.logging as L


BASE_PATH=Path(".scratch/experiments")


class Experiment:
    def __init__(self, cfg, load_checkpoint_filepath=None):
        self.cfg = cfg
        self.experiment_path = BASE_PATH / cfg.name
        
        # Create datamodule & model
        self.datamodule = DataModule()
        self.model = self.create_model(cfg)

        if load_checkpoint_filepath is not None:
            file_path =  self.experiment_path / "checkpoints" / load_checkpoint_filepath
            print(f" > Loading checkpoint: {file_path.as_posix()}")
            checkpoint = torch.load(
                file_path.as_posix(),
                map_location=torch.device("cpu")
            )
            self.model.load_state_dict(checkpoint["model"])
        else:
            print(f" > Created experiment : {cfg.name}")
            self.experiment_path.mkdir(parents=True, exist_ok=True)
            self.save_config(cfg, self.experiment_path)
            checkpoint = None

        self.trainer = Trainer(cfg, self.model)
        if checkpoint is not None:
            self.trainer.opt.load_state_dict(checkpoint["opt"])


    @staticmethod
    def from_folder(exp_name, version, checkpoint_epoch=None):
        experiment_path = BASE_PATH / str(exp_name) / str(version)
        config_filepath = experiment_path / "config.yaml"
        with config_filepath.open() as fp:
            config_dict = yaml.safe_load(fp)
            config_str = yaml.dump(config_dict)
            print(" > Loading configuration: ")
            print("--------------------------")
            print(config_str.strip())
            print("\n")
            cfg = Namespace(**config_dict)

        if checkpoint_epoch is None:
            checkpoint_filename = "last.pt"
        else:
            checkpoint_filename = f"checkpoint-{checkpoint_epoch:04d}.pt"

        experiment = Experiment(cfg, load_checkpoint_filepath=checkpoint_filename)


    def create_model(self, cfg):
        if cfg.model_architecture == "simple":
            model = SimpleConvModel(
                chin=3,
                channels=16,
                num_hidden=cfg.num_hidden,
                num_classes=self.datamodule.dataset_train.num_classes
            )
        elif cfg.model_architecture == "resnet":
            model = PretrainedConvModel(
                num_hidden=cfg.num_hidden,
                num_classes=self.datamodule.dataset_train.num_classes
            )
        else:
            raise Exception(f"Invalid architecture: {cfg.model_architecture}")
        
        print("Creating model: ")
        summary(
            model,
            input_size=(3,224,224),
            batch_size=1,
            device="cpu"
        )
        return model
    

    def save_config(self, cfg, exp_path):
        config_yaml = yaml.dump(vars(cfg))
        print(" > Training Configuration:")
        print("--------------------------")
        print(config_yaml.strip())
        print("\n")
        (exp_path / "config.yaml").write_text(config_yaml)


    def train(self):
        self.trainer.setup(
            datamodule=self.datamodule,
            logs=[
                L.CSVLog(self.experiment_path / "training.csv"),
                L.ReportCompiler(
                    filepath = self.experiment_path / "report.pdf",
                    source_filepath = self.experiment_path / "training.csv"
                ),
                L.ModelCheckpointer(self)
            ]
        )
        self.trainer.fit()


    def save_checkpoint(self, filename):
        checkpoint = {
            "model": self.model.state_dict(),
            "opt": self.trainer.opt.state_dict()
        }

        file_path = self.experiment_path / "checkpoints" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving checkpoint: {file_path.as_posix()}")
        torch.save(checkpoint, file_path.as_posix())




















class AGGCClassificationExperiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.experiment_path = BASE_PATH / cfg.name
        
        # Create datamodule & model
        self.datamodule = AGGC2022ClassificationDatamodule(cfg)
        self.model = self.create_model(cfg)

        continue_from_checkpoint = False
        if cfg.continue_from_checkpoint == "Yes":
            continue_from_checkpoint = True

        if continue_from_checkpoint:
            checkpoints_folder = self.experiment_path / "checkpoints"
            file_path = self.get_last_checkpoint(checkpoints_folder)
            print(f" > Loading checkpoint: {file_path.as_posix()}")
            checkpoint = torch.load(
                file_path.as_posix(),
                map_location=torch.device("cpu")
            )
            self.model.load_state_dict(checkpoint["model"])
        else:
            print(f" > Created experiment : {cfg.name}")
            self.experiment_path.mkdir(parents=True, exist_ok=True)
            self.save_config(cfg, self.experiment_path)
            checkpoint = None

        self.trainer = Trainer(cfg, self.model)
        if checkpoint is not None:
            self.trainer.opt.load_state_dict(checkpoint["opt"])


    @staticmethod
    def from_folder(exp_name, version, checkpoint_epoch=None):
        experiment_path = BASE_PATH / str(exp_name) / str(version)
        config_filepath = experiment_path / "config.yaml"
        with config_filepath.open() as fp:
            config_dict = yaml.safe_load(fp)
            config_str = yaml.dump(config_dict)
            print(" > Loading configuration: ")
            print("--------------------------")
            print(config_str.strip())
            print("\n")
            cfg = Namespace(**config_dict)

        if checkpoint_epoch is None:
            checkpoint_filename = "last.pt"
        else:
            checkpoint_filename = f"checkpoint-{checkpoint_epoch:04d}.pt"

        experiment = Experiment(cfg, load_checkpoint_filepath=checkpoint_filename)

    
    def get_last_checkpoint(self, checkpoints_folder):
        max_number = -1
        for file_name in os.listdir(checkpoints_folder):
            if file_name == "last.pt":
                highest_checkpoint = file_name
                break
            else:
                # Match checkpoint files with a pattern like 'checkpoint-XXXX.pt'
                match = re.match(r"checkpoint-(\d+)\.pt", file_name)
                if match:
                    checkpoint_number = int(match.group(1))
                    if checkpoint_number > max_number:
                        max_number = checkpoint_number
                        highest_checkpoint = file_name
        return checkpoints_folder / highest_checkpoint


    def create_model(self, cfg):

        # Determine the number of classes
        if cfg.gleason_handling == "Grouped":
            num_classes = 3
        else:
            num_classes = 5

        # Determine if to use pretrained model
        use_pretrained = True
        if cfg.use_pretrained_model == "No":
            use_pretrained = False

        # Determine if to use frozen model
        use_frozen = True
        if cfg.use_frozen_model == "No":
            use_frozen = False

        if cfg.model_architecture == "ResNet18":
            model = ResNet18Model(num_classes, use_pretrained)
        elif cfg.model_architecture == "ResNet50":
            model = ResNet50Model(num_classes, use_pretrained)
        elif cfg.model_architecture == "ViT":
            model = ViTModel(num_classes, use_pretrained, use_frozen)
        elif cfg.model_architecture == "EVA02":
            model = EVA02Model(num_classes, use_pretrained, use_frozen)
        else:
            raise Exception(f"Invalid architecture: {cfg.model_architecture}")
        
        print("Creating model: ")
        if cfg.model_architecture != "ViT":
            summary(
                model,
                input_size=(num_classes,512,512),
                batch_size=1,
                device="cpu"
            )
        else:
            summary(
                model,
                input_size=(num_classes,224,224),
                batch_size=1,
                device="cpu"
            )
        return model
    

    def save_config(self, cfg, exp_path):
        config_yaml = yaml.dump(vars(cfg))
        print(" > Training Configuration:")
        print("--------------------------")
        print(config_yaml.strip())
        print("\n")
        (exp_path / "config.yaml").write_text(config_yaml)


    def train(self):
        self.trainer.setup(
            datamodule=self.datamodule,
            logs=[
                L.CSVLog(self.experiment_path / "training.csv"),
                L.ReportCompiler(
                    filepath = self.experiment_path / "report.pdf",
                    source_filepath = self.experiment_path / "training.csv"
                ),
                L.ModelCheckpointer(self)
            ]
        )
        self.trainer.fit()


    def save_checkpoint(self, filename):
        checkpoint = {
            "model": self.model.state_dict(),
            "opt": self.trainer.opt.state_dict()
        }

        file_path = self.experiment_path / "checkpoints" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving checkpoint: {file_path.as_posix()}")
        torch.save(checkpoint, file_path.as_posix())