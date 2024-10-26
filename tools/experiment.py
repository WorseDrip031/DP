from pathlib import Path
import yaml
import torch
from torchsummary import summary
from argparse import Namespace

from tools.datamodule import DataModule, AGGC2022ClassificationDatamodule
from tools.model import SimpleConvModel, PretrainedConvModel, ResNet18Model, ResNet50Model
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
    def __init__(self, cfg, load_checkpoint_filepath=None):
        self.cfg = cfg
        self.experiment_path = BASE_PATH / cfg.name
        
        # Create datamodule & model
        self.datamodule = AGGC2022ClassificationDatamodule(cfg)
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
        use_pretrained = True
        if cfg.use_pretrained_model == "No":
            use_pretrained = False

        if cfg.model_architecture == "ResNet18":
            model = ResNet18Model(5, use_pretrained)
        elif cfg.model_architecture == "ResNet50":
            model = ResNet50Model(5, use_pretrained)
        else:
            raise Exception(f"Invalid architecture: {cfg.model_architecture}")
        
        print("Creating model: ")
        summary(
            model,
            input_size=(3,512,512),
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