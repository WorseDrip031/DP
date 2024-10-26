from argparse import ArgumentParser

from tools.experiment import Experiment, AGGCClassificationExperiment


def main(cfg):
    experiment = AGGCClassificationExperiment(cfg)
    experiment.train()


if __name__ == "__main__":

    p = ArgumentParser()

    # Experiment
    p.add_argument("--name", "-n", type=str, default="Non-Pretrained ResNet18 First Try", help="Experiment name")

    # Hyperparameters
    p.add_argument("--batch_size", "-bs", type=int, default=64, help="Batch size")
    p.add_argument("--num_workers", "-nw", type=int, default=2, help="Number of dataloader workers")
    p.add_argument("--max_epochs", "-e", type=int, default=10, help="Number of epochs to train")
    p.add_argument("--learning_rate", "-lr", type=float, default=0.1, help="Optimizer learning rate")
    p.add_argument("--num_hidden", "-nh", type=int, default=512, help="Number of hidden units")
    p.add_argument("--use_augmentations", "-ua", choices=["Yes", "No"], default="Yes", help="Use augmentations?")
    p.add_argument("--use_pretrained_model", "-upm", choices=["Yes", "No"], default="No", help="Use pretrained model?")

    # Model
    p.add_argument("--model_architecture", "-ma", choices=["ResNet18", "ResNet50"], default="ResNet18", help="Model Architecture")

    cfg = p.parse_args()
    main(cfg)