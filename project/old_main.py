from argparse import ArgumentParser

from tools.experiment import Experiment


def main(cfg):
    experiment = Experiment(cfg)
    experiment.train()


if __name__ == "__main__":

    p = ArgumentParser()

    # Experiment
    p.add_argument("--name", "-n", type=str, default="conv", help="Experiment name")

    # Hyperparameters
    p.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size")
    p.add_argument("--num_workers", "-nw", type=int, default=0, help="Number of dataloader workers")
    p.add_argument("--max_epochs", "-e", type=int, default=3, help="Number of epochs to train")
    p.add_argument("--learning_rate", "-lr", type=float, default=0.1, help="Optimizer learning rate")
    p.add_argument("--num_hidden", "-nh", type=int, default=512, help="Number of hidden units")

    # Model
    p.add_argument("--model_architecture", "-ma", choices=["simple", "resnet"], default="resnet", help="Model Architecture")

    cfg = p.parse_args()
    main(cfg)