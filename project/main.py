from argparse import ArgumentParser

from tools.experiment import Experiment, AGGCClassificationExperiment


def main(cfg):
    experiment = AGGCClassificationExperiment(cfg)
    experiment.train()


if __name__ == "__main__":

    p = ArgumentParser()

    # Experiment
    p.add_argument("--name", "-n", type=str, default="0009 - ResNet18 Pretrained Separate", help="Experiment name")
    p.add_argument("--project", "-p", choices=["DP-Classification"], default="DP-Classification", help="Project name")
    p.add_argument("--continue_from_checkpoint", "-cfc", choices=["Yes", "No"], default="No", help="Want to continue from last checkpoint?")
    p.add_argument("--run_id", "-rid", type=str, default="qj57dd2n", help="Wandb run ID for continuing experiment")

    # Hyperparameters
    p.add_argument("--batch_size", "-bs", type=int, default=128, help="Batch size")
    p.add_argument("--num_workers", "-nw", type=int, default=2, help="Number of dataloader workers")
    p.add_argument("--max_epochs", "-e", type=int, default=20, help="Number of epochs to train")
    p.add_argument("--learning_rate", "-lr", type=float, default=0.005, help="Optimizer learning rate")
    p.add_argument("--beta1", "-b1", type=float, default=0.5, help="Optimizer beta1")
    p.add_argument("--beta2", "-b2", type=float, default=0.999, help="Optimizer beta2")
    p.add_argument("--num_hidden", "-nh", type=int, default=512, help="Number of hidden units")
    p.add_argument("--use_augmentations", "-ua", choices=["Yes", "No"], default="Yes", help="Use augmentations?")
    p.add_argument("--gleason_handling", "-gh", choices=["Separate", "Grouped"], default="Separate", help="How to handle gleason classes")

    # Model
    p.add_argument("--model_architecture", "-ma", choices=["ResNet18", "ResNet50", "ViT", "EVA02"], default="ResNet18", help="Model Architecture")
    p.add_argument("--use_pretrained_model", "-upm", choices=["Yes", "No"], default="Yes", help="Use pretrained model?")
    p.add_argument("--use_frozen_model", "-ufm", choices=["Yes", "No"], default="No", help="Use frozen model?")
    p.add_argument("--vit_technique", "-vt", choices=["Downscale", "Crop", "QuintupleCrop", "N/A"], default="N/A", help="ViT data preparation technique")

    cfg = p.parse_args()
    main(cfg)