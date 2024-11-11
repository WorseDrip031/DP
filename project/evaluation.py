from captum.attr import IntegratedGradients, Saliency, visualization

from tools.model import ResNet18Model

if __name__ == "__main__":

    state_dict_path = 