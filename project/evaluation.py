import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from captum.attr import IntegratedGradients, Saliency, visualization, Occlusion

from tools.model import ResNet18Model

def load_model(state_dict_path, num_classes, use_prterained):
    model = ResNet18Model(num_classes, use_prterained)
    checkpoint = torch.load(
        state_dict_path,
        map_location=torch.device("cpu")
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = datasets.folder.default_loader(image_path)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def evaluate_with_captum(model, image_tensor, target_class):
    ig = IntegratedGradients(model)
    saliency = Saliency(model)
    occlusion = Occlusion(model)
    image_tensor.requires_grad = True
    baseline = torch.zeros_like(image_tensor)
    attributions_ig = ig.attribute(image_tensor, baseline, target=target_class, n_steps=50)
    attributions_saliency = saliency.attribute(image_tensor, target=target_class)
    attributions_occ = occlusion.attribute(image_tensor, target=target_class, strides=(3, 8, 8), sliding_window_shapes=(3, 15, 15), baselines=0)

    _ = visualization.visualize_image_attr(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="heat_map",
        sign="all",
        show_colorbar=True,
        title="Integrated Gradients Attribution"
    )

    _ = visualization.visualize_image_attr(
        np.transpose(attributions_saliency.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title="Saliency Map Attribution"
    )

    _ = visualization.visualize_image_attr(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="original_image",
        sign="all",
        show_colorbar=True,
        title="Original Image"
    )

    _ = visualization.visualize_image_attr(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="heat_map",
        sign="positive",
        show_colorbar=True,
        title="Positive Attribution"
    )

    _ = visualization.visualize_image_attr(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="heat_map",
        sign="negative",
        show_colorbar=True,
        title="Negative Attribution"
    )

    _ = visualization.visualize_image_attr(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="masked_image",
        sign="positive",
        show_colorbar=True,
        title="Masked Image"
    )

if __name__ == "__main__":

    # Load the model
    state_dict_path = ".scratch/experiments/0001 - Pretrained ResNet18 Grouped/checkpoints/checkpoint-0002.pt"
    num_classes = 3
    use_pretrained = True
    model = load_model(state_dict_path, num_classes, use_pretrained)

    # Load and preprocess images
    normal_image_path = ".scratch/data/AGGC-2022-Classification/train/normal/00030.png"
    normal_image_tensor = preprocess_image(normal_image_path)
    stroma_image_path = ".scratch/data/AGGC-2022-Classification/train/stroma/00030.png"
    stroma_image_tensor = preprocess_image(stroma_image_path)
    gleason_image_path = ".scratch/data/AGGC-2022-Classification/train/g3/00030.png"
    gleason_image_tensor = preprocess_image(gleason_image_path)

    # Evaluate the model with captum
    evaluate_with_captum(model, normal_image_tensor, 0)
    evaluate_with_captum(model, stroma_image_tensor, 1)
    evaluate_with_captum(model, gleason_image_tensor, 2)