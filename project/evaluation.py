import os
import re
import yaml
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Saliency, visualization, Occlusion
from tqdm import tqdm

from tools.model import ResNet18Model, ResNet50Model, ViTModel

import matplotlib
matplotlib.use('Agg')



def find_highest_checkpoint(base_path):

    list_checkpoints = []

    # Iterate over all the experiment folders inside the base path
    for experiment_folder in os.listdir(base_path):
        experiment_path = os.path.join(base_path, experiment_folder)

        # Check if the current folder is a directory (to skip files in base_path)
        if os.path.isdir(experiment_path):
            checkpoints_path = os.path.join(experiment_path, "checkpoints")
            
            # If 'checkpoints' folder exists inside the experiment folder
            if os.path.isdir(checkpoints_path):
                highest_checkpoint = None
                max_number = -1

                # List all files in the checkpoints directory
                for file_name in os.listdir(checkpoints_path):
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

                # If a highest checkpoint is found, add it to the list
                if highest_checkpoint:
                    highest_checkpoint_path = f"{base_path}/{experiment_folder}/checkpoints/{highest_checkpoint}"
                    list_checkpoints.append(highest_checkpoint_path)

    return list_checkpoints





def get_all_hyperparameters(base_path):
    hyperparameters_list = []

    # Iterate over all subdirectories in the base path
    for root, _, files in tqdm(os.walk(base_path), desc="Scanning directories for config.yaml"):
        # Check if 'config.yaml' is in the current folder
        if "config.yaml" in files:
            config_path = os.path.join(root, "config.yaml")
            try:
                # Open and read the YAML file
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)

                    # Retrieve the required variables
                    gleason_handling = config.get("gleason_handling", "N/A")
                    model_architecture = config.get("model_architecture", "N/A")
                    use_pretrained_model = config.get("use_pretrained_model", "N/A")
                    use_frozen_model = config.get("use_frozen_model", "N/A")
                    name = config.get("name", "N/A")
                    vit_technique = config.get("vit_technique", "N/A")

                    # Create a dictionary with the retrieved variables
                    hyperparameters = {
                        "gleason_handling": gleason_handling,
                        "model_architecture": model_architecture,
                        "use_pretrained_model": use_pretrained_model,
                        "use_frozen_model": use_frozen_model,
                        "name": name,
                        "vit_technique": vit_technique
                    }

                    # Add the dictionary to the list
                    hyperparameters_list.append(hyperparameters)
            except Exception as e:
                print(f"Error reading {config_path}: {e}")

    return hyperparameters_list





def load_models(list_state_dict_paths, list_dict_hyperparameters):

    models = []

    for i in tqdm(range(len(list_state_dict_paths)), desc="Loading models", ncols=100):
        state_dict_path = list_state_dict_paths[i]
        dict_hyperparameters = list_dict_hyperparameters[i]

        gleason_handling = dict_hyperparameters["gleason_handling"]
        model_architecture = dict_hyperparameters["model_architecture"]
        use_pretrained_model = dict_hyperparameters["use_pretrained_model"]
        use_frozen_model = dict_hyperparameters["use_frozen_model"]

        # Determine the number of classes
        if gleason_handling == "Grouped":
            num_classes = 3
        else:
            num_classes = 5

        # Determine if to use pretrained model
        use_pretrained = True
        if use_pretrained_model == "No":
            use_pretrained = False

        # Determine if to use frozen model
        use_frozen = True
        if use_frozen_model == "No":
            use_frozen = False

        if model_architecture == "ResNet18":
            model = ResNet18Model(num_classes, use_pretrained)
        elif model_architecture == "ResNet50":
            model = ResNet50Model(num_classes, use_pretrained)
        elif model_architecture == "ViT":
            model = ViTModel(num_classes, use_pretrained, use_frozen)
        else:
            raise Exception(f"Invalid architecture: {model_architecture}")

        checkpoint = torch.load(
            state_dict_path,
            map_location=torch.device("cpu")
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)

    return models





def preprocess_image(image_path, vit_technique):

    if vit_technique == "Downscale":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    image = datasets.folder.default_loader(image_path)

    if vit_technique == "Crop" or vit_technique == "QuintupleCrop":
        # Crop the image to the center (224x224)
        width, height = image.size
        left = (width - 224) // 2
        top = (height - 224) // 2
        right = (width + 224) // 2
        bottom = (height + 224) // 2
        image = image.crop((left, top, right, bottom))
        
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor





def find_correct_and_incorrect_images(model, folder_path, target_class, vit_technique):
    
    correct_image = None
    incorrect_image = None
    
    # Iterate over all images in the folder
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        
        # Open the image and apply transformations
        image_tensor = preprocess_image(image_path, vit_technique)
        
        # Make a prediction using the model
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        # Check if the prediction is correct or incorrect
        if predicted_class == target_class and correct_image is None:
            correct_image = image_path
        elif predicted_class != target_class and incorrect_image is None:
            incorrect_image = image_path

        # Stop if both images are found
        if correct_image and incorrect_image:
            break

    return correct_image, incorrect_image





def find_suitable_images(list_models, list_dict_hyperparameters):
    
    list_dict_image_tensors = []

    for i in tqdm(range(len(list_models)), desc="Finding suitable images", ncols=100):
        model = list_models[i]
        dict_hyperparameters = list_dict_hyperparameters[i]
        gleason_handling = dict_hyperparameters["gleason_handling"]
        vit_technique = dict_hyperparameters["vit_technique"]

        # Determine the number of classes
        if gleason_handling == "Grouped":
            num_classes = 3
        else:
            num_classes = 5

        normal_correct, normal_incorrect = find_correct_and_incorrect_images(model, ".scratch/data/AGGC-2022-Classification/train/normal", 0, vit_technique)
        stroma_correct, stroma_incorrect = find_correct_and_incorrect_images(model, ".scratch/data/AGGC-2022-Classification/train/stroma", 1, vit_technique)
        g3_correct, g3_incorrect = find_correct_and_incorrect_images(model, ".scratch/data/AGGC-2022-Classification/train/g3", 2, vit_technique)
        if num_classes == 5:
            g4_correct, g4_incorrect = find_correct_and_incorrect_images(model, ".scratch/data/AGGC-2022-Classification/train/g4", 3, vit_technique)
            g5_correct, g5_incorrect = find_correct_and_incorrect_images(model, ".scratch/data/AGGC-2022-Classification/train/g5", 4, vit_technique)

        dict_image_tensors = {
            "normal_correct": preprocess_image(normal_correct, vit_technique),
            "normal_incorrect": preprocess_image(normal_incorrect, vit_technique),
            "stroma_correct": preprocess_image(stroma_correct, vit_technique),
            "stroma_incorrect": preprocess_image(stroma_incorrect, vit_technique),
            "g3_correct": preprocess_image(g3_correct, vit_technique),
            "g3_incorrect": preprocess_image(g3_incorrect, vit_technique)
        }
        if num_classes == 5:
            dict_image_tensors.update({
                "g4_correct": preprocess_image(g4_correct, vit_technique),
                "g4_incorrect": preprocess_image(g4_incorrect, vit_technique),
                "g5_correct": preprocess_image(g5_correct, vit_technique),
                "g5_incorrect": preprocess_image(g5_incorrect, vit_technique)
            })
        
        list_dict_image_tensors.append(dict_image_tensors)

    return list_dict_image_tensors





def save_visualization(attribution, image, method, sign, title, file_name, output_dir):
    fig, ax = plt.subplots(figsize=(8, 8))
    _ = visualization.visualize_image_attr(
        np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method=method,
        sign=sign,
        show_colorbar=True,
        title=title,
        plt_fig_axis=(fig, ax)
    )
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    plt.close(fig)





def evaluate_with_captum(model, image_tensor, target_class, experiment_name, image_name):
    ig = IntegratedGradients(model)
    saliency = Saliency(model)
    occlusion = Occlusion(model)
    image_tensor.requires_grad = True
    baseline = torch.zeros_like(image_tensor)

    attributions_ig = ig.attribute(image_tensor, baseline, target=target_class, n_steps=50)
    attributions_saliency = saliency.attribute(image_tensor, target=target_class)
    attributions_occ = occlusion.attribute(image_tensor, target=target_class, strides=(3, 8, 8), sliding_window_shapes=(3, 15, 15), baselines=0)

    output_dir = f".scratch/evaluation-results/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save each visualization as a PNG image
    save_visualization(attributions_ig, image_tensor, "heat_map", "all", 
                       f"[{experiment_name}] - {image_name} - Integrated Gradients Attribution", 
                       f"{image_name}_integrated_gradients.png", output_dir)

    save_visualization(attributions_saliency, image_tensor, "heat_map", "absolute_value", 
                       f"[{experiment_name}] - {image_name} - Saliency Map Attribution", 
                       f"{image_name}_saliency_map.png", output_dir)

    save_visualization(attributions_occ, image_tensor, "original_image", "all", 
                       f"[{experiment_name}] - {image_name} - Original Image", 
                       f"{image_name}_original_image.png", output_dir)

    save_visualization(attributions_occ, image_tensor, "heat_map", "positive", 
                       f"[{experiment_name}] - {image_name} - Positive Attribution", 
                       f"{image_name}_positive_attribution.png", output_dir)

    save_visualization(attributions_occ, image_tensor, "heat_map", "negative", 
                       f"[{experiment_name}] - {image_name} - Negative Attribution", 
                       f"{image_name}_negative_attribution.png", output_dir)

    save_visualization(attributions_occ, image_tensor, "masked_image", "positive", 
                       f"[{experiment_name}] - {image_name} - Masked Image", 
                       f"{image_name}_masked_image.png", output_dir)





def evaluate_all_models(list_models, list_dict_hyperparameters, list_dict_image_tensors):
    
    for i in tqdm(range(len(list_models)), desc="Evaluating models", ncols=100):
        model = list_models[i]
        dict_image_tensors = list_dict_image_tensors[i]
        dict_hyperparameters = list_dict_hyperparameters[i]
        gleason_handling = dict_hyperparameters["gleason_handling"]
        experiment_name = dict_hyperparameters["name"]

        # Determine the number of classes
        if gleason_handling == "Grouped":
            num_classes = 3
        else:
            num_classes = 5

        image_name = "normal_correct"
        evaluate_with_captum(model, dict_image_tensors[image_name], 0, experiment_name, image_name)

        image_name = "normal_incorrect"
        evaluate_with_captum(model, dict_image_tensors[image_name], 0, experiment_name, image_name)

        image_name = "stroma_correct"
        evaluate_with_captum(model, dict_image_tensors[image_name], 1, experiment_name, image_name)

        image_name = "stroma_incorrect"
        evaluate_with_captum(model, dict_image_tensors[image_name], 1, experiment_name, image_name)

        image_name = "g3_correct"
        evaluate_with_captum(model, dict_image_tensors[image_name], 2, experiment_name, image_name)

        image_name = "g3_incorrect"
        evaluate_with_captum(model, dict_image_tensors[image_name], 2, experiment_name, image_name)

        if num_classes == 5:

            image_name = "g4_correct"
            evaluate_with_captum(model, dict_image_tensors[image_name], 3, experiment_name, image_name)

            image_name = "g4_incorrect"
            evaluate_with_captum(model, dict_image_tensors[image_name], 3, experiment_name, image_name)

            image_name = "g5_correct"
            evaluate_with_captum(model, dict_image_tensors[image_name], 4, experiment_name, image_name)

            image_name = "g5_incorrect"
            evaluate_with_captum(model, dict_image_tensors[image_name], 4, experiment_name, image_name)


        


if __name__ == "__main__":

    # Load all of the trained models
    list_state_dict_paths = find_highest_checkpoint(".scratch/experiments")
    list_dict_hyperparameters = get_all_hyperparameters(".scratch/experiments")
    list_models = load_models(list_state_dict_paths, list_dict_hyperparameters)
    
    # Load and preprocess images
    list_dict_image_tensors = find_suitable_images(list_models, list_dict_hyperparameters)

    # Evaluate the models with captum
    evaluate_all_models(list_models, list_dict_hyperparameters, list_dict_image_tensors)
    