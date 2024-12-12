import os
import re
import yaml
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Saliency, visualization, Occlusion
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
from torchmetrics import Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from tools.model import ResNet18Model, ResNet50Model, ViTModel, EVA02Model
from tools.datamodule import AGGC2022ClassificationDatamodule
from tools.statistics import Statistics

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
                    batch_size = config.get("batch_size", "N/A")
                    num_workers = config.get("num_workers", "N/A")
                    use_augmentations = config.get("use_augmentations", "N/A")

                    # Create a dictionary with the retrieved variables
                    hyperparameters = {
                        "gleason_handling": gleason_handling,
                        "model_architecture": model_architecture,
                        "use_pretrained_model": use_pretrained_model,
                        "use_frozen_model": use_frozen_model,
                        "name": name,
                        "vit_technique": vit_technique,
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "use_augmentations": use_augmentations
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
        elif model_architecture == "EVA02":
            model = EVA02Model(num_classes, use_pretrained, use_frozen)
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





def preprocess_image(image_path, vit_technique, model_architecture):

    if vit_technique == "Downscale":
        if model_architecture == "EVA02":
            transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor()
            ])
        else:
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
        crop_size = 224
        if model_architecture == "EVA02":
            crop_size = 448
        # Crop the image to the center
        width, height = image.size
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2
        image = image.crop((left, top, right, bottom))
        
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor





def find_all_classification_combinations(model, folder_path, target_class, vit_technique, model_architecture, num_combinations, images_per_combination):

    classification_combinations = defaultdict(list)
    
    # Initialize counters to track how many images we have for each combination
    combination_counters = defaultdict(int)
    
    # Iterate over all images in the folder
    for file_name in os.listdir(folder_path):
        # Stop early if all combinations have the required number of images
        if all(count == images_per_combination for count in combination_counters.values()) and len(combination_counters) == num_combinations:
            break

        image_path = os.path.join(folder_path, file_name)
        
        # Open the image and apply transformations
        image_tensor = preprocess_image(image_path, vit_technique, model_architecture)
        
        # Make a prediction using the model
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        # Record the combination
        combination = (target_class, predicted_class)
        
        # Add the image if we haven't yet reached the desired count for this combination
        if combination_counters[combination] < images_per_combination:
            classification_combinations[combination].append(image_path)
            combination_counters[combination] += 1

    return classification_combinations





def find_suitable_images(list_models, list_dict_hyperparameters, images_per_combination=1):

    list_dict_image_tensors = []

    for i in tqdm(range(len(list_models)), desc="Finding suitable images", ncols=100):
        model = list_models[i]
        dict_hyperparameters = list_dict_hyperparameters[i]
        gleason_handling = dict_hyperparameters["gleason_handling"]
        vit_technique = dict_hyperparameters["vit_technique"]
        model_architecture = dict_hyperparameters["model_architecture"]

        # Determine the number of classes
        if gleason_handling == "Grouped":
            num_classes = 3
        else:
            num_classes = 5

        # Initialize a dictionary to store all classifications
        all_classifications = defaultdict(list)

        classes = ["normal", "stroma", "g3", "g4", "g5"]

        # Loop through each class
        for target_class in range(num_classes):
            class_folder = f".scratch/data/AGGC-2022-Classification/train/{classes[target_class]}"
            combinations = find_all_classification_combinations(
                model, class_folder, target_class, vit_technique, model_architecture, num_classes, images_per_combination
            )
            
            # Combine results into the main dictionary
            for key, images in combinations.items():
                all_classifications[key].extend(preprocess_image(img, vit_technique, model_architecture) for img in images)

        list_dict_image_tensors.append(all_classifications)

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
        experiment_name = dict_hyperparameters["name"]

        classes = ["normal", "stroma", "g3", "g4", "g5"]

        # Iterate through all combinations in the dictionary
        for (true_class, predicted_class), image_tensors in dict_image_tensors.items():
            # Construct a descriptive image name
            combination_name = f"A-{classes[true_class]}_P-{classes[predicted_class]}"

            # Evaluate each image tensor in the current combination
            for idx, image_tensor in enumerate(image_tensors):
                evaluate_with_captum(
                    model, image_tensor, true_class, experiment_name, f"{combination_name}_image_{idx + 1}"
                )



        

def test_all_models(list_models, list_dict_hyperparameters):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()

    for i in range(len(list_models)):

        model = list_models[i]
        model = model.to(device)
        dict_hyperparameters = list_dict_hyperparameters[i]
        experiment_name = dict_hyperparameters["name"]
        acc = Accuracy("multiclass", num_classes=model.num_classes)

        p = ArgumentParser()
        p.add_argument("--batch_size", "-bs", type=int, default=dict_hyperparameters["batch_size"], help="Batch size")
        p.add_argument("--num_workers", "-nw", type=int, default=dict_hyperparameters["num_workers"], help="Number of dataloader workers")
        p.add_argument("--model_architecture", "-ma", choices=["ResNet18", "ResNet50", "ViT", "EVA02"], default=dict_hyperparameters["model_architecture"], help="Model Architecture")
        p.add_argument("--vit_technique", "-vt", choices=["Downscale", "Crop", "QuintupleCrop"], default=dict_hyperparameters["vit_technique"], help="ViT data preparation technique")
        p.add_argument("--use_augmentations", "-ua", choices=["Yes", "No"], default=dict_hyperparameters["use_augmentations"], help="Use augmentations?")
        p.add_argument("--gleason_handling", "-gh", choices=["Separate", "Grouped"], default=dict_hyperparameters["gleason_handling"], help="How to handle gleason classes")
        cfg = p.parse_args()

        datamodule = AGGC2022ClassificationDatamodule(cfg)
        dataloader_test = datamodule.dataloader_test
        stats = Statistics()

        # Lists to store true and predicted labels for the confusion matrix
        all_true_labels = []
        all_pred_labels = []

        model.eval()
        with tqdm(dataloader_test, desc=experiment_name) as progress:
            for x, y in progress:
                
                x = x.to(device)
                y = y.to(device)

                y_hat_logits = model(x)
                l = loss(y_hat_logits, y)

                # Convert logits to predictions
                preds = torch.argmax(torch.softmax(y_hat_logits, dim=1), dim=1)

                # Store true labels and predictions for confusion matrix
                all_true_labels.append(y.cpu().numpy())
                all_pred_labels.append(preds.cpu().numpy())

                a = acc(
                    torch.softmax(y_hat_logits, dim=1).cpu(),      # Predictions
                    torch.argmax(y, dim=1).cpu()                   # Classes
                )

                stats.step("loss_val", l.item())
                stats.step("acc_val", a.item())
                progress.set_postfix(stats.get())

        # Flatten the collected true and predicted labels
        all_true_labels = np.concatenate(all_true_labels, axis=0)
        all_pred_labels = np.concatenate(all_pred_labels, axis=0)

        # Compute confusion matrix
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        print(f"Confusion Matrix for {experiment_name}:")
        print(cm)

        # Calculate accuracy from the confusion matrix
        accuracy_from_cm = np.trace(cm) / np.sum(cm)
        print(f"Accuracy from Confusion Matrix for {experiment_name}: {accuracy_from_cm:.4f}")

        # Calculate precision, recall, and F1-score
        accuracy = accuracy_score(all_true_labels, all_pred_labels, normalize=True)
        precision = precision_score(all_true_labels, all_pred_labels, average='weighted')
        recall = recall_score(all_true_labels, all_pred_labels, average='weighted')
        f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')

        print(f"Accuracy for {experiment_name}: {accuracy:.4f}")
        print(f"Precision for {experiment_name}: {precision:.4f}")
        print(f"Recall for {experiment_name}: {recall:.4f}")
        print(f"F1-Score for {experiment_name}: {f1:.4f}")


        


EVALUATE_EXPLAINABILITY = False
EVALUATE_METRICS = True

if __name__ == "__main__":

    if EVALUATE_EXPLAINABILITY or EVALUATE_METRICS:

        # Load all of the trained models
        list_state_dict_paths = find_highest_checkpoint(".scratch/experiments")
        list_dict_hyperparameters = get_all_hyperparameters(".scratch/experiments")
        list_models = load_models(list_state_dict_paths, list_dict_hyperparameters)

    if EVALUATE_EXPLAINABILITY:
    
        # Load and preprocess images
        list_dict_image_tensors = find_suitable_images(list_models, list_dict_hyperparameters)

        # Evaluate the models with captum
        evaluate_all_models(list_models, list_dict_hyperparameters, list_dict_image_tensors)

    if EVALUATE_METRICS:

        # Evaluate the models on testing dataset
        test_all_models(list_models, list_dict_hyperparameters)
    