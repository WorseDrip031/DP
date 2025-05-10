import csv
import numpy as np
from PIL import Image
from pathlib import Path


def load_mask(mask_path:Path, use_print:bool=True):
    
    Image.MAX_IMAGE_PIXELS = None
    mask = Image.open(mask_path)
    binary_mask = np.array(mask)
    mask.close()
    if use_print:
        print(f"Loading complete of mask: {mask_path}")

    return binary_mask


def calculate_overlap(original_path:Path,
                      predicted_path:Path,
                      use_print:bool=True):
    
    original_mask = load_mask(original_path, use_print)
    predicted_mask = load_mask(predicted_path, use_print)

    original_bin = (original_mask > 0).astype(np.uint8)
    predicted_bin = (predicted_mask > 0).astype(np.uint8)

    annotated_area = (original_bin == 1)
    correct_predictions = np.logical_and(annotated_area, predicted_bin == 1)
    total_annotated = np.sum(annotated_area)

    matched = np.sum(correct_predictions)
    overlap_percentage = (matched / total_annotated) * 100.0
    
    if use_print:
        print(f"Overlap: {overlap_percentage:.2f}% ({matched}/{total_annotated})")
    
    return overlap_percentage


def evaluate_masks(original_folder:Path,
                   predicted_folder:Path,
                   use_print:bool=True):
    
    gleason_scores = []
    normal_scores = []
    stroma_scores = []
    names = []

    original_subfolders = [f for f in original_folder.iterdir() if f.is_dir()]
    predicted_subfolders = [f for f in predicted_folder.iterdir() if f.is_dir()]

    for i in range(len(original_subfolders)):
        
        o_subfolder = original_subfolders[i]
        p_subfolder = predicted_subfolders[i]

        names.append(o_subfolder.name)

        g_name = "Gleason_Mask.tif"
        n_name = "Normal_Mask.tif"
        s_name = "Stroma_Mask.tif"

        o_g_path = o_subfolder / g_name
        o_n_path = o_subfolder / n_name
        o_s_path = o_subfolder / s_name

        p_middle_name = Path("finetuned_masks") / "0007 - EVA02 Grouped Pretrained Frozne Downscale"

        p_g_path = p_subfolder / p_middle_name / g_name
        p_n_path = p_subfolder / p_middle_name / n_name
        p_s_path = p_subfolder / p_middle_name / s_name

        if o_g_path.exists():
            gleason_scores.append(calculate_overlap(o_g_path, p_g_path, use_print))
        else:
            gleason_scores.append(0)

        if o_n_path.exists():
            normal_scores.append(calculate_overlap(o_n_path, p_n_path, use_print))
        else:
            normal_scores.append(0)

        if o_s_path.exists():
            stroma_scores.append(calculate_overlap(o_s_path, p_s_path, use_print))
        else:
            stroma_scores.append(0)

    csv_path = Path("scores.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Name", "Gleason_Score", "Normal_Score", "Stroma_Score"])
        # Write rows
        for name, g_score, n_score, s_score in zip(names, gleason_scores, normal_scores, stroma_scores):
            writer.writerow([name, g_score, n_score, s_score])
    

    
