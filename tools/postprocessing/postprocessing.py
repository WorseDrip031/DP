from pathlib import Path

from .downsampleWSI import downsample
from .downsampleMasks import downsample_masks
from .finetuner import finetune_masks
from .upsampleMasks import upsample_masks
from .annotation_converter import convert_masks_to_geojson
from .wsi_converter import convert_wsi_to_pyramidal
from .cleaner import clean_inference_folder

def postprocess_predictions(wsi_file_path:Path,
                            inference_folder:Path,
                            masks_folder:Path,
                            downsample_size:int=20000):
    
    # Step 1: Donwsample WSI
    finetuned_masks_folder = inference_folder / "finetuned_masks" / masks_folder.stem
    temp_folder = finetuned_masks_folder / "temp"
    downsampled_wsi_path = temp_folder / f"{wsi_file_path.stem}_downsampled.tiff"
    temp_folder.mkdir(parents=True, exist_ok=True)
    downsample(wsi_file_path, downsampled_wsi_path, downsample_size)

    # Step 2: Downsample masks
    height, width = downsample_masks(masks_folder, temp_folder, downsample_size)

    # Step 3: Fine-tune the segmentation masks
    finetune_masks(downsampled_wsi_path, temp_folder)

    # Step 4: Upsample masks
    upsample_masks(temp_folder, finetuned_masks_folder, height, width)

    # Step 5: Create .geojson files from finetuned masks
    convert_masks_to_geojson(finetuned_masks_folder)

    # Step 6: Create pyramidal WSI from original WSI
    convert_wsi_to_pyramidal(wsi_file_path, inference_folder)

    # Step 7: Clean the inference folder from temporary files
    clean_inference_folder(inference_folder, temp_folder)