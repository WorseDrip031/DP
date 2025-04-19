import shutil
from pathlib import Path


def clean_inference_folder(inference_folder:Path,
                           temp_folder:Path):

    print("Inference folder cleaning started...")

    patches_folder = inference_folder / "patches"
    if patches_folder.exists():
        shutil.rmtree(patches_folder)

    classified_patches_folder = inference_folder / "classified_patches"
    if classified_patches_folder.exists():
        shutil.rmtree(classified_patches_folder)

    if temp_folder.exists():
        shutil.rmtree(temp_folder)

    print("Inference folder cleaning complete...")