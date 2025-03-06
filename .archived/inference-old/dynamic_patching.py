import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import cv2


def get_threshold_otsu(image):
    img = image.convert("L")
    img_array = np.array(img)
    thershold, _ = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(thershold)


def get_threshold(image):
    img = image.convert("L")
    img_array = np.array(img)
    # Adaptive thresholding (uses a small window to determine local thresholds)
    adaptive_thresh = cv2.adaptiveThreshold(
        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 5
    )
    return int(adaptive_thresh.mean())  # Use the mean as a dynamic threshold


def preprocess_patches(wsi_file_path, patch_size=512, overlap_percentage=0.5, tissue_coverage=0.5):

    # Required to open WSI images of huge sizes
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(wsi_file_path)

    print("-----------------------------")
    print("Determining optimal threshold")
    print("-----------------------------")

    # otsu_threshold = get_threshold(img)
    otsu_threshold = 217

    width, height = img.size
    overlap_size = int(patch_size * overlap_percentage)

    # Create new directory
    directory_path = Path("inference") / "output" / f"{wsi_file_path.stem}-{overlap_size}-tresh-101-10"
    directory_path.mkdir(parents=True, exist_ok=True)

    # Extract patches
    step = int(patch_size * (1 - overlap_percentage))

    for y in tqdm(range(0, height - patch_size + 1, step), desc="Processing rows", unit="row"):
        for x in range(0, width - patch_size + 1, step):

            # Define the box to crop the patch (left, upper, right, lower)
            box = (x, y, x + patch_size, y + patch_size)
            patch = img.crop(box)

            # Convert patch to grayscale to detect tissue
            grayscale_patch = patch.convert("L")
            grayscale_array = np.array(grayscale_patch)
            
            # Count the tissue pixels (those with intensity below the threshold)
            tissue_pixels = np.sum(grayscale_array < otsu_threshold)
            total_pixels = patch_size * patch_size
            
            # Determine if patch contains enough tissue to process further
            process_patch = "n"
            if tissue_pixels / total_pixels >= tissue_coverage:
                process_patch = "p"

            patch.save(directory_path / f"{y}_{x}_{process_patch}.png")

    return directory_path