import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from skimage.color import rgb2hsv
from skimage.filters import median
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.util import img_as_ubyte
from skimage.segmentation import felzenszwalb


def calculate_tissue_percentage(image_array, scale=1000, sigma=0.9, min_size=500):
    # Convert RGB to HSV and use the saturation channel
    hsv_image = rgb2hsv(image_array)
    saturation_channel = hsv_image[:, :, 1]

    # Apply median & entopy filter
    filtered_image = median(saturation_channel, disk(5))
    entropy_image = entropy(img_as_ubyte(filtered_image), disk(3))

    # Use superpixel segmentation
    segments = felzenszwalb(entropy_image, scale=scale, sigma=sigma, min_size=min_size)

    # Threshold Superpixels
    T = np.mean(filtered_image)
    output_image = np.zeros_like(filtered_image)
    for label in np.unique(segments):
        mask = segments == label
        if np.mean(filtered_image[mask]) > T:
            output_image[mask] = 255
        else:
            output_image[mask] = 0

    # Calculate tissue percentage
    total_pixels = output_image.size
    white_pixels = np.count_nonzero(output_image == 255)
    return white_pixels / total_pixels


def preprocess_patches(wsi_file_path, patch_size=512, overlap_percentage=0.5, tissue_coverage=0.5):

    # Required to open WSI images of huge sizes
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(wsi_file_path)
    width, height = img.size
    overlap_size = int(patch_size * overlap_percentage)

    # Create new directory
    directory_path = Path("inference") / "output" / f"{wsi_file_path.stem}-{height}-{width}-{overlap_size}"
    directory_path.mkdir(parents=True, exist_ok=True)

    # Extract patches
    step = int(patch_size * (1 - overlap_percentage))

    for y in tqdm(range(0, height - patch_size + 1, step), desc="Processing rows", unit="row"):
        for x in range(0, width - patch_size + 1, step):

            # Define the box to crop the patch (left, upper, right, lower)
            box = (x, y, x + patch_size, y + patch_size)
            patch = img.crop(box)

            # Convert patch to numpy array
            patch_array = np.array(patch)

            # Determine if patch contains enough tissue to process further
            process_patch = "n"
            if tissue_coverage <= calculate_tissue_percentage(patch_array):
                process_patch = "p"

            patch.save(directory_path / f"{y}_{x}_{process_patch}.png")