import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.color import rgb2hsv
from skimage.filters import median
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.util import img_as_ubyte
from skimage.segmentation import felzenszwalb


def calculate_tissue_percentage(wsi_file_path, output_directory_path, scale, sigma, min_size):

    # Read image
    image = cv2.imread(wsi_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert RGB to HSV and use the saturation channel
    hsv_image = rgb2hsv(image)
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
    tissue_percentage = (white_pixels / total_pixels) * 100

    # Save the resulting segmentation mask
    image_path = output_directory_path / f"{scale}_{sigma}_{min_size}_{tissue_percentage:.2f}.png"
    cv2.imwrite(image_path, output_image)


##### Main #####
input_folder_path = Path(".archived/superpixel_experiments/input")
files = sorted(list(input_folder_path.rglob("*.png")))

scales = [10, 50, 100, 500, 1000]
sigmas = [0.1, 0.25, 0.5, 0.75, 0.9]
min_sizes = [10, 50, 100, 500, 1000]

for file in tqdm(files, desc="Processing Files", unit="file"):
    directory_path = Path(".archived/superpixel_experiments/output") / file.stem
    directory_path.mkdir(parents=True, exist_ok=True)

    for scale in tqdm(scales, desc="Scale", unit="scale", leave=False):
        for sigma in sigmas:
            for min_size in min_sizes:
                calculate_tissue_percentage(file, directory_path, scale, sigma, min_size)