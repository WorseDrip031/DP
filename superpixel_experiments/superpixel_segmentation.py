from PIL import Image
from pathlib import Path
import cv2
import numpy as np
from pathlib import Path
from skimage.color import rgb2hsv
from skimage.filters import median
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.util import img_as_ubyte
from skimage.segmentation import felzenszwalb


DATASET_BASEPATH=Path(".scratch/data")

wsi_file_path = DATASET_BASEPATH / "AGGC-2022-Unprepared" / "train" / "Subset1_Train_075.tiff"

Image.MAX_IMAGE_PIXELS = None
img = Image.open(wsi_file_path)

width, height = img.size
scale_factor = 2000 / max(width, height)

if scale_factor < 1:
    new_size = (int(width * scale_factor), int(height * scale_factor))
    print("Downsampling started...")
    wsi = img.resize(new_size, Image.LANCZOS)
    wsi.save("superpixel_experiments/temp/75-downscaled.png")
    print("Downsampling finished")
else:
    print("Image is already smaller than the target size. No resizing needed.")

img.close()


# Read image
image = cv2.imread("superpixel_experiments/temp/75-downscaled.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert RGB to HSV and use the saturation channel
hsv_image = rgb2hsv(image)
saturation_channel = hsv_image[:, :, 1]

# Apply median & entopy filter
filtered_image = median(saturation_channel, disk(5))
entropy_image = entropy(img_as_ubyte(filtered_image), disk(3))

# Use superpixel segmentation
segments = felzenszwalb(entropy_image, scale=50, sigma=0.3, min_size=50)

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

cv2.imwrite("superpixel_experiments/temp/75-segmented.png", output_image)