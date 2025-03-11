import numpy as np
from PIL import Image
from pathlib import Path
from skimage.color import rgb2hsv
from skimage.filters import median
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.util import img_as_ubyte
from skimage.segmentation import felzenszwalb
from skimage.io import imsave, imread


def segment_tissue(dowsnampled_wsi:Image.Image,
                   segmented_wsi_path:Path
                   ) -> Image.Image:
    
    print("Tissue segmentation analysis started...")

    if segmented_wsi_path.exists():
        output_image = imread(segmented_wsi_path).astype(np.uint8)
        print("Tissue segmentation complete...")
        return output_image

    # Convert RGB to HSV and use the saturation channel
    wsi_array = np.array(dowsnampled_wsi)
    hsv_wsi = rgb2hsv(wsi_array)
    saturation_channel = hsv_wsi[:, :, 1]

    # Apply median & entopy filter
    filtered_wsi = median(saturation_channel, disk(5))
    entropy_wsi = entropy(img_as_ubyte(filtered_wsi), disk(3))

    print("Tissue segmentation started...")

    # Use superpixel segmentation
    segments = felzenszwalb(entropy_wsi, scale=50, sigma=0.3, min_size=50)

    # Threshold Superpixels
    T = np.mean(filtered_wsi)
    output_image = np.zeros_like(filtered_wsi)
    for label in np.unique(segments):
        mask = segments == label
        if np.mean(filtered_wsi[mask]) > T:
            output_image[mask] = 255
        else:
            output_image[mask] = 0

    print("Tissue segmented wsi saving...")
    
    # Save the segmentation mask
    imsave(segmented_wsi_path, output_image.astype(np.uint8))
    return output_image