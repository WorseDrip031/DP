import cv2
import numpy as np
import matplotlib.pyplot as plt

TISSUE_THRESHOLD = 235

def visualize_tissue_mask(patch):
    """
    Converts a patch to grayscale, applies Otsu's thresholding, and visualizes the mask.
    
    Parameters:
        patch (numpy array): The 512x512 image patch.
    
    Returns:
        None (displays the images).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    grayscale_array = np.array(gray)
            
    # Count the tissue pixels (those with intensity below the threshold)
    tissue_pixels = np.sum(grayscale_array < TISSUE_THRESHOLD)
    total_pixels = 512 * 512
    tissue_percentage = tissue_pixels / total_pixels

    # Plot the original image and binary mask
    _ , ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(patch)
    ax[0].set_title("Original Patch")
    ax[0].axis("off")

    ax[1].imshow(gray, cmap="gray")
    if tissue_percentage >= 0.25:
        ax[1].set_title(f"Tissue: {tissue_percentage:.2f}")
    else:
        ax[1].set_title(f"Background: {tissue_percentage:.2f}")
    ax[1].axis("off")

    plt.show()

# Example usage
patch = cv2.imread("superpixel_experiments/input/n_1.png")  # Load a patch
patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB if using OpenCV
visualize_tissue_mask(patch)
