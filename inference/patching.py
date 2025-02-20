import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

TISSUE_THRESHOLD = 235

def preprocess_patches(wsi_file_path, patch_size=512, overlap_percentage=0.5, tissue_coverage=0.2):

    # Required to open WSI images of huge sizes
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(wsi_file_path)
    width, height = img.size
    overlap_size = int(patch_size * overlap_percentage)

    # Create new directory
    directory_path = Path("inference") / "temp" / f"{wsi_file_path.stem}-{height}-{width}-{overlap_size}"
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
            tissue_pixels = np.sum(grayscale_array < TISSUE_THRESHOLD)
            total_pixels = patch_size * patch_size
            
            # Determine if patch contains enough tissue to process further
            process_patch = "n"
            if tissue_pixels / total_pixels >= tissue_coverage:
                process_patch = "p"

            patch.save(directory_path / f"{y}_{x}_{process_patch}.png")
