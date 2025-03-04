import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def visualize_blocks_to_process(folder, patch_size, overlap_percentage):
    
    print("Processing block visualization analysis started...")
    if (folder / "blocks_to_process.png").exists():
        print("Processing block visualization complete...")
        return

    # Step 1: Determine WSI Size
    max_x, max_y = 0, 0
    patches = []

    for filename in os.listdir(folder / "patches"):
        if filename.endswith(".png"):
            parts = filename[:-4].split("_")
            y, x, patch_type = int(parts[0]), int(parts[1]), parts[2]
            patches.append((y, x, patch_type, filename))
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    overlap_size = int(patch_size * overlap_percentage)

    wsi_width = max_x + patch_size
    wsi_height = max_y + patch_size

    red_width = wsi_width // overlap_size
    red_height = wsi_height // overlap_size

    # Step 2: Reconstruct the WSI with blending
    wsi_image = Image.new("RGB", (wsi_width, wsi_height), (0, 0, 0))
    red_regions = np.zeros((red_height, red_width), dtype=np.uint8)

    for y, x, patch_type, filename in tqdm(patches, desc="Processing patches"):
        patch_path = os.path.join((folder / "patches"), filename)
        patch = Image.open(patch_path).convert("RGB")
        wsi_image.paste(patch, (x, y))

        # If patch type is 'p', increase red overlay intensity
        if patch_type == 'p':
            r_y = y // overlap_size
            r_x = x // overlap_size
            red_regions[r_y:r_y+2, r_x:r_x+2] += 1

    # Step 3: Create red transparency overlay
    max_red_value = np.max(red_regions) + 1
    red_regions = (red_regions / max_red_value * 255).astype(np.uint8)
    red_mask = Image.new("RGBA", (wsi_width, wsi_height), (0, 0, 0, 0))

    for i in tqdm(range(len(red_regions)), desc="Generating red overlay"):
        for j in range(len(red_regions[i])):
            if red_regions[i, j] > 0:
                red_patch = Image.new("RGBA", (overlap_size, overlap_size), (255, 0, 0, red_regions[i][j]))
                red_mask.paste(red_patch, (j*overlap_size, i*overlap_size), red_patch)

    print("Processing block visualization saving...")
    wsi_image = wsi_image.convert('RGBA')
    wsi_image = Image.alpha_composite(wsi_image, red_mask)

    # Save final reconstructed image
    wsi_image.save((folder / "blocks_to_process.png"))