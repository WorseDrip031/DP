import os
import numpy as np
from PIL import Image
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None

PATCH_FOLDER = "inference/output/Subset1_Train_075-256-tresh-101-10"

# Step 1: Determine WSI Size
max_x, max_y = 0, 0
patches = []

for filename in os.listdir(PATCH_FOLDER):
    if filename.endswith(".png"):
        parts = filename[:-4].split("_")
        y, x, patch_type = int(parts[0]), int(parts[1]), parts[2]
        patches.append((y, x, patch_type, filename))
        max_x = max(max_x, x)
        max_y = max(max_y, y)

PATCH_SIZE = 512
OVERLAP_SIZE = PATCH_SIZE // 2

WSI_WIDTH = max_x + PATCH_SIZE
WSI_HEIGHT = max_y + PATCH_SIZE

RED_WIDTH = WSI_WIDTH // OVERLAP_SIZE
RED_HEIGHT = WSI_HEIGHT // OVERLAP_SIZE

# Step 2: Reconstruct the WSI with blending
wsi_image = Image.new("RGB", (WSI_WIDTH, WSI_HEIGHT), (0, 0, 0))
red_regions = np.zeros((RED_HEIGHT, RED_WIDTH), dtype=np.uint8)

for y, x, patch_type, filename in tqdm(patches, desc="Processing patches"):
    patch_path = os.path.join(PATCH_FOLDER, filename)
    patch = Image.open(patch_path).convert("RGB")
    wsi_image.paste(patch, (x, y))

    # If patch type is 'p', increase red overlay intensity
    if patch_type == 'p':
        r_y = y // OVERLAP_SIZE
        r_x = x // OVERLAP_SIZE
        red_regions[r_y:r_y+2, r_x:r_x+2] += 1

# Step 3: Create red transparency overlay
max_red_value = np.max(red_regions)
red_regions = (red_regions / max_red_value * 255).astype(np.uint8)
red_mask = Image.new("RGBA", (WSI_WIDTH, WSI_HEIGHT), (0, 0, 0, 0))

for i in tqdm(range(len(red_regions)), desc="Generating red overlay"):
    for j in range(len(red_regions[i])):
        if red_regions[i, j] > 0:
            red_patch = Image.new("RGBA", (OVERLAP_SIZE, OVERLAP_SIZE), (255, 0, 0, red_regions[i][j]))
            red_mask.paste(red_patch, (j*OVERLAP_SIZE, i*OVERLAP_SIZE), red_patch)

wsi_image = wsi_image.convert('RGBA')
wsi_image = Image.alpha_composite(wsi_image, red_mask)

# Save final reconstructed image
wsi_image.save("inference/output/75-101-10.png")
print("WSI reconstruction complete. Saved as reconstructed_wsi.png")
