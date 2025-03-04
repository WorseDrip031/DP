import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def visualize_regions(folder, patch_size, overlap_percentage, modes):
    
    if all((folder.parent / f"{mode}.png").exists() for mode in modes):
        print("Visualization complete...")
        return
    
    # Step 1: Determine WSI Size
    max_x, max_y = 0, 0
    patches = []

    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            parts = filename[:-4].split("_")
            y, x, patch_type = int(parts[0]), int(parts[1]), parts[2]
            patches.append((y, x, patch_type, filename))
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    overlap_size = int(patch_size * overlap_percentage)

    wsi_width = max_x + patch_size
    wsi_height = max_y + patch_size

    map_width = wsi_width // overlap_size
    map_height = wsi_height // overlap_size

    # Step 2: Reconstruct the WSI with blending
    wsi_image = Image.new("RGB", (wsi_width, wsi_height), (0, 0, 0))
    normal_regions = np.zeros((map_height, map_width), dtype=np.uint8)
    stroma_regions = np.zeros((map_height, map_width), dtype=np.uint8)
    gleason_regions = np.zeros((map_height, map_width), dtype=np.uint8)

    for y, x, patch_type, filename in tqdm(patches, desc="Processing patches"):
        patch_path = os.path.join((folder), filename)
        patch = Image.open(patch_path).convert("RGB")
        wsi_image.paste(patch, (x, y))

        if patch_type != 'n':
            m_y = y // overlap_size
            m_x = x // overlap_size

            if patch_type == "normal":
                normal_regions[m_y:m_y+2, m_x:m_x+2] += 1
            if patch_type == "stroma":
                stroma_regions[m_y:m_y+2, m_x:m_x+2] += 1
            if patch_type == "gleason":
                gleason_regions[m_y:m_y+2, m_x:m_x+2] += 1

    # Step 3: Create red transparency overlay
    if "normal" in modes:
        normal_overlay = np.zeros((wsi_height, wsi_width, 4), dtype=np.uint8)
        normal_overlay[:, :] = [255, 0, 0, 128]
        normal_mask = Image.fromarray(normal_overlay)

    if "stroma" in modes:
        stroma_overlay = np.zeros((wsi_height, wsi_width, 4), dtype=np.uint8)
        stroma_overlay[:, :] = [0, 0, 255, 128]  # Blue with transparency
        stroma_mask = Image.fromarray(stroma_overlay)

    if "gleason" in modes:
        gleason_overlay = np.zeros((wsi_height, wsi_width, 4), dtype=np.uint8)
        gleason_overlay[:, :] = [0, 255, 0, 128]  # Green with transparency
        gleason_mask = Image.fromarray(gleason_overlay)

    if "all" in modes:
        # Combine overlays for the "all" mode, ensuring the alpha channel does not exceed 255
        combined_overlay = np.zeros((wsi_height, wsi_width, 4), dtype=np.uint8)

        # Red + Blue = Magenta (255, 0, 255)
        combined_overlay[normal_regions > 0] += [255, 0, 0, 0]  # Red
        combined_overlay[stroma_regions > 0] += [0, 0, 255, 0]  # Blue
        combined_overlay[gleason_regions > 0] += [0, 255, 0, 0]  # Green

        # After blending the RGB channels, set the alpha to 128
        combined_overlay[:, :, 3] = np.minimum(combined_overlay[:, :, 3] + 128, 255)  # Ensure alpha doesn't exceed 255

        combined_mask = Image.fromarray(combined_overlay)

    # Step 4: Apply overlays and save images for selected modes
    print("Processing block visualization saving...")
    wsi_image = wsi_image.convert('RGBA')

    # Save overlay for each mode if selected
    if "normal" in modes:
        normal_wsi = Image.alpha_composite(wsi_image, normal_mask)
        normal_wsi.save(folder / "normal.png")
    
    if "stroma" in modes:
        stroma_wsi = Image.alpha_composite(wsi_image, stroma_mask)
        stroma_wsi.save(folder / "stroma.png")
    
    if "gleason" in modes:
        gleason_wsi = Image.alpha_composite(wsi_image, gleason_mask)
        gleason_wsi.save(folder / "gleason.png")
    
    if "all" in modes:
        all_wsi = Image.alpha_composite(wsi_image, combined_mask)
        all_wsi.save(folder / "all.png")

    print("Visualization complete.")