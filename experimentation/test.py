import numpy as np
import tifffile as tiff
from PIL import Image

def save_mask_as_png(mask, filename="experimentation/binary_mask.png"):
    img = Image.fromarray(mask)
    img.save(filename, optimize=True)

def create_circular_mask(height, width, fill_ratio=0.25):
    mask = np.zeros((height, width), dtype=np.uint8)
    
    center_x, center_y = width // 2, height // 2
    total_pixels = height * width
    circle_radius = int((fill_ratio * total_pixels / np.pi) ** 0.5)
    
    y, x = np.ogrid[:height, :width]
    mask_area = (x - center_x) ** 2 + (y - center_y) ** 2 <= circle_radius ** 2
    
    mask[mask_area] = 255
    return mask

def save_mask_as_tiff(mask, filename="experimentation/binary_mask_deflate.tif"):
    tiff.imwrite(filename, mask, compression="deflate") 

if __name__ == "__main__":
    height, width = 50000, 40000
    mask = create_circular_mask(height, width)
    save_mask_as_tiff(mask)
    print("Binary mask saved as binary_mask")