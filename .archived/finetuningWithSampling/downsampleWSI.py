from pathlib import Path
from PIL import Image


def downsample(image_path:Path,
               output_path:Path,
               max_size:int):
    
    if output_path.exists():
        return
    
    # Open image
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_path)

    # Determine new size
    width, height = image.size
    scale_factor = max_size / max(width, height)
    new_size = (int(width * scale_factor), int(height * scale_factor))

    # Downsample
    downsampled_image = image.resize(new_size, Image.LANCZOS)
    downsampled_image.save(output_path)
    downsampled_image.close()
    image.close()
