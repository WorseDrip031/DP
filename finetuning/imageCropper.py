from PIL import Image
from pathlib import Path


def crop_wsi(image_path:Path,
             output_path:Path,
             size:int):
    
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_path)
    
    cropped = image.crop((0,10000,size,size+10000))
    cropped.save(output_path)

    image.close()


