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


wsi_path = Path("experimentation") / "test" / "Subset1_Test_001.tiff"
wsi_crop_path = Path("experimentation") / "test" / "Subset1_Test_001_cropped.tiff"

mask_path = Path("experimentation") / "test" / "Stroma_Mask.tif"
mask_crop_path = Path("experimentation") / "test" / "Stroma_Mask_cropped.tif"

print("Start")
crop_wsi(wsi_path, wsi_crop_path, 10000)
print("Start2")
crop_wsi(mask_path, mask_crop_path, 10000)
print("End")