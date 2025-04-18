from pathlib import Path

from .grabCut import grabCutMask
from .imageCropper import crop_wsi

sizes = [10000, 20000, 30000, 40000, 50000]



for size in sizes:




wsi_path = Path("experimentation") / "test" / "Subset1_Test_001.tiff"
wsi_crop_path = Path("experimentation") / "test" / "Subset1_Test_001_cropped.tiff"

mask_path = Path("experimentation") / "test" / "Stroma_Mask.tif"
mask_crop_path = Path("experimentation") / "test" / "Stroma_Mask_cropped.tif"

print("Start")
crop_wsi(wsi_path, wsi_crop_path, 10000)
print("Start2")
crop_wsi(mask_path, mask_crop_path, 10000)
print("End")

mask_path = Path("experimentation") / "test" / "Stroma_Mask_cropped.tif"
original_image_path = Path("experimentation") / "test" / "Subset1_Test_001_cropped.tiff"
output_mask_path = Path("experimentation") / "test" / "Stroma_Mask_cropped_finetuned.tif"