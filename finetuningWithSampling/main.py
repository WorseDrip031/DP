from pathlib import Path

from downsampleWSI import downsample
from downsampleMask import downsample_mask
from finetune import finetune_segmentation
from upsampleMask import upsample_mask

wsi_path = Path(".scratch") / "data" / "AGGC-2022-Unprepared" / "test" / "Subset1_Test_001.tiff"
mask_path = Path(".scratch") / "inference" / "Subset1_Test_001-512-0.5-0.5" / "masks" / "0007 - EVA02 Grouped Pretrained Frozne Downscale" / "Stroma_Mask.tif"

downsample_size = 20000
downsampled_wsi_path = Path("finetuningWithSampling") / "output" / f"{wsi_path.stem}_{downsample_size}.tiff"
downsampled_mask_path = Path("finetuningWithSampling") / "output" / f"{wsi_path.stem}_{downsample_size}.tiff"

downsampled_finetuned_mask_path = Path("finetuningWithSampling") / "output" / f"{wsi_path.stem}_{downsample_size}_df.tiff"
finetuned_mask_path = Path("finetuningWithSampling") / "output" / f"{wsi_path.stem}_{downsample_size}_f.tiff"

# Step 1: Downsample WSI
print("WSI downsampling started...")
downsample(wsi_path, downsampled_wsi_path, downsample_size)
print("WSI downsampling complete...")

# Step 2: Downsample Mask
print("Mask downsample started...")
height, width = downsample_mask(mask_path, downsampled_mask_path, downsample_size)
print("Mask downsample complete...")

# Step 3: Finetune segmentation
print("Segmentation finetuning started...")
finetune_segmentation(downsampled_wsi_path, downsampled_mask_path, downsampled_finetuned_mask_path)
print("Segmentation finetuning complete...")

# Step 4: Upsample Mask
print("Mask upsampling started...")
upsample_mask(downsampled_finetuned_mask_path, finetuned_mask_path, height, width)
print("Mask upsampling complete...")