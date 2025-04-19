from pathlib import Path

from grabCut import grabCutMask
from imageCropper import crop_wsi

sizes = [15000, 20000, 30000, 40000, 50000]

wsi_path = Path(".scratch") / "data" / "AGGC-2022-Unprepared" / "test" / "Subset1_Test_001.tiff"
mask_path = Path(".scratch") / "inference" / "Subset1_Test_001-512-0.5-0.5" / "masks" / "0007 - EVA02 Grouped Pretrained Frozne Downscale" / "Stroma_Mask.tif"

for size in sizes:

    wsi_crop_path = Path("finetuning") / "output" / f"{wsi_path.stem}_{size}.tiff"
    mask_crop_path = Path("finetuning") / "output" / f"Stroma_Mask_{size}.tif"

    print("Start wsi crop")
    crop_wsi(wsi_path, wsi_crop_path, size)
    print("Start mask crop")
    crop_wsi(mask_path, mask_crop_path, size)

    output_mask_path = Path("finetuning") / "output" / f"Stroma_Mask_{size}_finetuned.tif"
    print("Start grabCut")
    grabCutMask(mask_crop_path, wsi_crop_path, output_mask_path)
    print()
