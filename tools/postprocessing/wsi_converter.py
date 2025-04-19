import pyvips
from pathlib import Path


def convert_wsi_to_pyramidal(wsi_file_path:Path,
                             inference_folder:Path):
    
    print("Pyramidal WSI conversion analysis started...")

    output_filepath = inference_folder / f"{wsi_file_path.stem}_pyramidal.tiff"

    if output_filepath.exists():
        print("Pyramidal WSI conversion complete...")
        return
    
    # Load the large WSI file
    image = pyvips.Image.new_from_file(str(wsi_file_path), access="sequential")

    # Ensure the image is in RGB format
    if image.bands == 4:  # Drop alpha channel if present
        image = image[:3]
    elif image.interpretation not in ["srgb", "rgb"]:
        image = image.colourspace("srgb")

    # Save as pyramidal TIFF
    image.tiffsave(
        str(output_filepath),
        compression="jpeg",
        Q=100,
        tile=True,
        tile_width=512,
        tile_height=512,
        pyramid=True,
        bigtiff=True
    )

    print("Pyramidal WSI conversion complete...")