import json
from PIL import Image
from pathlib import Path
from typing import Tuple

def downsample_image(image:Image.Image,
                     output_file_path:Path,
                     save_scale_factor:bool=True,
                     max_size:int=2000
                     ) -> Tuple[Image.Image, float]:

    print("Downasmpling analysis started...")

    # If downsampled image already exists, simply load into memory
    if output_file_path.exists():
        image = Image.open(output_file_path)
        scale_factor = 1.0

        if save_scale_factor:
            json_file_path = output_file_path.parent / f"{output_file_path.stem}.json"
            if json_file_path.exists():
                with open(json_file_path, "r") as f:
                    data = json.load(f)
                scale_factor = data["scale_factor"]
        print("Downsampling complete...")

        return image, scale_factor

    # Analyse 
    width, height = image.size
    scale_factor = max_size / max(width, height)
    new_size = (int(width * scale_factor), int(height * scale_factor))

    print("Downsampling started...")

    image = image.resize(new_size, Image.LANCZOS)

    print("Downsampled wsi saving...")

    image.save(output_file_path)
    if save_scale_factor:
        data = {
            "scale_factor": scale_factor
        }
        json_file_path = output_file_path.parent / f"{output_file_path.stem}.json"
        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=1)

    return image, scale_factor