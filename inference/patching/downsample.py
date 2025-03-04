import json
from PIL import Image

def downsample_wsi(wsi, output_file_path, max_size=2000):

    print("Downasmpling analysis started...")

    if output_file_path.exists():
        wsi = Image.open(output_file_path)
        json_file_path = f"{output_file_path.with_suffix('')}.json"
        with open(json_file_path, "r") as f:
            data = json.load(f)
        scale_factor = data["scale_factor"]
        print("Downsampling complete...")
        return wsi, scale_factor

    width, height = wsi.size
    scale_factor = max_size / max(width, height)

    print("Downsampling started...")
    new_size = (int(width * scale_factor), int(height * scale_factor))
    wsi = wsi.resize(new_size, Image.LANCZOS)

    print("Downsampled wsi saving...")
    wsi.save(output_file_path)
    data = {
        "scale_factor": scale_factor
    }
    json_file_path = f"{output_file_path.with_suffix('')}.json"
    with open(json_file_path, "w") as f:
        json.dump(data, f, indent=1)

    return wsi, scale_factor