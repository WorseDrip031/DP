from pathlib import Path
from PIL import Image

print("Downasmpling analysis started...")

base_path = Path("inference/patching/temp")
image_path = base_path / "Subset1_Train_071" / "blocks_to_process.png"
max_size = 2000

Image.MAX_IMAGE_PIXELS = None
image = Image.open(image_path)
width, height = image.size
scale_factor = max_size / max(width, height)

print("Downsampling started...")

new_size = (int(width * scale_factor), int(height * scale_factor))
image = image.resize(new_size, Image.LANCZOS)

print("Downsampled wsi saving...")

new_file_path = image_path.parent / f"{image_path.stem}-downscaled.png"
image.save(new_file_path)