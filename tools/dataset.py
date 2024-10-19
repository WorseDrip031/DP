import urllib.request
import zipfile
import cv2
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as tnf
from PIL import Image
import math


DATASET_BASEPATH=Path(".scratch/data")
ANIMALS_DOWNLOAD_URL="https://vggnas.fiit.stuba.sk/download/datasets/animals/animals.zip"


class NumpyToTensor(Callable):
    def __call__(self, x):
        x = np.transpose(x, axes=(2,0,1))   # HWC -> CHW
        x = torch.from_numpy(x) / 255.0     # <0;255>UINT8 -> <0;1>
        return x.float()                    # cast as 32-bit float


class ClassToOneHot(Callable):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, x):
        y = torch.LongTensor([x])[0]
        y = tnf.one_hot(y, num_classes=self.num_classes)
        return y.float()


class AnimalsDataset(Dataset):
    def __init__(self, subset_path: Path, image_transform):
        self.class_names = {}
        self.files = sorted(list(subset_path.rglob("*.jpg")))
        self.image_transform = image_transform

        for filepath in self.files:
            class_name = filepath.parent.name
            if not class_name in self.class_names:
                self.class_names[class_name] = len(self.class_names)

        self.num_classes = len(self.class_names)
        self.label_transform = ClassToOneHot(self.num_classes)


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        file_path = self.files[index]
        class_name = file_path.parent.name
        label_class = self.class_names[class_name]

        image = self.load_image(file_path)
        if self.image_transform is not None:
            image = self.image_transform(image)

        label = self.label_transform(label_class)
        return image, label
    

    def load_image(self, file_path):
        image = cv2.imread(file_path.as_posix(), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


def download_file(url, local_file):
    try:
        print("Downloading file: ", url)
        urllib.request.urlretrieve(url, local_file)
        return True
    except:
        print("Error downloading: ", url)
        return False
    

def extract_archive(file_path):
    print(f"Extracting {file_path.as_posix()}...")
    if (file_path.suffix == ".zip"):
        target_dir = file_path.parent
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)


def create_animals_dataset(is_train=True, **kwargs):
    local_dataset_folder = DATASET_BASEPATH / "animals"
    local_dataset_file = DATASET_BASEPATH / "animals.zip"
    check_file = local_dataset_folder / ".done"
    if not check_file.exists():
        ok = download_file(ANIMALS_DOWNLOAD_URL, local_dataset_file.as_posix())
        if not ok:
            return None
        extract_archive(local_dataset_file)
        check_file.touch()
    
    if is_train:
        image_fodler = local_dataset_folder / "train"
    else:
        image_fodler = local_dataset_folder / "val"

    return AnimalsDataset(image_fodler, **kwargs)










class AGGC2022Dataset(Dataset):
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.files = sorted(list(dataset_path.rglob("*.png")))

        # Remove every file which starts with "00000" as they are only for visualization
        self.files = [file for file in self.files if not file.name.startswith("00000")]

        # Divide the files into 6 categories:
        # 1. Files which end with "image.png" for images
        # 2. Files which end with "g3.png" for masks
        # 3. Files which end with "g4.png" for masks
        # 4. Files which end with "g5.png" for masks
        # 5. Files which end with "normal.png" for masks
        # 6. Files which end with "stroma.png" for masks
        self.images = sorted([file for file in self.files if file.name.endswith("image.png")])
        self.g3 = sorted([file for file in self.files if file.name.endswith("g3.png")])
        self.g4 = sorted([file for file in self.files if file.name.endswith("g4.png")])
        self.g5 = sorted([file for file in self.files if file.name.endswith("g5.png")])
        self.normal = sorted([file for file in self.files if file.name.endswith("normal.png")])
        self.stroma = sorted([file for file in self.files if file.name.endswith("stroma.png")])


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        image_path = self.images[index]
        g3_path = self.g3[index]
        g4_path = self.g4[index]
        g5_path = self.g5[index]
        normal_path = self.normal[index]
        stroma_path = self.stroma[index]

        image = np.array(Image.open(image_path).convert("RGB"))
        g3_mask = np.array(Image.open(g3_path).convert("L"))
        g4_mask = np.array(Image.open(g4_path).convert("L"))
        g5_mask = np.array(Image.open(g5_path).convert("L"))
        normal_mask = np.array(Image.open(normal_path).convert("L"))
        stroma_mask = np.array(Image.open(stroma_path).convert("L"))

        mask = np.stack([stroma_mask, normal_mask, g3_mask, g4_mask, g5_mask], axis=-1)

        return image, mask 


def contains_mask(image, mask_percent):
    if image is None:
        return False
    pixels = list(image.getdata())
    total_pixels = len(pixels)
    count_255 = pixels.count(255)
    return count_255 >= (mask_percent * total_pixels)


def save_annotation(patch, patch_size, sub_folder, name, patch_num):
    if patch is None:
        # Create an empty annotation 
        patch = Image.new('L', (patch_size, patch_size), 0)
    patch_file = sub_folder / f"{patch_num:05d}_{name}.png"
    patch.save(patch_file)


def pre_process_subset(original_folder, target_folder):
    target_folder.mkdir(parents=True, exist_ok=True)
    files = sorted(list(original_folder.rglob("*.tiff")))

    # Hyperparameters of pre-processing
    Image.MAX_IMAGE_PIXELS = None
    overlap = 0.5
    patch_size = 512
    mask_percent = 0.75

    # For every ".tiff" file in the subset create patches
    for file in files:

        # Create sub-folder for a concrete ".tiff" image
        sub_folder = target_folder / file.stem
        sub_folder.mkdir(parents=True, exist_ok=True)

        # Open ".tiff" image and also it's annotations
        annotations_folder = original_folder / file.stem
        g3_file = annotations_folder / "G3_Mask.tif"
        g4_file = annotations_folder / "G4_Mask.tif"
        g5_file = annotations_folder / "G5_Mask.tif"
        normal_file = annotations_folder / "Normal_Mask.tif"
        stroma_file = annotations_folder / "Stroma_Mask.tif"

        print(f"\nOpening {file.stem}.tiff\n")
        img = Image.open(file)

        # Create a downsized version and save it
        img_copy = img.copy()
        size = 2048, 2048
        img_copy.thumbnail(size)
        img_copy.save(sub_folder / "00000_image.png")

        g3_img, g4_img, g5_img, normal_img, stroma_img = None, None, None, None, None

        try:
            g3_img = Image.open(g3_file)
            img_copy = g3_img.copy()
            img_copy.thumbnail(size)
            img_copy.save(sub_folder / "00000_g3.png")
        except:
            print(f"There is no G3 Mask for {file.stem}\n")

        try:
            g4_img = Image.open(g4_file)
            img_copy = g4_img.copy()
            img_copy.thumbnail(size)
            img_copy.save(sub_folder / "00000_g4.png")
        except:
            print(f"There is no G4 Mask for {file.stem}\n")

        try:
            g5_img = Image.open(g5_file)
            img_copy = g5_img.copy()
            img_copy.thumbnail(size)
            img_copy.save(sub_folder / "00000_g5.png")
        except:
            print(f"There is no G5 Mask for {file.stem}\n")

        try:
            normal_img = Image.open(normal_file)
            img_copy = normal_img.copy()
            img_copy.thumbnail(size)
            img_copy.save(sub_folder / "00000_normal.png")
        except:
            print(f"There is no Normal Mask for {file.stem}\n")

        try:
            stroma_img = Image.open(stroma_file)
            img_copy = stroma_img.copy()
            img_copy.thumbnail(size)
            img_copy.save(sub_folder / "00000_stroma.png")
        except:
            print(f"There is no Stroma Mask for {file.stem}\n")

        # Pre-compute the possible number of patches along width and height
        width, height = img.size
        step = int(patch_size * (1 - overlap))

        num_patches_x = math.floor((width - patch_size) / step) + 1
        num_patches_y = math.floor((height - patch_size) / step) + 1
        total_patches = num_patches_x * num_patches_y

        # Loop through the image and create patches
        patch_num = 0
        created_patches = 0
        for y in range(0, height - patch_size + 1, step):
            for x in range(0, width - patch_size + 1, step):

                # Define the box to crop the patch (left, upper, right, lower)
                box = (x, y, x + patch_size, y + patch_size)
                
                # Crop the patch from the image and it's annotations
                g3_patch, g4_patch, g5_patch, normal_patch, stroma_patch = None, None, None, None, None
                patch = img.crop(box)
                if g3_img is not None:
                    g3_patch = g3_img.crop(box)
                if g4_img is not None:
                    g4_patch = g4_img.crop(box)
                if g5_img is not None:
                    g5_patch = g5_img.crop(box)
                if normal_img is not None:
                    normal_patch = normal_img.crop(box)
                if stroma_img is not None:
                    stroma_patch = stroma_img.crop(box)

                # If at least one annotation contains a predefined percentage of mask pixels, save the patch
                if contains_mask(g3_patch, mask_percent) or contains_mask(g4_patch, mask_percent) or contains_mask(g5_patch, mask_percent) or contains_mask(normal_patch, mask_percent) or contains_mask(stroma_patch, mask_percent):
                    created_patches += 1
                    patch_file = sub_folder / f"{created_patches:05d}_image.png"
                    patch.save(patch_file)
                    save_annotation(g3_patch, patch_size, sub_folder, "g3", created_patches)
                    save_annotation(g4_patch, patch_size, sub_folder, "g4", created_patches)
                    save_annotation(g5_patch, patch_size, sub_folder, "g5", created_patches)
                    save_annotation(normal_patch, patch_size, sub_folder, "normal", created_patches)
                    save_annotation(stroma_patch, patch_size, sub_folder, "stroma", created_patches)
                
                # Track and report progress periodically
                if patch_num % 100 == 0:
                    print(f"{file.stem} - Creating patches: {patch_num} / {total_patches}   ->   {(patch_num/total_patches*100):.2f}%")

                patch_num += 1
        
        # Closing images to prevent memory leak
        if img is not None:
            img.close()
        if g3_img is not None:
            g3_img.close()
        if g4_img is not None:
            g4_img.close()
        if g5_img is not None:
            g5_img.close()
        if normal_img is not None:
            normal_img.close()
        if stroma_img is not None:
            stroma_img.close()


def pre_process_dataset(dataset_folder):
    dataset_folder.mkdir(parents=True, exist_ok=True)
    original_folder = DATASET_BASEPATH / "AGGC-2022-Unprepared"

    check_file = dataset_folder / "train" / ".done"
    if not check_file.exists():
        pre_process_subset(original_folder / "train", dataset_folder / "train")
        check_file.touch()

    check_file = dataset_folder / "val" / ".done"
    if not check_file.exists():
        pre_process_subset(original_folder / "val", dataset_folder / "val")
        check_file.touch()

    check_file = dataset_folder / "test" / ".done"
    if not check_file.exists():
        pre_process_subset(original_folder / "test", dataset_folder / "test")
        check_file.touch()


def create_aggc_dataset(type="train", **kwargs):

    # Pre-process dataset if neccessary
    dataset_folder = DATASET_BASEPATH / "AGGC-2022"
    check_file = dataset_folder / ".done"
    if not check_file.exists():
        #pre_process_dataset(dataset_folder)
        check_file.touch()

    # Create dataset from folder
    if type == "train":
        images_folder = dataset_folder / "train"
    elif type == "val":
        images_folder = dataset_folder / "val"
    elif type == "test":
        images_folder = dataset_folder / "test"
    else:
        return None
    
    return AGGC2022Dataset(images_folder)