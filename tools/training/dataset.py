import cv2
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable
import numpy as np
import torch
from PIL import Image
import math
from tqdm import tqdm
import random


DATASET_BASEPATH=Path(".scratch/data")


#************************#
#    Helper Functions    #
#************************#


random.seed(42)





class NumpyToTensor(Callable):
    def __call__(self, x):
        x = np.transpose(x, axes=(2,0,1))   # HWC -> CHW
        x = torch.from_numpy(x) / 255.0     # <0;255>UINT8 -> <0;1>
        return x.float()                    # cast as 32-bit float
    




def is_mask_fully_white(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')
        return all(pixel == 255 for pixel in img.getdata())





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





























#********************************#
#   AGGC Segmentation Dataset    #
#********************************#


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

        g3_img, g4_img, g5_img, normal_img, stroma_img = None, None, None, None, None

        try:
            g3_img = Image.open(g3_file)
        except:
            print(f"There is no G3 Mask for {file.stem}\n")

        try:
            g4_img = Image.open(g4_file)
        except:
            print(f"There is no G4 Mask for {file.stem}\n")

        try:
            g5_img = Image.open(g5_file)
        except:
            print(f"There is no G5 Mask for {file.stem}\n")

        try:
            normal_img = Image.open(normal_file)
        except:
            print(f"There is no Normal Mask for {file.stem}\n")

        try:
            stroma_img = Image.open(stroma_file)
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
    dataset_folder = DATASET_BASEPATH / "AGGC-2022-Patches"
    check_file = dataset_folder / ".done"
    if not check_file.exists():
        pre_process_dataset(dataset_folder)
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






























#********************************#
#  AGGC Classification Dataset   #
#********************************#


class AGGC2022ClassificationDataset(Dataset):
    def __init__(self, cfg, dataset_path: Path, image_transform, augmentation_transform):
        self.dataset_path = dataset_path
        self.image_transform = image_transform
        self.augmentation_transform = augmentation_transform
        self.cfg = cfg

        if self.cfg.gleason_handling == "Grouped":
            self.num_classes = 3
        else:
            self.num_classes = 5

        # Find the smallest group and sample randomly from each group accordingly
        self.normal_path = dataset_path / "normal"
        self.normal_patches = sorted(list(self.normal_path.rglob("*png")))

        self.stroma_path = dataset_path / "stroma"
        self.stroma_patches = sorted(list(self.stroma_path.rglob("*png")))

        self.g3_path = dataset_path / "g3"
        self.g3_patches = sorted(list(self.g3_path.rglob("*png")))

        self.g4_path = dataset_path / "g4"
        self.g4_patches = sorted(list(self.g4_path.rglob("*png")))

        self.g5_path = dataset_path / "g5"
        self.g5_patches = sorted(list(self.g5_path.rglob("*png")))

        if self.num_classes == 5:
            self.patch_lists = [self.normal_patches, self.stroma_patches, self.g3_patches, self.g4_patches, self.g5_patches]
            self.list_min_length = min(len(lst) for lst in self.patch_lists)

            # Quintiple the dataset if the option is chosen
            if self.cfg.model_architecture == "ViT" and self.cfg.vit_technique == "QuintupleCrop":
                self.files = []
                for _ in range(5):
                    files = []
                    for lst in self.patch_lists:
                        files.extend(random.sample(lst, self.list_min_length))
                    self.files.extend(files)
            else:
                self.files = []
                for lst in self.patch_lists:
                    self.files.extend(random.sample(lst, self.list_min_length))
        else:
            self.g_patches = self.g3_patches + self.g4_patches + self.g5_patches

            self.patch_lists = [self.normal_patches, self.stroma_patches, self.g_patches]
            self.list_min_length = min(len(lst) for lst in self.patch_lists)

            # Quintiple the dataset if the option is chosen
            if self.cfg.model_architecture == "ViT" and self.cfg.vit_technique == "QuintupleCrop":
                self.files = []
                for _ in range(5):
                    files = []
                    for lst in self.patch_lists:
                        files.extend(random.sample(lst, self.list_min_length))
                    self.files.extend(files)
            else:
                self.files = []
                for lst in self.patch_lists:
                    self.files.extend(random.sample(lst, self.list_min_length))

            
            

    def __len__(self):
        return len(self.files)
 

    def __getitem__(self, index):
        image_path = self.files[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if (self.cfg.model_architecture == "ViT" or self.cfg.model_architecture == "EVA02") and self.cfg.vit_technique == "Crop":
            # Define crop dimensions for a patch from the center
            height, width, _ = image.shape
            crop_size = 224
            if self.cfg.model_architecture == "EVA02":
                crop_size = 448
            start_x = (width - crop_size) // 2  # Starting x-coordinate for center crop
            start_y = (height - crop_size) // 2  # Starting y-coordinate for center crop
            image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]

        if (self.cfg.model_architecture == "ViT" or self.cfg.model_architecture == "EVA02") and self.cfg.vit_technique == "QuintupleCrop":
            batch_size = len(self.files) / 5
            order_of_batch = index // batch_size

            height, width, _ = image.shape
            crop_size = 224
            if self.cfg.model_architecture == "EVA02":
                crop_size = 448

            # Determine the starting x and y coordinates for cropping based on order_of_batch
            if order_of_batch == 0:  # Center crop
                start_x = (width - crop_size) // 2
                start_y = (height - crop_size) // 2
            elif order_of_batch == 1:  # Top-left corner
                start_x = 0
                start_y = 0
            elif order_of_batch == 2:  # Top-right corner
                start_x = width - crop_size
                start_y = 0
            elif order_of_batch == 3:  # Bottom-left corner
                start_x = 0
                start_y = height - crop_size
            elif order_of_batch == 4:  # Bottom-right corner
                start_x = width - crop_size
                start_y = height - crop_size
            else:
                raise ValueError("order_of_batch must be between 0 and 4")

            image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]

        label = torch.zeros(self.num_classes)
        if self.num_classes == 5:
            if image_path.parent.name == "normal":
                label[0] = 1.0
            elif image_path.parent.name == "stroma":
                label[1] = 1.0
            elif image_path.parent.name == "g3":
                label[2] = 1.0
            elif image_path.parent.name == "g4":
                label[3] = 1.0
            elif image_path.parent.name == "g5":
                label[4] = 1.0
        else:
            if image_path.parent.name == "normal":
                label[0] = 1.0
            elif image_path.parent.name == "stroma":
                label[1] = 1.0
            elif image_path.parent.name == "g3":
                label[2] = 1.0
            elif image_path.parent.name == "g4":
                label[2] = 1.0
            elif image_path.parent.name == "g5":
                label[2] = 1.0

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.augmentation_transform is not None:
            image = self.augmentation_transform(image)

        return image, label 





def pre_process_classification_subset(original_folder, target_folder):
    target_folder.mkdir(parents=True, exist_ok=True)
    normal_folder = target_folder / "normal"
    normal_folder.mkdir(parents=True, exist_ok=True)
    stroma_folder = target_folder / "stroma"
    stroma_folder.mkdir(parents=True, exist_ok=True)
    g3_folder = target_folder / "g3"
    g3_folder.mkdir(parents=True, exist_ok=True)
    g4_folder = target_folder / "g4"
    g4_folder.mkdir(parents=True, exist_ok=True)
    g5_folder = target_folder / "g5"
    g5_folder.mkdir(parents=True, exist_ok=True)
    files = sorted(list(original_folder.rglob("*.png")))

    # Remove every file which starts with "00000" as they are only for visualization
    files = [file for file in files if not file.name.startswith("00000")]

    # Divide the files into 6 categories:
    # 1. Files which end with "image.png" for images
    # 2. Files which end with "g3.png" for masks
    # 3. Files which end with "g4.png" for masks
    # 4. Files which end with "g5.png" for masks
    # 5. Files which end with "normal.png" for masks
    # 6. Files which end with "stroma.png" for masks
    images = sorted([file for file in files if file.name.endswith("image.png")])
    g3 = sorted([file for file in files if file.name.endswith("g3.png")])
    g4 = sorted([file for file in files if file.name.endswith("g4.png")])
    g5 = sorted([file for file in files if file.name.endswith("g5.png")])
    normal = sorted([file for file in files if file.name.endswith("normal.png")])
    stroma = sorted([file for file in files if file.name.endswith("stroma.png")])

    normal_count, stroma_count, g3_count, g4_count, g5_count = 0, 0, 0, 0, 0

    # Pre-process the dataset
    for i, _ in enumerate(tqdm(images)):
        img = Image.open(images[i])

        if is_mask_fully_white(normal[i]):
            file_path = normal_folder / f"{normal_count:05d}.png"
            img.save(file_path)
            normal_count += 1
        elif is_mask_fully_white(stroma[i]):
            file_path = stroma_folder / f"{stroma_count:05d}.png"
            img.save(file_path)
            stroma_count += 1
        elif is_mask_fully_white(g3[i]):
            file_path = g3_folder / f"{g3_count:05d}.png"
            img.save(file_path)
            g3_count += 1
        elif is_mask_fully_white(g4[i]):
            file_path = g4_folder / f"{g4_count:05d}.png"
            img.save(file_path)
            g4_count += 1
        elif is_mask_fully_white(g5[i]):
            file_path = g5_folder / f"{g5_count:05d}.png"
            img.save(file_path)
            g5_count += 1






def pre_process_classification_dataset(dataset_folder):
    dataset_folder.mkdir(parents=True, exist_ok=True)
    original_folder = DATASET_BASEPATH / "AGGC-2022-Patches"

    check_file = dataset_folder / "train" / ".done"
    if not check_file.exists():
        pre_process_classification_subset(original_folder / "train", dataset_folder / "train")
        check_file.touch()

    check_file = dataset_folder / "val" / ".done"
    if not check_file.exists():
        pre_process_classification_subset(original_folder / "val", dataset_folder / "val")
        check_file.touch()

    check_file = dataset_folder / "test" / ".done"
    if not check_file.exists():
        pre_process_classification_subset(original_folder / "test", dataset_folder / "test")
        check_file.touch()







def create_aggc_classification_dataset(cfg, type="train", **kwargs):

    # Pre-process dataset if neccessary
    dataset_folder = DATASET_BASEPATH / "AGGC-2022-Patches"
    check_file = dataset_folder / ".done"
    if not check_file.exists():
        pre_process_dataset(dataset_folder)
        check_file.touch()

    # Pre-process dataset if neccessary
    dataset_folder = DATASET_BASEPATH / "AGGC-2022-Classification"
    check_file = dataset_folder / ".done"
    if not check_file.exists():
        pre_process_classification_dataset(dataset_folder)
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
    
    return AGGC2022ClassificationDataset(cfg, images_folder, **kwargs)