from tools.dataset import create_aggc_dataset
dataset = create_aggc_dataset()

image, mask = dataset[0]

print(len(dataset))

import torch
print(torch.cuda.is_available()) 


from pathlib import Path
DATASET_BASEPATH=Path(".scratch/data")
path = DATASET_BASEPATH / "AGGC-2022" / "train"
patches = sorted(list(path.rglob("*png")))
a = patches[0]
print(a)
print(a.parent.name)

if a.parent.name == "Subset1_Train_001":
    print("yes")