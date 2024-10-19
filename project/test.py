from tools.dataset import create_aggc_dataset
dataset = create_aggc_dataset()

image, mask = dataset[0]

print(len(dataset))

import torch
print(torch.cuda.is_available())