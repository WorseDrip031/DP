from tools.dataset import create_aggc_classification_dataset
dataset = create_aggc_classification_dataset()

image, mask = dataset[0]

print(len(dataset))

import torch
print(torch.cuda.is_available()) 