from tools.dataset import create_aggc_dataset
dataset = create_aggc_dataset()

import torch
print(torch.cuda.is_available())