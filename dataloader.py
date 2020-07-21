import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

ytb_train = os.listdir()

train_path = "ignore/data/ytb_vos/train/"

with open(train_path + "train.json") as data_file:    
	train_json = json.load(data_file)

print(len(train_json))