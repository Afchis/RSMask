import os
import json
import random
import argparse
import numpy as np

from PIL import Image, ImageDraw, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from .utils.label_helper import ScoreLabelHelper


class TrainYtb(Dataset):
	def __init__(self, data_path):
		super().__init__()
		self.path = data_path
		with open(data_path + "ytb.json") as data_file:
			self.json = json.load(data_file)
		self.border = (0, 280, 0, 280)
		self.label_helper = ScoreLabelHelper()

	def __len__(self):
		return len(self.json)

	def target_choice(self, idx, search_anno):
		target_anno = random.choice(self.json[idx]['object_anno'])
		if target_anno['bbox'] == None:
			target_anno = self.target_choice(idx, search_anno)
		return target_anno

	def crop_target(self, target, target_anno):
		[x, y, w, h] = target_anno['bbox']
		y = y + 280
		pic_w, pic_h = target.size
		center = [x + w/2, y + h/2]
		if w/16 < h/9:
			crop_size = h/2
		else:
			crop_size = w/2
		left = center[0] - crop_size
		top = center[1] - crop_size
		right = center[0] + crop_size
		bottom = center[1] + crop_size
		crop_target = target.crop((left, top, right, bottom))
		return crop_target

	def Rle_to_numpy(self, RLE, size):
		width, height = size[1], size[0]
		NOT_RLE = []
		try:
			for i, data in enumerate(RLE):
				if i % 2 == 0:
					x = 0
				else:
					x = 1
				for j in range(data):
					NOT_RLE.append(x)
			np_array = np.asarray(NOT_RLE)
			np_array = np_array.reshape(width, height).T#.tolist()
			np_array = np.uint8(np_array*255)
		except TypeError:
			np_array = np.zeros((width, height))
		return np_array

	def pil_to_tensor(self, pil_img, size=256):
		trans = transforms.Compose([
			transforms.Resize((size, size), interpolation=0),
			transforms.ToTensor()
		])
		return trans(pil_img) 

	def __getitem__(self, idx):
		'''
		TO DO:
		* Image.convert('RGB') if search and target gray scale.
		* ImageOps.expand(search, border=self.border) add gray border.
		'''
		# random Search img
		search_anno = random.choice(self.json[idx]['object_anno'])
		search = Image.open(self.path + 'JPEGImages/' + search_anno['file_name'])
		search = ImageOps.expand(search, border=self.border)
		# random not None Target img
		anno_check = search_anno
		target_anno = self.target_choice(idx, search_anno)
		target = Image.open(self.path + 'JPEGImages/' + target_anno['file_name'])
		target = ImageOps.expand(target, border=self.border)
		target = self.crop_target(target, target_anno)
		# build mask for search (RLE to gray scale:[0, 1] PIL.Image) 1x1280x1280
		if search_anno['object_size'] != None:
			mask = self.Rle_to_numpy(search_anno['segmentation']['counts'], search_anno['segmentation']['size'])
			mask = Image.fromarray(mask)
			mask = ImageOps.expand(mask, border=self.border)
		# transforms with save size ratio
		search = self.pil_to_tensor(search)
		target = self.pil_to_tensor(target)
		if search_anno['object_size'] != None:
			mask = self.pil_to_tensor(mask)
		else:
			mask = torch.zeros([1, 256, 256])
		# build score label
		score_labels = self.label_helper.BuildLabels(mask, search_anno['object_size'])
		return target, search, mask, score_labels


def Loader(data_path, batch_size, num_workers, shuffle=True):
	print("Initiate DataLoader")
	train_dataset = TrainYtb(data_path)
	train_loader = DataLoader(dataset=train_dataset,
							  batch_size=batch_size,
							  num_workers=num_workers,
							  shuffle=shuffle)
	print("Iters in epoch: ", len(train_loader))
	return train_loader


def main():
	data = TrainYtb()
	target, search, mask, score_label = data[16]
	return score_label
# print('dataloader.py TODO: DataLoader and show PIL.Images')


if __name__ == '__main__':
	main()

 	