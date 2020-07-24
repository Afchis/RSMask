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

from utils.label_helper import ScoreLabelHelper


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="ignore/data/ytb_vos/train/", help = "data path")
args = parser.parse_args()

with open(args.data_path + "ytb.json") as data_file:
	train_json = json.load(data_file)


class TrainYtb(Dataset):
	def __init__(self):
		super().__init__()
		self.path = args.data_path
		self.json = train_json
		self.border = (0, 280, 0, 280)
		self.label_helper = ScoreLabelHelper()

	def __len__(self):
		return len(self.json)

	def target_choice(self, idx, search_anno):
		'''
		TODO:
		* nearly always target image is search image, because we take the condition: 
			'target image cant be smaller then search image.'
		'''
		target_anno = random.choice(self.json[idx]['object_anno'])
		if target_anno['bbox'] == None:
			target_anno = self.target_choice(idx, search_anno)
		if target_anno['bbox'][1] < search_anno['bbox'][1] or target_anno['bbox'][3] < search_anno['bbox'][3]:
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

	def pil_to_tensor(self, pil_img):
		_, size = pil_img.size
		size = int(round(size * 0.2)) # 0.2 because: 1280 / 256 = 0.2
		trans = transforms.Compose([
			transforms.Resize((size, size), interpolation=0),
			transforms.ToTensor()
		])
		return trans(pil_img) 

	def __getitem__(self, idx):
		'''
		TO DO:
		** !!! make score label !!!
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
		"""
		# build target with bounding box
		target_wb = target.copy()
		target_wb_drow = ImageDraw.Draw(target_wb)
		target_wb_drow.rectangle([target_anno['bbox'][0], target_anno['bbox'][1]+280, 
								  target_anno['bbox'][0]+target_anno['bbox'][2], 
								  target_anno['bbox'][1]+target_anno['bbox'][3]+280], 
								 fill=None, outline=(255, 0, 0), width=5)
		"""
		# build mask for search (RLE to gray scale:[0, 1] PIL.Image) 1x1280x1280
		mask = self.Rle_to_numpy(search_anno['segmentation']['counts'], search_anno['segmentation']['size'])
		mask = Image.fromarray(mask)
		mask = ImageOps.expand(mask, border=self.border)
		# transforms with save size ratio
		search = self.pil_to_tensor(search)
		target = self.pil_to_tensor(target)
		mask = self.pil_to_tensor(mask)
		# build score label
		score_label = self.label_helper.build_score_label(target, mask)
		return target, search, mask, score_label

def main():
	data = TrainYtb()
	target, search, mask, score_label = data[16]
	return score_label
print('TODO: DataLoader and show PIL.Images')

if __name__ == '__main__':
	score_label = main()
	print(score_label)
	