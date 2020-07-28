import torch
import torch.nn.functional as F


class ScoreLabelHelper():
	'''
	TODO:
	* Make more competemt score builder
	'''
	def __init__(self):
		self.Lpad = 256/4
		self.Lcorr= 5
		self.Lstr = (2*self.Lpad)/(5-1)
		self.Mpad = 256/8
		self.Mcorr= 17
		self.Mstr = (256+2*self.Mpad-128)/(17-1)
		self.Spad = 0
		self.Scorr= 25
		self.Sstr = (256-64)/(25-1)

	def _size_list_(self, mask, size):
		'''
		Size_list --> [large, medium, small]
		'''
		if size == None:
			size_list = [0, 0, 0]
		elif size == 'large':
			size_list = [1, 0, 0]
		elif size == 'medium':
			size_list = [0, 1, 0]
		elif size == 'small':
			size_list = [0, 0, 1]
		else:
			print('SizeError: Wrong size in json file!')
			quit()
		return size_list

	def _norm_label_(self, x):
		max_value = x.max()
		out = x / max_value
		ones = (out == 1.).float()
		half = (out >= 0.6).float()
		out = (ones + half) / 2
		return out

	def _build_label_(self, mask, size, corr_size, pad, stride):
		mask = F.pad(input=mask, pad=(pad, pad, pad, pad), mode='constant', value=0)
		j_tensors_list = list()
		for j in range(corr_size):
			j_tensor = torch.tensor([])
			for i in range(corr_size):
				ij = mask[:, int(j*stride):int(j*stride+size), int(i*stride):int(i*stride+size)].sum(dim=1).sum(dim=1)
				j_tensor = torch.cat([j_tensor, ij], dim=0)
			j_tensors_list.append(j_tensor)
		out_tensor = torch.stack(j_tensors_list)
		out_tensor = self._norm_label_(out_tensor)
		print(out_tensor.shape)
		return out_tensor

	def BuildLabels(self, mask, size):
		s_l = self._size_list_(mask, size)
		large_score = self._build_label_(mask, size=256, corr_size=self.Lcorr, pad=int(self.Lpad), stride=self.Lstr)*s_l[0]
		medium_score = self._build_label_(mask, size=128, corr_size=self.Mcorr, pad=int(self.Mpad), stride=self.Mstr)*s_l[1]
		small_score = self._build_label_(mask, size=64, corr_size=self.Scorr, pad=int(self.Spad), stride=self.Sstr)*s_l[2]
		score_dict = {
			'large' : large_score,
			'medium' : medium_score,
			'small' : small_score
		}
		return score_dict




if __name__ == '__main__':
	target = torch.rand([3, 256, 256])
	search = torch.rand([3, 256, 256])
	mask = torch.rand([1, 256, 256])
	score_helper = ScoreLabelHelper()
	out = score_helper.BuildLabels(mask, size='large')
	print(out['small'])

