import torch


class ScoreLabelHelper():
	def __init__(self):
		self.search_size = 256
		self.search_p3 = 32

	def round_up(self, num, div):
		return int(num / div) + int(num % div > 0)

	def backbone_outs(self, size):
		p0 = self.round_up(size, 2)
		p1 = self.round_up(p0, 2)
		p2 = self.round_up(p1, 2)
		p3 = p2
		return p0, p1, p2, p3

	def corr_feat_size(self, target):
		target_size = target.size(2)
		_, _, _, p3 = self.backbone_outs(target_size)
		return p3

	def conv_for_mask(self, target, mask):
		'''
		kernel size = target size
		'''
		p3 = self.corr_feat_size(target)
		delta = self.search_size - target.size(2)
		stride = delta / (p3 - 1)
		j_tensors_list = list()
		for j in range(p3):
			j_tensor = torch.tensor([])
			for i in range(p3):
				ij = mask[:, int(j*stride):int(j*stride+target.size(2)), int(i*stride):int(i*stride+target.size(2))].sum(dim=1).sum(dim=1)
				j_tensor = torch.cat([j_tensor, ij], dim=0)
			j_tensors_list.append(j_tensor)
		out_tensor = torch.stack(j_tensors_list).unsqueeze(0)
		return out_tensor


	def build_score_label(self, target, mask):
		out_tensor = self.conv_for_mask(target, mask)
		max_value = out_tensor.max()
		out_tensor = out_tensor / max_value
		ones = (out_tensor == 1.).float()
		half = (out_tensor >= 0.6).float()
		score_label = (ones + half) / 2
		return score_label

if __name__ == '__main__':
	target = torch.rand([3, 128, 128])
	search = torch.rand([3, 256, 256])
	mask = torch.rand([1, 256, 256])
	score_helper = ScoreLabelHelper()
	out = score_helper.build_score_label(target, mask)
	print(out)
	