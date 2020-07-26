import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50 


class Backbone(nn.Module):
	def __init__(self):
		super(Backbone, self).__init__()
		self.backbone = resnet50(pretrained=True)
		self.adjust = nn.Conv2d(1024, 256, kernel_size=1)

	def forward(self, x):
		_, _, _, p3 = self.backbone(x)
		p3 = self.adjust(p3)
		return p3

	def forward_sharp(self, x):
		p0, p1, p2, p3 = self.backbone(x)
		p3 = self.adjust(p3)
		return p0, p1, p2, p3


class ScoreBranch(nn.Module):
	def __init__(self):
		self.score_branch = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 1),
			nn.Sigmoid()
			)

	def forward(self, x):
		return self.branch(x)

	def forward_sharp(self, x):
		raise NotImplementedError


class MaskBranch(nn.Module):
	def __init__(self):
		super(MaskBranch, self).__init__()
		self.mask_branch = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 64*64, kernel_size=1)
			)

	def forward(self, x):
		return self.mask_branch(x)

	def forward_sharp(self, x):
		raise NotImplementedError


def Correlation_func(self, s_f, t_f): # s_f-->search_feat, t_f-->target_feat
	t_f = t_f.reshape(-1, 1, t_f.size(2), t_f.size(3))
	s_f = s_f.reshape(1, -1, s_f.size(2), s_f.size(3))
	out = F.conv2d(out, t_f, group=t_f.size(0))
	out = out.shape(-1, s_f.size(1), out.size(2), out.size(3))
	return out