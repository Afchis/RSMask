import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50 


def Correlation_func(t_f, s_f, padding=0): # s_f-->search_feat, t_f-->target_feat
	b = t_f.size(0)
	t_f = t_f.reshape(-1, 1, t_f.size(2), t_f.size(3))
	s_f = s_f.reshape(1, -1, s_f.size(2), s_f.size(3))
	out = F.conv2d(s_f, t_f, groups=t_f.size(0), padding=padding)
	out = out.reshape(b, -1, out.size(2), out.size(3))
	return out


class Backbone(nn.Module):
	def __init__(self):
		super(Backbone, self).__init__()
		self.backbone = resnet50(pretrained=True)
		self.adjust = nn.Conv2d(1024, 256, kernel_size=1)
		self.avgpool = nn.AvgPool2d(2, stride=2)

	def AvgPools(self, feat):
		whole = feat
		half = self.avgpool(feat)
		fourth = self.avgpool(half)
		return whole, half, fourth

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
		super(ScoreBranch, self).__init__()
		self.conv_target = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			)
		self.conv_search = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			)
		self.score_branch = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 1, kernel_size=1),
			nn.Sigmoid()
			)

	def _conv_target_(self, target_feats):
		large_feat, medium_feat, small_feat = target_feats
		large_feat = self.conv_target(large_feat)
		medium_feat = self.conv_target(medium_feat)
		small_feat = self.conv_target(small_feat)
		return large_feat, medium_feat, small_feat

	def _corr_(self, target_feats, search_feat):
		'''
		TODO:
		** do padding for medium_corr and check padding for large_corr:
			* for medium_corr padding need change score_labels.shape in dataloader.py
		'''
		large_feat, medium_feat, small_feat = target_feats
		large_corr = Correlation_func(large_feat, search_feat, padding=2)
		medium_corr = Correlation_func(medium_feat, search_feat)
		small_corr = Correlation_func(small_feat, search_feat)
		return large_corr, medium_corr, small_corr

	def _branch_(self, outs):
		large_out, medium_out, small_out = outs
		large_out = self.score_branch(large_out)
		medium_out = self.score_branch(medium_out)
		small_out = self.score_branch(small_out)
		# large_out = F.log_softmax(large_out, dim=1)
		# medium_out = F.log_softmax(medium_out, dim=1)
		# small_out = F.log_softmax(small_out, dim=1)
		return large_out, medium_out, small_out

	def forward(self, target_feats, search_feat):
		target_feats = self._conv_target_(target_feats)
		search_feat = self.conv_search(search_feat)
		outs = self._corr_(target_feats, search_feat)
		outs = self._branch_(outs)
		return outs

	def forward_sharp(self, x):
		raise NotImplementedError


class MaskBranch(nn.Module):
	def __init__(self):
		super(MaskBranch, self).__init__()
		self.conv_target = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			)
		self.conv_search = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			)
		self.mask_branch = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 64*64, kernel_size=1),
			nn.Sigmoid()
			)

	def _conv_target_(self, target_feats):
		large_feat, medium_feat, small_feat = target_feats
		large_feat = self.conv_target(large_feat)
		medium_feat = self.conv_target(medium_feat)
		small_feat = self.conv_target(small_feat)
		return large_feat, medium_feat, small_feat

	def _corr_(self, target_feats, search_feat):
		'''
		TODO:
		** do padding for medium_corr and check padding for large_corr:
			* for medium_corr padding need change score_labels.shape in dataloader.py
		'''
		large_feat, medium_feat, small_feat = target_feats
		large_corr = Correlation_func(large_feat, search_feat, padding=2)
		medium_corr = Correlation_func(medium_feat, search_feat)
		small_corr = Correlation_func(small_feat, search_feat)
		return large_corr, medium_corr, small_corr

	def _branch_(self, outs):
		large_out, medium_out, small_out = outs
		large_out = self.mask_branch(large_out)
		medium_out = self.mask_branch(medium_out)
		small_out = self.mask_branch(small_out)
		return large_out, medium_out, small_out

	def BaseModelReshape(self, x):
		raise NotImplementedError

	def forward(self, target_feats, search_feat):
		target_feats = self._conv_target_(target_feats)
		search_feat = self.conv_search(search_feat)
		outs = self._corr_(target_feats, search_feat)
		outs = self._branch_(outs)
		return outs

	def forward_sharp(self, x):
		raise NotImplementedError

