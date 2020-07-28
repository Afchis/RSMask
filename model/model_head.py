import torch
import torch.nn as nn

from .model_parts import *


class RSiamMask(nn.Module):
	def __init__(self, model='base'):
		super(RSiamMask, self).__init__()
		self.model = model
		self.backbone = Backbone()
		self.score_branch = ScoreBranch()
		self.mask_branch = MaskBranch()

	def _base_model_(self, target, search):
		'''
		TODO: 
		*** Mask Branch
		'''
		# Backbone:
		target_feats = self.backbone(target)
		target_feats = self.backbone.AvgPools(target_feats)
		search_feat = self.backbone(search)
		# Score Branch:
		pred_scores = self.score_branch(target_feats, search_feat)
		# Mask Branch:
		return pred_scores

	def _sharp_model_(self, target, search):
		raise NotImplementedError

	def forward(self, x):
		if self.model == 'base': 
			return self._base_model_(inputs)
		elif self.model == 'sharp':
			return self._sharp_model_(inputs)
		else
			print("Please choise model in parser: 'base' or 'sharp'")
			quit()
