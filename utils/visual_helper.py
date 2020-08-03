import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms


to_pil = transforms.ToPILImage()


def VisualMaskHelper(pred_masks, score_labels):
	labels_large_scores, labels_medium_scores, labels_small_scores = score_labels
	pred_large_masks, pred_medium_masks, pred_small_masks = pred_masks

	labels_large_scores = labels_large_scores.permute(0, 2, 3, 1).reshape(-1)
	labels_medium_scores = labels_medium_scores.permute(0, 2, 3, 1).reshape(-1)
	labels_small_scores = labels_small_scores.permute(0, 2, 3, 1).reshape(-1)

	pred_large_masks = pred_large_masks.permute(0, 2, 3, 1).reshape(-1, 4096)
	pred_medium_masks = pred_medium_masks.permute(0, 2, 3, 1).reshape(-1, 4096)
	pred_small_masks = pred_small_masks.permute(0, 2, 3, 1).reshape(-1, 4096)

	pos_large = Variable(labels_large_scores.data.eq(1).nonzero().squeeze())
	pos_medium = Variable(labels_medium_scores.data.eq(1).nonzero().squeeze())
	pos_small = Variable(labels_small_scores.data.eq(1).nonzero().squeeze())
	if pos_large.nelement() != 0:
		preds = torch.index_select(pred_large_masks, 0, pos_large)
	elif pos_medium.nelement() != 0:
		preds = torch.index_select(pred_medium_masks, 0, pos_medium)
	elif pos_small.nelement() != 0:
		preds = torch.index_select(pred_small_masks, 0, pos_small)
	else:
		preds = torch.zeros([5, 4096])
	len_preds = preds.size(0)
	rand_mask = random.randint(0, len_preds-1)
	mask = preds[rand_mask].reshape(1, 1, 64, 64)
	return mask


def Visual(pred_scores, pred_masks, score_labels, mask_label, visual_iter, treshhold=True):
	mask_pred = VisualMaskHelper(pred_masks, score_labels)
	pred_large_scores, pred_medium_scores, pred_small_scores = pred_scores
	pred_large_scores = nn.UpsamplingBilinear2d(size=[25, 25])(pred_large_scores)
	pred_medium_scores = nn.UpsamplingBilinear2d(size=[25, 25])(pred_medium_scores)
	pred_large_scores, pred_medium_scores, pred_small_scores = to_pil(pred_large_scores[0, 0].cpu()), to_pil(pred_medium_scores[0, 0].cpu()), to_pil(pred_small_scores[0, 0].cpu())
	mask_label = to_pil(mask_label[0, 0].cpu())
	if treshhold == True:
		mask_pred = to_pil((mask_pred[0, 0] > 0.5).float().cpu())
	else:
		mask_pred = to_pil(mask_pred[0, 0].cpu())
	pred_large_scores.save("ignore/visual/%ipred_scores_large.png" % visual_iter)
	pred_medium_scores.save("ignore/visual/%ipred_scores_medium.png" % visual_iter)
	pred_small_scores.save("ignore/visual/%ipred_scores_small.png" % visual_iter)
	mask_label.save("ignore/visual/%imask_label.png" % visual_iter)
	mask_pred.save("ignore/visual/%imask_pred.png" % visual_iter)
	return print("Save visual: /ignore/visual/%iimgs.png" % visual_iter)

