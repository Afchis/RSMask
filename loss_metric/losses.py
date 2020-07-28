import torch
import torch.nn.functional as F
from torch.autograd import Variable

def get_cls_loss(pred, label, select):
	if select.nelement() == 0: return pred.sum()*0.
	pred = torch.index_select(pred, 0, select)
	label = torch.index_select(label, 0, select)

	return F.nll_loss(pred, label.long())


def select_cross_entropy_loss(pred, label):
	# pred = pred.permute(0, 2, 3, 1)
	# print(pred.shape, label.shape)
	pred = pred.view(-1, 2)
	label = label.view(-1)
	pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
	neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()

	loss_pos = get_cls_loss(pred, label, pos)
	loss_neg = get_cls_loss(pred, label, neg)
	return loss_pos * 0.5 + loss_neg * 0.5

def cross_entropy_loss(pred, label):
	return F.binary_cross_entropy(pred, label)

def ScoreLoss(preds, labels):
	pred_large, pred_medium, pred_small = preds
	label_large, label_medium, label_small = labels
	loss_large = cross_entropy_loss(pred_large, label_large)
	loss_medium = cross_entropy_loss(pred_medium, label_medium)
	loss_small = cross_entropy_loss(pred_small, label_small)
	score_loss = (loss_large + loss_medium + loss_small)/3
	return score_loss
