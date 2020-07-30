import torch
import torch.nn as nn
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
	label = label.reshape(pred.shape)
	return F.binary_cross_entropy(pred, label).mean()

def ScoreLoss(preds, labels):
	pred_large, pred_medium, pred_small = preds
	label_large, label_medium, label_small = labels
	loss_large = cross_entropy_loss(pred_large, label_large)
	loss_medium = cross_entropy_loss(pred_medium, label_medium)
	loss_small = cross_entropy_loss(pred_small, label_small)
	score_loss = (loss_large + loss_medium + loss_small)/3
	return score_loss

def make_masks(mask_label, kernel_size, padding, corr_size):
	stride = (256 + 2 * padding - kernel_size)/(corr_size - 1)
	mask_label = F.pad(input=mask_label, pad=(padding, padding, padding, padding), mode='constant', value=0)
	tensors_list = list()
	for j in range(corr_size):
		for i in range(corr_size):
			ij = mask_label[:, :, int(j*stride):int(j*stride+kernel_size), int(i*stride):int(i*stride+kernel_size)]
			tensors_list.append(ij)		
	out_tensor = torch.stack(tensors_list)
	return out_tensor

def select_mask_logistic_loss(preds, mask_label, score_label, kernel_size=256, padding=128, corr_size=5):
	score_label = score_label.reshape(-1)
	pos = Variable(score_label.data.eq(1).nonzero().squeeze())
	if pos.nelement() == 0: return 0

	preds = preds.permute(0, 2, 3, 1).contiguous().reshape(-1, 1, 64, 64)
	preds = torch.index_select(preds, 0, pos)

	mask_labels = make_masks(mask_label, kernel_size=kernel_size, padding=padding, corr_size=corr_size)
	mask_labels = mask_labels.permute(1, 0, 2, 3, 4).contiguous().reshape(-1, 1, kernel_size, kernel_size)
	mask_labels = torch.index_select(mask_labels, 0, pos)

	if kernel_size != 64:
		preds = nn.UpsamplingBilinear2d(size=[kernel_size, kernel_size])(preds)
	return F.binary_cross_entropy(preds, mask_labels).mean()



	# weight = weight.view(-1)
	# pos = Variable(weight.data.eq(1).nonzero().squeeze())
	# if pos.nelement() == 0: return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0

	# p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
	# p_m = torch.index_select(p_m, 0, pos)

	# p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
	# p_m = p_m.view(-1, g_sz * g_sz)

	# mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)
	# mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)

	# mask_uf = torch.index_select(mask_uf, 0, pos)
	# afchi_label = mask_uf
	# afchi_mask = (p_m >= 0).float()
	# loss = F.soft_margin_loss(p_m, mask_uf)
	# return loss, afchi_label, afchi_mask

def MaskLoss(preds, mask_label, score_labels):
	pred_large, pred_medium, pred_small = preds
	label_large, label_medium, label_small = score_labels
	loss_large = select_mask_logistic_loss(pred_large, mask_label, label_large, kernel_size=256, padding=128, corr_size=5)
	loss_medium = select_mask_logistic_loss(pred_medium, mask_label, label_medium, kernel_size=128, padding=32, corr_size=17)
	loss_small = select_mask_logistic_loss(pred_small, mask_label, label_small, kernel_size=64, padding=0, corr_size=25)
	mask_loss = (loss_large + loss_medium + loss_small)/3
	return mask_loss


