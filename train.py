import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from dataloader.dataloader import Loader
from model.model_head import RSiamMask
from loss_metric.losses import ScoreLoss, MaskLoss


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="ignore/data/ytb_vos/train/", help="data path")
parser.add_argument("--b", type=int, default=32, help="Batch size")
parser.add_argument("--n_w", type=int, default=12, help="Num workers")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
parser.add_argument("--model", type=str, default="base", help="Choise model: 'base' or 'sharp'")
parser.add_argument("--w", type=str, default="default_weights", help="Weights name")
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
parser.add_argument("--lr_step", type=int, default=100, help="Learning rate scheduler step")
parser.add_argument("--gamma", type=int, default=0.9, help="Learning rate scheduler gamma")
parser.add_argument("--epochs", type=int, default=1000, help="Num epochs")
parser.add_argument("--tb", type=str, default=None, help="Tensorboard")
PARS = parser.parse_args()


# init tensorboard: --logdir=runs
if PARS.tb != None:
	writer = SummaryWriter()

# init model
device = torch.device(PARS.device if torch.cuda.is_available() else "cpu")
model = RSiamMask()
model = model.to(device)


# load weights
print("Load weights: ", "%s.pth" % PARS.w)
try:
	model.load_state_dict(torch.load('ignore/weights/%s.pth' % PARS.w), )#strict=False
except FileNotFoundError:
	print("!!!Create new weights!!!: ", "%s.pth" % PARS.w)
	pass


# init dataloader
train_loader = Loader(data_path=PARS.data, batch_size=PARS.b, num_workers=PARS.n_w)


# init optimizer and lr_scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=PARS.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=PARS.lr_step, gamma=PARS.gamma)


def main():
	for epoch in range(PARS.epochs):
		tb_iter = 0
		e_iter = 0
		e_loss = 0
		for i, data in enumerate(train_loader):
			e_iter += 1
			target, search, mask_label, score_labels = data
			target, search, mask_label = target.to(device), search.to(device), mask_label.to(device)
			large_score, medium_score, small_score = score_labels
			large_score, medium_score, small_score = large_score.to(device), medium_score.to(device), small_score.to(device)
			score_labels = large_score, medium_score, small_score
			pred_scores, pred_masks = model(target, search)
			score_loss = ScoreLoss(pred_scores, score_labels)
			mask_loss = MaskLoss(pred_masks, mask_label, score_labels)
			loss = score_loss + 36*mask_loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			e_loss += loss.item()
			if e_iter % 10 == 0:
				print("epoch: ", epoch, "iter: ", e_iter, "loss: ", e_loss/e_iter, "|||", loss.item())
				if PARS.tb != None:
					tb_iter += 1
					writer.add_scalars('%s_loss' % PARS.tb, {'train' : loss}, iter)
		torch.save(model.state_dict(), 'ignore/weights/%s.pth' % PARS.w)


if __name__ == '__main__':
	main()

