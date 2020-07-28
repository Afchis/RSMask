import argparse

import torch

from dataloader.dataloader import Loader
from model.model_head import RSiamMask
from loss_metric.losses import ScoreLoss


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="ignore/data/ytb_vos/train/", help="data path")
parser.add_argument("--b", type=int, default=1, help="batch size")
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument("--model", type=str, default="base", help="Choise model: 'base' or 'sharp'")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--lr_step", type=int, default=3000, help="Learning rate scheduler step")
parser.add_argument("--gamma", type=int, default=0.9, help="Learning rate scheduler gamma")
parser.add_argument("--epochs", type=int, default=1000, help="Num epochs")
PARS = parser.parse_args()


# init model
device = torch.device(PARS.device if torch.cuda.is_available() else "cpu")
model = RSiamMask()
model = model.to(device)


# init dataloader
train_loader = Loader(data_path=PARS.data, batch_size=PARS.b)


# init optimizer and lr_scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=PARS.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=PARS.lr_step, gamma=PARS.gamma)


def main():
	for epoch in range(PARS.epochs):

		e_iter = 0
		e_loss = 0
		for i, data in enumerate(train_loader):
			e_iter += 1
			target, search, mask, score_labels = data
			target, search, mask = target.to(device), search.to(device), mask.to(device)
			large_score, medium_score, small_score = score_labels
			large_score, medium_score, small_score = large_score.to(device), medium_score.to(device), small_score.to(device)
			score_labels = large_score, medium_score, small_score
			pred_scores = model(target, search)
			loss = ScoreLoss(pred_scores, score_labels)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			e_loss += loss.item()
			if e_iter % 10 == 0:
				print("epoch: ", epoch, "iter: ", e_iter, "loss: ", e_loss/e_iter, "|||", loss.item())


if __name__ == '__main__':
	main()

