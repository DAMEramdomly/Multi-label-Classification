import torch
from wider import get_subsets
from torch.autograd import Variable
from sklearn import metrics
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

def get_dataset(opts):
	trainset, valset, testset = get_subsets(opts["dataset_path"], opts["scale"])

	return trainset, valset, testset


def adjust_learning_rate(optimizer, epoch, args):

	#lr = args.learning_rate * (args.decay ** (epoch // args.stepsize))
	lr = args.learning_rate * (1 + math.cos(1.0 * epoch / args.epoch_max * math.pi)) / 2
	print("Current learning rate is: {:.5f}".format(lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def SigmoidCrossEntropyLoss(x, y):
	loss = 0.0
	batch = y.size(0)
	x = torch.sigmoid(x)
	for i in range(batch):
		temp = - (y[i] * (1 / (1 + (-x[i]).exp())).log() +
				  (1 - y[i]) * ((-x[i]).exp() / (1 + (-x[i]).exp())).log())
		loss += temp.sum()

	loss = loss / batch
	return loss


def Recall_loss(x, y):

	batch = y.size(0)
	recall = np.zeros(batch)
	x = torch.sigmoid(x)
	for i in range(batch):
		true = torch.squeeze(y[i, :]).cpu()
		pred = torch.squeeze(x[i, :]).cpu()

		threshold = 0.5
		pred = (pred > threshold).float()
		true[true == -1.] = 0

		true = true.flatten()
		pred = pred.flatten()

		if torch.sum(true).item() == 0 or torch.sum(pred).item() == 0:
			recall[i] = 0.0
		else:
			recall[i] = metrics.recall_score(true, pred, zero_division=0)

	avg_recall = np.mean(recall)
	loss = 1 - avg_recall
	return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss