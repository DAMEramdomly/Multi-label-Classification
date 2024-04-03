import torch
import numpy as np
import time
import sklearn.metrics as metrics
from sklearn.metrics import average_precision_score, confusion_matrix
from wider import get_subsets
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt

def calc_average_precision(y_true, y_score):

	num_attr = y_true.size(0)
	aps = np.zeros(num_attr)

	for i in range(num_attr):
		true = torch.squeeze(y_true[i, :])
		score = torch.squeeze(y_score[i, :])

		true[true == -1.] = 0
		true = true.flatten()
		score = score.flatten()

		aps[i] = average_precision_score(true, score, average='macro')

	avg_pre = np.mean(aps)
	return avg_pre


def calc_acc_pr_f1(y_true, y_pred):
	num_attr = y_true.size(0)
	precision = np.zeros(num_attr)
	recall = np.zeros(num_attr)
	accuracy = np.zeros(num_attr)
	f1 = np.zeros(num_attr)

	for i in range(num_attr):

		true = torch.squeeze(y_true[i, :])
		pred = torch.squeeze(y_pred[i, :])

		threshold = 0.5
		pred = (pred > threshold).float()
		true[true == -1.] = 0

		true = true.flatten()
		pred = pred.flatten()

		precision[i] = metrics.precision_score(true, pred, zero_division=1, average='macro')
		recall[i] = metrics.recall_score(true, pred, zero_division=1, average='macro')
		accuracy[i] = metrics.accuracy_score(true, pred)
		f1[i] = metrics.f1_score(true, pred, average='macro')

	avg_pre = np.mean(precision)
	avg_recall = np.mean(recall)
	avg_acc = np.mean(accuracy)
	avg_f1 = np.mean(f1)

	return avg_pre, avg_recall, avg_acc, avg_f1

def eval_example(y_true, y_pred):
	N = y_true.shape[1]

	acc = 0.
	prec = 0.
	rec = 0.
	f1 = 0.

	for i in range(N):
		true_exam = y_true[:,i]		# column: labels for an example
		pred_exam = y_pred[:,i]

		temp = true_exam + pred_exam

		yi = true_exam.sum()	# number of attributes for i
		fi = pred_exam.sum()	# number of predicted attributes for i
		ui = (temp > 0).sum()	# temp == 1 or 2 means the union of attributes in yi and fi
		ii = (temp == 2).sum()	# temp == 2 means the intersection

		if ui != 0:
			acc += 1.0 * ii / ui
		if fi != 0:
			prec += 1.0 * ii / fi
		if yi != 0:
			rec += 1.0 * ii / yi

	acc /= N
	prec /= N
	rec /= N
	f1 = 2.0 * prec * rec / (prec + rec)
	return acc, prec, rec, f1

def validate(model, test_loader, epoch):
	print("validataing ... ")

	all_precisions, all_recalls, all_accuracies, all_f1_scores, all_aps = [], [], [], [], []

	for i, (img, labels) in enumerate(test_loader):
		labels = labels.cuda()
		img = img.cuda()

		with torch.no_grad():
			outputs = model(img)
			outputs = torch.sigmoid(outputs)


		aps = calc_average_precision(labels.cpu(), outputs.cpu())

		precision, recall, accuracy, f1 = calc_acc_pr_f1(labels.cpu(), outputs.cpu())


		all_aps.append(aps.mean())
		all_precisions.append(precision.mean())
		all_recalls.append(recall.mean())
		all_accuracies.append(accuracy.mean())
		all_f1_scores.append(f1.mean())

	mean_recall = np.mean(all_recalls)

	print("Average AP: {}".format(np.mean(all_aps)))
	print('Average F1-C: {}'.format(np.mean(all_f1_scores)))
	print('Average P-C: {}'.format(np.mean(all_precisions)))
	print('Average R-C: {}'.format(np.mean(all_recalls)))
	print('Average Accuracy: {}'.format(np.mean(all_accuracies)))

	print(f"Now the validation recall in validation is {mean_recall}")
	model.train()
	print('Validation finished ....')

	return mean_recall, epoch



def plot_and_save_confusion_matrices(y_true_all, y_pred_all, save_path):
    label_names = ['ILMD', 'IRS', 'ORS', 'RD']

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5), sharey=True)

    vmax = 600
    vmin = 0

    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    for label_index in range(4):
        y_true = y_true_all[:, label_index]
        y_pred = y_pred_all[:, label_index]

        conf_mat = confusion_matrix(y_true, y_pred)
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=axes[label_index],
                    annot_kws={"size": 20}, vmin=vmin, vmax=vmax,
                    cbar=label_index == 3, cbar_ax=None if label_index < 3 else cbar_ax)

        axes[label_index].set_title(f'Confusion Matrix for {label_names[label_index]}', fontsize=14)
        axes[label_index].set_xlabel('Predicted', fontsize=12)
        if label_index == 0:
            axes[label_index].set_ylabel('True', fontsize=12)

    plt.subplots_adjust(right=0.9)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'combined_confusion_matrices.png'))
    plt.close()



def test(model, test_loader, model_name):
	print("testing ... ")

	probs = torch.FloatTensor().cuda()
	gtruth = torch.FloatTensor().cuda()

	all_precisions, all_recalls, all_accuracies, all_f1_scores, all_aps = [], [], [], [], []


	for i, (img, labels) in enumerate(test_loader):
		labels = labels.cuda()
		img = img.cuda()

		with torch.no_grad():
			outputs = model(img)
			outputs = torch.sigmoid(outputs)
		probs = torch.cat((probs, outputs), dim=0)
		gtruth = torch.cat((gtruth, labels), dim=0)

		aps = calc_average_precision(labels.cpu(), outputs.cpu())
		precision, recall, accuracy, f1 = calc_acc_pr_f1(labels.cpu(), outputs.cpu())

		all_aps.append(aps.mean())
		all_precisions.append(precision.mean())
		all_recalls.append(recall.mean())
		all_accuracies.append(accuracy.mean())
		all_f1_scores.append(f1.mean())

	save_path = f"./results2/{model_name}"

	y_true = gtruth.cpu().numpy()  # 直接访问正确的维度
	y_pred = probs.cpu().numpy() > 0.5  # 同上
	plot_and_save_confusion_matrices(y_true, y_pred, save_path)


	results = {
		'mean Average Precision' : np.mean(all_aps),
		'Average Precision': np.mean(all_precisions),
		'Average Recall': np.mean(all_recalls),
		'Average Accuracy': np.mean(all_accuracies),
		'Average F1 Score': np.mean(all_f1_scores)
	}

	results_path = f"./results/{model_name}/test_results.json"
	os.makedirs(os.path.dirname(results_path), exist_ok=True)
	with open(results_path, 'w') as f:
		json.dump(results, f, indent=5)

	print("Test results saved to:", results_path)

	print("Average AP: {}".format(np.mean(all_aps) * 100))
	print('Average F1-C: {}'.format(np.mean(all_f1_scores) * 100))
	print('Average P-C: {}'.format(np.mean(all_precisions) * 100))
	print('Average R-C: {}'.format(np.mean(all_recalls) * 100))
	print('Average Accuracy: {}'.format(np.mean(all_accuracies) * 100))


	model.train()
	print('Test finished ....')



def test_original(model, test_loader):
	print("testing ... ")

	probs = torch.FloatTensor().cuda()
	gtruth = torch.FloatTensor().cuda()

	all_precisions, all_recalls, all_accuracies, all_f1_scores, all_aps = [], [], [], [], []

	for i, (img, labels) in enumerate(test_loader):
		labels = labels.cuda().unsqueeze(2).unsqueeze(3)
		img = img.cuda()

		with torch.no_grad():
			outputs = model(img)
			outputs = torch.sigmoid(outputs)

		probs = torch.cat((probs, outputs), dim=0)
		gtruth = torch.cat((gtruth, labels), dim=0)

		aps = calc_average_precision(labels.cpu(), outputs.cpu())
		precision, recall, accuracy, f1 = calc_acc_pr_f1(labels.cpu(), outputs.cpu())

		all_aps.append(aps.mean())
		all_precisions.append(precision.mean())
		all_recalls.append(recall.mean())
		all_accuracies.append(accuracy.mean())
		all_f1_scores.append(f1.mean())

if __name__ == "__main__":
	np.random.seed(0)  # 确保可重复性
	y_true = torch.tensor(np.random.randint(0, 2, (100, 5)).astype(float))  # 假设有100个样本，5个类别
	y_score = torch.tensor(np.random.rand(100, 5))  # 同样的形状，但是是随机的预测得分

	avg_pre = calc_average_precision(y_true, y_score)
	print("平均精确度（mAP）：", avg_pre)